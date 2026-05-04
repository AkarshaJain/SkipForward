"""Segmentation logic.

Combines per-second multimodal evidence into:

1) An ``ad_score(t)`` curve in [-inf, +inf] (z-scored, weighted sum).
2) A binary ad mask after smoothing + thresholding.
3) Connected ad regions, snapped to the nearest shot boundaries.
4) Final labelled segments covering the entire timeline:
   ``core_content`` / ``ad`` / ``intro`` / ``outro`` / ``silence`` / ``filler``.

The label assignment is rule-based but every rule is interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing, binary_opening

from . import config
from .fusion import FusedFeatures


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float
    notes: dict | None = None

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.end - self.start, 3),
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            **({"notes": self.notes} if self.notes else {}),
        }


def _smooth(arr: np.ndarray, sigma_sec: float) -> np.ndarray:
    sigma = max(sigma_sec / 2.355, 0.5)  # FWHM → sigma
    return gaussian_filter1d(arr, sigma=sigma, mode="nearest")


def compute_ad_score(features: FusedFeatures,
                      weights: dict[str, float] = None) -> np.ndarray:
    """Weighted z-score sum, smoothed across ``SMOOTHING_WINDOW_SEC``."""
    weights = weights or config.SCORE_WEIGHTS
    z_by_channel = {c: features.z[:, i] for i, c in enumerate(features.channels)}

    score = np.zeros(features.z.shape[0], dtype=np.float64)
    for ch, w in weights.items():
        if ch not in z_by_channel:
            continue
        score += w * z_by_channel[ch]

    # If the audio_rms channel is all zeros (extremely rare), avoid bias.
    return _smooth(score, config.SMOOTHING_WINDOW_SEC)


# ---------------------------------------------------------------------------
# Mask → contiguous regions
# ---------------------------------------------------------------------------

def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _snap_to_shots(start: float, end: float, shots: np.ndarray,
                    max_snap_sec: float = 4.0) -> tuple[float, float]:
    """Snap a region's edges to the closest shot boundary if within ``max_snap_sec``."""
    if shots.size == 0:
        return start, end
    s_idx = int(np.argmin(np.abs(shots - start)))
    e_idx = int(np.argmin(np.abs(shots - end)))
    if abs(shots[s_idx] - start) <= max_snap_sec:
        start = float(shots[s_idx])
    if abs(shots[e_idx] - end) <= max_snap_sec:
        end = float(shots[e_idx])
    return start, end


def _normalize_score(score: np.ndarray) -> np.ndarray:
    """Map score to [0, 1] using min/max with a small buffer."""
    if score.size == 0:
        return score
    lo, hi = float(np.percentile(score, 2)), float(np.percentile(score, 98))
    if hi - lo < 1e-6:
        return np.zeros_like(score)
    out = (score - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Multi-modality consensus
# ---------------------------------------------------------------------------

# Channels we group into independent modalities. A second only counts as
# "ad-like" if at least N modalities agree (precision filter).
_MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "visual":      ("shot_rate", "saturation", "motion", "edge_density"),
    "audio":       ("audio_rms", "spectral_flux", "music_likeness", "silence"),
    "speech":      ("ad_keyword", "speech_density", "transcript_garble"),
    "cross":       ("local_outlierness",),
}


def _modality_consensus(features: FusedFeatures) -> np.ndarray:
    """Return per-second integer count of modalities agreeing on 'ad-like'.

    A modality "votes" when at least one of its z-scored channels crosses
    ``MODALITY_VOTE_Z`` (with a sign that means 'more ad-like'). This is a
    deliberately weak per-modality test; the strength comes from requiring
    several modalities to agree at the same time.
    """
    z_by_channel = {c: features.z[:, i] for i, c in enumerate(features.channels)}
    n = features.z.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    # Smooth modality votes over a few seconds so a transient blip in one
    # channel doesn't flicker the consensus on/off.
    sigma = max(config.SMOOTHING_WINDOW_SEC / 2.355, 1.0)

    votes = np.zeros(n, dtype=np.int32)
    for _name, channels in _MODALITY_GROUPS.items():
        modality_z = np.zeros(n, dtype=np.float64)
        for ch in channels:
            if ch not in z_by_channel:
                continue
            sign = -1.0 if ch == "audio_rms" and False else 1.0  # audio_rms higher = louder = ad-like
            # 'silence' channel is positive when ad-like. 'audio_rms' is
            # also higher when louder which often correlates with ads.
            # All other channels are positive-when-ad-like by construction.
            modality_z = np.maximum(modality_z, sign * z_by_channel[ch])
        modality_z = gaussian_filter1d(modality_z, sigma=sigma, mode="nearest")
        votes += (modality_z >= config.MODALITY_VOTE_Z).astype(np.int32)
    return votes


# ---------------------------------------------------------------------------
# Public entry: produce labelled segments
# ---------------------------------------------------------------------------

def segment(features: FusedFeatures,
            *,
            shot_times: np.ndarray,
            duration_sec: float,
            threshold: float = config.AD_SCORE_THRESHOLD,
            silence_intervals: Iterable[tuple[float, float]] = (),
            splice_pairs: Iterable[tuple[float, float, float]] = (),
            ) -> tuple[list[Segment], dict]:
    """Run the full segmentation. Returns (segments, debug_info)."""

    score = compute_ad_score(features)
    score_norm = _normalize_score(score)
    mask = score_norm >= threshold

    # Precision filter (optional): require at least N modalities to agree
    # before we accept a second as ad-like. Only active when >= 2 -- with
    # the default of 1 the filter would erroneously drop seconds where the
    # *fused* score crossed threshold from many small contributions but
    # no single channel reached the per-modality vote z.
    if config.MIN_MODALITY_CONSENSUS >= 2:
        consensus = _modality_consensus(features)
        mask = mask & (consensus >= config.MIN_MODALITY_CONSENSUS)
    else:
        consensus = np.zeros(features.z.shape[0], dtype=np.int32)

    # Add splice-pair regions to the mask so long ads bracketed by
    # cut+silence splice points are captured even when their content is
    # statistically similar to the surrounding video.
    for a, b, _conf in splice_pairs:
        ai = max(0, int(round(a)))
        bi = min(len(mask), int(round(b)))
        if bi > ai:
            mask[ai:bi] = True

    # Morphological smoothing on the binary mask:
    #   - close gaps shorter than MERGE_GAP_SEC
    #   - drop runs shorter than MIN_NONCONTENT_DURATION_SEC
    close_w = max(int(config.MERGE_GAP_SEC), 1)
    open_w = max(int(config.MIN_NONCONTENT_DURATION_SEC), 1)
    mask = binary_closing(mask, structure=np.ones(close_w))
    mask = binary_opening(mask, structure=np.ones(open_w))

    runs = _runs_of_true(mask.astype(bool))

    # Helper: lookup confidence for a splice-pair containing the given range
    splice_lookup = list(splice_pairs)

    def _splice_conf(a: int, b: int) -> float:
        for sa, sb, sc in splice_lookup:
            if a >= int(round(sa)) - 1 and b <= int(round(sb)) + 1:
                return float(sc)
        return 0.0

    # Build raw non-content regions (label = "ad" tentatively, will be reclassified)
    n = features.z.shape[0]
    nc_regions: list[tuple[float, float, float]] = []
    for a, b in runs:
        a_t, b_t = float(a), float(b)
        a_t, b_t = _snap_to_shots(a_t, b_t, shot_times,
                                   max_snap_sec=config.MERGE_GAP_SEC)
        if b_t - a_t < config.MIN_NONCONTENT_DURATION_SEC:
            continue
        # Confidence: average score within region, OR splice-pair confidence
        # if the region was added by a splice pair (score might be low).
        score_conf = float(score_norm[a:b].mean()) if b > a else 0.0
        sp_conf = _splice_conf(a, b)
        conf = max(score_conf, sp_conf)
        nc_regions.append((a_t, b_t, conf))

    # Add long silences as their own non-content regions only if they are
    # sustained -- natural lectures/podcasts have many short pauses.
    # Also cap each silence at SILENCE_MAX_SEGMENT_SEC so a video that
    # opens with several minutes of quiet music doesn't produce one huge
    # silence band.
    sil_regions: list[tuple[float, float]] = []
    sil_min = config.SILENCE_AS_SEGMENT_MIN_SEC
    sil_max = config.SILENCE_MAX_SEGMENT_SEC
    for s, e in silence_intervals:
        if e - s < sil_min:
            continue
        e_capped = min(e, s + sil_max)
        sil_regions.append((float(s), float(e_capped)))

    # Build labelled segments end-to-end ----------------------------------
    segments = _label_and_fill(
        nc_regions=nc_regions,
        sil_regions=sil_regions,
        duration=duration_sec,
        intro_window=config.INTRO_WINDOW_SEC,
        outro_window=config.OUTRO_WINDOW_SEC,
    )

    # Sub-type reclassification: turn generic 'ad'/'silence' segments into
    # 'transition' or 'holding_screen' when their multimodal signature
    # matches. This brings previously-dead labels in the taxonomy to life.
    segments = _reclassify_subtypes(segments, features=features)

    # Precision filters on "ad" segments specifically (intro / outro /
    # silence have their own labelling logic and aren't subject to these).
    segments = _filter_ad_segments(segments, features=features,
                                    splice_pairs=splice_lookup)

    debug = {
        "score": score.tolist(),
        "score_norm": score_norm.tolist(),
        "mask": mask.astype(int).tolist(),
        "consensus": consensus.tolist(),
        "threshold": threshold,
    }
    return segments, debug


# ---------------------------------------------------------------------------
# Sub-type reclassification
# ---------------------------------------------------------------------------

def _region_stat(arr: np.ndarray | None, a: int, b: int) -> float:
    if arr is None or arr.size == 0:
        return 0.0
    a = max(0, min(a, arr.size))
    b = max(a, min(b, arr.size))
    if b <= a:
        return 0.0
    return float(np.mean(arr[a:b]))


def _reclassify_subtypes(segments: list[Segment],
                          *, features: FusedFeatures) -> list[Segment]:
    """Re-label generic non-content segments with more specific sub-types.

    Rules (each strict, only one rule matches per segment):

    * **transition** -- short (TRANSITION_MIN_SEC..TRANSITION_MAX_SEC)
      segment that is mostly black AND mostly silent. Typical for
      hard scene cuts with a fade-to-black bridge.
    * **holding_screen** -- long, visually static (very low motion),
      silent, with no detectable speech. Typical for "starting soon"
      / "be right back" cards.

    Segments that match neither rule keep their original label.
    """
    raw = features.raw
    z_by = {c: features.z[:, i] for i, c in enumerate(features.channels)}

    black = raw.get("black_frame")
    silence = raw.get("silence")
    speech_density = raw.get("speech_density")
    motion_z = z_by.get("motion")

    eligible = {config.LABEL_AD, config.LABEL_SILENCE,
                config.LABEL_INTRO, config.LABEL_OUTRO}

    out: list[Segment] = []
    for seg in segments:
        if seg.label not in eligible:
            out.append(seg)
            continue

        a, b = int(round(seg.start)), int(round(seg.end))
        dur = seg.end - seg.start

        black_ratio = _region_stat(black, a, b)
        silence_ratio = _region_stat(silence, a, b)
        speech_ratio = _region_stat(speech_density, a, b)
        motion_z_mean = _region_stat(motion_z, a, b)

        # --- Transition -------------------------------------------------
        if (config.TRANSITION_MIN_SEC <= dur <= config.TRANSITION_MAX_SEC
                and black_ratio >= config.TRANSITION_BLACK_RATIO
                and silence_ratio >= config.TRANSITION_SILENCE_RATIO):
            notes = dict(seg.notes or {})
            notes.update({
                "subtype_rule": "black+silence bridge",
                "black_ratio": round(black_ratio, 3),
                "silence_ratio": round(silence_ratio, 3),
            })
            out.append(Segment(seg.start, seg.end, config.LABEL_TRANSITION,
                                confidence=max(seg.confidence, 0.85),
                                notes=notes))
            continue

        # --- Holding screen --------------------------------------------
        if (dur >= config.HOLDING_MIN_SEC
                and motion_z_mean <= config.HOLDING_MAX_MOTION_Z
                and speech_ratio <= config.HOLDING_MAX_SPEECH
                and silence_ratio >= config.HOLDING_MIN_SILENCE):
            notes = dict(seg.notes or {})
            notes.update({
                "subtype_rule": "static+silent+no-speech",
                "motion_z_mean": round(motion_z_mean, 3),
                "silence_ratio": round(silence_ratio, 3),
                "speech_ratio": round(speech_ratio, 3),
            })
            out.append(Segment(seg.start, seg.end, config.LABEL_HOLDING,
                                confidence=max(seg.confidence, 0.80),
                                notes=notes))
            continue

        out.append(seg)
    return out


# ---------------------------------------------------------------------------
# "ad"-only precision filter
# ---------------------------------------------------------------------------

def _filter_ad_segments(segments: list[Segment],
                         *, features: FusedFeatures,
                         splice_pairs: list[tuple[float, float, float]],
                         ) -> list[Segment]:
    """Apply post-hoc precision filters to ``ad``-labelled segments only.

    Filters (each independently configurable in ``pipeline/config.py``):

    1. **Confidence floor.** Drop ads with mean score-confidence below
       ``MIN_AD_CONFIDENCE``. Real ads have high confidence; smoothing-
       artifact regions don't.
    2. **Duration cap.** Truncate ads longer than ``MAX_AD_DURATION_SEC``
       (real ads in the dataset are <= 120s; longer is suspect).
    3. **Long-ad confirmation.** An ad longer than ``LONG_AD_CONFIRM_SEC``
       must have at least one independent confirmation: ad-keyword hit,
       splice-pair endpoint within range, or a transcript_garble spike.
       Without confirmation it's downgraded to ``core_content``.

    Filtered-out ads become ``core_content`` (rather than disappearing
    from the timeline) so the wall-to-wall coverage invariant is kept.
    Adjacent core_content segments are merged afterwards in postprocess.
    """
    raw = features.raw
    z_by = {c: features.z[:, i] for i, c in enumerate(features.channels)}
    ad_keyword = raw.get("ad_keyword")
    transcript_garble_z = z_by.get("transcript_garble")

    splice_endpoints: list[float] = []
    for sa, sb, _sc in splice_pairs:
        splice_endpoints.append(float(sa))
        splice_endpoints.append(float(sb))

    def _has_keyword_hit(a: int, b: int) -> bool:
        return ad_keyword is not None and ad_keyword[a:b].sum() > 0

    def _has_garble_spike(a: int, b: int) -> bool:
        if transcript_garble_z is None or transcript_garble_z.size == 0:
            return False
        return float(transcript_garble_z[a:b].max()) >= 1.5

    def _has_splice_endpoint(s: float, e: float) -> bool:
        for ep in splice_endpoints:
            if s - 5.0 <= ep <= e + 5.0:
                return True
        return False

    out: list[Segment] = []
    for seg in segments:
        if seg.label != config.LABEL_AD:
            out.append(seg)
            continue

        a, b = int(round(seg.start)), int(round(seg.end))
        dur = seg.end - seg.start

        # Filter 1: confidence floor
        if seg.confidence < config.MIN_AD_CONFIDENCE:
            out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                confidence=0.85))
            continue

        # Filter 2: hard duration cap. Truncate to MAX_AD_DURATION_SEC and
        # leave the tail as core_content (postprocess will absorb it into
        # adjacent core regions).
        cap = config.MAX_AD_DURATION_SEC
        tail_seg: Segment | None = None
        if cap and dur > cap:
            tail_seg = Segment(seg.start + cap, seg.end, config.LABEL_CORE,
                                confidence=0.85)
            seg = Segment(seg.start, seg.start + cap, seg.label,
                           confidence=seg.confidence, notes=seg.notes)
            b = int(round(seg.end))
            dur = cap

        # Filter 3: long ads must be confirmed by an independent signal.
        if dur >= config.LONG_AD_CONFIRM_SEC:
            confirmed = (_has_keyword_hit(a, b)
                          or _has_garble_spike(a, b)
                          or _has_splice_endpoint(seg.start, seg.end))
            if not confirmed:
                out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                    confidence=0.85))
                if tail_seg is not None:
                    out.append(tail_seg)
                continue

        out.append(seg)
        if tail_seg is not None:
            out.append(tail_seg)
    return out


def _label_and_fill(*,
                     nc_regions: list[tuple[float, float, float]],
                     sil_regions: list[tuple[float, float]],
                     duration: float,
                     intro_window: float,
                     outro_window: float) -> list[Segment]:
    """Turn region lists into a wall-to-wall labelled timeline."""
    # Combine non-content and silence regions, deduplicate overlaps (silence
    # is the more specific label and wins over generic ad-likeness).
    combined: list[tuple[float, float, str, float]] = []
    for s, e, c in nc_regions:
        combined.append((s, e, "__candidate__", c))
    for s, e in sil_regions:
        combined.append((s, e, config.LABEL_SILENCE, 0.95))

    combined.sort()
    merged: list[tuple[float, float, str, float]] = []
    for seg in combined:
        s, e, lbl, c = seg
        if merged and s <= merged[-1][1] + 0.01:
            ps, pe, plbl, pc = merged[-1]
            # If overlapping, prefer the more specific label
            new_label = plbl if plbl != "__candidate__" else lbl
            new_conf = max(pc, c)
            merged[-1] = (ps, max(pe, e), new_label, new_conf)
        else:
            merged.append(seg)

    # Reclassify __candidate__ as intro/outro/ad based on position
    relabelled: list[tuple[float, float, str, float]] = []
    for s, e, lbl, c in merged:
        if lbl != "__candidate__":
            relabelled.append((s, e, lbl, c))
            continue
        if s < intro_window:
            relabelled.append((s, e, config.LABEL_INTRO, c))
        elif e > duration - outro_window:
            relabelled.append((s, e, config.LABEL_OUTRO, c))
        else:
            relabelled.append((s, e, config.LABEL_AD, c))

    # Now fill gaps between non-content segments with core_content
    result: list[Segment] = []
    cursor = 0.0
    for s, e, lbl, c in relabelled:
        s = max(s, cursor)
        if s > cursor:
            result.append(Segment(cursor, s, config.LABEL_CORE,
                                  confidence=0.85))
        if e > s:
            result.append(Segment(s, min(e, duration), lbl, confidence=c))
            cursor = min(e, duration)
    if cursor < duration:
        result.append(Segment(cursor, duration, config.LABEL_CORE,
                              confidence=0.85))

    # Drop any zero-length segments (rounding artefacts)
    return [seg for seg in result if seg.end - seg.start > 0.05]
