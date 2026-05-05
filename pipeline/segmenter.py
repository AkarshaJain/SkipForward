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
        recovery_eligible = consensus >= config.MIN_MODALITY_CONSENSUS
    else:
        consensus = np.zeros(features.z.shape[0], dtype=np.int32)
        recovery_eligible = np.ones(features.z.shape[0], dtype=bool)

    # Add splice-pair regions to the mask so long ads bracketed by
    # cut+silence splice points are captured even when their content is
    # statistically similar to the surrounding video.
    splice_mask_cap = float(config.SPLICE_MASK_MAX_SPAN_SEC)
    for a, b, _conf in splice_pairs:
        if float(b) - float(a) > splice_mask_cap:
            continue
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
    # Short splice-pair corridors (below the morphology survivability cutoff) get
    # wiped by opening alone; graft them back so brief insert detectors remain.
    splice_rearm_lim = float(config.MIN_NONCONTENT_DURATION_SEC)
    for a, b, _conf in splice_pairs:
        ai = max(0, int(round(a)))
        bi = min(len(mask), int(round(b)))
        if bi > ai and (bi - ai) <= splice_rearm_lim:
            mask[ai:bi] = True

    runs = _runs_of_true(mask.astype(bool))

    # Helper: lookup confidence for a splice-pair containing the given range
    splice_lookup = list(splice_pairs)

    def _max_splice_conf_overlapping(ai: int, bi: int) -> float:
        """Best splice-pair confidence among pairs that intersect integer [ai, bi)."""
        best = 0.0
        hi = max(ai + 1, bi)
        for sa, sb, sc in splice_lookup:
            la = max(0, int(round(sa)))
            rb = max(la + 1, min(len(mask), int(round(sb))))
            if max(ai, la) < min(hi, rb):
                best = max(best, float(sc))
        return best

    def _splice_conf_inner(a: int, b: int) -> float:
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
        dur_raw = b_t - a_t
        splice_bracket_conf = _max_splice_conf_overlapping(int(a), int(b))
        if dur_raw < float(config.MIN_NONCONTENT_DURATION_SEC):
            # Splice-only inserts can be shorter than the morphology floor; keep
            # them when an accepted splice-pair clearly brackets the run.
            if splice_bracket_conf < 0.32 or dur_raw < 10.0:
                continue
        # Confidence: average score within region, OR splice-pair confidence
        # if the region was added by a splice pair (score might be low).
        score_conf = float(score_norm[a:b].mean()) if b > a else 0.0
        sp_conf = max(_splice_conf_inner(a, b), splice_bracket_conf)
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
        score_norm=score_norm,
    )

    # Sub-type reclassification: turn generic 'ad'/'silence' segments into
    # 'transition' or 'holding_screen' when their multimodal signature
    # matches. This brings previously-dead labels in the taxonomy to life.
    segments = _reclassify_subtypes(segments, features=features)

    # Demote phantom early-region "ads" entirely inside/adjacent intro window,
    # and weak positional intro/outro bands.
    segments = _demote_spurious_intro_ads(segments)

    segments = _apply_ad_spacing_and_recovery(
        segments,
        features=features,
        splice_pairs=splice_lookup,
        score_norm=score_norm,
        recovery_eligible=recovery_eligible,
        shot_times=shot_times,
    )

    # Precision filters on "ad" segments specifically (intro / outro /
    # silence have their own labelling logic and aren't subject to these).
    segments = _filter_ad_segments(
        segments, features=features, splice_pairs=splice_lookup,
        score_norm=score_norm,
    )

    segments = _extend_ad_tails_toward_splice(
        segments,
        duration_sec=duration_sec,
        score_norm=score_norm,
        splice_pairs=splice_lookup,
        shot_times=shot_times,
    )

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
                         score_norm: np.ndarray | None = None,
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
    4. **Mid-roll precision.** Short / mid ads that lack strict secondary
       evidence (high-conf splice, keyword, speech garble, sustained music or
       anomaly) must exceed a mean fused-score floor; otherwise lecture
       texture is demoted to ``core_content``.

    Filtered-out ads become ``core_content`` (rather than disappearing
    from the timeline) so the wall-to-wall coverage invariant is kept.
    Adjacent core_content segments are merged afterwards in postprocess.
    """
    raw = features.raw
    z_by = {c: features.z[:, i] for i, c in enumerate(features.channels)}
    ad_keyword = raw.get("ad_keyword")
    transcript_garble_z = z_by.get("transcript_garble")
    music_z_fb = z_by.get("music_likeness")
    outlier_z_fb = z_by.get("local_outlierness")

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

        # Recovered candidates need a pragmatic minimum duration; many FPs land
        # at ~20-26 s whereas real placements in labelled data are slightly longer.
        if (seg.notes or {}).get("recovered_peak"):
            if dur < float(config.MIN_RECOVERED_AD_DURATION_SEC):
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

        # Filter 4: mid-roll mask texture — without strict secondary evidence,
        # the fused region must ride clearly above baseline on norm score.
        if (score_norm is not None and score_norm.size > 0
                and dur < float(config.LONG_AD_CONFIRM_SEC) + 32.0):
            strict_ev = _secondary_ad_evidence(
                a, b, splice_pairs=splice_pairs, keyword=ad_keyword,
                garble_z=transcript_garble_z,
                music_z=music_z_fb, outlier_z=outlier_z_fb,
                splice_pair_min_conf=float(
                    config.STRICT_SECONDARY_SPLICE_MIN_CONF))
            aa = max(0, min(a, score_norm.size - 1))
            bb = max(aa + 1, min(b, score_norm.size))
            mu = float(score_norm[aa:bb].mean()) if bb > aa else 0.0
            floor_mu = float(
                config.MIDROLL_AD_MEAN_FLOOR_WITHOUT_STRICT_EVIDENCE)
            if (seg.notes or {}).get("recovered_peak"):
                floor_mu -= float(
                    config.MIDROLL_AD_MEAN_FLOOR_RECOVERED_PEAK_DELTA)
            if not strict_ev and mu < floor_mu:
                out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                    confidence=0.85))
                if tail_seg is not None:
                    out.append(tail_seg)
                continue

        out.append(seg)
        if tail_seg is not None:
            out.append(tail_seg)
    return out


def _extend_ad_tails_toward_splice(
        segments: list[Segment],
        *,
        duration_sec: float,
        score_norm: np.ndarray,
        splice_pairs: list[tuple[float, float, float]],
        shot_times: np.ndarray,
        ) -> list[Segment]:
    """Where recovery stops short of a strong splice bookend, lengthen the ad."""
    if not segments or score_norm.size == 0:
        return segments

    min_sc = float(config.AD_TAIL_EXTEND_SPLICE_ENDPOINT_MIN_PAIR_CONF)
    splice_eps: list[float] = []
    for sa, sb, sc in splice_pairs:
        if float(sc) < min_sc:
            continue
        splice_eps.extend([float(sa), float(sb)])
    splice_eps = sorted(set(splice_eps))
    if not splice_eps:
        return segments

    ordered = sorted(segments, key=lambda s: s.start)
    mut: list[Segment] = [
        Segment(s.start, s.end, s.label, s.confidence,
                dict(s.notes) if s.notes else None)
        for s in ordered]

    hard_cap = float(config.MAX_AD_DURATION_SEC or 99999.0)

    for i, seg in enumerate(mut):
        if seg.label != config.LABEL_AD:
            continue
        notes = dict(seg.notes or {})
        if not notes.get("recovered_peak"):
            continue

        hi = float(seg.end)
        j = i + 1

        # Wall-to-wall timelines put the trailing ``core_content`` flush with the
        # ad ``end``. Using ``next_segment.start`` as a hard cap would freeze
        # extensions entirely; bounded forward expansion + splice/mean guards
        # are safer (test_001 first insert vs splice ~210).
        forward_cap = hi + float(config.AD_TAIL_EXTEND_MAX_FORWARD_SEC)
        dur_cap = float(seg.start) + hard_cap
        cap_end = min(forward_cap, dur_cap, float(duration_sec))
        if j < len(mut) and mut[j].label == config.LABEL_AD:
            cap_end = min(cap_end, float(mut[j].start))

        best_ep = hi
        for ep in splice_eps:
            if ep <= hi + 2.5:
                continue
            if ep > cap_end + 1e-6:
                break
            ia = max(0, int(round(hi)))
            ib = min(score_norm.size, int(round(ep)))
            if ib <= ia + 2:
                continue
            mu_chunk = float(score_norm[ia:ib].mean())
            chunk_floor = float(config.AD_TAIL_EXTEND_MIN_CHUNK_MEAN_NORM)
            if notes.get("recovered_peak"):
                chunk_floor -= 0.09
            if mu_chunk < chunk_floor:
                trust_ep = False
                hi_trust = float(
                    config.AD_TAIL_HIGH_TRUST_SPLICE_PAIR_CONF)
                for sa, sb, sc in splice_pairs:
                    if float(sc) < hi_trust:
                        continue
                    fsb = float(sb)
                    fsa = float(sa)
                    if abs(ep - fsb) <= 2.0 or abs(ep - fsa) <= 2.0:
                        trust_ep = True
                        break
                if not trust_ep:
                    continue
            best_ep = max(best_ep, ep)

        if best_ep <= hi + 2.5:
            continue

        _, snapped_e = _snap_to_shots(seg.start, best_ep, shot_times,
                                      max_snap_sec=7.0)
        new_end = float(snapped_e)
        if new_end <= hi + 2.5:
            new_end = best_ep
        new_end = min(new_end, cap_end)
        if new_end <= hi + 2.5:
            continue

        aa = max(0, int(round(seg.start)))
        bb = max(aa + 1, min(score_norm.size, int(round(new_end))))
        mn_full = float(score_norm[aa:bb].mean())
        notes["tail_extended_to_splice"] = True
        mut[i] = Segment(float(seg.start), new_end, seg.label,
                          confidence=max(float(seg.confidence), mn_full),
                          notes=notes)

        if j < len(mut) and mut[j].label == config.LABEL_CORE:
            nxt = mut[j]
            if nxt.start < new_end - 0.05:
                n_notes = dict(nxt.notes) if nxt.notes else None
                mut[j] = Segment(new_end, nxt.end, nxt.label,
                                  nxt.confidence, n_notes)

    mut.sort(key=lambda s: s.start)
    return mut


def _merge_nearby_ads_through_tiny_core(
        segments: list[Segment],
        ) -> list[Segment]:
    """Fuse `[ad][short core][ad]` when morphology splits a single placement."""
    if len(segments) < 3:
        return segments
    gap_max = float(config.MIDROLL_NEAR_AD_GAP_MERGE_CORE_MAX_SEC)
    ea_max = float(config.MIDROLL_NEAR_AD_MERGE_EACH_MAX_SEC)
    span_cap = float(config.MIDROLL_NEAR_AD_MERGED_SPAN_CAP_SEC)

    cur = list(segments)
    while len(cur) >= 3:
        ordered = sorted(cur, key=lambda s: s.start)
        new_out: list[Segment] = []
        did_fuse = False
        i = 0
        while i < len(ordered):
            if i + 2 >= len(ordered):
                new_out.extend(ordered[i:])
                break
            a0, mid, b0 = ordered[i], ordered[i + 1], ordered[i + 2]
            if not (a0.label == config.LABEL_AD and mid.label == config.LABEL_CORE
                    and b0.label == config.LABEL_AD):
                new_out.append(a0)
                i += 1
                continue

            gap = mid.end - mid.start
            da = a0.end - a0.start
            db = b0.end - b0.start
            merged_span = b0.end - a0.start

            if (gap > gap_max or da > ea_max or db > ea_max
                    or merged_span > span_cap):
                new_out.append(a0)
                i += 1
                continue

            notes_m: dict[str, object] = {}
            if a0.notes:
                notes_m.update(dict(a0.notes))
            if b0.notes:
                notes_m.update(dict(b0.notes))
            notes_m["merged_adjacent_core_gap_sec"] = round(gap, 3)
            if (a0.notes or {}).get("recovered_peak") or (
                    b0.notes or {}).get("recovered_peak"):
                notes_m["recovered_peak"] = True

            fused = Segment(
                float(a0.start), float(b0.end), config.LABEL_AD,
                confidence=max(float(a0.confidence), float(b0.confidence)),
                notes=notes_m,
            )
            new_out.append(fused)
            i += 3
            did_fuse = True
        cur = new_out
        if not did_fuse:
            break

    return cur


def _label_and_fill(*,
                     nc_regions: list[tuple[float, float, float]],
                     sil_regions: list[tuple[float, float]],
                     duration: float,
                     intro_window: float,
                     outro_window: float,
                     score_norm: np.ndarray | None,
                     ) -> list[Segment]:
    """Turn region lists into a wall-to-wall labelled timeline."""
    def _slice_conf(t_lo: float, t_hi: float, c_fallback: float) -> float:
        """Mean normalized score confidence on integer indices [floor(t_lo), ceil(t_hi))."""
        if score_norm is None or score_norm.size == 0 or t_hi <= t_lo:
            return c_fallback
        a = max(0, int(np.floor(t_lo)))
        b = max(a + 1, min(score_norm.size, int(np.ceil(t_hi))))
        if b <= a:
            return c_fallback
        return float(score_norm[a:b].mean())

    def _split_candidates(
            rs: float, re: float, rc: float,
            ) -> list[tuple[float, float, str, float]]:
        """Split merged __candidate__ by intro / mid-body / outro windows."""
        parts: list[tuple[float, float, str, float]] = []
        body_hi = duration - outro_window
        cur = rs
        if cur >= re:
            return parts

        # Solely in trailing outro window ⇒ outro label only (not "ad").
        if cur >= body_hi:
            conf_p = _slice_conf(cur, re, rc)
            parts.append((cur, re, config.LABEL_OUTRO, conf_p))
            return parts

        # Intro-zone prefix -------------------------------------------------
        if cur < intro_window and re > cur:
            split = min(re, intro_window)
            conf_p = _slice_conf(cur, split, rc)
            if split - cur >= 0.95:
                if cur >= config.INTRO_LATEST_LEGITIMATE_START_SEC:
                    parts.append((cur, split, config.LABEL_AD, conf_p))
                else:
                    parts.append((cur, split, config.LABEL_INTRO, conf_p))
            cur = split

        # Mid‑body ---------------------------------------------------------
        if cur < body_hi and re > cur:
            split = min(re, body_hi)
            if split - cur >= 0.05:
                conf_p = _slice_conf(cur, split, rc)
                parts.append((cur, split, config.LABEL_AD, conf_p))
            cur = split

        # Outro tail -------------------------------------------------------
        if cur < duration and re > cur:
            conf_p = _slice_conf(cur, re, rc)
            parts.append((cur, re, config.LABEL_OUTRO, conf_p))
        return parts

    # Combine non-content and silence regions --------------------------------
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
            # Prefer score/splice candidates over raw silence so a short quiet
            # tail inside a splice-bracketed insert does not erase the ad cue.
            if plbl == "__candidate__" or lbl == "__candidate__":
                new_label = "__candidate__"
            else:
                new_label = plbl if plbl != "__candidate__" else lbl
            new_conf = max(pc, c)
            merged[-1] = (ps, max(pe, e), new_label, new_conf)
        else:
            merged.append(seg)

    expanded: list[tuple[float, float, str, float]] = []
    for s, e, lbl, c in merged:
        if lbl == "__candidate__":
            expanded.extend(_split_candidates(s, e, c))
        else:
            expanded.append((s, e, lbl, c))

    # Optional: drop micro-slivers mistaken for positional intro/outro
    trimmed: list[tuple[float, float, str, float]] = []
    for part in expanded:
        s0, e0, lb, cf = part
        if lb in (config.LABEL_INTRO, config.LABEL_OUTRO) and (e0 - s0 < 5.0):
            continue  # disappears into surrounding core gaps
        trimmed.append((s0, e0, lb, cf))

    # Fill gaps -------------------------------------------------------------
    relabelled_sorted = sorted(trimmed)

    result: list[Segment] = []
    cursor = 0.0
    for s, e, lbl, c in relabelled_sorted:
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

    # Drop zero-length artefacts
    return [seg for seg in result if seg.end - seg.start > 0.05]


def _demote_spurious_intro_ads(segments: list[Segment]) -> list[Segment]:
    """Remove weak positional bands that confuse viewers (e.g. 58-81 s bumps)."""

    iw = config.INTRO_WINDOW_SEC
    out: list[Segment] = []
    for seg in segments:
        if seg.label == config.LABEL_OUTRO:
            if (seg.confidence < config.MIN_INTRO_OUTRO_DISPLAY_CONFIDENCE
                    and seg.end - seg.start < config.SHORT_AD_NEED_EVIDENCE_MAX_SEC):
                out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                    confidence=0.85))
                continue
            out.append(seg)
            continue

        if seg.label == config.LABEL_INTRO:
            if (seg.start >= config.INTRO_LATEST_LEGITIMATE_START_SEC
                    and seg.confidence < config.MIN_INTRO_OUTRO_DISPLAY_CONFIDENCE):
                out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                    confidence=0.85))
                continue
            out.append(seg)
            continue

        if seg.label == config.LABEL_AD:
            dur = seg.end - seg.start
            if (dur < float(config.SHORT_AD_NEED_EVIDENCE_MAX_SEC)
                    and iw - 25.0 < seg.start <
                    float(config.NEAR_INTRO_VOLATILE_SHORT_AD_DEMOTE_MAX_START_SEC)
                    and seg.start >= config.INTRO_LATEST_LEGITIMATE_START_SEC
                    and seg.confidence < config.MIN_INTRO_OUTRO_DISPLAY_CONFIDENCE):
                out.append(Segment(seg.start, seg.end, config.LABEL_CORE,
                                    confidence=0.85))
                continue
            out.append(seg)
            continue

        out.append(seg)
    return out


def _secondary_ad_evidence(
        a: int, b: int, *,
        splice_pairs: list[tuple[float, float, float]],
        keyword: np.ndarray | None,
        garble_z: np.ndarray | None,
        music_z: np.ndarray | None,
        outlier_z: np.ndarray | None,
        splice_pair_min_conf: float = 0.0,
        ) -> bool:
    tol = 12.5
    strict = splice_pair_min_conf > 1e-9
    music_thr = (float(config.STRICT_SECONDARY_MUSIC_LIKELINESS_MEAN_Z)
                 if strict else float(config.SECONDARY_MUSIC_LIKELINESS_MEAN_Z))
    out_thr = (float(config.STRICT_SECONDARY_OUTLIER_MAX_Z)
               if strict else float(config.SECONDARY_OUTLIER_MAX_Z))

    s_f, e_f = float(a), float(b)
    seg_dur = max(0.0, e_f - s_f)
    min_ov = min(6.0, max(3.5, 0.20 * seg_dur)) if seg_dur > 1e-6 else 3.5

    def _splice_supports() -> bool:
        for sa, sb, sc in splice_pairs:
            if float(sc) < splice_pair_min_conf:
                continue
            sa_f, sb_f = float(sa), float(sb)
            if strict:
                lo = max(s_f, sa_f)
                hi_ = min(e_f, sb_f)
                if hi_ - lo >= min_ov:
                    return True
            else:
                for ep in (sa_f, sb_f):
                    if (s_f - tol) <= ep <= (e_f + tol):
                        return True
        return False

    if _splice_supports():
        return True
    if keyword is not None and keyword[a:b].sum() > 0:
        return True
    if garble_z is not None and float(garble_z[a:b].max()) >= 1.35:
        return True
    if music_z is not None and float(music_z[a:b].mean()) >= music_thr:
        return True
    if outlier_z is not None and float(outlier_z[a:b].max()) >= out_thr:
        return True
    return False


def _runs_above_threshold(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i, n = 0, flags.size
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flags[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _fuse_recovered_ads_wall_to_wall(
        baseline: list[Segment],
        recovered: list[Segment],
        ) -> list[Segment]:
    """Carve recovered ads out of baseline ``core_content`` so the timeline stays
    wall-to-wall without temporal overlap (recovered ads replace core time)."""

    recovered_ads = [r for r in recovered if r.label == config.LABEL_AD]
    base_sorted = sorted(baseline, key=lambda s: s.start)
    if not recovered_ads:
        return base_sorted

    def _merge_overlapping_ad_spans(
            spans: list[tuple[float, float, float, dict | None]],
            ) -> list[tuple[float, float, float, dict | None]]:
        spans.sort(key=lambda t: t[0])
        blocks: list[list] = []
        for a, b, conf, notes in spans:
            if not blocks or a > blocks[-1][1] + 0.05:
                blocks.append([a, b, conf, notes])
                continue
            blocks[-1][1] = max(blocks[-1][1], b)
            blocks[-1][2] = max(blocks[-1][2], conf)
            if notes:
                prev_n = blocks[-1][3]
                merge_n = {**(prev_n or {}), **notes}
                blocks[-1][3] = merge_n
        return [
            (float(x[0]), float(x[1]), float(x[2]), x[3])
            for x in blocks
        ]

    rebuilt: list[Segment] = []
    for seg in base_sorted:
        if seg.label != config.LABEL_CORE:
            rebuilt.append(seg)
            continue

        raw_spans: list[tuple[float, float, float, dict | None]] = []
        for r in recovered_ads:
            a = max(r.start, seg.start)
            bb = min(r.end, seg.end)
            if bb - a <= 1.0:
                continue
            raw_spans.append(
                (a, bb, float(r.confidence),
                 dict(r.notes) if r.notes else None))
        if not raw_spans:
            rebuilt.append(seg)
            continue

        merged_spans = _merge_overlapping_ad_spans(raw_spans)
        cur = seg.start
        for a, b, conf, notes in sorted(merged_spans, key=lambda t: t[0]):
            if cur + 1e-6 < a:
                rebuilt.append(Segment(cur, a, config.LABEL_CORE,
                                        confidence=seg.confidence))
            rebuilt.append(Segment(a, b, config.LABEL_AD, confidence=conf,
                                    notes=notes))
            cur = b
        if cur + 1e-6 < seg.end:
            rebuilt.append(Segment(cur, seg.end, config.LABEL_CORE,
                                    confidence=seg.confidence))

    rebuilt.sort(key=lambda s: s.start)
    return rebuilt


def _apply_ad_spacing_and_recovery(segments: list[Segment],
                                    *,
                                    features: FusedFeatures,
                                    splice_pairs: list[tuple[float, float, float]],
                                    score_norm: np.ndarray,
                                    recovery_eligible: np.ndarray,
                                    shot_times: np.ndarray,
                                    ) -> list[Segment]:
    """Suppress clustered phantom ads + recover weak-but-real inserts in long core gaps."""

    keyword = features.raw.get("ad_keyword")
    z_by = {c: features.z[:, i] for i, c in enumerate(features.channels)}
    garble_z = z_by.get("transcript_garble")
    music_z = z_by.get("music_likeness")
    outlier_z = z_by.get("local_outlierness")

    ordered = sorted(segments, key=lambda s: s.start)
    out_pass1: list[Segment] = []
    last_ad_end: float | None = None

    for seg in ordered:
        if seg.label != config.LABEL_AD:
            out_pass1.append(seg)
            continue

        ai = int(max(0, round(seg.start)))
        bi = int(min(score_norm.shape[0], max(ai + 1, round(seg.end))))
        dur = seg.end - seg.start
        evidence = _secondary_ad_evidence(
            ai, bi, splice_pairs=splice_pairs, keyword=keyword,
            garble_z=garble_z, music_z=music_z, outlier_z=outlier_z,
            splice_pair_min_conf=float(config.STRICT_SECONDARY_SPLICE_MIN_CONF),
        )

        suppress = False
        if last_ad_end is not None:
            delta = seg.start - last_ad_end
            short = dur <= float(config.SHORT_AD_NEED_EVIDENCE_MAX_SEC)
            if delta < float(config.MIN_GAP_BETWEEN_ADS_SEC) and short and not evidence:
                suppress = True

        if suppress:
            out_pass1.append(Segment(seg.start, seg.end,
                                      config.LABEL_CORE, confidence=0.85))
        else:
            out_pass1.append(seg)
            last_ad_end = max(last_ad_end or -1e9, seg.end)

    baseline_ads = [s for s in out_pass1 if s.label == config.LABEL_AD]

    def _ad_overlap(t0: int, t1: int, pool: list[Segment]) -> bool:
        for ad in pool:
            aa = int(round(ad.start))
            bb = int(round(ad.end))
            if not (t1 <= aa + 15 or t0 >= bb - 15):
                return True
        return False

    recovered: list[Segment] = []
    timeline_sorted = sorted(out_pass1, key=lambda s: s.start)
    last_ad_end_seen: float | None = None

    for seg in timeline_sorted:
        if seg.label == config.LABEL_AD:
            last_ad_end_seen = max(last_ad_end_seen or -1e9, seg.end)
            continue
        if seg.label != config.LABEL_CORE:
            continue
        if seg.end - seg.start < float(config.MIN_CORE_SPAN_FOR_RECOVERY_SCAN_SEC):
            continue

        ws = max(0, int(round(seg.start)))
        we = min(score_norm.shape[0], max(ws + 1, int(round(seg.end))))
        if we <= ws:
            continue

        flags = (
            score_norm[ws:we] >= float(config.AD_RECOVERY_NORM_THRESHOLD)
        ) & recovery_eligible[ws:we]
        if not flags.any():
            continue

        ad_pool = baseline_ads + recovered

        for ra, rb in _runs_above_threshold(flags):
            ra_g, rb_g = ws + ra, ws + rb
            run_sec = rb_g - ra_g
            if run_sec < 18 or run_sec > 132:
                continue
            if float(score_norm[ra_g:rb_g].max()) < float(config.AD_RECOVERY_PEAK_THRESHOLD):
                continue

            if _ad_overlap(ra_g - 50, rb_g + 50, ad_pool):
                continue

            if last_ad_end_seen is not None and (ra_g - last_ad_end_seen) < (
                    float(config.MIN_GAP_BETWEEN_ADS_SEC) * 0.55):
                continue

            if ra_g < float(config.RECOVERY_REQUIRES_STRICT_BEFORE_SEC):
                strict_early = _secondary_ad_evidence(
                    ra_g, rb_g,
                    splice_pairs=splice_pairs, keyword=keyword,
                    garble_z=garble_z, music_z=music_z, outlier_z=outlier_z,
                    splice_pair_min_conf=float(
                        config.STRICT_SECONDARY_SPLICE_MIN_CONF),
                )
                if not strict_early:
                    continue

            if not _secondary_ad_evidence(
                    ra_g, rb_g,
                    splice_pairs=splice_pairs, keyword=keyword,
                    garble_z=garble_z, music_z=music_z, outlier_z=outlier_z,
            ):
                if float(score_norm[ra_g:rb_g].mean()) < 0.612:
                    continue

            snapped_s, snapped_e = _snap_to_shots(
                float(ra_g), float(rb_g), shot_times,
                max_snap_sec=5.5,
            )
            mu = float(score_norm[max(ra_g - 50, 0): min(rb_g + 50,
                                                         score_norm.size)].mean())
            conf_val = float(np.clip(mu + 0.08, 0.71, 0.92))
            rec = Segment(
                snapped_s, snapped_e, config.LABEL_AD,
                confidence=conf_val,
                notes={"recovered_peak": True},
            )
            if rec.end - rec.start < 17.5:
                continue
            recovered.append(rec)
            ad_pool = baseline_ads + recovered
            last_ad_end_seen = max(last_ad_end_seen or -1e9, rec.end)

    merged_out = _fuse_recovered_ads_wall_to_wall(timeline_sorted, recovered)
    return merged_out
