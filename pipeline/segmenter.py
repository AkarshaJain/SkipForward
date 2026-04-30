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
    # sustained — natural lectures/podcasts have many short pauses.
    sil_regions: list[tuple[float, float]] = []
    sil_min = config.SILENCE_AS_SEGMENT_MIN_SEC
    for s, e in silence_intervals:
        if e - s < sil_min:
            continue
        sil_regions.append((float(s), float(e)))

    # Build labelled segments end-to-end ----------------------------------
    segments = _label_and_fill(
        nc_regions=nc_regions,
        sil_regions=sil_regions,
        duration=duration_sec,
        intro_window=config.INTRO_WINDOW_SEC,
        outro_window=config.OUTRO_WINDOW_SEC,
    )

    debug = {
        "score": score.tolist(),
        "score_norm": score_norm.tolist(),
        "mask": mask.astype(int).tolist(),
        "threshold": threshold,
    }
    return segments, debug


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
