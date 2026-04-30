"""Splice-pair segmenter.

Inserted ads typically appear between two splice points (hard cut +
co-located silence). This module enumerates candidate (splice_start,
splice_end) pairs and accepts those whose enclosed region looks
*statistically different* from the rest of the video on a small set of
robust features.

The accepted regions are returned as ``(start, end, confidence)`` tuples
to be merged with the score-based segmenter's output.
"""

from __future__ import annotations

import numpy as np

from . import config


# Reasonable bounds for inserted ads in seconds.
MIN_AD_DURATION = 12.0
MAX_AD_DURATION = 240.0

# How far apart (s) two consecutive splice points may be.
PAIR_MIN = MIN_AD_DURATION
PAIR_MAX = MAX_AD_DURATION


def _z(arr: np.ndarray) -> np.ndarray:
    sd = arr.std()
    if sd < 1e-6:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


def detect_splice_pair_ads(
    splice_times: list[float],
    *,
    music_likeness: np.ndarray,
    saturation: np.ndarray,
    motion: np.ndarray,
    audio_rms: np.ndarray,
    speech_density: np.ndarray,
    ad_keyword: np.ndarray,
    feature_matrix: np.ndarray,
    duration_sec: float,
    margin_sec: int = 30,
) -> list[tuple[float, float, float]]:
    """Return list of ``(start, end, confidence)`` triples.

    Strategy (conservative, low false-positive)
    -------------------------------------------
    Inserted ads are bounded by *immediately consecutive* splice points,
    not by arbitrary far-apart pairs. We therefore only consider:

      * Each consecutive splice pair ``(s_i, s_{i+1})``.
      * Any splice pair ``(s_i, s_j)`` with j > i+1 only if every splice
        point between them falls inside a single homogeneous region
        (i.e. ``s_{i+1}, ..., s_{j-1}`` are content-similar to ``[s_i, s_j]``
        and dissimilar to the surrounding context).

    A candidate is accepted only if its multimodal feature vector is
    significantly distinct from both its left and right ``margin_sec``-second
    contexts.
    """
    if len(splice_times) < 2:
        return []

    splice_times = sorted(set(int(round(t)) for t in splice_times))
    n = feature_matrix.shape[0]

    F = feature_matrix.copy()
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-6
    Fz = (F - mu) / sd

    feats = {
        "music":      _z(music_likeness),
        "saturation": _z(saturation),
        "motion":     _z(motion),
        "rms":        _z(audio_rms),
        "speech":     _z(speech_density),
    }
    keyword = ad_keyword

    def _candidate_score(a_idx: int, b_idx: int) -> float | None:
        if b_idx - a_idx < PAIR_MIN or b_idx - a_idx > PAIR_MAX:
            return None
        left_a = max(0, a_idx - margin_sec)
        right_b = min(n, b_idx + margin_sec)
        inside = Fz[a_idx:b_idx]
        left_ctx = Fz[left_a:a_idx]
        right_ctx = Fz[b_idx:right_b]
        if min(inside.shape[0], left_ctx.shape[0], right_ctx.shape[0]) < 4:
            return None

        inside_mean = inside.mean(axis=0)
        d_left = float(np.linalg.norm(inside_mean - left_ctx.mean(axis=0)))
        d_right = float(np.linalg.norm(inside_mean - right_ctx.mean(axis=0)))
        local_distinct = 0.5 * (d_left + d_right)

        # Demand that BOTH sides differ meaningfully — rules out simple
        # genre changes (e.g. transition into a quiet section that only
        # differs on one side).
        if min(d_left, d_right) < 1.0:
            return None

        inside_slice = slice(a_idx, b_idx)
        ad_bonus = (
            0.3 * feats["music"][inside_slice].mean()
            + 0.2 * feats["saturation"][inside_slice].mean()
            + 0.2 * feats["motion"][inside_slice].mean()
            + 1.5 * float(keyword[inside_slice].mean())
            - 0.2 * feats["speech"][inside_slice].mean()
        )
        return local_distinct + 0.4 * ad_bonus

    # Consider all (i, j) pairs but score with the strict criterion above.
    candidates: list[tuple[float, tuple[int, int]]] = []
    for i, a in enumerate(splice_times):
        for j in range(i + 1, len(splice_times)):
            b = splice_times[j]
            if b - a > PAIR_MAX:
                break
            if b - a < PAIR_MIN:
                continue
            s = _candidate_score(int(a), int(b))
            if s is not None:
                candidates.append((s, (int(a), int(b))))

    # Greedy non-overlapping selection with a STRONG threshold.
    candidates.sort(reverse=True)
    accepted: list[tuple[int, int, float]] = []
    SCORE_FLOOR = 2.0  # ~2 z-units of combined distinctiveness
    for score, (a, b) in candidates:
        if score < SCORE_FLOOR:
            break
        # reject candidates that touch or overlap an already-accepted region
        too_close = any(not (b + 5 <= ax or a >= bx + 5) for ax, bx, _ in accepted)
        if too_close:
            continue
        conf = float(np.clip(1.0 / (1.0 + np.exp(-(score - 2.5))), 0.05, 0.99))
        accepted.append((a, b, conf))

    return [(float(a), float(b), c) for (a, b, c) in accepted]
