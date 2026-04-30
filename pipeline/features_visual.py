"""Visual feature extraction.

Produces, per second of video, a feature vector capturing:

- ``shot_rate``      : shot cuts per minute, smoothed locally
- ``saturation``     : mean HSV saturation
- ``brightness``     : mean V channel
- ``motion``         : mean optical-flow magnitude between consecutive sampled frames
- ``edge_density``   : Canny edge fraction (proxy for text overlays / busy graphics)
- ``black_frame``    : 0/1 flag for near-black frames
- ``hist_diff``      : raw histogram delta from previous sampled frame

It also returns the list of detected shot boundaries (in seconds), used as
candidate segment edges by the segmenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .preprocessing import VideoMeta, iter_sampled_frames, iter_dense_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-like distance bounded in [0, 1]."""
    diff = np.abs(a - b)
    return float(np.sum(diff)) / max(np.sum(a) + np.sum(b), 1e-6)


# ---------------------------------------------------------------------------
# Shot-boundary detection (dense)
# ---------------------------------------------------------------------------

def detect_shots(video_path: Path | str,
                 *,
                 stride: int = config.SHOT_DETECT_STRIDE,
                 threshold: float = config.SHOT_CUT_THRESHOLD,
                 min_gap_sec: float = config.SHOT_MIN_GAP_SEC,
                 progress: bool = True) -> list[float]:
    """Return list of shot-boundary timestamps in seconds.

    Compares HSV histograms of strided frames; flags frame pairs whose
    distance is far above the local median (robust to lighting).
    """
    times: list[float] = []
    diffs: list[float] = []
    prev_hist: np.ndarray | None = None

    iterator = iter_dense_frames(video_path, stride=stride, resize=(160, 90))
    if progress:
        iterator = tqdm(iterator, desc="shot-detect", leave=False, unit="f")

    for t, frame in iterator:
        h = _hsv_hist(frame)
        if prev_hist is not None:
            d = _hist_distance(prev_hist, h)
            diffs.append(d)
            times.append(t)
        prev_hist = h

    if not diffs:
        return []

    diffs_arr = np.asarray(diffs)
    times_arr = np.asarray(times)

    # Adaptive threshold: max(global_threshold, median + 6*MAD)
    median = float(np.median(diffs_arr))
    mad = float(np.median(np.abs(diffs_arr - median)) + 1e-6)
    adaptive = median + 6.0 * mad
    cut_thresh = max(threshold, adaptive)

    candidate_idx = np.where(diffs_arr > cut_thresh)[0]
    if candidate_idx.size == 0:
        return []

    # Suppress duplicates closer than min_gap_sec (keep the strongest).
    cuts: list[tuple[float, float]] = []
    for i in candidate_idx:
        cuts.append((float(times_arr[i]), float(diffs_arr[i])))

    cuts.sort()
    pruned: list[float] = []
    last_t = -1e9
    for t, _d in cuts:
        if t - last_t >= min_gap_sec:
            pruned.append(t)
            last_t = t
    return pruned


# ---------------------------------------------------------------------------
# Per-second feature extraction (sparse)
# ---------------------------------------------------------------------------

@dataclass
class VisualFeatures:
    times: np.ndarray            # (N,) seconds
    saturation: np.ndarray       # (N,)
    brightness: np.ndarray       # (N,)
    motion: np.ndarray           # (N,)
    edge_density: np.ndarray     # (N,)
    black_frame: np.ndarray      # (N,) 0/1
    hist_diff: np.ndarray        # (N,)
    shot_times: np.ndarray       # (M,) shot boundary timestamps


def extract_visual_features(video_path: Path | str,
                            meta: VideoMeta,
                            *,
                            sample_fps: float = config.SAMPLE_FPS,
                            progress: bool = True) -> VisualFeatures:
    sat_l: list[float] = []
    bri_l: list[float] = []
    mot_l: list[float] = []
    edge_l: list[float] = []
    black_l: list[float] = []
    hist_l: list[float] = []
    times: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    iterator = iter_sampled_frames(video_path, sample_fps=sample_fps)
    total_estimate = max(int(meta.duration * sample_fps), 1)
    if progress:
        iterator = tqdm(iterator, total=total_estimate,
                        desc="visual feats", leave=False, unit="f")

    for t, frame in iterator:
        times.append(t)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sat_l.append(float(hsv[..., 1].mean()))
        bri_l.append(float(hsv[..., 2].mean()))
        edges = cv2.Canny(gray, 80, 180)
        edge_l.append(float(edges.mean()) / 255.0)
        black_l.append(1.0 if hsv[..., 2].mean() < config.BLACK_FRAME_LUMA else 0.0)

        if prev_gray is not None:
            # Coarse optical flow proxy: mean absolute diff (cheap, robust).
            diff = cv2.absdiff(gray, prev_gray)
            mot_l.append(float(diff.mean()) / 255.0)
        else:
            mot_l.append(0.0)

        h = _hsv_hist(frame)
        if prev_hist is not None:
            hist_l.append(_hist_distance(prev_hist, h))
        else:
            hist_l.append(0.0)

        prev_gray = gray
        prev_hist = h

    shot_times = detect_shots(video_path, progress=progress)

    return VisualFeatures(
        times=np.asarray(times, dtype=np.float64),
        saturation=np.asarray(sat_l, dtype=np.float64),
        brightness=np.asarray(bri_l, dtype=np.float64),
        motion=np.asarray(mot_l, dtype=np.float64),
        edge_density=np.asarray(edge_l, dtype=np.float64),
        black_frame=np.asarray(black_l, dtype=np.float64),
        hist_diff=np.asarray(hist_l, dtype=np.float64),
        shot_times=np.asarray(shot_times, dtype=np.float64),
    )


def shot_rate_per_second(shot_times: np.ndarray,
                          duration_sec: float,
                          window_sec: float = 15.0) -> np.ndarray:
    """Number of cuts in a sliding window centred on each second."""
    n = max(int(np.ceil(duration_sec)), 1)
    rate = np.zeros(n, dtype=np.float64)
    if shot_times.size == 0:
        return rate
    half = window_sec / 2.0
    for s in range(n):
        lo, hi = s - half, s + half
        rate[s] = float(np.sum((shot_times >= lo) & (shot_times <= hi)))
    rate /= window_sec / 60.0
    return rate


# ---------------------------------------------------------------------------
# Splice-boundary detection
# ---------------------------------------------------------------------------
# An "inserted ad" almost always shows a characteristic signature at its
# in-point and out-point: a hard shot cut coincident with brief silence
# (or a black frame). We detect these splice times explicitly because they
# are extremely interpretable and let us bracket long ad regions whose
# *content* is otherwise hard to distinguish from the surrounding video.

def detect_splice_boundaries(
    shot_times: np.ndarray,
    silence_intervals: list[tuple[float, float]],
    black_frame_per_sec: np.ndarray,
    duration_sec: float,
    *,
    silence_window_sec: float = 3.0,
    feature_matrix: np.ndarray | None = None,
    discontinuity_window_sec: int = 8,
    discontinuity_z: float = 1.8,
) -> list[float]:
    """Return seconds of shot cuts that look like *splices between unrelated clips*.

    A splice is accepted if at least ONE of these conditions holds:
      (a) silence boundary within ``silence_window_sec`` of the cut, OR
      (b) co-located black/dark frame within ±1s, OR
      (c) the per-second feature matrix (if provided) shows a large
          content discontinuity across the cut, i.e. the mean feature
          vector ``W`` seconds before differs strongly from the mean
          ``W`` seconds after (z-score relative to the global distribution
          of such cross-cut distances).

    Conditions (a)+(b) catch classic ad inserts that drop audio at the join.
    Condition (c) catches inserts whose audio overlaps the main track but
    whose visual content is clearly from another source clip — exactly the
    case of a long video advertisement spliced into a podcast/lecture.
    """
    if shot_times.size == 0:
        return []

    sil_starts = np.array([s for s, _ in silence_intervals], dtype=np.float64)
    sil_ends = np.array([e for _, e in silence_intervals], dtype=np.float64)

    # Pre-compute cross-cut content distance for every shot (if features given)
    cross_dists: dict[float, float] = {}
    if feature_matrix is not None and feature_matrix.size and shot_times.size:
        n = feature_matrix.shape[0]
        # z-normalise channel-wise so distance is dimensionless
        F = feature_matrix.copy()
        mu = F.mean(axis=0, keepdims=True)
        sd = F.std(axis=0, keepdims=True) + 1e-6
        F = (F - mu) / sd

        W = max(int(discontinuity_window_sec), 2)
        for t in shot_times:
            i = int(round(t))
            a0, a1 = max(0, i - W), max(0, i)
            b0, b1 = min(n, i + 1), min(n, i + 1 + W)
            if a1 - a0 < 2 or b1 - b0 < 2:
                continue
            before = F[a0:a1].mean(axis=0)
            after = F[b0:b1].mean(axis=0)
            cross_dists[float(t)] = float(np.linalg.norm(before - after))

        # Convert to z-score relative to the distribution of all cross-cut dists.
        if cross_dists:
            vals = np.array(list(cross_dists.values()))
            mu_d, sd_d = vals.mean(), vals.std() + 1e-6
            cross_dists = {k: (v - mu_d) / sd_d for k, v in cross_dists.items()}

    splice: list[float] = []
    for t in shot_times:
        # (a) silence proximity
        near_sil = False
        if sil_starts.size:
            if (np.min(np.abs(sil_starts - t)) <= silence_window_sec
                    or np.min(np.abs(sil_ends - t)) <= silence_window_sec):
                near_sil = True

        # (b) black/dark frame proximity
        i = int(round(t))
        lo, hi = max(0, i - 1), min(len(black_frame_per_sec), i + 2)
        near_black = bool(black_frame_per_sec[lo:hi].max() > 0.5) if hi > lo else False

        # (c) content discontinuity
        big_jump = cross_dists.get(float(t), 0.0) >= discontinuity_z

        if near_sil or near_black or big_jump:
            splice.append(float(t))
    return splice


def splice_signal_per_second(splice_times: list[float],
                              duration_sec: float,
                              kernel_sec: float = 2.0) -> np.ndarray:
    """Return a soft 0..1 indicator that decays away from each splice point."""
    n = max(int(np.ceil(duration_sec)), 1)
    sig = np.zeros(n, dtype=np.float64)
    for t in splice_times:
        i = int(round(t))
        lo = max(0, i - int(kernel_sec))
        hi = min(n, i + int(kernel_sec) + 1)
        for j in range(lo, hi):
            sig[j] = max(sig[j], 1.0 - abs(j - t) / max(kernel_sec, 1e-6))
    return sig
