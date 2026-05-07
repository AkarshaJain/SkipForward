"""Multimodal fusion → per-step feature matrix.

All modality outputs are aligned to a common ``config.SAMPLE_FPS``-Hz
timeline of length ``N`` (one row per ``1/SAMPLE_FPS`` seconds of video).
We z-score each channel against the *video's own* distribution so the
segmenter is robust to absolute differences across videos (a quiet
podcast still has 'loud ads' relative to itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from . import config
from .features_audio import AudioFeatures
from .features_visual import VisualFeatures, shot_rate_per_second
from .features_speech import SpeechFeatures


CHANNELS = (
    "shot_rate",
    "saturation",
    "motion",
    "edge_density",
    "audio_rms",
    "spectral_flux",
    "music_likeness",
    "ad_keyword",
    "silence",
    "black_frame",
    "speech_density",
    "transcript_garble",
    "local_outlierness",
)


@dataclass
class FusedFeatures:
    times: np.ndarray
    matrix: np.ndarray                     # (N, C) raw values
    z: np.ndarray                          # (N, C) z-scored values
    channels: tuple[str, ...] = CHANNELS
    raw: dict[str, np.ndarray] = field(default_factory=dict)


def _align_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate a 1-D array to length ``n``."""
    arr = np.asarray(arr, dtype=np.float64).flatten()
    if arr.size == n:
        return arr
    if arr.size > n:
        return arr[:n]
    out = np.zeros(n, dtype=np.float64)
    out[: arr.size] = arr
    return out


def _zscore(arr: np.ndarray) -> np.ndarray:
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-6:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _local_outlierness(z_features: np.ndarray, window: int = 120) -> np.ndarray:
    """How anomalous is each step's feature vector vs. its local context?

    ``window`` is given in *grid steps* (already scaled by SAMPLE_FPS by
    the caller). Computed as the L2 distance between the row at ``t``
    and the mean row of an outer ring ``[t-W, t-W/2] ∪ [t+W/2, t+W]``
    (skipping the inner half so a wide ad doesn't suppress its own
    outlierness).
    """
    n = z_features.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    half = max(window // 4, 6)
    full = max(window // 2, 12)
    for t in range(n):
        lo_a, lo_b = max(0, t - full), max(0, t - half)
        hi_a, hi_b = min(n, t + half), min(n, t + full)
        ring = []
        if lo_b > lo_a:
            ring.append(z_features[lo_a:lo_b])
        if hi_b > hi_a:
            ring.append(z_features[hi_a:hi_b])
        if not ring:
            out[t] = 0.0
            continue
        ctx_mean = np.concatenate(ring, axis=0).mean(axis=0)
        out[t] = float(np.linalg.norm(z_features[t] - ctx_mean))
    return out


def fuse(
    duration_sec: float,
    visual: VisualFeatures,
    audio: AudioFeatures,
    speech: SpeechFeatures,
) -> FusedFeatures:
    """Build the (N, C) feature matrix at ``config.SAMPLE_FPS`` Hz."""
    n = max(int(np.ceil(duration_sec * config.SAMPLE_FPS)), 1)

    shot_rate = shot_rate_per_second(visual.shot_times, duration_sec)

    garble = (speech.transcript_garble
              if speech.transcript_garble.size
              else np.zeros(n))

    raw: dict[str, np.ndarray] = {
        "shot_rate":         _align_length(shot_rate, n),
        "saturation":        _align_length(visual.saturation, n),
        "motion":            _align_length(visual.motion, n),
        "edge_density":      _align_length(visual.edge_density, n),
        "audio_rms":         _align_length(audio.rms_db, n),
        "spectral_flux":     _align_length(audio.spectral_flux, n),
        "music_likeness":    _align_length(audio.music_likeness, n),
        "ad_keyword":        _align_length(speech.ad_keyword_score, n),
        "silence":           _align_length(audio.silence, n),
        "black_frame":       _align_length(visual.black_frame, n),
        "speech_density":    _align_length(speech.speech_density, n),
        "transcript_garble": _align_length(garble, n),
    }

    # local_outlierness: at each second t, distance from the mean feature
    # vector in a wider context window. This catches sustained anomalous
    # regions (e.g. a long ad whose features are all moderately different
    # from the surrounding video — invisible to per-second z-scoring).
    pre_matrix = np.column_stack(
        [_zscore(raw[c]) for c in (
            "saturation", "motion", "edge_density",
            "audio_rms", "music_likeness", "speech_density",
        )])
    # 120 *seconds* of context, scaled to grid steps.
    raw["local_outlierness"] = _local_outlierness(
        pre_matrix, window=config.sec_to_width(120))

    matrix = np.column_stack([raw[c] for c in CHANNELS])
    z = np.column_stack([_zscore(raw[c]) for c in CHANNELS])

    # times is in seconds (each row = step / SAMPLE_FPS).
    times = np.arange(n, dtype=np.float64) / config.SAMPLE_FPS

    return FusedFeatures(
        times=times,
        matrix=matrix,
        z=z,
        raw=dict(raw),
    )
