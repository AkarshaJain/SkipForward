"""Fusion / alignment tests.

Proves that fuse() correctly handles per-modality arrays of slightly
different lengths and produces a clean (N, C) matrix at 1 Hz.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.fusion import (
    CHANNELS, FusedFeatures, fuse, _align_length, _zscore, _local_outlierness,
)
from pipeline.features_visual import VisualFeatures
from pipeline.features_audio import AudioFeatures
from pipeline.features_speech import SpeechFeatures


def _stub_visual(n: int, *, shot_times: np.ndarray | None = None) -> VisualFeatures:
    return VisualFeatures(
        times=np.arange(n, dtype=np.float64),
        saturation=np.full(n, 50.0),
        brightness=np.full(n, 100.0),
        motion=np.full(n, 0.05),
        edge_density=np.full(n, 0.1),
        black_frame=np.zeros(n),
        hist_diff=np.zeros(n),
        shot_times=shot_times if shot_times is not None else np.array([]),
    )


def _stub_audio(n: int) -> AudioFeatures:
    return AudioFeatures(
        times=np.arange(n, dtype=np.float64),
        rms_db=np.full(n, -20.0),
        zcr=np.full(n, 0.05),
        spectral_centroid=np.full(n, 1000.0),
        spectral_flatness=np.full(n, 0.4),
        spectral_flux=np.full(n, 0.1),
        music_likeness=np.full(n, 0.2),
        silence=np.zeros(n),
        sr=16000,
        duration=float(n),
    )


def _stub_speech(n: int) -> SpeechFeatures:
    return SpeechFeatures(
        ad_keyword_score=np.zeros(n),
        speech_density=np.full(n, 0.6),
        transcript_garble=np.zeros(n),
        available=False,
    )


def test_align_length_pads_short_arrays():
    a = _align_length(np.array([1.0, 2.0, 3.0]), 6)
    assert a.shape == (6,)
    assert list(a) == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]


def test_align_length_truncates_long_arrays():
    a = _align_length(np.arange(10, dtype=np.float64), 5)
    assert a.shape == (5,)
    assert list(a) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_zscore_constant_array_is_zero():
    z = _zscore(np.full(20, 7.0))
    assert np.allclose(z, 0.0)


def test_local_outlierness_zero_when_constant():
    # All-equal feature vectors -> nothing is locally anomalous.
    n = 200
    z = np.zeros((n, 5))
    out = _local_outlierness(z, window=60)
    assert out.shape == (n,)
    assert np.allclose(out, 0.0)


def test_local_outlierness_high_in_anomalous_window():
    n = 300
    z = np.zeros((n, 4))
    z[140:160, :] = 5.0  # 20-second anomalous window
    out = _local_outlierness(z, window=80)
    # Centre of the window must be much more outlying than the edges.
    assert out[150] > out[10]
    assert out[150] > out[290]


def test_fuse_produces_correct_shape_and_channel_order():
    n = 60
    fused = fuse(duration_sec=float(n),
                  visual=_stub_visual(n),
                  audio=_stub_audio(n),
                  speech=_stub_speech(n))
    assert fused.matrix.shape == (n, len(CHANNELS))
    assert fused.z.shape == (n, len(CHANNELS))
    assert fused.channels == CHANNELS

    # All raw channels are constant -> z-scored values are exactly 0.
    assert np.allclose(fused.z, 0.0)


def test_fuse_handles_off_by_one_channel_lengths():
    n = 60
    visual = _stub_visual(n - 1)        # short by 1
    audio = _stub_audio(n + 2)          # long by 2
    speech = _stub_speech(n)
    fused = fuse(duration_sec=float(n),
                  visual=visual, audio=audio, speech=speech)
    assert fused.matrix.shape == (n, len(CHANNELS))
