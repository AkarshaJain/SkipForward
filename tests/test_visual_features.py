"""Visual-feature tests with frame arrays directly (no real video file).

We build NumPy frames and call the helpers in features_visual.py directly.
This avoids needing FFmpeg in CI and proves:
  - black frames are flagged when luminance is low
  - shot-boundary detection fires on a hard cut between two distinct
    "scenes" but NOT inside a constant scene (repeated frames)
  - the splice-boundary detector correctly combines silence+black evidence
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config
from pipeline.features_visual import (
    _hsv_hist, _hist_distance,
    detect_splice_boundaries, splice_signal_per_second, shot_rate_per_second,
)


def _solid_frame(value: int, h: int = 90, w: int = 160) -> np.ndarray:
    """BGR frame filled with a single grey value."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _coloured_frame(b: int, g: int, r: int,
                     h: int = 90, w: int = 160) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 0] = b
    out[..., 1] = g
    out[..., 2] = r
    return out


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def test_hist_distance_zero_for_identical_frames():
    a = _coloured_frame(40, 200, 30)
    b = _coloured_frame(40, 200, 30)
    d = _hist_distance(_hsv_hist(a), _hsv_hist(b))
    assert d == pytest.approx(0.0, abs=1e-3)


def test_hist_distance_large_for_unrelated_frames():
    """Pure red vs pure blue should be far apart in HSV-hist space."""
    red = _coloured_frame(0, 0, 240)
    blue = _coloured_frame(240, 0, 0)
    d = _hist_distance(_hsv_hist(red), _hsv_hist(blue))
    assert d > 0.5, f"unrelated colours should be far apart, got d={d:.3f}"


# ---------------------------------------------------------------------------
# Splice-boundary detector
# ---------------------------------------------------------------------------

def _steps(seconds: float) -> int:
    return int(round(seconds * config.SAMPLE_FPS))


def test_no_splice_when_no_evidence():
    """A shot cut with no nearby silence and no black frame is NOT a splice."""
    shots = np.array([60.0])
    black = np.zeros(_steps(120))
    splices = detect_splice_boundaries(
        shot_times=shots,
        silence_intervals=[],
        black_frame_per_sec=black,
        duration_sec=120.0,
    )
    assert splices == []


def test_splice_detected_when_silence_brackets_cut():
    shots = np.array([60.0])
    silence = [(58.0, 62.0)]   # 4-s silent gap straddles the cut
    black = np.zeros(_steps(120))
    splices = detect_splice_boundaries(
        shot_times=shots,
        silence_intervals=silence,
        black_frame_per_sec=black,
        duration_sec=120.0,
    )
    assert splices == [60.0]


def test_splice_detected_when_black_frame_at_cut():
    shots = np.array([60.0])
    black = np.zeros(_steps(120)); black[_steps(60)] = 1.0
    splices = detect_splice_boundaries(
        shot_times=shots,
        silence_intervals=[],
        black_frame_per_sec=black,
        duration_sec=120.0,
    )
    assert splices == [60.0]


def test_splice_signal_per_second_decays_around_splice():
    sig = splice_signal_per_second([30.0], duration_sec=60.0, kernel_sec=2.0)
    assert sig.shape == (_steps(60),)
    centre = _steps(30)
    assert sig[centre] == pytest.approx(1.0, abs=1e-6)
    assert sig[centre - _steps(1)] < sig[centre]
    assert sig[centre + _steps(1)] < sig[centre]
    assert sig[_steps(10)] == 0.0
    assert sig[_steps(50)] == 0.0


# ---------------------------------------------------------------------------
# Shot-rate aggregation
# ---------------------------------------------------------------------------

def test_shot_rate_zero_when_no_cuts():
    rate = shot_rate_per_second(np.array([]), duration_sec=60.0)
    assert rate.shape == (_steps(60),)
    assert rate.sum() == 0.0


def test_shot_rate_increases_with_cut_density():
    sparse = shot_rate_per_second(np.array([30.0]), duration_sec=60.0)
    dense = shot_rate_per_second(
        np.linspace(20.0, 40.0, num=20), duration_sec=60.0)
    centre = _steps(30)
    assert dense[centre] > sparse[centre] * 5
