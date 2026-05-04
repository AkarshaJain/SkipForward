"""Pytest fixtures + shared synthetic-input builders.

Every test in this folder is fully synthetic -- no .mp4 / .mkv files from
the dataset are read. The intent is to give the segmenter known inputs
where the *correct* output is unambiguous.
"""

from __future__ import annotations

import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable regardless of where pytest is invoked
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Synthetic WAV builders (used by audio + speech feature tests)
# ---------------------------------------------------------------------------

@dataclass
class WavSpec:
    duration_sec: float
    sr: int = 16_000


def _write_wav_int16(path: Path, samples: np.ndarray, sr: int) -> Path:
    """Write a 16-bit PCM WAV using only the stdlib (no soundfile dep)."""
    samples = np.clip(samples, -1.0, 1.0)
    samples_i16 = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_i16.tobytes())
    return path


@pytest.fixture
def silent_wav(tmp_path: Path):
    def _make(seconds: float = 5.0, sr: int = 16_000) -> Path:
        n = int(seconds * sr)
        return _write_wav_int16(tmp_path / "silent.wav",
                                np.zeros(n, dtype=np.float64), sr)
    return _make


@pytest.fixture
def tone_wav(tmp_path: Path):
    """Steady 440-Hz sine -- music-like, definitely not silent."""
    def _make(seconds: float = 5.0, sr: int = 16_000,
              freq: float = 440.0, amp: float = 0.4) -> Path:
        n = int(seconds * sr)
        t = np.arange(n) / sr
        y = amp * np.sin(2 * np.pi * freq * t)
        return _write_wav_int16(tmp_path / "tone.wav", y, sr)
    return _make


@pytest.fixture
def noise_wav(tmp_path: Path):
    """White noise -- definitely not silent, definitely not music."""
    def _make(seconds: float = 5.0, sr: int = 16_000, amp: float = 0.3) -> Path:
        n = int(seconds * sr)
        rng = np.random.default_rng(seed=0)
        y = amp * rng.standard_normal(n)
        return _write_wav_int16(tmp_path / "noise.wav", y, sr)
    return _make


@pytest.fixture
def mixed_wav(tmp_path: Path):
    """Speech-band-ish mid-low noise + intermittent silence regions."""
    def _make(seconds: float = 30.0, sr: int = 16_000,
              silent_ranges=((10.0, 20.0),)) -> Path:
        n = int(seconds * sr)
        rng = np.random.default_rng(seed=42)
        y = 0.25 * rng.standard_normal(n)
        for s, e in silent_ranges:
            a, b = int(s * sr), int(e * sr)
            y[a:b] = 0.0
        return _write_wav_int16(tmp_path / "mixed.wav", y, sr)
    return _make
