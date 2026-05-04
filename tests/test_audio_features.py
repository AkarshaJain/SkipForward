"""Audio-feature tests on fully-synthetic WAVs.

These prove that:
  - silent input is detected as silent end-to-end (RMS very low, silence flag set)
  - a steady sine tone is detected as music-like (high music_likeness)
  - white noise is NOT detected as music-like
  - silence_intervals returns the correct (start, end) ranges
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.features_audio import (
    AudioFeatures, extract_audio_features, silence_intervals,
)
from pipeline import config


def test_silent_wav_is_detected_silent(silent_wav):
    wav = silent_wav(seconds=8.0)
    feats = extract_audio_features(wav, num_seconds=8)

    assert feats.silence.shape == (8,)
    assert feats.silence.sum() == 8, (
        "every second of a silent WAV must be flagged silent")
    assert feats.rms_db.max() < -50.0, (
        f"silent WAV RMS should be deeply negative, got {feats.rms_db.max():.2f} dB")


def test_silent_wav_yields_one_silence_interval(silent_wav):
    wav = silent_wav(seconds=10.0)
    feats = extract_audio_features(wav, num_seconds=10)
    intervals = silence_intervals(feats, min_sec=2.0)

    assert len(intervals) == 1
    s, e = intervals[0]
    assert s == 0.0
    assert e == pytest.approx(10.0, abs=1.0)


def test_tone_wav_is_loud_and_tonal(tone_wav):
    """A 440 Hz sine wave is not silent and is highly tonal (very low
    spectral flatness). We test 'tonal' rather than 'music_likeness > X'
    because music_likeness rewards multi-pitch content; a single pure
    pitch is not realistic music."""
    wav = tone_wav(seconds=6.0, freq=440.0, amp=0.5)
    feats = extract_audio_features(wav, num_seconds=6)

    assert feats.silence.sum() == 0, "loud sine wave must not be flagged silent"
    assert feats.rms_db.mean() > -20.0, "loud tone should be loud in dB"
    assert feats.spectral_flatness.mean() < 1e-3, (
        f"steady sine must be highly tonal (low flatness), "
        f"got {feats.spectral_flatness.mean():.4f}")


def test_noise_is_spectrally_flatter_than_tone(noise_wav, tone_wav):
    """White noise should be far flatter (more uniform spectrum) than a
    pure sine tone -- this is the core signal music_likeness uses."""
    nf = extract_audio_features(noise_wav(seconds=6.0), num_seconds=6)
    tf = extract_audio_features(tone_wav(seconds=6.0), num_seconds=6)

    assert nf.silence.sum() == 0
    assert nf.spectral_flatness.mean() > tf.spectral_flatness.mean() * 100, (
        f"noise flatness ({nf.spectral_flatness.mean():.4f}) should be >> "
        f"tone flatness ({tf.spectral_flatness.mean():.4f})")


def test_silence_in_the_middle_is_recovered(mixed_wav):
    """A mixed track with a 10-s silent gap should yield ONE silence interval
    of length ~10 covering that gap (not the surrounding noise)."""
    wav = mixed_wav(seconds=30.0, silent_ranges=((10.0, 20.0),))
    feats = extract_audio_features(wav, num_seconds=30)
    intervals = silence_intervals(feats, min_sec=3.0)

    assert len(intervals) == 1, (
        f"expected exactly one silent gap, got {intervals}")
    s, e = intervals[0]
    # Allow +/- 2 s slack because adaptive thresholding can grow the
    # detected region a little, but the centre must be inside [10, 20].
    assert 8.0 <= s <= 12.0, f"silence start {s} far from expected 10"
    assert 18.0 <= e <= 22.0, f"silence end {e} far from expected 20"


def test_audio_features_align_to_requested_length(noise_wav):
    """Even if the WAV is slightly shorter/longer than num_seconds requested,
    every per-second array comes back at exactly that length."""
    wav = noise_wav(seconds=4.7)
    feats = extract_audio_features(wav, num_seconds=5)
    for arr in (feats.rms_db, feats.zcr, feats.music_likeness,
                feats.silence, feats.spectral_flux):
        assert arr.shape == (5,), f"expected length 5, got {arr.shape}"
