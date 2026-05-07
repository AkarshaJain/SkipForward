"""Audio feature extraction.

Per-step features computed from the WAV produced by ``preprocessing``,
where 1 step = 1 / ``config.SAMPLE_FPS`` seconds. With the default
SAMPLE_FPS=2 this is 0.5 s per row.

- ``rms_db``         : loudness in decibels (full scale)
- ``zcr``            : zero-crossing rate (proxy for noisiness)
- ``spectral_centroid``  : spectral brightness
- ``spectral_flatness``  : noise-vs-tonal indicator
- ``spectral_flux``  : magnitude change between consecutive frames
- ``music_likeness`` : 0..1 score combining flatness + chroma stability,
                       higher = more music-like, lower = more speech-like
- ``silence``        : 0/1 flag for low-RMS steps
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class AudioFeatures:
    times: np.ndarray            # (N,) seconds
    rms_db: np.ndarray
    zcr: np.ndarray
    spectral_centroid: np.ndarray
    spectral_flatness: np.ndarray
    spectral_flux: np.ndarray
    music_likeness: np.ndarray
    silence: np.ndarray
    sr: int
    duration: float


def _aggregate_per_step(values: np.ndarray, hop: int, sr: int,
                         num_steps: int, reducer="mean") -> np.ndarray:
    """Reduce a frame-rate feature down to a per-step array, where
    each step is ``1 / config.SAMPLE_FPS`` seconds."""
    if values.size == 0:
        return np.zeros(num_steps)
    frames_per_sec = sr / hop
    frames_per_step = frames_per_sec / config.SAMPLE_FPS
    out = np.zeros(num_steps, dtype=np.float64)
    for s in range(num_steps):
        a = int(s * frames_per_step)
        b = int((s + 1) * frames_per_step)
        chunk = values[a:b] if b > a else values[a:a + 1]
        if chunk.size == 0:
            out[s] = 0.0
            continue
        if reducer == "mean":
            out[s] = float(chunk.mean())
        elif reducer == "max":
            out[s] = float(chunk.max())
        else:
            raise ValueError(reducer)
    return out


def extract_audio_features(audio_wav: Path | str,
                            *,
                            sr: int = config.AUDIO_SR,
                            num_seconds: int | None = None) -> AudioFeatures:
    """Compute the per-step audio feature vector at ``config.SAMPLE_FPS``
    Hz. Requires ``librosa``.

    ``num_seconds`` is the *video duration* in whole seconds; the output
    arrays have length ``ceil(num_seconds * SAMPLE_FPS)``.
    """
    import librosa  # local import: the rest of the pipeline doesn't need it

    y, sr = librosa.load(str(audio_wav), sr=sr, mono=True)
    duration = len(y) / sr if sr else 0.0
    secs = num_seconds or int(np.ceil(duration))
    secs = max(secs, 1)
    n = max(int(np.ceil(secs * config.SAMPLE_FPS)), 1)

    hop = config.AUDIO_HOP_LENGTH
    nfft = config.AUDIO_FRAME_LENGTH

    # RMS loudness (in dB FS)
    rms = librosa.feature.rms(y=y, frame_length=nfft, hop_length=hop)[0]
    rms_db_frame = 20.0 * np.log10(np.clip(rms, 1e-6, None))

    # Zero-crossing rate
    zcr_frame = librosa.feature.zero_crossing_rate(
        y, frame_length=nfft, hop_length=hop)[0]

    # Spectral features
    centroid_frame = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=nfft, hop_length=hop)[0]
    flatness_frame = librosa.feature.spectral_flatness(
        y=y, n_fft=nfft, hop_length=hop)[0]

    # Spectral flux (positive change of magnitude spectrum)
    S = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop))
    flux_frame = np.zeros(S.shape[1])
    if S.shape[1] > 1:
        diff = np.diff(S, axis=1)
        diff[diff < 0] = 0.0
        flux_frame[1:] = diff.sum(axis=0)
        # normalize
        flux_frame /= (flux_frame.max() + 1e-6)

    # Music-vs-speech proxy:
    #   Music tends to have stable chroma (low chroma stddev) and
    #   moderate flatness. Speech tends to have high zero-crossing-rate
    #   variation and lower chroma stability.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                         n_fft=nfft, hop_length=hop)
    chroma_std = chroma.std(axis=0)  # per-frame stddev across pitch classes
    chroma_score = np.exp(-chroma_std * 5.0)  # high = stable = music-like
    flatness_score = 1.0 - flatness_frame      # tonal > noisy = music
    flatness_score = np.clip(flatness_score, 0.0, 1.0)
    zcr_norm = zcr_frame / (zcr_frame.max() + 1e-6)
    speech_score = zcr_norm
    music_frame = np.clip(0.5 * chroma_score + 0.5 * flatness_score
                          - 0.5 * speech_score, 0.0, 1.0)

    rms_db = _aggregate_per_step(rms_db_frame, hop, sr, n, "mean")
    zcr = _aggregate_per_step(zcr_frame, hop, sr, n, "mean")
    centroid = _aggregate_per_step(centroid_frame, hop, sr, n, "mean")
    flatness = _aggregate_per_step(flatness_frame, hop, sr, n, "mean")
    flux = _aggregate_per_step(flux_frame, hop, sr, n, "max")
    music = _aggregate_per_step(music_frame, hop, sr, n, "mean")

    # Adaptive silence detection — compares to the video's own RMS
    # distribution so it works on quiet podcasts AND loud commercials.
    # A second is silence iff its RMS is BOTH below the absolute -40 dB
    # ceiling AND below (median_rms - 12 dB). That second condition makes
    # quiet narration sections not get falsely flagged.
    valid = rms_db[rms_db > -80.0]
    median_rms = float(np.median(valid)) if valid.size else -50.0
    relative_thresh = median_rms - 12.0
    threshold = min(config.SILENCE_DB_THRESHOLD, relative_thresh)
    silence = (rms_db < threshold).astype(np.float64)

    # times in seconds (each step covers 1 / SAMPLE_FPS seconds).
    times = np.arange(n, dtype=np.float64) / config.SAMPLE_FPS
    return AudioFeatures(
        times=times,
        rms_db=rms_db,
        zcr=zcr,
        spectral_centroid=centroid,
        spectral_flatness=flatness,
        spectral_flux=flux,
        music_likeness=music,
        silence=silence,
        sr=sr,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Silence segments
# ---------------------------------------------------------------------------

def silence_intervals(audio: AudioFeatures,
                       min_sec: float = config.SILENCE_MIN_SEC,
                       ) -> list[tuple[float, float]]:
    """Return list of (start, end) seconds of contiguous silence.

    The flag array is at SAMPLE_FPS Hz; indices are converted back to
    seconds via ``idx_to_sec``.
    """
    flags = audio.silence.astype(bool)
    intervals: list[tuple[float, float]] = []
    in_run = False
    start = 0
    min_steps = config.sec_to_width(min_sec)
    for i, v in enumerate(flags):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            if i - start >= min_steps:
                intervals.append((config.idx_to_sec(start),
                                   config.idx_to_sec(i)))
            in_run = False
    if in_run and len(flags) - start >= min_steps:
        intervals.append((config.idx_to_sec(start),
                           config.idx_to_sec(len(flags))))
    return intervals
