"""Preprocessing: extract frames and audio from a video.

Frames are streamed via OpenCV (no temp files), audio is dumped via ffmpeg
into a deterministic location under ``outputs/intermediate/<video_id>/``.

This module is the only place that talks directly to ffmpeg / OpenCV; the
rest of the pipeline operates on numpy arrays and the audio WAV path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import config
from .ffmpeg_utils import run_ffmpeg


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


def probe_video(path: Path | str) -> VideoMeta:
    """Cheap probe via OpenCV (no ffprobe dependency)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 1e-3:
        # Fallback when OpenCV cannot read fps from the container.
        fps = 25.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return VideoMeta(fps=fps, frame_count=frame_count, duration=duration,
                     width=width, height=height)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def iter_sampled_frames(
    path: Path | str,
    sample_fps: float = config.SAMPLE_FPS,
    resize: tuple[int, int] | None = config.FRAME_RESIZE,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(timestamp_sec, frame_bgr)`` at ~``sample_fps`` Hz.

    Uses ``CAP_PROP_POS_MSEC`` -based seeking which is well-supported across
    container types and avoids extracting frames we won't use.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    meta = probe_video(path)
    duration = meta.duration if meta.duration > 0 else 1e9
    step = 1.0 / max(sample_fps, 1e-3)

    try:
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            yield t, frame
            t += step
    finally:
        cap.release()


def iter_dense_frames(
    path: Path | str,
    stride: int = config.SHOT_DETECT_STRIDE,
    resize: tuple[int, int] | None = (160, 90),
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield every ``stride``-th decoded frame at native fps (used for shots)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                yield idx / fps, frame
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio_wav(
    video_path: Path | str,
    output_path: Path | str,
    sr: int = config.AUDIO_SR,
    overwrite: bool = False,
) -> Path:
    """Extract mono PCM WAV at ``sr`` Hz using bundled ffmpeg."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    run_ffmpeg([
        "-i", str(video_path),
        "-vn",                       # no video
        "-ac", "1",                  # mono
        "-ar", str(sr),              # sample rate
        "-acodec", "pcm_s16le",
        str(output_path),
    ])
    return output_path


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def num_seconds(meta: VideoMeta) -> int:
    return int(math.ceil(meta.duration))
