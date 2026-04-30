"""Discover input videos and (optionally) load matching ground-truth metadata.

The dataset layout is expected to be::

    videos_with_ads/test_001.mp4
    video_info/test_001.json   (optional, ground truth)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import config


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


@dataclass
class VideoItem:
    """A single video to process."""

    video_id: str
    path: Path
    ground_truth_path: Path | None = None

    def load_ground_truth(self) -> dict | None:
        """Return the matching ground-truth dict if it exists, else None."""
        if self.ground_truth_path and self.ground_truth_path.exists():
            with self.ground_truth_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None


def discover_videos(
    videos_dir: Path | str = config.VIDEOS_DIR,
    ground_truth_dir: Path | str = config.GROUND_TRUTH_DIR,
    only: Iterable[str] | None = None,
) -> list[VideoItem]:
    """Return every video in ``videos_dir`` (optionally filtered by ``only``)."""
    videos_dir = Path(videos_dir)
    ground_truth_dir = Path(ground_truth_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    only_set = set(only) if only else None
    items: list[VideoItem] = []

    for path in sorted(videos_dir.iterdir()):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        video_id = path.stem
        if only_set and video_id not in only_set:
            continue
        gt_path = ground_truth_dir / f"{video_id}.json"
        items.append(
            VideoItem(
                video_id=video_id,
                path=path,
                ground_truth_path=gt_path if gt_path.exists() else None,
            )
        )

    return items
