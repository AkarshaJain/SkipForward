"""Metadata generator: turns ``Segment`` lists into the final JSON output.

Output schema (matches the requirement in the project spec):

    {
      "video_id": "test_001",
      "video_filename": "test_001.mp4",
      "duration_seconds": 1458.6,
      "segments": [
        {
          "start": 0.0,
          "end": 12.5,
          "duration": 12.5,
          "label": "intro",
          "confidence": 0.92
        },
        ...
      ],
      "skip_recommendations": [
        { "start": 106.1, "end": 224.4, "label": "ad", "reason": "..."}
      ],
      "summary": {
        "total_segments": 7,
        "labels": { "core_content": 4, "ad": 3 },
        "core_content_seconds": 1280.0,
        "non_content_seconds": 178.6,
        "non_content_ratio": 0.122
      },
      "timeline_map": [
        { "label": "core_content", "start": 0, "end": 106.1 },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from . import config
from .segmenter import Segment


_REASONS = {
    config.LABEL_AD:      "Detected as advertisement (visual + audio + speech cues).",
    config.LABEL_INTRO:   "Detected as intro segment near video start.",
    config.LABEL_OUTRO:   "Detected as outro segment near video end.",
    config.LABEL_SILENCE: "Long silence / dead air.",
    config.LABEL_FILLER:  "Low-information filler segment.",
    config.LABEL_TRANSITION: "Transition / interstitial.",
    config.LABEL_RECAP:   "Likely repeated/recap content.",
}


def build_metadata(*,
                    video_id: str,
                    video_filename: str,
                    duration_seconds: float,
                    segments: Sequence[Segment],
                    extra: dict | None = None,
                    ) -> dict:
    label_counts = Counter(s.label for s in segments)
    core_secs = sum(s.end - s.start
                     for s in segments if s.label == config.LABEL_CORE)
    nc_secs = sum(s.end - s.start
                   for s in segments if s.label in config.NON_CONTENT_LABELS)

    skip_recommendations = [
        {
            "start":  round(s.start, 3),
            "end":    round(s.end, 3),
            "label":  s.label,
            "reason": _REASONS.get(s.label, "Detected as non-content."),
        }
        for s in segments
        if s.label in config.NON_CONTENT_LABELS
    ]

    timeline_map = [
        {
            "label": s.label,
            "color": config.LABEL_COLORS.get(s.label, "#888888"),
            "start": round(s.start, 3),
            "end":   round(s.end, 3),
        }
        for s in segments
    ]

    return {
        "video_id": video_id,
        "video_filename": video_filename,
        "duration_seconds": round(duration_seconds, 3),
        "segments": [s.to_dict() for s in segments],
        "skip_recommendations": skip_recommendations,
        "summary": {
            "total_segments": len(segments),
            "labels": dict(label_counts),
            "core_content_seconds":  round(core_secs, 3),
            "non_content_seconds":   round(nc_secs, 3),
            "non_content_ratio":     round(nc_secs / max(duration_seconds, 1e-6), 4),
        },
        "timeline_map": timeline_map,
        **({"extra": extra} if extra else {}),
    }


def save_metadata(metadata: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path
