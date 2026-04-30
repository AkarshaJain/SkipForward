"""Post-processing: merge tiny adjacent same-label segments, enforce minimum
durations, and produce a clean, presentable timeline.
"""

from __future__ import annotations

from . import config
from .segmenter import Segment


def merge_adjacent(segments: list[Segment],
                    max_gap: float = 0.25) -> list[Segment]:
    """Merge consecutive segments that share a label and have a tiny gap."""
    if not segments:
        return segments
    out = [segments[0]]
    for cur in segments[1:]:
        prev = out[-1]
        if prev.label == cur.label and cur.start - prev.end <= max_gap:
            prev.end = max(prev.end, cur.end)
            prev.confidence = max(prev.confidence, cur.confidence)
        else:
            out.append(cur)
    return out


def absorb_tiny_core(segments: list[Segment],
                      min_core_sec: float = 6.0) -> list[Segment]:
    """A tiny ``core_content`` between two ad segments is almost certainly
    an artefact: absorb it into the surrounding non-content."""
    if len(segments) < 3:
        return segments
    out = [segments[0]]
    i = 1
    while i < len(segments) - 1:
        prev = out[-1]
        cur = segments[i]
        nxt = segments[i + 1]
        if (cur.label == config.LABEL_CORE
                and cur.end - cur.start < min_core_sec
                and prev.label in config.NON_CONTENT_LABELS
                and nxt.label in config.NON_CONTENT_LABELS
                and prev.label == nxt.label):
            # Extend prev over cur and nxt
            prev.end = nxt.end
            prev.confidence = max(prev.confidence, nxt.confidence)
            i += 2
        else:
            out.append(cur)
            i += 1
    if i == len(segments) - 1:
        out.append(segments[-1])
    return out


def enforce_min_duration(segments: list[Segment],
                          min_sec: float = config.MIN_NONCONTENT_DURATION_SEC,
                          ) -> list[Segment]:
    """Convert too-short non-content regions into ``core_content`` then merge."""
    cleaned: list[Segment] = []
    for s in segments:
        if (s.label in config.NON_CONTENT_LABELS
                and s.label != config.LABEL_SILENCE
                and (s.end - s.start) < min_sec):
            cleaned.append(Segment(s.start, s.end, config.LABEL_CORE,
                                    confidence=0.5))
        else:
            cleaned.append(s)
    return merge_adjacent(cleaned)


def finalize(segments: list[Segment]) -> list[Segment]:
    """Apply the full clean-up sequence."""
    s = merge_adjacent(segments)
    s = absorb_tiny_core(s)
    s = enforce_min_duration(s)
    s = merge_adjacent(s)
    return s
