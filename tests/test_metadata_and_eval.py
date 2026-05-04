"""Schema + evaluator tests.

Proves:
  - the JSON schema produced by metadata.build_metadata matches the spec:
    contains video_id / duration / segments / skip_recommendations / summary
    / timeline_map, and every segment has a colour code.
  - evaluator.evaluate computes the right precision / recall / F1 / IoU on
    a tiny hand-crafted ground truth + prediction pair.
"""

from __future__ import annotations

import math

import pytest

from pipeline import config
from pipeline.metadata import build_metadata
from pipeline.segmenter import Segment
from pipeline.evaluator import evaluate


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------

def _three_segments():
    return [
        Segment(0.0, 30.0, config.LABEL_INTRO, 0.9),
        Segment(30.0, 90.0, config.LABEL_CORE, 0.85),
        Segment(90.0, 120.0, config.LABEL_AD, 0.92),
    ]


def test_metadata_schema_keys():
    md = build_metadata(
        video_id="t", video_filename="t.mp4",
        duration_seconds=120.0, segments=_three_segments(),
    )
    for key in ("video_id", "video_filename", "duration_seconds",
                 "segments", "skip_recommendations", "summary",
                 "timeline_map"):
        assert key in md, f"missing required key {key}"


def test_metadata_summary_counts_match_segments():
    md = build_metadata(
        video_id="t", video_filename="t.mp4",
        duration_seconds=120.0, segments=_three_segments(),
    )
    s = md["summary"]
    assert s["total_segments"] == 3
    assert s["labels"][config.LABEL_CORE] == 1
    assert s["labels"][config.LABEL_AD] == 1
    assert s["labels"][config.LABEL_INTRO] == 1
    assert s["core_content_seconds"] == pytest.approx(60.0)
    assert s["non_content_seconds"] == pytest.approx(60.0)
    assert s["non_content_ratio"] == pytest.approx(0.5)


def test_skip_recommendations_only_for_non_content():
    md = build_metadata(
        video_id="t", video_filename="t.mp4",
        duration_seconds=120.0, segments=_three_segments(),
    )
    skip_labels = {r["label"] for r in md["skip_recommendations"]}
    assert config.LABEL_CORE not in skip_labels
    assert config.LABEL_AD in skip_labels
    assert config.LABEL_INTRO in skip_labels


def test_timeline_map_has_colours():
    md = build_metadata(
        video_id="t", video_filename="t.mp4",
        duration_seconds=120.0, segments=_three_segments(),
    )
    for entry in md["timeline_map"]:
        assert entry["color"].startswith("#") and len(entry["color"]) == 7


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _gt_with_ad(start: float, end: float) -> dict:
    return {"timeline_segments": [{
        "type": "ad",
        "final_video_start_seconds": start,
        "final_video_end_seconds": end,
    }]}


def _pred(label: str, start: float, end: float) -> dict:
    return {"label": label, "start": start, "end": end, "confidence": 0.9}


def test_evaluator_perfect_match_gives_f1_1():
    gt = _gt_with_ad(50.0, 100.0)
    pred = [_pred(config.LABEL_AD, 50.0, 100.0)]
    rep = evaluate(ground_truth=gt, predicted_segments=pred,
                   duration_sec=200.0)
    assert rep.per_second_precision == pytest.approx(1.0)
    assert rep.per_second_recall == pytest.approx(1.0)
    assert rep.per_second_f1 == pytest.approx(1.0)
    assert rep.region_detection_rate == pytest.approx(1.0)


def test_evaluator_no_overlap_gives_zero():
    gt = _gt_with_ad(50.0, 100.0)
    pred = [_pred(config.LABEL_AD, 110.0, 150.0)]
    rep = evaluate(ground_truth=gt, predicted_segments=pred,
                   duration_sec=200.0)
    assert rep.per_second_f1 == 0.0
    assert rep.region_detection_rate == 0.0


def test_evaluator_counts_silence_as_skip_target():
    """The evaluator MUST treat any non-content label (incl. 'silence')
    as a candidate for matching against ground-truth ads, because the
    user-facing goal is 'skip me', not 'classify me'."""
    gt = _gt_with_ad(50.0, 100.0)
    pred = [_pred(config.LABEL_SILENCE, 50.0, 100.0)]
    rep = evaluate(ground_truth=gt, predicted_segments=pred,
                   duration_sec=200.0)
    assert rep.per_second_f1 == pytest.approx(1.0)


def test_evaluator_partial_overlap_intermediate_score():
    gt = _gt_with_ad(50.0, 150.0)               # 100 s ad
    pred = [_pred(config.LABEL_AD, 100.0, 200.0)]  # 50 s overlap
    rep = evaluate(ground_truth=gt, predicted_segments=pred,
                   duration_sec=300.0)
    # Per-second precision = 50/100, recall = 50/100, F1 = 0.5
    assert rep.per_second_precision == pytest.approx(0.5, abs=1e-3)
    assert rep.per_second_recall == pytest.approx(0.5, abs=1e-3)
    assert rep.per_second_f1 == pytest.approx(0.5, abs=1e-3)
    # IoU = 50 / 150 = 0.333
    assert rep.per_second_iou == pytest.approx(1/3, abs=1e-3)
