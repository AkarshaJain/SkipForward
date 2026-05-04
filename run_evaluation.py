"""Re-evaluate already-produced segmentation outputs against ground truth.

Usage::
    python run_evaluation.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline import config
from pipeline.data_loader import discover_videos
from pipeline.evaluator import evaluate


def main() -> int:
    items = discover_videos()
    rows = []
    for item in items:
        seg_path = config.SEGMENTS_DIR / f"{item.video_id}.json"
        if not seg_path.exists():
            continue
        gt = item.load_ground_truth()
        if gt is None:
            continue
        meta = json.loads(seg_path.read_text(encoding="utf-8"))
        report = evaluate(
            ground_truth=gt,
            predicted_segments=meta["segments"],
            duration_sec=float(meta["duration_seconds"]),
        )
        eval_path = config.EVAL_DIR / f"{item.video_id}.json"
        eval_path.write_text(json.dumps(report.to_dict(), indent=2),
                              encoding="utf-8")
        rows.append((item.video_id, report))

    print()
    print("=" * 78)
    print("RE-EVALUATION RESULTS")
    print("=" * 78)
    print(f"{'video_id':<14} {'duration':>9} {'F1':>5} {'IoU':>5} "
          f"{'P':>5} {'R':>5} {'detect':>7} {'mIoU':>5}  notes")
    print("-" * 78)
    macro = {"f1": 0.0, "iou": 0.0, "detect": 0.0, "miou": 0.0}
    for vid, r in rows:
        print(f"{vid:<14} {r.duration_sec:>8.1f}s "
              f"{r.per_second_f1:.2f} {r.per_second_iou:.2f} "
              f"{r.per_second_precision:.2f} {r.per_second_recall:.2f} "
              f"{r.region_detection_rate:>6.0%}  {r.region_mean_iou:.2f}  "
              f"matches={[round(m['iou'], 2) for m in r.matched_regions]}")
        macro["f1"] += r.per_second_f1
        macro["iou"] += r.per_second_iou
        macro["detect"] += r.region_detection_rate
        macro["miou"] += r.region_mean_iou
    n = max(len(rows), 1)
    print("-" * 78)
    print(f"MACRO AVG    {'':9} {macro['f1']/n:.2f} {macro['iou']/n:.2f}"
          f" {' ':5} {' ':5} {macro['detect']/n:>6.0%}  {macro['miou']/n:.2f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
