"""Evaluation against the provided ground-truth JSONs in ``video_info/``.

We compare predicted *non-content* (specifically ``ad``) regions vs. the
ground-truth ``timeline_segments`` of type ``"ad"``.

Metrics reported:
    - per-second precision / recall / F1 / IoU on the binary ad mask
    - per-region matching: each GT ad gets the best-overlapping predicted
      ad segment; we report mean IoU and detection rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class EvalReport:
    duration_sec: float
    per_second_precision: float
    per_second_recall: float
    per_second_f1: float
    per_second_iou: float
    region_detection_rate: float
    region_mean_iou: float
    matched_regions: list[dict]
    summary: str

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 3),
            "per_second": {
                "precision": round(self.per_second_precision, 4),
                "recall":    round(self.per_second_recall, 4),
                "f1":        round(self.per_second_f1, 4),
                "iou":       round(self.per_second_iou, 4),
            },
            "regions": {
                "detection_rate": round(self.region_detection_rate, 4),
                "mean_iou":       round(self.region_mean_iou, 4),
                "matches":        self.matched_regions,
            },
            "summary": self.summary,
        }


def _gt_ad_intervals(gt: dict) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for seg in gt.get("timeline_segments", []):
        if seg.get("type") == "ad":
            out.append((float(seg["final_video_start_seconds"]),
                        float(seg["final_video_end_seconds"])))
    return out


def _pred_ad_intervals(pred_segments: list[dict]) -> list[tuple[float, float]]:
    """All predicted *non-content* regions, regardless of fine label.

    The user-facing goal is to skip non-core content; whether our system
    labelled a region as ``ad`` vs. ``silence`` vs. ``intro`` is internal
    bookkeeping. From a viewer's standpoint they're all "skip me".
    """
    out: list[tuple[float, float]] = []
    for s in pred_segments:
        if s["label"] in config.NON_CONTENT_LABELS:
            out.append((float(s["start"]), float(s["end"])))
    return out


def _intervals_to_mask(intervals, duration_sec: float) -> np.ndarray:
    n = max(int(np.ceil(duration_sec)), 1)
    mask = np.zeros(n, dtype=bool)
    for s, e in intervals:
        a = max(0, int(np.floor(s)))
        b = min(n, int(np.ceil(e)))
        if b > a:
            mask[a:b] = True
    return mask


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-6)


def evaluate(*,
             ground_truth: dict,
             predicted_segments: list[dict],
             duration_sec: float,
             ) -> EvalReport:
    gt_intervals = _gt_ad_intervals(ground_truth)
    pred_intervals = _pred_ad_intervals(predicted_segments)

    gt_mask = _intervals_to_mask(gt_intervals, duration_sec)
    pr_mask = _intervals_to_mask(pred_intervals, duration_sec)

    tp = int(np.sum(gt_mask & pr_mask))
    fp = int(np.sum(~gt_mask & pr_mask))
    fn = int(np.sum(gt_mask & ~pr_mask))
    union = int(np.sum(gt_mask | pr_mask))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    iou = tp / max(union, 1)

    # Region matching
    matches: list[dict] = []
    detected = 0
    ious: list[float] = []
    for gt in gt_intervals:
        if not pred_intervals:
            best = (0.0, None)
        else:
            best = max(((_interval_iou(gt, p), p) for p in pred_intervals),
                       key=lambda x: x[0])
        gt_iou, gt_match = best
        if gt_iou > 0.30:
            detected += 1
        ious.append(gt_iou)
        matches.append({
            "gt_start": round(gt[0], 3),
            "gt_end": round(gt[1], 3),
            "best_pred_start": round(gt_match[0], 3) if gt_match else None,
            "best_pred_end":   round(gt_match[1], 3) if gt_match else None,
            "iou": round(gt_iou, 3),
        })

    detection_rate = detected / max(len(gt_intervals), 1)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    summary = (
        f"GT ads: {len(gt_intervals)} | "
        f"Predicted non-content regions: {len(pred_intervals)} | "
        f"Detected (IoU>0.30): {detected}/{len(gt_intervals)} "
        f"({detection_rate:.0%}) | "
        f"Mean region IoU: {mean_iou:.2f} | "
        f"Per-second F1: {f1:.2f}"
    )

    return EvalReport(
        duration_sec=duration_sec,
        per_second_precision=precision,
        per_second_recall=recall,
        per_second_f1=f1,
        per_second_iou=iou,
        region_detection_rate=detection_rate,
        region_mean_iou=mean_iou,
        matched_regions=matches,
        summary=summary,
    )
