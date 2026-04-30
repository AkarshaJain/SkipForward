"""Generate a timeline PNG that visualises the segmentation result.

Top row: predicted segments (coloured strip).
Below:   the ad-likeness score curve.
If ground truth is provided, a second strip shows GT ads for comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from . import config


def _draw_strip(ax, segments, *, label_to_color, y_low, y_high, title):
    for s in segments:
        ax.add_patch(mpatches.Rectangle(
            (s["start"], y_low), s["end"] - s["start"], y_high - y_low,
            facecolor=label_to_color.get(s["label"], "#888"), edgecolor="none",
        ))
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_yticks([])


def render_timeline(*,
                     metadata: dict,
                     score: np.ndarray | None = None,
                     ground_truth: dict | None = None,
                     output_path: Path | str) -> Path:
    duration = float(metadata["duration_seconds"])
    segments = metadata["timeline_map"]

    fig_h = 4.5 if (score is not None or ground_truth is not None) else 2.0
    fig, axes = plt.subplots(
        2 if score is not None else 1, 1,
        figsize=(14, fig_h),
        gridspec_kw={"height_ratios": [1, 2]} if score is not None else None,
        sharex=True,
    )
    if score is None:
        axes = [axes]

    ax_strip = axes[0]
    _draw_strip(ax_strip, segments,
                label_to_color=config.LABEL_COLORS,
                y_low=0.6, y_high=1.0,
                title=f"Predicted segments — {metadata['video_id']}")
    ax_strip.set_xlim(0, duration)
    ax_strip.set_ylim(0, 1.0)

    if ground_truth is not None:
        gt_segs = []
        for seg in ground_truth.get("timeline_segments", []):
            if seg.get("type") == "ad":
                gt_segs.append({
                    "start": seg["final_video_start_seconds"],
                    "end":   seg["final_video_end_seconds"],
                    "label": "ad",
                })
            else:
                gt_segs.append({
                    "start": seg["final_video_start_seconds"],
                    "end":   seg["final_video_end_seconds"],
                    "label": "core_content",
                })
        _draw_strip(ax_strip, gt_segs,
                    label_to_color=config.LABEL_COLORS,
                    y_low=0.05, y_high=0.45,
                    title="")
        ax_strip.text(0, 0.25, "GT", va="center", ha="right", fontsize=9)
        ax_strip.text(0, 0.8, "Pred", va="center", ha="right", fontsize=9)

    if score is not None:
        ax_score = axes[1]
        t = np.arange(len(score))
        ax_score.plot(t, score, color="#222", linewidth=1.0)
        ax_score.axhline(config.AD_SCORE_THRESHOLD, color="#c62828",
                          linewidth=0.8, linestyle="--",
                          label=f"threshold ({config.AD_SCORE_THRESHOLD:.2f})")
        ax_score.fill_between(t, 0, score,
                                where=score >= config.AD_SCORE_THRESHOLD,
                                color="#c62828", alpha=0.20, step="mid")
        ax_score.set_xlim(0, duration)
        ax_score.set_ylim(0, 1)
        ax_score.set_xlabel("seconds")
        ax_score.set_ylabel("normalised\nad score")
        ax_score.legend(loc="upper right", fontsize=8)
        ax_score.grid(True, alpha=0.2)

    # Legend with all labels actually used
    used = sorted({s["label"] for s in segments})
    handles = [mpatches.Patch(color=config.LABEL_COLORS.get(l, "#888"), label=l)
               for l in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(used),
               fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
