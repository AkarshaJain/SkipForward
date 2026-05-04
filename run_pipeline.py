"""Command-line entry point for the multimodal segmentation pipeline.

Examples
--------
Process every video in ``videos_with_ads/`` (with Whisper if installed)::

    python run_pipeline.py

Process a single video already in the dataset::

    python run_pipeline.py --only test_001

Process ANY video file (anywhere on disk) -- it will be copied into
``videos_with_ads/`` so the player can serve it::

    python run_pipeline.py --video "D:/somewhere/my_lecture.mp4"
    python run_pipeline.py --video clip.mkv --id my_clip   # custom video_id

Skip Whisper (e.g. if you don't have the model weights)::

    python run_pipeline.py --no-whisper

Force re-extraction of audio (ignore cache)::

    python run_pipeline.py --no-cache
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr so the script does not crash on Windows when
# printing non-ASCII characters (the default cp1252 console codec cannot
# encode common arrows / em-dashes / accented characters).
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline import config
from pipeline.pipeline import run_all, run_video


def _import_external_video(src: Path, video_id: str | None = None) -> str:
    """Copy ``src`` into ``videos_with_ads/`` so the pipeline + player
    can find it. Returns the resolved video_id (filename stem).
    """
    if not src.exists():
        sys.exit(f"ERROR: video not found: {src}")
    suffix = src.suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".avi", ".webm"}:
        sys.exit(f"ERROR: unsupported video format: {suffix}")
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    vid = video_id or src.stem
    dest = config.VIDEOS_DIR / f"{vid}{suffix}"
    if dest.resolve() != src.resolve():
        if dest.exists():
            print(f"NOTE: {dest.name} already exists in videos_with_ads/, "
                  "reusing it (delete first if you want to re-import).")
        else:
            print(f"Copying {src} -> {dest}")
            shutil.copy2(src, dest)
    return vid


def main() -> int:
    p = argparse.ArgumentParser(description="Multimodal video segmenter.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Only process these video ids (e.g. test_001 test_003).")
    p.add_argument("--video", type=str, default=None,
                   help="Process ANY video file by path (will be imported "
                        "into videos_with_ads/).")
    p.add_argument("--id", type=str, default=None,
                   help="Custom video_id when used with --video "
                        "(default = filename stem).")
    p.add_argument("--no-whisper", action="store_true",
                   help="Disable Whisper speech transcription.")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-extracting audio even if cached.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress most logs (still prints final summary).")
    args = p.parse_args()

    if args.video:
        vid = _import_external_video(Path(args.video), args.id)
        only_filter = [vid]
    else:
        only_filter = args.only

    results = run_all(
        only=only_filter,
        use_whisper=not args.no_whisper,
        cache=not args.no_cache,
        quiet=args.quiet,
    )

    print()
    print("=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    print(f"{'video_id':<14} {'duration':>9} {'segments':>9} "
          f"{'nc_secs':>8} {'F1':>5} {'IoU':>5} {'detect':>7}  elapsed")
    print("-" * 78)
    for r in results:
        m = r.metadata
        nc = m["summary"]["non_content_seconds"]
        eval_ = r.eval_report or {}
        per_s = eval_.get("per_second", {})
        regs = eval_.get("regions", {})
        f1 = per_s.get("f1")
        iou = per_s.get("iou")
        det = regs.get("detection_rate")
        print(f"{r.video_id:<14} {m['duration_seconds']:>8.1f}s "
              f"{m['summary']['total_segments']:>9} "
              f"{nc:>7.1f}s "
              f"{f1 if f1 is not None else '-':>5} "
              f"{iou if iou is not None else '-':>5} "
              f"{det if det is not None else '-':>7}  "
              f"{r.elapsed_sec:.1f}s")
    print("=" * 78)
    print(f"Outputs: ./outputs/segments/  ./outputs/timelines/  ./outputs/evaluation/")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
