"""Locate the ffmpeg binary.

We rely on ``imageio_ffmpeg`` which bundles a static ffmpeg build, so the
pipeline works on machines without a system ffmpeg install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_ffmpeg_binary() -> str:
    """Return path to a usable ffmpeg executable.

    Order of preference:
      1) ``$FFMPEG_BINARY`` env var (explicit override).
      2) ``imageio_ffmpeg`` bundled binary (preferred, no system deps).
      3) ``ffmpeg`` on PATH.
    """
    override = os.environ.get("FFMPEG_BINARY")
    if override and Path(override).exists():
        return override

    try:
        import imageio_ffmpeg  # type: ignore

        bin_path = imageio_ffmpeg.get_ffmpeg_exe()
        if bin_path and Path(bin_path).exists():
            return bin_path
    except Exception:
        pass

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    raise RuntimeError(
        "No ffmpeg binary found. Install with `pip install imageio-ffmpeg` "
        "or set FFMPEG_BINARY to your ffmpeg.exe path."
    )


def run_ffmpeg(args: list[str], *, quiet: bool = True) -> None:
    """Run ffmpeg with the given args (without the binary name)."""
    cmd = [get_ffmpeg_binary(), "-hide_banner", "-y", *args]
    if quiet:
        cmd.insert(1, "-loglevel")
        cmd.insert(2, "error")
    subprocess.run(cmd, check=True)
