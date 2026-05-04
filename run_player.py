"""Build the player's app.json (index of all segmented videos), start a tiny
local HTTP server in the project root (so the browser can fetch app.json
and the video files), and open the player in your default browser.

Usage::
    python run_player.py                # build + serve + open browser
    python run_player.py --no-open      # serve without opening browser
    python run_player.py --build-only   # just write app.json and exit
    python run_player.py --port 9000    # use a custom port (default 8000)

Why a server?  Browsers block ``fetch()`` from ``file://`` URLs (same-origin
policy treats them as the ``null`` origin), so opening ``player.html``
directly fails with "Could not load player/app.json". A trivial static HTTP
server fixes this without adding any dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows (default cp1252 cannot encode common
# arrows / em-dashes that may appear in URLs or log messages).
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline import config


def build_app_json() -> Path:
    """Combine every segments/<video>.json into player/app.json."""
    seg_dir = config.SEGMENTS_DIR
    out_path = config.PLAYER_DIR / "app.json"

    videos = []
    for p in sorted(seg_dir.glob("*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        # Path is served from the project root so the URL is absolute-ish:
        rel = f"/{config.VIDEOS_DIR.name}/{meta['video_filename']}"
        videos.append({
            **meta,
            "video_url": rel,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"videos": videos}, indent=2),
                         encoding="utf-8")
    return out_path


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class _RangeHandler(SimpleHTTPRequestHandler):
    """A static-file handler that supports HTTP byte-range requests.

    The default ``SimpleHTTPRequestHandler`` does NOT honour the ``Range``
    header. Browsers require it to seek inside ``<video>`` files: setting
    ``video.currentTime`` issues a partial-content request, and if the
    server returns the whole file the seek silently fails. This subclass
    adds 206-Partial-Content support.
    """

    # Always send no-cache so users see the latest player.html / app.json
    # without needing to hard-refresh.
    extensions_map = SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        ".mp4":  "video/mp4",
        ".m4v":  "video/mp4",
        ".mkv":  "video/x-matroska",
        ".webm": "video/webm",
        ".mov":  "video/quicktime",
        ".json": "application/json",
    })

    def log_message(self, format, *args):  # noqa: A002
        return

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        """Override to honour the Range header for partial-content GETs."""
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            file_size = fs.st_size
            ctype = self.guess_type(path)

            range_header = self.headers.get("Range")
            if not range_header:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Last-Modified",
                                 self.date_time_string(fs.st_mtime))
                self.end_headers()
                return f

            m = _RANGE_RE.match(range_header)
            if not m:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                f.close()
                return None
            start_s, end_s = m.group(1), m.group(2)
            if start_s == "" and end_s == "":
                self.send_error(400, "Invalid Range")
                f.close()
                return None
            if start_s == "":
                # last N bytes
                length = int(end_s)
                start = max(file_size - length, 0)
                end = file_size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else file_size - 1
            if start >= file_size or end >= file_size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                f.close()
                return None
            length = end - start + 1
            f.seek(start)
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified",
                             self.date_time_string(fs.st_mtime))
            self.end_headers()
            # Cap response body to the requested range
            return _LimitedReader(f, length)
        except Exception:
            f.close()
            raise


class _LimitedReader:
    """Wraps a file-like object so ``copyfile`` only reads ``limit`` bytes."""

    def __init__(self, fp, limit: int):
        self.fp = fp
        self.remaining = limit

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        data = self.fp.read(size)
        self.remaining -= len(data)
        return data

    def close(self):
        try:
            self.fp.close()
        except Exception:
            pass


def _find_free_port(preferred: int) -> int:
    """Try ``preferred``; if taken, pick any free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(*, port: int = 8000, open_browser: bool = True) -> None:
    """Start a static file server in the project root."""
    root = config.PROJECT_ROOT
    port = _find_free_port(port)

    handler = partial(_RangeHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/player/player.html"

    print(f"Serving {root} at http://127.0.0.1:{port}/")
    print(f"Player: {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        # Open after a short delay so the server is ready when the page asks.
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        print("\nServer stopped.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-open", action="store_true",
                   help="Serve but don't open the browser.")
    p.add_argument("--build-only", action="store_true",
                   help="Only write app.json — don't start the server.")
    p.add_argument("--port", type=int, default=8000,
                   help="HTTP server port (default 8000).")
    args = p.parse_args()

    out = build_app_json()
    print(f"Wrote {out}")

    if args.build_only:
        return 0

    html = config.PLAYER_DIR / "player.html"
    if not html.exists():
        print(f"ERROR: player file not found at {html}", file=sys.stderr)
        return 1

    serve(port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
