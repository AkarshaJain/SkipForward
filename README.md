# SkipForward — multimodal video segmentation

I wanted a practical way to mark **which parts of a long video are “the show” versus everything else**—ads, long silences, intros/outros, and similar breaks—using **sight, sound, and speech** together, without training a heavyweight model or calling paid APIs.

This repo is a **Python pipeline plus a small offline web player**. You run it on your machine; it writes JSON timelines, optional timeline images, and can stack up against ground-truth ad markers when you have them.

Originally built for a **CSCI 576 (Spring 2026) multimedia** project. The ideas here are reusable for anyone who cares about explainable skips rather than black-box tagging.

---

## What you get

- **Labels** along the timeline: things like core content, ads, intros/outros, silences, and a few transitional types—all rule-based from fused features, not an opaque classifier score you can’t inspect.
- **Whisper optional**: transcription helps (keywords, speech density, “garbly” stretches that often correlate with sung/jingle-heavy ads). The pipeline still runs without it if you skip the install or pass `--no-whisper`.
- **A browser player** (`player/player.html`) that reads a generated index, draws a coloured strip under the video, and supports skip / jump / “play mostly the meat” style controls. No build step, no React—just HTML + JS served over localhost so the browser’s `fetch` and video seeking behave.

---

## Try it quickly

```bash
python -m pip install -r requirements.txt

# Whisper is optional but improves speech-based cues:
python -m pip install -U openai-whisper

# Process everything in videos_with_ads/
python run_pipeline.py

# Or one id, or any file on disk (copied into videos_with_ads/ for serving):
python run_pipeline.py --only test_001
python run_pipeline.py --video "path/to/clip.mp4" --id my_demo

# Optional: re-score cached outputs vs video_info/ ground truth
python run_evaluation.py

# Build the player manifest and open the UI (local HTTP server)
python run_player.py

# Fast checks — no dataset required
python -m pytest tests/ -q
```

`ffmpeg` comes along via **imageio-ffmpeg**; you don’t need a separate system install for the default path.

---

## Where files land

| Path | What |
|------|------|
| `outputs/segments/*.json` | Segments, summary, skip hints |
| `outputs/timelines/*.png` | Timeline + score curve plots |
| `outputs/evaluation/*.json` | Metrics when GT exists in `video_info/` |
| `outputs/intermediate/*/` | Cached audio (e.g. WAV) per video |

---

## How it’s put together (short version)

Video and audio are sampled into **per-second features** (shots, motion, edges, loudness, music-ish vs speech-ish cues, etc.). Speech adds another slice when Whisper is there. Everything is **z-scored within each video** so quiet and loud sources both get a fair shot.

A smoothed weighted score picks candidate non-content regions; extras like **splice-style boundaries** help when an insert doesn’t look “loud” or “busy” compared to the rest. Post-processing merges fragments and enforces minimum lengths. The design is intentionally **readable**—thresholds and weights live in `pipeline/config.py` if you want to tune behavior.

---

## Player notes

Run `python run_player.py`. It writes `player/app.json`, serves the project root on **127.0.0.1** (port may shift if the default is busy), and opens the player. Range requests are supported so **seeking in long MP4s** actually works in Chrome and friends.

There’s also a **local file picker** and optional URL paste in the sidebar for playback without prior segmentation—the timeline stays empty unless you ran the pipeline for that clip.

*(If you see Windows socket errors after closing a tab mid-stream—that’s the client dropping the connection while a range response is streaming; harmless.)*

---

## Tests

`tests/` holds **pytest** suites with synthetic inputs (silence, tones, fabricated feature patterns, etc.). They’re meant to freeze extractor and fusion behavior—run them anytime, CI-friendly, no course videos required.

---

## Honest limits

Videos where the **ad looks and sounds almost like the main program** are still hard—you’re trading interpretability against perfect recall. The pipeline is biased toward setups where ads create a separable multimodal fingerprint or a clean structural boundary.

---

## Credits / stack

**Python**: NumPy/SciPy, OpenCV, librosa, optionally Whisper • **Serving**: stdlib HTTP • All **offline-friendly** dependencies; no telemetry in this repo.

If this project is useful for your own skip logic or demos, fork it or borrow the fuse + segment patterns—happy building.
