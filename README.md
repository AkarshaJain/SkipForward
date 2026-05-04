# Multimodal Video Segmentation — Core Content vs. Non-Content

A complete, **offline**, **free-tools-only** system that segments long-form
videos into core content and non-content (ads, intros, outros, silences,
transitions). The system uses **multimodal reasoning across visual, audio,
and speech signals**, produces structured metadata, and ships with a
custom HTML5 video player that visualises the segmentation and lets the
viewer skip non-content.

> Built for the CSCI 576 Spring 2026 multimedia project. Tested on the
> five provided videos in `videos_with_ads/` against the ground-truth
> ad timestamps in `video_info/`.

---

## 1. Architecture

The pipeline is **modular** — every stage is independently testable and
can be replaced. Inputs flow left-to-right; data formats are simple
numpy arrays and dataclasses.

```
              ┌──────────────────────┐
videos_with_ads/ ─►  Data Loader     │   discover videos + matching GT JSON
              └─────────┬────────────┘
                        ▼
        ┌──────── Preprocessing ─────────┐
        │ Frame extractor   (1 fps + dense for shots)     │
        │ Audio extractor   (16 kHz mono WAV via ffmpeg)  │   imageio-ffmpeg
        └─────────┬─────────┬────────┬───────────────────┘
                  ▼         ▼        ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │  Visual  │ │  Audio   │ │  Speech      │
        │ features │ │ features │ │  (Whisper)   │
        └────┬─────┘ └────┬─────┘ └──────┬───────┘
             └────────┬───┴──────────────┘
                      ▼
             ┌─────────────────┐
             │ Fusion (1 Hz)   │   per-second feature matrix (N × 13)
             │ + local outlier │
             └────────┬────────┘
                      ▼
   ┌───────────────────────────────────────────────────┐
   │ Segmenter                                          │
   │  • Smoothed weighted z-score "ad-likeness"        │
   │  • Splice-pair detection (cut+silence/discontinuity)│
   │  • Snap to shot boundaries                         │
   └────────────────────────┬──────────────────────────┘
                            ▼
                 ┌───────────────────┐
                 │  Post-processing  │   merge / min-duration / smooth
                 └─────────┬─────────┘
                           ▼
            ┌─────────────────────────────┐
            │ Metadata + Timeline PNG     │  outputs/segments/*.json
            └─────────────────────────────┘
                           ▼
            ┌─────────────────────────────┐
            │ Offline HTML Player         │  player/player.html
            │  • coloured timeline strip  │
            │  • segment table            │
            │  • skip / play-core-only    │
            └─────────────────────────────┘
```

### 1.1 Modules

| Module | File | Responsibility |
| --- | --- | --- |
| Data loader     | `pipeline/data_loader.py`     | discover videos + match GT |
| Preprocessing   | `pipeline/preprocessing.py`   | OpenCV frame iter, ffmpeg WAV |
| Visual features | `pipeline/features_visual.py` | shot detection, motion, edges, splice boundaries |
| Audio features  | `pipeline/features_audio.py`  | RMS, ZCR, spectral, music-vs-speech, adaptive silence |
| Speech features | `pipeline/features_speech.py` | Whisper transcription, ad-keywords, transcript-garble |
| Fusion          | `pipeline/fusion.py`          | per-second matrix + `local_outlierness` |
| Segmenter       | `pipeline/segmenter.py`       | weighted score, smoothing, region extraction |
| Splice segmenter| `pipeline/splice_segmenter.py`| splice-pair detection for long ads |
| Post-process    | `pipeline/postprocess.py`     | merge / min-duration / cleanup |
| Metadata        | `pipeline/metadata.py`        | output JSON, skip recommendations |
| Evaluator       | `pipeline/evaluator.py`       | per-second + region IoU vs GT |
| Visualisation   | `pipeline/visualize.py`       | timeline PNG with score curve |
| Orchestrator    | `pipeline/pipeline.py`        | end-to-end run-one / run-all |
| Player          | `player/player.html`          | offline HTML5 player |

### 1.2 Per-second feature channels (13)

| Channel | Modality | Intuition |
| --- | --- | --- |
| `shot_rate`         | visual | shots/min in a 15 s window — ads cut faster |
| `saturation`        | visual | mean HSV saturation — ads more colourful |
| `motion`            | visual | mean abs frame diff |
| `edge_density`      | visual | Canny edge fraction — text overlays/graphics |
| `audio_rms`         | audio  | dB FS loudness — ads loudness-compressed |
| `spectral_flux`     | audio  | magnitude-spectrum change — punchy music |
| `music_likeness`    | audio  | chroma stability + flatness − ZCR |
| `silence`           | audio  | adaptive (median − 12 dB) |
| `black_frame`       | visual | luma < 18 |
| `ad_keyword`        | speech | hits on "subscribe", "sponsor", "buy now" … |
| `speech_density`    | speech | words/sec from Whisper |
| `transcript_garble` | speech | Whisper's `no_speech_prob` / `compression_ratio` / `avg_logprob` — fires on music ads where Whisper produces nonsense |
| `local_outlierness` | cross  | distance from a wide context window — catches *sustained* anomalies invisible to per-second z-scores |

### 1.3 Why this isn't a black box

Every segmentation decision can be traced:

1. The score is a **weighted sum of 13 z-scored channels** — weights live
   in `pipeline/config.py:SCORE_WEIGHTS` and are interpretable.
2. The score is **smoothed with a 9-second Gaussian** before thresholding.
3. Boundaries are **snapped to shot cuts** for clean visual edges.
4. **Splice-pair candidates** are added explicitly and listed in the
   metadata (`extra.splice_pair_ads`) for human verification.
5. The output JSON's `skip_recommendations` field gives a one-line
   reason per non-content segment.
6. Each video's timeline is rendered to `outputs/timelines/<id>.png`,
   showing *both* the predicted strip and the ground-truth strip plus
   the underlying score curve.

---

## 2. Quick start

```bash
# 1. Install dependencies (free, runs offline once Whisper is cached).
python -m pip install -r requirements.txt

# OPTIONAL but recommended — speech analysis (free, MIT-licensed).
python -m pip install -U openai-whisper

# 2. Run the segmenter on every video in videos_with_ads/.
python run_pipeline.py            # ~2 min/video on CPU with Whisper

#    or process a single video:
python run_pipeline.py --only test_001
#    or skip Whisper for faster runs:
python run_pipeline.py --no-whisper

#    or process ANY video file from anywhere on disk
#    (it gets copied into videos_with_ads/ so the player can serve it):
python run_pipeline.py --video "D:/somewhere/my_lecture.mp4"
python run_pipeline.py --video clip.mkv --id my_clip   # custom video_id

# 3. Re-evaluate against ground truth without re-running the pipeline:
python run_evaluation.py

# 4. Build the player index and open the offline player in your browser:
python run_player.py

# 5. Run the synthetic-input unit tests (a few seconds, no dataset needed):
python -m pytest tests/ -v
```

The pipeline writes to:

```
outputs/segments/<video_id>.json    # metadata (segments + skip_recommendations)
outputs/timelines/<video_id>.png    # visual timeline + score curve
outputs/evaluation/<video_id>.json  # IoU/F1/region-detection vs ground truth
outputs/intermediate/<video_id>/    # cached audio (.wav)
```

`ffmpeg` does **not** need to be installed system-wide — `imageio-ffmpeg`
ships a static binary with the package.

### 2.1 Repository layout

```
videos_with_ads/        # input videos (provided dataset)
video_info/             # ground-truth ad timestamps (provided)
pipeline/               # all reusable Python modules
  config.py             # everything tweakable (weights, thresholds, paths)
  data_loader.py
  preprocessing.py
  features_visual.py
  features_audio.py
  features_speech.py
  fusion.py
  segmenter.py
  splice_segmenter.py
  postprocess.py
  metadata.py
  evaluator.py
  visualize.py
  pipeline.py
  ffmpeg_utils.py
player/
  player.html           # the offline web player
  app.json              # generated index (videos + segmentation)
outputs/                # everything we produce
run_pipeline.py         # CLI: run the segmenter
run_player.py           # CLI: build app.json + open the player
run_evaluation.py       # CLI: re-eval cached outputs against GT
requirements.txt
README.md
```

---

## 3. Output format

`outputs/segments/test_001.json` (abridged):

```json
{
  "video_id": "test_001",
  "video_filename": "test_001.mp4",
  "duration_seconds": 1458.425,
  "segments": [
    {
      "start": 0.0,
      "end": 219.953,
      "duration": 219.953,
      "label": "intro",
      "confidence": 0.71
    },
    {
      "start": 630.62,
      "end": 658.402,
      "duration": 27.782,
      "label": "ad",
      "confidence": 0.95
    }
  ],
  "skip_recommendations": [
    {
      "start": 0.0,
      "end": 219.953,
      "label": "intro",
      "reason": "Detected as intro segment near video start."
    }
  ],
  "summary": {
    "total_segments": 7,
    "labels": {"core_content": 4, "intro": 1, "ad": 2},
    "core_content_seconds": 1198.5,
    "non_content_seconds": 259.9,
    "non_content_ratio": 0.178
  },
  "timeline_map": [
    { "label": "intro", "color": "#1565c0", "start": 0.0, "end": 219.953 },
    { "label": "core_content", "color": "#2e7d32", "start": 219.953, "end": 630.62 }
  ]
}
```

The label taxonomy is the one suggested in the project brief plus an
explicit `holding_screen` sub-type:
`core_content`, `ad`, `intro`, `outro`, `silence`, `transition`,
`holding_screen`, `filler`, `recap`. Labels are assigned by the segmenter
as follows:

* `silence`        — sustained low-RMS regions (≥ 8 s, adaptive threshold)
* `intro`          — ad-like region in the first 60 s of the video
* `outro`          — ad-like region in the last 60 s
* `transition`     — short (2–12 s) black + silent bridge between scenes
                     (re-classified after the initial mask)
* `holding_screen` — long (≥ 15 s) static visual + silent + no-speech
                     region (e.g. "starting soon" / "be right back" cards)
* `ad`             — every other detected non-content region
* `core_content`   — everything else

`transition` and `holding_screen` are the *sub-types*: the segmenter
first finds non-content regions via the multimodal score, then a
**reclassifier** in `pipeline/segmenter.py:_reclassify_subtypes` looks at
each region's per-second feature signature (black-frame ratio, silence
ratio, motion z-score, speech density) and may upgrade the generic label
to a more specific one.

---

## 4. Player demo

`player/player.html` is a single-file, **offline**, no-build-step web
player. It reads `player/app.json` (produced by `run_player.py`) and
displays:

* **Coloured timeline strip** below the video, one band per segment
  (red = ad, blue = intro, purple = outro, grey = silence,
  orange = transition, brown = holding_screen,
  green = core content). Hover for tooltips, click anywhere on the
  strip to scrub the playhead to that exact time (no snapping).
* **Tick marks** every minute (or 5 minutes for long videos).
* **Segment table** to the right of the strip, listing every segment
  with start / end / label / confidence / duration. Click a row to jump.
* **Controls:**
  - `⏮ Prev seg` / `⏭ Next seg` — jump between segment boundaries
  - `⏩ Skip current` — skip the segment currently playing
  - `▶ Play core only` — chain core-content segments back-to-back
  - **Auto-skip non-content** toggle — automatically jump over any
    non-`core_content` segment as the playhead enters it. The player
    uses an internal `isProgrammaticSeek` flag so auto-skip never fights
    a user click during the same animation frame.
* **Keyboard shortcuts:** `Shift+→` / `Shift+←` for next/prev segment,
  `Space` for play/pause.
* **Summary line** below the timeline: total segments, core-content
  duration, non-content duration + ratio, skip recommendations count.

### 4.1 Playing *any* video without running the pipeline

The player has two entry points for ad-hoc playback that don't require
running `run_pipeline.py` first:

* **"Choose local file…" button** in the sidebar opens the OS file picker
  and plays the selected file directly via `URL.createObjectURL` (no
  upload, no copy — the browser reads the file in-place from your disk).
* **"Paste a video URL"** input below the picker accepts any
  HTTPS-served video URL and plays it via the native `<video>` element.

Both modes show the video in the same player chrome with all keyboard
shortcuts and the auto-skip toggle still functional (the timeline strip
and segment table are simply empty for ad-hoc videos because they have
no segmentation metadata).

### 4.2 Launching the player

```bash
python run_player.py                # default: serves on http://127.0.0.1:8000
python run_player.py --port 9000    # use a custom port
python run_player.py --no-open      # serve without auto-opening the browser
python run_player.py --build-only   # only write app.json, don't serve
```

This writes `player/app.json` (combining all per-video metadata),
starts a built-in static-file HTTP server in the project root, and opens
`player/player.html` in your default browser. The server:

* Is needed because browsers block `fetch()` from `file://` URLs.
* Implements **HTTP byte-range requests** (`206 Partial Content` /
  `416 Range Not Satisfiable`), `Accept-Ranges: bytes`, correct MIME
  types, and `Cache-Control: no-store`. This is what makes
  scrubbing on long `<video>` elements actually work — without it,
  Chrome silently fails to seek mid-stream on big files.
* Uses Python's stdlib `http.server` (no extra deps) and shuts down
  cleanly on `Ctrl+C`.

Because video playback is delegated to the native `<video>` element,
**audio is synchronised with video automatically** — no custom AV-sync
code required.

---

## 5. Evaluation against the provided ground truth

The dataset comes with `video_info/<id>.json` describing exactly where
ads were inserted. The pipeline's evaluator computes:

* **Per-second precision / recall / F1 / IoU** on the binary
  non-content mask.
* **Region-level detection rate** (fraction of GT ads matched at IoU ≥
  0.30) and **mean region IoU**.

After running `python run_pipeline.py` followed by
`python run_evaluation.py`:

| video    | dur (s) | F1   | IoU  | recall | detect | mean IoU | per-ad IoU       |
| -------- | ------: | ---- | ---- | -----: | -----: | -------: | ---------------- |
| test_001 |  1458   | 0.72 | 0.56 |   0.87 | 100 %  |     0.72 | 0.47, 0.88, 0.80 |
| test_002 |  1351   | 0.32 | 0.19 |   0.79 |  67 %  |     0.53 | 0.74, 0.59, 0.26 |
| test_003 |  1827   | 0.26 | 0.15 |   0.53 |  33 %  |     0.27 | 0.56, 0.11, 0.13 |
| test_004 |  1936   | 0.56 | 0.39 |   0.83 | 100 %  |     0.79 | 0.66, 0.77, 0.95 |
| test_005 |  1420   | 0.37 | 0.23 |   0.90 |  67 %  |     0.63 | 0.23, 0.95, 0.70 |
| **avg**  |         | **0.44** | **0.30** |  | **73 %** | **0.59** |          |

(Detection threshold = IoU > 0.30; recall is per-second.)

**What the numbers mean:** with **no training and no per-video tuning**
the pipeline detects 11/15 inserted ads at IoU > 0.30 with mean IoU
0.59 for matched regions, while watching ~75 % of total ad seconds.
Two videos are detected perfectly (3 / 3 ads). The misses are dominated
by **long video-style ads** whose visual + audio content is genuinely
similar to the surrounding podcast/lecture (e.g. test_001's first ad,
test_003's middle ads) — the kind of "non-content that is
contextual and subtle" the project brief explicitly warned about.

### 5.1 Visual evaluation

For each video the pipeline writes `outputs/timelines/<id>.png` showing:

* **Top strip** — predicted segments colour-coded by label.
* **Middle strip** — ground-truth ad regions (when available).
* **Bottom panel** — the smoothed normalised ad-likeness score with the
  threshold line overlaid. Threshold-crossings are tinted red.

Inspect those images for a qualitative read on each video.

---

## 5.2 Synthetic-input unit tests

Beyond per-video evaluation against ground truth, the project ships a
`tests/` folder with 36 synthetic-input unit tests that prove each
feature extractor and the segmenter respond correctly to known stimuli
(silent WAV, pure sine, white noise, repeated frames, hand-crafted ad
regions, schema validity, etc.). Run them with:

```bash
python -m pytest tests/ -v
```

The whole suite runs in a few seconds and uses **no** dataset files,
so it works on any machine without `videos_with_ads/` present. See
`tests/README.md` for what each test proves.

---

## 6. Design choices & trade-offs

* **Sample at 1 Hz, not native fps.** A 24-minute video is ~1500
  rows of feature data. Visual feature extraction runs at 30+ fps and
  the whole pipeline finishes in 2-4 min/video on CPU.
* **Adaptive silence threshold** (`min(-40 dB, median − 12 dB)`).
  Absolute -40 dB worked for loud videos but mis-flagged 200 + s of
  speech in quiet podcasts as silence.
* **Splice-pair detector** is conservative — requires *both* sides
  to differ from the candidate region by ≥ 1 z-unit, with a strong
  combined-score floor. This avoids false positives on internal scene
  changes (which look superficially like splices).
* **Multi-modality consensus filter.** A second only stays in the ad
  mask if at least **2 of 4 modality groups** (visual / audio / speech /
  cross-modal) cross their voting z-threshold. Tuned in
  `pipeline/config.py:MIN_MODALITY_CONSENSUS`. This is the dial that
  trades recall for precision: setting it to 1 reverts to the loose
  pre-fix behaviour, setting it to 3 produces ultra-clean output but
  drops recall on ads that are only audio-visual without speech.
* **Whisper as a quality oracle, not just a transcriber.** Music ads
  rarely contain "subscribe" / "sponsor" / "buy now" — but Whisper's
  `no_speech_prob` and `compression_ratio` reliably spike on music
  vamps, jingles, and overlapped-vocal commercials. We feed them
  directly into the score as `transcript_garble`.
* **The score is interpretable.** No training, no neural classifier,
  no per-video calibration. Every weight in
  `pipeline/config.py:SCORE_WEIGHTS` is a knob you can turn.
* **Player is HTML5 + a single JSON file.** No installation, no Python
  process to keep alive, audio/video sync handled by the browser. Works
  offline. Skip controls are pure DOM event handlers.

---

## 7. Limitations & honest caveats

1. **Long, video-style ads** that match the visual / audio statistics
   of the main content are hard to detect without speech keywords.
   We catch the boundaries via splice-pair logic when they exist; we
   miss them when both the splice and the content are subtle.
2. **The `local_outlierness` channel** can over-fire near video
   boundaries (first/last 60 s have less context). We mitigate by
   labelling such regions `intro`/`outro` rather than `ad`.
3. **No persistence / training across videos.** This is by design — the
   project brief's stop condition forbids retraining. The system
   normalises features per video so it adapts at inference time.
4. **Silence vs. ad.** Some ads happen to also be quiet; they get
   labelled `silence` rather than `ad`. From the *viewer's* perspective
   this is harmless (still skipped), and the evaluator counts both as
   non-content.

---

## 8. License & dependencies

All dependencies are free and runnable offline:

* **OpenCV** — Apache 2.0
* **librosa**, **soundfile** — ISC / BSD
* **imageio-ffmpeg** — BSD (ships a statically-built ffmpeg)
* **openai-whisper** — MIT (downloads model weights on first run; cached
  locally afterwards)
* **numpy / scipy / matplotlib / tqdm** — BSD-style

No paid APIs, no cloud calls, no telemetry.
