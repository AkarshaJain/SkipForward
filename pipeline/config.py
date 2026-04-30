"""Global configuration: paths, hyper-parameters, label taxonomy.

Everything tweakable lives here so other modules import a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEOS_DIR = PROJECT_ROOT / "videos_with_ads"
GROUND_TRUTH_DIR = PROJECT_ROOT / "video_info"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
TIMELINES_DIR = OUTPUT_DIR / "timelines"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
EVAL_DIR = OUTPUT_DIR / "evaluation"

PLAYER_DIR = PROJECT_ROOT / "player"

for _d in (OUTPUT_DIR, SEGMENTS_DIR, TIMELINES_DIR, INTERMEDIATE_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sampling / preprocessing
# ---------------------------------------------------------------------------

# We analyse video at a coarse temporal grid of 1 frame / second.
# Shot detection internally uses a denser grid (every Nth frame at native fps)
# but the fused feature matrix is at 1 Hz to keep memory + compute small.
SAMPLE_FPS: float = 1.0

# Frame size used for visual feature extraction (smaller = faster, still expressive).
FRAME_RESIZE: Tuple[int, int] = (320, 180)

# Audio sampling rate for librosa analysis (16 kHz is standard for speech & MFCC).
AUDIO_SR: int = 16_000

# librosa frame size (in samples). 1024 @ 16 kHz = 64 ms windows.
AUDIO_FRAME_LENGTH: int = 1024
AUDIO_HOP_LENGTH: int = 512


# ---------------------------------------------------------------------------
# Shot-boundary detection
# ---------------------------------------------------------------------------

# Stride (in native video frames) for histogram-based shot detection.
SHOT_DETECT_STRIDE: int = 5

# Histogram-difference threshold above which a frame pair is flagged as a cut.
# Chi-square distance on HSV histograms. Tuned empirically.
SHOT_CUT_THRESHOLD: float = 0.35

# Minimum gap (in seconds) between two reported cuts to suppress flicker.
SHOT_MIN_GAP_SEC: float = 0.5


# ---------------------------------------------------------------------------
# Silence / black-frame detection
# ---------------------------------------------------------------------------

# RMS below this dB level (relative to full scale) is considered silence.
SILENCE_DB_THRESHOLD: float = -40.0

# Minimum duration (s) of consecutive silence to be reported as a silence event.
# Only LONG sustained silences indicate non-content (dead air, transitions).
# Short pauses are normal in natural speech and lectures.
SILENCE_MIN_SEC: float = 5.0

# Minimum duration to label a silence as its own non-content segment.
SILENCE_AS_SEGMENT_MIN_SEC: float = 8.0

# Mean luminance below this is treated as a "black/dark" frame.
BLACK_FRAME_LUMA: float = 18.0  # 0..255


# ---------------------------------------------------------------------------
# Segmenter / fusion
# ---------------------------------------------------------------------------

# Weights for the multimodal "ad-likeness" score. These are interpretable and
# tuned against the provided ground-truth videos. Setting any weight to 0
# effectively disables that signal.
SCORE_WEIGHTS = {
    # Visual
    "shot_rate":         1.0,   # ads cut faster than typical content
    "saturation":        0.8,   # ads tend to be more saturated
    "motion":            0.7,   # ads have more motion / camera change
    "edge_density":      0.5,   # text overlays / busy graphics
    # Audio
    "audio_rms":         0.6,   # ads loudness-compressed
    "spectral_flux":     0.6,   # punchy music transitions
    "music_likeness":    1.3,   # music vs. speech ratio
    # Speech
    "ad_keyword":        2.0,   # "subscribe", "sponsor", "discount" etc.
    "transcript_garble": 1.6,   # Whisper produced garbled / repetitive text
                                # (very strong signal for music ads & jingles)
    # Cross-modal local anomaly
    "local_outlierness": 1.5,   # sustained difference from surrounding video
}

# Threshold on the normalised ad-likeness score above which a second is
# considered "ad-like".
AD_SCORE_THRESHOLD: float = 0.60

# Smoothing window (seconds, full width) applied to the fused score with a
# Gaussian kernel before thresholding.
SMOOTHING_WINDOW_SEC: float = 9.0

# Minimum duration (s) for any non-content segment to be reported.
MIN_NONCONTENT_DURATION_SEC: float = 12.0

# Maximum gap (s) between two ad-like regions that should be merged.
MERGE_GAP_SEC: float = 5.0


# ---------------------------------------------------------------------------
# Intro / outro heuristics
# ---------------------------------------------------------------------------

# Anything in the first INTRO_WINDOW_SEC that looks like non-content is
# reclassified as "intro" instead of "ad".
INTRO_WINDOW_SEC: float = 60.0

# Anything in the last OUTRO_WINDOW_SEC similarly becomes "outro".
OUTRO_WINDOW_SEC: float = 60.0


# ---------------------------------------------------------------------------
# Speech / Whisper
# ---------------------------------------------------------------------------

# Whisper model size. "tiny" / "base" are fastest. The pipeline silently skips
# speech analysis if Whisper is not installed.
WHISPER_MODEL: str = "tiny"
WHISPER_LANGUAGE: str | None = "en"  # None = autodetect

# Keywords / phrases that strongly suggest advertising or self-promotion.
# Scoring: per-second binary indicator → smoothed across a window.
AD_KEYWORDS: tuple[str, ...] = (
    # call to action
    "subscribe", "like and subscribe", "smash that like", "hit the bell",
    "sign up", "click the link", "link in the description", "link below",
    "promo code", "use code", "discount code", "coupon",
    # commerce
    "sponsor", "sponsored by", "brought to you by", "today's video is sponsored",
    "save", "% off", "percent off", "free trial", "limited time", "offer",
    "buy now", "order now", "shop now", "available now", "in stores",
    # brand-style fillers
    "advertisement", "commercial",
)


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

LABEL_CORE = "core_content"
LABEL_AD = "ad"
LABEL_INTRO = "intro"
LABEL_OUTRO = "outro"
LABEL_SILENCE = "silence"
LABEL_TRANSITION = "transition"
LABEL_FILLER = "filler"
LABEL_RECAP = "recap"

NON_CONTENT_LABELS = {
    LABEL_AD, LABEL_INTRO, LABEL_OUTRO,
    LABEL_SILENCE, LABEL_TRANSITION, LABEL_FILLER, LABEL_RECAP,
}

# Visual colour code for each label, used by the player + timeline PNG.
LABEL_COLORS = {
    LABEL_CORE:       "#2e7d32",  # green
    LABEL_AD:         "#c62828",  # red
    LABEL_INTRO:      "#1565c0",  # blue
    LABEL_OUTRO:      "#6a1b9a",  # purple
    LABEL_SILENCE:    "#757575",  # grey
    LABEL_TRANSITION: "#ef6c00",  # orange
    LABEL_FILLER:     "#9e9d24",  # olive
    LABEL_RECAP:      "#00838f",  # teal
}


@dataclass
class PipelineOptions:
    """Per-run options that override config defaults."""

    use_whisper: bool = True
    cache: bool = True
    sample_fps: float = SAMPLE_FPS
    ad_score_threshold: float = AD_SCORE_THRESHOLD
    extra: dict = field(default_factory=dict)
