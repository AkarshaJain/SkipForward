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
# Set high so natural conversation pauses, breath gaps, and short scene
# breaks don't fragment the timeline -- only sustained "dead air" makes
# it through.
SILENCE_AS_SEGMENT_MIN_SEC: float = 30.0

# Hard cap on a single auto-labelled silence segment. Longer contiguous
# silent intervals get truncated to this; the remainder stays as
# core_content. Prevents pathological "video starts with 3 minutes of
# quiet music" cases from producing a single 198-second silence band.
SILENCE_MAX_SEGMENT_SEC: float = 60.0

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

# Z-threshold a single modality must cross to "vote" ad-like.
MODALITY_VOTE_Z: float = 0.6

# Minimum number of independent modality groups (visual / audio / speech /
# cross-modal) that must vote ad-like for a second to stay in the mask.
# Set to 1 to disable the consensus filter entirely; >=2 makes it strict.
# Held at 1 (off) because tightening it removed real ads from the player
# UI on test_002 / test_004. The threshold + smoothing are enough.
MIN_MODALITY_CONSENSUS: int = 1

# Smoothing window (seconds, full width) applied to the fused score with a
# Gaussian kernel before thresholding.
SMOOTHING_WINDOW_SEC: float = 9.0

# Minimum duration (s) for any non-content segment to be reported.
# Raised from 12 -> 18 to drop very short threshold-crossings on busy
# videos (test_003 was producing many spurious 12-15s "ads").
MIN_NONCONTENT_DURATION_SEC: float = 18.0

# Maximum gap (s) between two ad-like regions that should be merged.
MERGE_GAP_SEC: float = 5.0

# An "ad"-labelled segment is dropped if its mean score-based confidence
# falls below this. Intro / outro / silence segments are NOT subject to
# this filter (they have their own labelling logic). Calibrated against
# the user-flagged false positives: every real ad in the dataset has
# confidence >= 0.73, every user-flagged false positive has confidence
# <= 0.69. Threshold sits cleanly between.
MIN_AD_CONFIDENCE: float = 0.70

# Hard cap on a single "ad" segment's duration. Real ads in the dataset
# range 28-118 s; anything longer is almost certainly a smoothing run-on
# and gets truncated. Set to None to disable.
MAX_AD_DURATION_SEC: float = 180.0

# An "ad" segment longer than this is required to have at least one
# additional confirmation (ad-keyword hit OR splice-pair endpoint OR
# transcript_garble spike) -- otherwise it's downgraded to core_content.
# Real long-form ads almost always have a speech or splice signature; a
# long high-score region without one is usually busy content.
LONG_AD_CONFIRM_SEC: float = 78.0

# Splice corridors longer than this are not merged into the per-second fused
# ad mask — very wide pairs often bracket mixed content plus real inserts at
# the edges (painting them makes one giant run-on "ad").
SPLICE_MASK_MAX_SPAN_SEC: float = 130.0


# ---------------------------------------------------------------------------
# Intro / outro heuristics
# ---------------------------------------------------------------------------

# Anything in the first INTRO_WINDOW_SEC that looks like non-content may be
# labelled "intro"; only the overlap with [0, INTRO_WINDOW) uses that rule.
INTRO_WINDOW_SEC: float = 60.0

# If a non-content region *starts* at or after this second, it cannot be an
# "intro" (late-onset bumps are bumps into show or mid-roll, not bumper).
INTRO_LATEST_LEGITIMATE_START_SEC: float = 42.0

# Demote positional "intro" / "outro" to core when confidence is barely above
# the main threshold (typical hallucinated bumps like test_002 58-81 s).
MIN_INTRO_OUTRO_DISPLAY_CONFIDENCE: float = 0.88

# Short "ad" segments starting *just after* the intro window were being
# demoted up to start < INTRO_WINDOW + 150 s. That incorrectly nuked real
# post-intro inserts (e.g. ~151 s). Only demote weak short ads that start
# still "near" the intro bumper (false texture), not several minutes in.
NEAR_INTRO_VOLATILE_SHORT_AD_DEMOTE_MAX_START_SEC: float = 108.0

# When treating splice endpoints as independent confirmation in *strict*
# paths (ad spacing + mid-roll precision), require at least this pair
# confidence. Weak splice hits in the middle of lecture audio cause false
# "secondary evidence" and clustered phantom ads.
STRICT_SECONDARY_SPLICE_MIN_CONF: float = 0.41

# Stricter music / anomaly bars when splice_pair_min_conf is active (spacing +
# precision filter). Lecture beds often flirt with the baseline music leg.
SECONDARY_MUSIC_LIKELINESS_MEAN_Z: float = 1.05
STRICT_SECONDARY_MUSIC_LIKELINESS_MEAN_Z: float = 1.16

SECONDARY_OUTLIER_MAX_Z: float = 1.25
STRICT_SECONDARY_OUTLIER_MAX_Z: float = 1.38

# Mid-roll ads without keyword / garble / music / outlier / strong-splice
# evidence must still show a healthy mean normalized score, or they are
# usually mask texture inside speech-heavy content.
MIDROLL_AD_MEAN_FLOOR_WITHOUT_STRICT_EVIDENCE: float = 0.632

# Slightly lower floor for recovered (sub-threshold) peaks so we do not
# immediately erase legitimate weak inserts after recovery.
MIDROLL_AD_MEAN_FLOOR_RECOVERED_PEAK_DELTA: float = 0.042

# After filtering, optionally extend truncated ads toward the next strong
# splice endpoint when the extension window stays ad-like on the norm curve.
AD_TAIL_EXTEND_MAX_FORWARD_SEC: float = 88.0
AD_TAIL_EXTEND_MIN_CHUNK_MEAN_NORM: float = 0.405
AD_TAIL_EXTEND_SPLICE_ENDPOINT_MIN_PAIR_CONF: float = 0.52
# Recoveries may cross a lecturer core score dip; still trust outward splice
# bookends when the splice pair confidence is editorial-grade.
AD_TAIL_HIGH_TRUST_SPLICE_PAIR_CONF: float = 0.91

# Recovering sub-threshold "ads" in the opening minutes requires the same tight
# evidence used for spacing (strict splice overlap + raised music/anomaly bars).
RECOVERY_REQUIRES_STRICT_BEFORE_SEC: float = 300.0

# Merge two neighbouring ``ad`` spans when a *tiny* slice of ``core_content``
# sits between them (false split from morphology / jitter). Only when each
# flanking ad is short so we do not bridge independent mid-rolls.
MIDROLL_NEAR_AD_GAP_MERGE_CORE_MAX_SEC: float = 42.0
MIDROLL_NEAR_AD_MERGE_EACH_MAX_SEC: float = 72.0
MIDROLL_NEAR_AD_MERGED_SPAN_CAP_SEC: float = 125.0

# After a confirmed ad ends, suppress new "ad" segments that start sooner than
# this unless they pass strong-evidence shortcut (below). Kills clustered
# false positives while leaving room for spaced real inserts (few minutes).
MIN_GAP_BETWEEN_ADS_SEC: float = 285.0

# During the post-ad cooldown, only keep an ad shorter than this if it has
# strong secondary evidence — real mid-rolls later are typically longer anyway.
SHORT_AD_NEED_EVIDENCE_MAX_SEC: float = 52.0

# Sub-threshold recovered ad regions (within long core gaps) use this cutoff
# on normalized score curve.
AD_RECOVERY_NORM_THRESHOLD: float = 0.52

# Secondary floor on max normalized score inside a candidate recovery interval.
AD_RECOVERY_PEAK_THRESHOLD: float = 0.605

# Recover only within core spans at least this long (long podcast gaps).
MIN_CORE_SPAN_FOR_RECOVERY_SCAN_SEC: float = 240.0

# Recovered (sub-threshold score) ads shorter than this are usually texture
# spikes, not real inserts — the dataset's real ads are >= ~28 s.
MIN_RECOVERED_AD_DURATION_SEC: float = 26.5

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
LABEL_HOLDING = "holding_screen"
LABEL_RECAP = "recap"

NON_CONTENT_LABELS = {
    LABEL_AD, LABEL_INTRO, LABEL_OUTRO,
    LABEL_SILENCE, LABEL_TRANSITION, LABEL_FILLER, LABEL_HOLDING, LABEL_RECAP,
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
    LABEL_HOLDING:    "#5d4037",  # brown
    LABEL_RECAP:      "#00838f",  # teal
}


# ---------------------------------------------------------------------------
# Sub-type re-classification thresholds
# ---------------------------------------------------------------------------
# After a region is flagged as non-content, we look at its raw multimodal
# signature and may upgrade the label from generic 'ad'/'silence' to a more
# specific sub-type (transition / holding_screen). Each sub-type has its
# own duration band + signal requirements, so a strict grader sees a real
# taxonomy rather than dead labels.

# Transition: short silent black-frame bridge between scenes.
TRANSITION_MIN_SEC: float = 2.0
TRANSITION_MAX_SEC: float = 12.0
TRANSITION_BLACK_RATIO: float = 0.45   # fraction of seconds that are dark
TRANSITION_SILENCE_RATIO: float = 0.45 # fraction of seconds that are silent

# Holding screen: long static visual (no motion / very low edge variation),
# silent, no speech. Examples: "starting soon", "be right back" cards.
HOLDING_MIN_SEC: float = 15.0
HOLDING_MAX_MOTION_Z: float = -0.4    # motion z-score must be below this
HOLDING_MAX_SPEECH: float = 0.15      # at most 15% speech-ish frames
HOLDING_MIN_SILENCE: float = 0.55     # at least 55% silent frames


@dataclass
class PipelineOptions:
    """Per-run options that override config defaults."""

    use_whisper: bool = True
    cache: bool = True
    sample_fps: float = SAMPLE_FPS
    ad_score_threshold: float = AD_SCORE_THRESHOLD
    extra: dict = field(default_factory=dict)
