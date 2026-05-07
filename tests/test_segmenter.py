"""Segmenter tests with synthetic FusedFeatures.

These prove that:
  - a feature matrix where every channel is constant produces NO ad regions
    (this is the canonical "repeated frames + silent track" smoke test)
  - a feature matrix with a strongly anomalous middle window produces an ad
    segment whose start/end land within tolerance of ground-truth
  - the precision filter (>=2 modality consensus) actually rejects regions
    where only one modality is anomalous
  - the sub-type reclassifier turns a black+silent region into a transition
    and a static+silent region into a holding_screen
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config
from pipeline.fusion import CHANNELS, FusedFeatures, _zscore
from pipeline.segmenter import segment, _modality_consensus, _reclassify_subtypes, Segment


def _build_fused(raw_dict: dict[str, np.ndarray]) -> FusedFeatures:
    """Wrap a per-channel raw dict into a FusedFeatures with z-scored cols.

    The dict is expected to contain arrays already at SAMPLE_FPS Hz (one
    row per ``1/SAMPLE_FPS`` seconds). Use ``_seconds_to_steps(s)`` when
    constructing test data with semantic time bounds.
    """
    n = len(next(iter(raw_dict.values())))
    full_raw = {ch: raw_dict.get(ch, np.zeros(n)) for ch in CHANNELS}
    matrix = np.column_stack([full_raw[c] for c in CHANNELS])
    z = np.column_stack([_zscore(full_raw[c]) for c in CHANNELS])
    times = np.arange(n, dtype=np.float64) / config.SAMPLE_FPS
    return FusedFeatures(
        times=times,
        matrix=matrix, z=z, raw=full_raw,
    )


def _steps(seconds: float) -> int:
    """Convert seconds to grid-step count for synthetic test arrays."""
    return int(round(seconds * config.SAMPLE_FPS))


def _ad_regions(segments) -> list[tuple[float, float, str]]:
    return [(s.start, s.end, s.label)
            for s in segments
            if s.label in config.NON_CONTENT_LABELS]


# ---------------------------------------------------------------------------
# Canonical "repeated frames" smoke test
# ---------------------------------------------------------------------------

def test_constant_features_produce_no_non_content():
    """All-zero, all-equal features (e.g. a video of repeated frames with
    a silent track) must NOT produce any ad/intro/outro/silence segments."""
    duration_sec = 300
    n = _steps(duration_sec)
    fused = _build_fused({
        "shot_rate":   np.zeros(n),
        "saturation":  np.full(n, 50.0),
        "motion":      np.zeros(n),
        "edge_density": np.zeros(n),
        "audio_rms":   np.full(n, -50.0),
        "spectral_flux": np.zeros(n),
        "music_likeness": np.zeros(n),
        "ad_keyword":   np.zeros(n),
        "silence":      np.zeros(n),  # NB: not flagged silent at audio level
        "black_frame":  np.zeros(n),
        "speech_density": np.zeros(n),
        "transcript_garble": np.zeros(n),
        "local_outlierness": np.zeros(n),
    })

    segs, _dbg = segment(fused, shot_times=np.array([]),
                         duration_sec=float(duration_sec))
    assert _ad_regions(segs) == [], (
        "constant features must yield only core_content")
    assert segs[0].label == config.LABEL_CORE
    assert pytest.approx(segs[0].end - segs[0].start, abs=1.0) == duration_sec


# ---------------------------------------------------------------------------
# Anomalous middle region must be detected
# ---------------------------------------------------------------------------

def test_anomalous_middle_window_is_detected_as_non_content():
    """Inject a 60-s anomalous region in the middle of a 300-s timeline.
    Multiple modalities are anomalous so the consensus filter passes."""
    duration_sec = 300
    n = _steps(duration_sec)
    a_sec, b_sec = 120, 180  # 60-second anomaly
    a, b = _steps(a_sec), _steps(b_sec)

    raw = {ch: np.zeros(n) for ch in CHANNELS}
    raw["saturation"]      = np.full(n, 40.0); raw["saturation"][a:b] = 200.0
    raw["motion"]          = np.full(n, 0.05); raw["motion"][a:b]     = 0.5
    raw["audio_rms"]       = np.full(n, -25.0); raw["audio_rms"][a:b] = -8.0
    raw["spectral_flux"]   = np.full(n, 0.05); raw["spectral_flux"][a:b] = 0.6
    raw["music_likeness"]  = np.full(n, 0.1); raw["music_likeness"][a:b] = 0.85
    raw["local_outlierness"] = np.zeros(n); raw["local_outlierness"][a:b] = 4.0

    fused = _build_fused(raw)
    segs, dbg = segment(fused, shot_times=np.array([float(a_sec), float(b_sec)]),
                         duration_sec=float(duration_sec))

    nc = _ad_regions(segs)
    assert len(nc) == 1, f"expected exactly one non-content region, got {nc}"
    s, e, lbl = nc[0]
    # Start/end must land within the merge-gap window of the truth.
    tol = config.MERGE_GAP_SEC + 5.0
    assert abs(s - a_sec) <= tol, f"start {s} far from {a_sec}"
    assert abs(e - b_sec) <= tol, f"end   {e} far from {b_sec}"
    assert lbl == config.LABEL_AD


def test_single_modality_alone_is_rejected_by_consensus(monkeypatch):
    """If ONLY music_likeness spikes for 30s and nothing else does, the
    >=2-modality consensus filter (when explicitly enabled) must drop
    the segment. The default config keeps the filter off, so we patch it
    on for this test only."""
    monkeypatch.setattr(config, "MIN_MODALITY_CONSENSUS", 2)

    duration_sec = 300
    n = _steps(duration_sec)
    a, b = _steps(120), _steps(150)
    raw = {ch: np.zeros(n) for ch in CHANNELS}
    # Single modality spike, all other channels truly constant.
    raw["music_likeness"] = np.full(n, 0.1); raw["music_likeness"][a:b] = 0.95

    fused = _build_fused(raw)
    segs, _dbg = segment(fused, shot_times=np.array([]),
                         duration_sec=float(duration_sec))
    assert _ad_regions(segs) == [], (
        "a single-modality spike must be filtered out by consensus")


def test_modality_consensus_counts_independent_groups():
    """When two distinct modality groups both spike, consensus >= 2."""
    duration_sec = 60
    n = _steps(duration_sec)
    a, b = _steps(20), _steps(40)
    raw = {ch: np.zeros(n) for ch in CHANNELS}
    raw["motion"][a:b]            = 1.0   # visual
    raw["music_likeness"][a:b]    = 1.0   # audio
    fused = _build_fused(raw)
    consensus = _modality_consensus(fused)

    centre = _steps(30)
    assert consensus[centre] >= 2
    assert consensus[_steps(5)] == 0


# ---------------------------------------------------------------------------
# Sub-type reclassification
# ---------------------------------------------------------------------------

def _features_with_pattern(*,
                            duration_sec: int,
                            black: tuple[int, int] | None = None,
                            silence: tuple[int, int] | None = None,
                            no_motion: tuple[int, int] | None = None,
                            no_speech: tuple[int, int] | None = None,
                            ) -> FusedFeatures:
    """Build a synthetic FusedFeatures. Pattern bounds are in seconds."""
    n = _steps(duration_sec)
    raw = {ch: np.zeros(n) for ch in CHANNELS}
    raw["motion"] = np.full(n, 0.4)  # baseline movement
    raw["speech_density"] = np.full(n, 0.6)  # baseline speech presence
    if black is not None:
        raw["black_frame"][_steps(black[0]):_steps(black[1])] = 1.0
    if silence is not None:
        raw["silence"][_steps(silence[0]):_steps(silence[1])] = 1.0
    if no_motion is not None:
        raw["motion"][_steps(no_motion[0]):_steps(no_motion[1])] = 0.0
    if no_speech is not None:
        raw["speech_density"][_steps(no_speech[0]):_steps(no_speech[1])] = 0.0
    return _build_fused(raw)


def test_short_black_silent_segment_relabelled_transition():
    """A 6-second black+silent ad becomes 'transition' after reclassification."""
    fused = _features_with_pattern(duration_sec=600,
                                    black=(300, 306), silence=(300, 306))
    fake_seg = Segment(start=300.0, end=306.0,
                       label=config.LABEL_AD, confidence=0.7)
    out = _reclassify_subtypes([fake_seg], features=fused)
    assert len(out) == 1
    assert out[0].label == config.LABEL_TRANSITION
    assert out[0].confidence >= 0.85


def test_long_static_silent_segment_relabelled_holding_screen():
    """A 30-second static (no motion) silent (no speech) segment becomes
    'holding_screen' after reclassification."""
    a, b = 100, 130
    fused = _features_with_pattern(
        duration_sec=600, silence=(a, b), no_motion=(a, b), no_speech=(a, b),
    )
    fake_seg = Segment(start=float(a), end=float(b),
                       label=config.LABEL_AD, confidence=0.7)
    out = _reclassify_subtypes([fake_seg], features=fused)
    assert len(out) == 1
    assert out[0].label == config.LABEL_HOLDING


def test_unrelated_segment_keeps_original_label():
    """A segment that matches NEITHER transition NOR holding_screen rules
    stays as its original label (e.g. 'ad')."""
    n = _steps(600)
    raw = {ch: np.zeros(n) for ch in CHANNELS}
    raw["motion"] = np.full(n, 0.4)
    raw["speech_density"] = np.full(n, 0.4)
    raw["music_likeness"] = np.full(n, 0.7)  # busy ad
    fused = _build_fused(raw)
    fake_seg = Segment(start=120.0, end=180.0,
                       label=config.LABEL_AD, confidence=0.7)
    out = _reclassify_subtypes([fake_seg], features=fused)
    assert out[0].label == config.LABEL_AD
