"""Speech-to-text + linguistic ad-cue detection.

Uses ``openai-whisper`` (free, MIT-licensed, runs offline once cached).
Gracefully degrades if Whisper is not installed: returns an empty transcript
and a zero ``ad_keyword_score`` so the rest of the pipeline still works.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config


log = logging.getLogger(__name__)


# Cache directory for raw Whisper transcripts. Storing them lets us re-run
# the pipeline deterministically and skip the (slow, slightly-stochastic
# due to temperature fallback) Whisper transcription step on subsequent
# runs of the same audio.
_TRANSCRIPT_CACHE_DIR = config.INTERMEDIATE_DIR


def _transcript_cache_path(audio_wav: Path, model_name: str) -> Path:
    return audio_wav.parent / f"{audio_wav.stem}.whisper-{model_name}.json"


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    avg_logprob: float = 0.0


@dataclass
class SpeechFeatures:
    segments: list[SpeechSegment] = field(default_factory=list)
    full_text: str = ""
    ad_keyword_score: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    speech_density: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    # Transcription "garble" score per second — high when Whisper produces
    # nonsense/repetitive output (a strong cue for music or non-speech audio
    # interleaved with vocals, typical of jingles and music ads).
    transcript_garble: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64))
    available: bool = False


def _whisper_available() -> bool:
    try:
        import whisper  # noqa: F401  (lazy import test)
        return True
    except Exception:
        return False


def transcribe(audio_wav: Path | str,
               *,
               model_name: str = config.WHISPER_MODEL,
               language: str | None = config.WHISPER_LANGUAGE,
               num_seconds: int | None = None,
               cache: bool = True,
               ) -> SpeechFeatures:
    """Run Whisper and produce per-second linguistic features.

    The raw transcript is cached to disk next to the audio WAV so that
    repeat pipeline runs are fully reproducible (Whisper's temperature
    fallback can otherwise produce slightly different transcripts each
    run, which propagates into different segmentation outputs).

    Returns an empty (but valid) ``SpeechFeatures`` if Whisper or its model
    weights are not available.
    """
    n = max(int(num_seconds or 0), 1)
    audio_path = Path(audio_wav)

    # Try cache first.
    raw_segments: list | None = None
    cache_path = _transcript_cache_path(audio_path, model_name)
    if cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw_segments = cached.get("segments") or []
            log.info("Loaded cached Whisper transcript: %s (%d segments)",
                      cache_path.name, len(raw_segments))
        except Exception as exc:
            log.warning("Failed to read transcript cache %s: %s",
                         cache_path, exc)
            raw_segments = None

    if raw_segments is None:
        if not _whisper_available():
            log.warning("Whisper not installed - skipping speech analysis. "
                        "Install with: pip install -U openai-whisper")
            return SpeechFeatures(
                ad_keyword_score=np.zeros(n),
                speech_density=np.zeros(n),
                available=False,
            )

        try:
            import whisper  # type: ignore
            import soundfile as sf  # type: ignore

            log.info("Loading Whisper model '%s'...", model_name)
            model = whisper.load_model(model_name)

            # Load audio ourselves to bypass Whisper's internal ffmpeg call
            # (which fails on Windows when ffmpeg isn't on PATH). Whisper
            # expects float32 mono @ 16 kHz.
            audio_array, sr = sf.read(str(audio_path), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16_000:
                import librosa  # type: ignore
                audio_array = librosa.resample(
                    audio_array.astype(np.float32),
                    orig_sr=sr, target_sr=16_000,
                )
            result = model.transcribe(
                audio_array,
                language=language,
                verbose=False,
                fp16=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            log.warning("Whisper failed (%s) - running without speech features.", exc)
            return SpeechFeatures(
                ad_keyword_score=np.zeros(n),
                speech_density=np.zeros(n),
                available=False,
            )

        raw_segments = result.get("segments") or []
        if cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({
                    "model": model_name,
                    "language": language,
                    "segments": [
                        {
                            "start": float(s.get("start", 0.0)),
                            "end": float(s.get("end", 0.0)),
                            "text": str(s.get("text", "")),
                            "no_speech_prob": float(s.get("no_speech_prob", 0.0)),
                            "compression_ratio": float(s.get("compression_ratio", 0.0)),
                            "avg_logprob": float(s.get("avg_logprob", 0.0)),
                        }
                        for s in raw_segments
                    ],
                }, indent=2), encoding="utf-8")
                log.info("Cached Whisper transcript -> %s", cache_path.name)
            except Exception as exc:
                log.warning("Failed to write transcript cache %s: %s",
                             cache_path, exc)

    segments = [
        SpeechSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=str(s.get("text", "")).strip(),
            no_speech_prob=float(s.get("no_speech_prob", 0.0)),
            compression_ratio=float(s.get("compression_ratio", 0.0)),
            avg_logprob=float(s.get("avg_logprob", 0.0)),
        )
        for s in raw_segments
    ]
    full_text = " ".join(s.text for s in segments)

    keyword_score = np.zeros(n, dtype=np.float64)
    speech_density = np.zeros(n, dtype=np.float64)
    garble = np.zeros(n, dtype=np.float64)

    keyword_lc = [k.lower() for k in config.AD_KEYWORDS]

    for seg in segments:
        a = max(0, int(np.floor(seg.start)))
        b = min(n, int(np.ceil(seg.end)))
        if b <= a:
            continue
        text_lc = seg.text.lower()
        words = max(len(text_lc.split()), 1)
        speech_density[a:b] += words / max(seg.end - seg.start, 1.0)

        hit = sum(1 for k in keyword_lc if k in text_lc)
        if hit:
            keyword_score[a:b] += min(hit, 3) / 3.0

        # "Garble" cues, all standard Whisper quality indicators:
        #   - high no_speech_prob  → audio looks non-speech
        #   - high compression_ratio (>2.4) → text is repetitive (music vamp)
        #   - very low avg_logprob (<-1.0) → low confidence transcription
        garble_score = 0.0
        if seg.no_speech_prob > 0.4:
            garble_score = max(garble_score, seg.no_speech_prob)
        if seg.compression_ratio > 2.4:
            garble_score = max(
                garble_score, min(1.0, (seg.compression_ratio - 2.4) / 1.5))
        if seg.avg_logprob < -1.0:
            garble_score = max(garble_score, min(1.0, (-1.0 - seg.avg_logprob) / 1.0))
        if garble_score > 0:
            garble[a:b] = np.maximum(garble[a:b], garble_score)

    return SpeechFeatures(
        segments=segments,
        full_text=full_text,
        ad_keyword_score=np.clip(keyword_score, 0.0, 1.0),
        speech_density=speech_density,
        transcript_garble=np.clip(garble, 0.0, 1.0),
        available=True,
    )
