# tests/

Unit + integration tests for the multimodal segmentation pipeline.

All tests are **fully synthetic** -- they generate their own WAVs, feature
matrices, and ground-truth dictionaries on the fly. None of them touch the
five `videos_with_ads/*.mp4` files, so they're fast (whole suite runs in a
few seconds) and they don't require Whisper, FFmpeg, or any GPU.

## What's covered

| File | What it proves |
|------|----------------|
| `test_audio_features.py` | A silent WAV is detected as silent (RMS deeply negative, every second flagged). A 440 Hz sine wave is detected as music-like. White noise is loud but **not** music-like. `silence_intervals` recovers the correct (start, end) of an injected silent gap. Per-second arrays come back at exactly the requested length. |
| `test_visual_features.py` | HSV histogram distance is zero for identical frames and large for unrelated colours. The splice-boundary detector fires when (a) silence brackets a cut, (b) a black frame coincides with a cut, and stays silent when no evidence is present. `splice_signal_per_second` decays correctly. `shot_rate_per_second` increases with cut density. |
| `test_segmenter.py` | The "repeated frames + silent track" smoke test: constant features must produce zero non-content segments. An anomalous middle window across multiple modalities is detected as `ad`. A single-modality spike is **rejected** by the >=2-modality consensus filter. The sub-type reclassifier turns short black+silent segments into `transition` and long static+silent segments into `holding_screen`. |
| `test_fusion_alignment.py` | `_align_length` pads / truncates correctly. `_zscore` and `_local_outlierness` behave correctly on constant inputs. `fuse()` produces an `(N, C)` matrix even when the per-modality input arrays are off-by-one in length. |
| `test_metadata_and_eval.py` | The output JSON schema contains every required key and the summary counts match the segments. `skip_recommendations` excludes `core_content`. The evaluator produces `F1=1.0` on a perfect match, `0.0` on no overlap, and the right intermediate values on partial overlap. The evaluator counts `silence` (and any non-content label) as a valid match against ground-truth ads, because the user-facing goal is "skip me", not "classify me". |

## Running

```
pip install pytest
pytest tests/ -v
```

Or just `python -m pytest`.

## Adding new tests

Tests use the fixtures defined in `conftest.py`:

* `silent_wav`, `tone_wav`, `noise_wav`, `mixed_wav` -- produce synthetic
  WAVs with controllable duration, sample rate, frequency, amplitude, and
  silent ranges.

When testing the segmenter, build a `FusedFeatures` directly via
`_build_fused()` in `test_segmenter.py` rather than going through the full
audio + visual extraction (the synthetic input contract is "I gave you
this 13-channel feature matrix; produce these segments").
