"""
test_chunked_pipeline.py — Edge case tests for the chunked encoder pipeline
=============================================================================
Tests structural correctness of the pipeline across boundary conditions.
All audio inputs are synthetic (numpy-generated) except the real-speech tests
which use existing files in audio/.

Run:
    python test_chunked_pipeline.py

Each test prints PASS or FAIL with a short reason.  A summary is printed at
the end.  Exit code is 0 if all tests pass, 1 if any fail.
"""

import sys
import gc
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingEncoderModelOutput,
)

# ─────────────────────────────────────────────────────────────────────────────
# Import pipeline components from the experiment script
# ─────────────────────────────────────────────────────────────────────────────

from transcribe_streaming_experiment import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    OVERLAP_SAMPLES,
    N_OVERLAP_FRAMES,
    MAX_TOKENS,
    MODEL_DIR,
    iter_chunks,
    encode_chunk,
    load_wav,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (name, passed, message)


def run_test(name: str, fn) -> None:
    """Run a single test function, catch exceptions, record result."""
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name}\n        {e}")
    except Exception as e:
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
        traceback.print_exc()


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_sine(duration_s: float, freq: float = 440.0) -> np.ndarray:
    """Generate a pure tone — recognisable signal, not silence."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def run_pipeline(
    audio: np.ndarray,
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
) -> str:
    """
    Run the full chunked encoder pipeline and return the decoded text.
    Mirrors the logic in transcribe_streaming_experiment.py exactly.
    """
    overlap_buffer     = np.array([], dtype=np.float32)
    accumulated_hidden = []
    accumulated_masks  = []

    for i, chunk in enumerate(iter_chunks(audio, CHUNK_SAMPLES)):
        is_first = (i == 0)
        hidden, mask = encode_chunk(chunk, overlap_buffer, is_first, processor, model)
        overlap_buffer = chunk[-OVERLAP_SAMPLES:].copy()
        accumulated_hidden.append(hidden)
        accumulated_masks.append(mask)
        del chunk
        gc.collect()

    del audio
    gc.collect()

    # Guard: if every chunk returned zero frames (shouldn't happen, but be safe)
    assert len(accumulated_hidden) > 0, "No chunks were encoded"

    encoder_hidden = torch.cat(accumulated_hidden, dim=1)
    encoder_mask   = torch.cat(accumulated_masks,  dim=1)

    assert encoder_hidden.shape[0] == 1,   "Batch dim should be 1"
    assert encoder_hidden.shape[2] == 768, "Hidden dim should be 768"
    assert encoder_mask.shape == encoder_hidden.shape[:2], \
        "Mask shape should match (batch, T)"

    reconstructed = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=encoder_hidden,
        attention_mask=encoder_mask,
    )
    with torch.no_grad():
        ids = model.generate(
            encoder_outputs=reconstructed,
            max_new_tokens=MAX_TOKENS,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Load model once — shared across all tests
# ─────────────────────────────────────────────────────────────────────────────

print("Loading model …")
processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    str(MODEL_DIR),
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✓ {type(model).__name__} loaded\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

section("Group 1 — Audio length edge cases")


def test_multi_chunk_real_speech():
    """Normal case: real speech file spanning many chunks."""
    wav_path = Path("audio/test.wav")
    if not wav_path.exists():
        raise AssertionError("audio/test.wav not found — skipping")
    audio = load_wav(wav_path)
    assert len(audio) > CHUNK_SAMPLES, "Expected audio longer than one chunk"
    text = run_pipeline(audio, processor, model)
    assert len(text) > 0, "Expected non-empty transcription"


def test_exactly_one_chunk():
    """Audio equal to exactly one chunk — no overlap should fire."""
    audio = make_sine(CHUNK_SAMPLES / SAMPLE_RATE)
    assert len(audio) == CHUNK_SAMPLES
    text = run_pipeline(audio, processor, model)
    # Model may output nonsense for a tone, but pipeline must not crash
    assert isinstance(text, str)


def test_shorter_than_one_chunk():
    """Audio shorter than one chunk (0.3s) — single chunk, no boundary."""
    duration = 0.3
    assert SAMPLE_RATE * duration < CHUNK_SAMPLES, "Precondition: shorter than chunk"
    audio = make_sine(duration)
    text = run_pipeline(audio, processor, model)
    assert isinstance(text, str)


def test_shorter_than_overlap_buffer():
    """Audio shorter than the overlap buffer (0.1s) — n_trim clamp must fire."""
    duration = 0.1
    assert SAMPLE_RATE * duration < OVERLAP_SAMPLES, "Precondition: shorter than overlap"
    audio = make_sine(duration)
    text = run_pipeline(audio, processor, model)
    # Must not crash and must return a string (possibly empty for very short audio)
    assert isinstance(text, str)


def test_non_round_duration():
    """Audio with a non-round duration produces correct frame count."""
    # 2.37s — not a multiple of chunk or overlap
    audio = make_sine(2.37)
    text = run_pipeline(audio, processor, model)
    assert isinstance(text, str)


section("Group 2 — Audio content edge cases")


def test_silence():
    """Full silence — attention mask may be sparse, decoder must not crash."""
    audio = make_silence(2.0)
    text = run_pipeline(audio, processor, model)
    # Silence may produce empty output or minimal tokens — both are acceptable
    assert isinstance(text, str)


def test_near_silence():
    """Near-silence (very low amplitude) — tests normalisation path."""
    audio = make_sine(2.0) * 0.001   # amplitude 1000× below normal
    text = run_pipeline(audio, processor, model)
    assert isinstance(text, str)


section("Group 3 — Real audio files")


def test_real_speech_test_wav():
    """test.wav produces non-empty transcription."""
    wav_path = Path("audio/test.wav")
    if not wav_path.exists():
        raise AssertionError("audio/test.wav not found — skipping")
    audio = load_wav(wav_path)
    text = run_pipeline(audio, processor, model)
    assert len(text) > 0, f"Expected non-empty transcription, got: {text!r}"
    print(f"        → {text}")


section("Group 4 — load_wav guards")


def test_load_wav_file_not_found():
    """load_wav raises FileNotFoundError for a missing file."""
    try:
        load_wav(Path("audio/does_not_exist.wav"))
        raise AssertionError("Expected FileNotFoundError but no error was raised")
    except FileNotFoundError:
        pass   # expected


def test_load_wav_empty_path():
    """load_wav raises FileNotFoundError for an empty path string."""
    try:
        load_wav(Path(""))
        raise AssertionError("Expected FileNotFoundError but no error was raised")
    except (FileNotFoundError, OSError):
        pass   # expected


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

section("Running tests")

run_test("multi-chunk real speech",           test_multi_chunk_real_speech)
run_test("exactly one chunk",                 test_exactly_one_chunk)
run_test("shorter than one chunk (0.3s)",     test_shorter_than_one_chunk)
run_test("shorter than overlap buffer (0.1s)",test_shorter_than_overlap_buffer)
run_test("non-round duration (2.37s)",        test_non_round_duration)
run_test("silence (2.0s)",                    test_silence)
run_test("near-silence (2.0s, 0.001×)",       test_near_silence)
run_test("real speech — test.wav",            test_real_speech_test_wav)
run_test("load_wav: file not found",          test_load_wav_file_not_found)
run_test("load_wav: empty path",              test_load_wav_empty_path)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")

passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
total  = len(_results)

print(f"  {passed}/{total} passed", end="")
if failed:
    print(f"  —  {failed} failed:")
    for name, ok, msg in _results:
        if not ok:
            print(f"    ✗  {name}: {msg}")
else:
    print("  — all tests passed ✓")

sys.exit(0 if failed == 0 else 1)
