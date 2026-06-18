"""
transcribe_streaming_experiment.py  —  Step 5: chunked encoder pipeline
=========================================================================
Validates the privacy-first transcription architecture:

  • Audio is processed in ≤0.9s at a time (0.6s chunk + 0.3s overlap buffer).
  • Each chunk is encoded and immediately deleted before the next is loaded.
  • A small overlap buffer (300ms, 15 frames) carries boundary context across
    chunks — the only raw audio that persists between iterations.
  • Accumulated encoder hidden states (no raw audio) are decoded in one pass.

The chunked output is compared against a full-audio baseline to verify that
chunked encoding produces equivalent transcription.

Run:
    python transcribe_streaming_experiment.py [path/to/audio.wav]
    Default: audio/test.wav  (must be 16 kHz mono WAV)
"""

import gc
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingEncoderModelOutput,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR         = Path("models/moonshine-streaming-medium")
SAMPLE_RATE       = 16_000
MAX_TOKENS        = 448
SAMPLES_PER_FRAME = 320     # effective conv frontend stride for moonshine-streaming-medium
                            # = 80-sample input frame × 4× internal downsampling
N_OVERLAP_FRAMES  = 15      # one full sliding-window context worth of frames
OVERLAP_SAMPLES   = N_OVERLAP_FRAMES * SAMPLES_PER_FRAME   # 4,800 samples = 300ms
CHUNK_SAMPLES     = 9_600   # 0.6s — combined with 0.3s overlap = 0.9s peak in memory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}\n")


def load_wav(path: Path) -> np.ndarray:
    """Load a 16 kHz mono WAV file into float32 [-1, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        raise ValueError(
            f"Expected mono audio, got {audio.ndim} channels. "
            f"Convert with:  ffmpeg -i {path} -ac 1 mono.wav"
        )
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. "
            f"Convert with:  ffmpeg -i {path} -ar 16000 -ac 1 resampled.wav"
        )
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — Chunk iterator (simulates a live audio stream)
# ─────────────────────────────────────────────────────────────────────────────

def iter_chunks(
    audio_array: np.ndarray,
    chunk_samples: int,
) -> Generator[np.ndarray, None, None]:
    """
    Yield successive ≤chunk_samples slices of audio_array as independent copies.

    In production this generator would be replaced by a live capture loop
    (e.g. sounddevice.InputStream).  Downstream code is identical either way.
    """
    pos = 0
    while pos < len(audio_array):
        end = min(pos + chunk_samples, len(audio_array))
        yield audio_array[pos:end].copy()
        pos = end


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 — Per-chunk encoder (privacy-critical)
# ─────────────────────────────────────────────────────────────────────────────

def encode_chunk(
    audio_chunk: np.ndarray,
    overlap_buffer: np.ndarray,
    is_first: bool,
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode one audio chunk and immediately discard all raw audio.

    Overlap is prepended before encoding so that boundary frames have
    correct local context.  The overlap frames are trimmed from the output
    so they are not double-counted in the accumulation buffer.

    Returns
    -------
    hidden : torch.Tensor  shape [1, T_chunk, 768]
    mask   : torch.Tensor  shape [1, T_chunk]
        Both on CPU, no raw audio recoverable.
    """
    # Prepend overlap from the previous chunk's tail
    if not is_first and len(overlap_buffer) > 0:
        chunk_with_overlap = np.concatenate([overlap_buffer, audio_chunk])
        n_trim = N_OVERLAP_FRAMES
    else:
        chunk_with_overlap = audio_chunk
        n_trim = 0

    # Preprocess — raw audio still in memory up to this point
    inputs = processor(
        chunk_with_overlap,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values   = inputs["input_values"]    # float32 waveform tensor
    attention_mask = inputs["attention_mask"]

    # ── Privacy checkpoint 1: delete raw audio before encoder runs ──────────
    del chunk_with_overlap, inputs
    gc.collect()

    # Encode: waveform → abstract hidden states
    with torch.no_grad():
        encoder_out = model.model.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )
    # encoder_out.last_hidden_state  [1, T, 768]  — not invertible to audio
    # encoder_out.attention_mask     [1, T]        — binary frame mask

    # ── Privacy checkpoint 2: delete waveform tensor after encode ───────────
    del input_values, attention_mask
    gc.collect()

    # Trim prepended overlap frames — they were context only, not new content.
    # Guard: if the chunk is shorter than the overlap (can happen on very short
    # audio), n_trim may exceed the actual frame count — clamp to avoid an
    # empty or negative slice.
    total_frames = encoder_out.last_hidden_state.shape[1]
    n_trim = min(n_trim, total_frames)

    hidden = encoder_out.last_hidden_state[:, n_trim:, :]   # [1, T_chunk, 768]
    mask   = encoder_out.attention_mask[:, n_trim:]         # [1, T_chunk]

    return hidden, mask


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — only runs when executed directly, not when imported
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Module 1: Load model ─────────────────────────────────────────────────

    section("1. Load model")

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  ✓ {type(model).__name__} loaded")

    # ── Module 2: Baseline ───────────────────────────────────────────────────

    section("2. Baseline — full audio encode + generate()")

    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audio/test.wav")
    audio = load_wav(audio_path)
    print(f"  ✓ {audio_path.name}: {len(audio) / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz")
    print(f"  Overlap buffer   : {N_OVERLAP_FRAMES} frames = "
          f"{OVERLAP_SAMPLES} samples = {1000 * OVERLAP_SAMPLES / SAMPLE_RATE:.0f}ms")

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        baseline_ids = model.generate(
            input_values=inputs["input_values"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_TOKENS,
        )
    result_baseline = processor.batch_decode(baseline_ids, skip_special_tokens=True)[0].strip()
    print(f"\n  Transcription:\n    {result_baseline}")

    # ── Modules 4 + 6: Main encoding loop ────────────────────────────────────

    section("3. Chunked encoder pipeline")

    total_samples = len(audio)

    # Guard: audio shorter than the overlap buffer still works — the overlap
    # prepend is skipped on the first (only) chunk, and n_trim is clamped
    # inside encode_chunk.  Log a notice so the caller is aware.
    if total_samples < OVERLAP_SAMPLES:
        print(f"  ⚠  Audio ({total_samples / SAMPLE_RATE:.2f}s) is shorter than the overlap "
              f"buffer ({OVERLAP_SAMPLES / SAMPLE_RATE:.2f}s). "
              f"Boundary context will be partial for this recording.\n")

    n_chunks = int(np.ceil(total_samples / CHUNK_SAMPLES))
    print(f"  Audio     : {total_samples / SAMPLE_RATE:.2f}s")
    print(f"  Chunks    : {n_chunks} × ≤{CHUNK_SAMPLES / SAMPLE_RATE:.1f}s")
    print(f"  Overlap   : {OVERLAP_SAMPLES} samples = {N_OVERLAP_FRAMES} frames = "
          f"{1000 * OVERLAP_SAMPLES / SAMPLE_RATE:.0f}ms per boundary\n")

    overlap_buffer     = np.array([], dtype=np.float32)
    accumulated_hidden = []
    accumulated_masks  = []

    for i, chunk in enumerate(iter_chunks(audio, CHUNK_SAMPLES)):
        is_first = (i == 0)

        hidden, mask = encode_chunk(chunk, overlap_buffer, is_first, processor, model)

        overlap_buffer = chunk[-OVERLAP_SAMPLES:].copy()
        accumulated_hidden.append(hidden)
        accumulated_masks.append(mask)

        n_trimmed = N_OVERLAP_FRAMES if not is_first else 0
        print(f"  Chunk {i+1:2d}/{n_chunks}: {len(chunk)/SAMPLE_RATE:.2f}s raw → "
              f"{hidden.shape[1]:3d} frames kept  "
              f"(trimmed {n_trimmed} overlap frames)")

        del chunk
        gc.collect()

    del audio
    gc.collect()
    print(f"\n  ✓ All {n_chunks} chunk(s) encoded. Raw audio fully released.")

    # ── Module 7: Reconstruct encoder output ─────────────────────────────────

    encoder_hidden = torch.cat(accumulated_hidden, dim=1)   # [1, T_total, 768]
    encoder_mask   = torch.cat(accumulated_masks,  dim=1)   # [1, T_total]

    print(f"\n  Reconstructed encoder states : {list(encoder_hidden.shape)}")
    print(f"  Reconstructed encoder mask   : {list(encoder_mask.shape)}")

    reconstructed_encoder_out = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=encoder_hidden,
        attention_mask=encoder_mask,
    )

    # ── Module 8: Decode ─────────────────────────────────────────────────────

    section("4. Decoding")

    with torch.no_grad():
        chunked_ids = model.generate(
            encoder_outputs=reconstructed_encoder_out,
            max_new_tokens=MAX_TOKENS,
        )

    result_chunked = processor.batch_decode(chunked_ids, skip_special_tokens=True)[0].strip()
    print(f"  Transcription:\n    {result_chunked}")

    # ── Module 9: Comparison ─────────────────────────────────────────────────

    section("5. Comparison")

    print(f"  Baseline (full audio)  : {result_baseline}")
    print(f"  Chunked encoder        : {result_chunked}")
