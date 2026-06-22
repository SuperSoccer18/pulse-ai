"""
benchmark_encode_chunk.py  —  Latency benchmark for encode_chunk
=================================================================
Measures how long encode_chunk takes at several chunk sizes and compares
against the real-time budget (chunk_duration seconds per chunk).

The system is real-time viable if:
    mean encode time  <  chunk duration  (no backlog builds)
    max  encode time  <  chunk duration  (no single spike causes catchup)

Run:
    python benchmark_encode_chunk.py
"""

import gc
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from transcribe_streaming_experiment import (
    SAMPLE_RATE,
    OVERLAP_SAMPLES,
    N_OVERLAP_FRAMES,
    MODEL_DIR,
    encode_chunk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

TRIALS        = 20      # timed trials per chunk size (after warm-up)
WARMUP_TRIALS = 3       # discarded warm-up trials (JIT / cache effects)

# Chunk sizes to benchmark.  Each entry: (label, chunk_samples)
CHUNK_SIZES = [
    ("0.3s  (4,800 samples)",  int(0.3 * SAMPLE_RATE)),
    ("0.6s  (9,600 samples)",  int(0.6 * SAMPLE_RATE)),
    ("1.0s (16,000 samples)",  int(1.0 * SAMPLE_RATE)),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_sine(n_samples: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def section(title: str) -> None:
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}\n")


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:6.1f} ms"


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

print("Loading model …")
processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    str(MODEL_DIR),
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"  ✓ {type(model).__name__} loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

section("Benchmark: encode_chunk latency vs real-time budget")

results = []

for label, chunk_samples in CHUNK_SIZES:
    chunk_duration_s = chunk_samples / SAMPLE_RATE
    overlap_audio    = make_sine(OVERLAP_SAMPLES)   # realistic overlap buffer

    # Warm-up — discard these times
    for _ in range(WARMUP_TRIALS):
        chunk = make_sine(chunk_samples)
        encode_chunk(chunk, overlap_audio, is_first=False,
                     processor=processor, model=model)
        del chunk
        gc.collect()

    # Timed trials — is_first=False is the steady-state case (has overlap)
    times = []
    for _ in range(TRIALS):
        chunk = make_sine(chunk_samples, freq=440.0 + len(times))  # vary freq slightly
        gc.collect()
        t0 = time.perf_counter()
        encode_chunk(chunk, overlap_audio, is_first=False,
                     processor=processor, model=model)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        del chunk
        gc.collect()

    mean_t = statistics.mean(times)
    std_t  = statistics.stdev(times)
    min_t  = min(times)
    max_t  = max(times)
    slack  = chunk_duration_s - mean_t
    viable = "✓ viable" if max_t < chunk_duration_s else "✗ too slow"

    results.append((label, chunk_duration_s, mean_t, std_t, min_t, max_t, slack, viable))

    print(f"  Chunk size  : {label}")
    print(f"  Budget      : {fmt_ms(chunk_duration_s)}")
    print(f"  Mean ± std  : {fmt_ms(mean_t)} ± {fmt_ms(std_t)}")
    print(f"  Min / Max   : {fmt_ms(min_t)} / {fmt_ms(max_t)}")
    print(f"  Slack (mean): {fmt_ms(slack)}  {'← positive = keeping up' if slack > 0 else '← NEGATIVE = falling behind'}")
    print(f"  Real-time   : {viable}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")

header = f"  {'Chunk':<26}  {'Budget':>8}  {'Mean':>8}  {'Max':>8}  {'Slack':>8}  {'Status'}"
print(header)
print(f"  {'─' * 26}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")

for label, budget, mean_t, std_t, min_t, max_t, slack, viable in results:
    print(f"  {label:<26}  {fmt_ms(budget):>8}  {fmt_ms(mean_t):>8}  "
          f"{fmt_ms(max_t):>8}  {fmt_ms(slack):>8}  {viable}")

print()
print("  Note: 'viable' = max encode time < chunk duration.")
print("  In a threaded capture loop, mean slack determines average queue depth.")
print("  Negative slack means the encoder cannot keep up with the audio stream.")
