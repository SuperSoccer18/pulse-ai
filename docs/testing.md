# Testing & Benchmarking

There is no automated test suite wired into CI (no CI config exists in this
repo at all). What exists are standalone scripts, each runnable directly.
None are pytest-based — each implements its own PASS/FAIL/exit-code
reporting.

## `test_chunked_pipeline.py`

The closest thing to a real regression test in this repo. Tests
**structural correctness** of the chunked encoder pipeline
(`encode_chunk()`, `iter_chunks()` from `transcribe_streaming_experiment.py`)
across boundary conditions — synthetic numpy-generated audio for most
cases, real speech files from `audio/` for a subset.

```bash
python test_chunked_pipeline.py
```

Prints PASS/FAIL with a short reason per test, plus a summary. Exit code 0
if all pass, 1 if any fail — safe to wire into CI as-is if that's ever set
up. Run this after any change to `encode_chunk()`, chunk-size constants, or
overlap handling.

## `transcribe_streaming_experiment.py`

Not a test in the assert sense — a runnable comparison script. Loads a WAV
file, runs the full chunked pipeline, and prints the chunked transcription
side-by-side with a full-audio baseline transcription for manual
comparison.

```bash
python transcribe_streaming_experiment.py [path/to/audio.wav]
# defaults to audio/test.wav (must be 16 kHz mono WAV)
```

Useful as a quick sanity check that chunked encoding still produces
equivalent output to a full-audio pass — the property the whole streaming
architecture depends on. If this ever visibly diverges, that's a signal
worth chasing before it shows up as a Cortex extraction accuracy problem.

## `benchmark_encode_chunk.py`

Latency benchmark for `encode_chunk()` across several chunk sizes, checking
real-time viability:

```bash
python benchmark_encode_chunk.py
```

System is real-time viable if `mean encode time < chunk duration` (no
backlog builds) and `max encode time < chunk duration` (no single spike
causes catch-up lag). Re-run this if `CHUNK_SAMPLES` changes, if the model
changes, or if deploying to different hardware than it was tuned on — the
whole live-transcription UX depends on encode staying ahead of real time.

## `veeva_integration_test.py`

Not a test suite — an exploration/provisioning script. See
[veeva-integration.md](veeva-integration.md) for what it actually does
(seeding demo data, dumping field-level permissions via `describe()`). Only
run pieces of this deliberately; it creates/deletes real Veeva sandbox
records.

## `testAgents.py`

Scratch script for creating a Cortex model and testing raw
`POST /model/ask/<name>` calls directly, bypassing `cortexAgents.py`'s
response-envelope parsing. Useful for isolating whether a problem is on the
Cortex side or in the parsing logic — see
[cortex-agents.md](cortex-agents.md).

## What's not covered

- No test exercises the FastAPI endpoints directly (no `TestClient`/httpx
  test hitting `/api/calls`, `/api/check-compliance`, `/api/submit-call`,
  or `/ws/transcribe`).
- No test exercises the Cortex agents' actual output quality/correctness —
  only the plumbing around them.
- No test exercises `_extraction_to_veeva_fields()`'s mapping logic
  directly (e.g. the datetime branching).
