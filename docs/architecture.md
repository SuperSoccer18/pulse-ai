# Architecture

## Overview

Pulse AI is a single FastAPI app (`server.py`) serving a static single-page
frontend and a WebSocket transcription endpoint. There is no database — all
state either lives in-memory for the duration of one WebSocket connection,
or in Veeva CRM (the system of record for call data).

```
Browser (static/index.html + recorder-worklet.js)
   │
   │  1. GET /api/calls           ───────────────► Veeva CRM (read)
   │  2. mic → AudioWorklet → WS /ws/transcribe
   ▼
FastAPI server (server.py)
   │  resample 48kHz → 16kHz, chunk into 0.6s windows
   ▼
Moonshine streaming encoder (per chunk, on a dedicated thread pool)
   │  raw audio discarded immediately after each chunk is encoded
   ▼
accumulated encoder hidden states (no raw audio, in-memory per connection)
   │  on "stop": one decode pass (model.generate)
   ▼
transcript (string)
   │
   ▼
Cortex pipeline (cortexAgents.py) — runs on the same thread pool
   1. correct_transcript()   — lilly-keyword-recognizer
   2. extract_call_fields()  — field-extractor
   ▼
result sent back over the WS: { transcript, correction, extraction }
   │
   ▼
Rep reviews/edits fields in the browser (review screen)
   │
   │  3. POST /api/check-compliance  (checking-state spinner, no write)
   │  4. POST /api/submit-call       (real gate + write)
   ▼
check_compliance()  — compliance-gate Cortex agent
   │  PROCEED ──────────────────────────► update_call_in_veeva() (Veeva write)
   │  HOLD ────────────────────────────► violations returned to review screen
```

## Component map

| File | Role |
|---|---|
| `server.py` | FastAPI app: static file serving, REST endpoints, `/ws/transcribe` WebSocket handler, audio resampling, chunk/decode orchestration, extraction→Veeva field mapping |
| `cortexAgents.py` | Thin client for the three Cortex agents (correction, extraction, compliance). Owns the Cortex response-envelope parsing. |
| `veeva_client.py` | Veeva auth (client-credentials OAuth) + SOQL read (`fetch_calls_for_rep`) + PATCH write (`update_call_in_veeva`) used by `server.py`. |
| `transcribe_streaming_experiment.py` | Defines the chunking constants (`SAMPLE_RATE`, `CHUNK_SAMPLES`, `OVERLAP_SAMPLES`, etc.) and `encode_chunk()` — imported directly by `server.py`, not just a standalone experiment despite the filename. |
| `transcribe_base.py` | Standalone CLI script for the older non-streaming `moonshine-base` ONNX model. Not used by `server.py`. Kept for comparison/reference. |
| `static/index.html` | Entire frontend UI (call list, recording screen, processing animation, review screen, compliance modal) — single file, vanilla JS, no framework/build step. |
| `static/recorder-worklet.js` | AudioWorklet processor: buffers mic PCM into fixed-size frames (`FRAME_SAMPLES = 4800`, 0.1s @ 48kHz) and posts them to the main thread for WS transmission. |
| `testAgents.py`, `veeva_integration_test.py`, `test_chunked_pipeline.py`, `benchmark_encode_chunk.py` | See [testing.md](testing.md). |
| `models/moonshine-streaming-medium/` | The model `server.py` actually loads. **Not checked into git** — see [setup.md](setup.md) for the download step. |
| `models/moonshine-base/` | Older ONNX model used only by `transcribe_base.py`. |

## Where state lives

- **Per-recording state** (`accumulated_hidden`, `accumulated_masks`,
  `overlap_buffer`, `resample_buf`) lives in local variables inside the
  `ws_transcribe` coroutine — scoped to one WebSocket connection, garbage
  collected on disconnect/completion. Nothing persists across recordings.
- **Model + processor** (`_processor`, `_model`) are module-level globals in
  `server.py`, loaded once at process startup and shared read-only across all
  connections.
- **Concurrency control**: a single-worker `ThreadPoolExecutor`
  (`_executor`, `max_workers=1`) serializes every `encode_chunk`/`generate()`
  call across *all* connections — the model is not thread-safe for concurrent
  inference. This means two reps recording simultaneously would queue behind
  each other, not run in parallel. Fine for a demo; a real concurrency
  bottleneck if this needs to support multiple simultaneous reps.
- **Call data** is never cached server-side — `/api/calls` hits Veeva fresh
  on every request; `/api/submit-call` writes straight through. No local DB.
- **Rep identity**: hardcoded to one demo rep (`ARTHUR_ID` in
  `veeva_client.py`). There is no login/session/auth layer in `server.py`
  itself.

## Request flow detail

See [api-reference.md](api-reference.md) for endpoint-level detail and
[asr-pipeline.md](asr-pipeline.md) for the encode/decode internals.
