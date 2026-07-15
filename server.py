"""
server.py  —  Pulse AI FastAPI backend
=======================================
Endpoints:
  GET  /                      → serves static/index.html
  GET  /api/calls             → dummy call list (JSON)
  POST /api/pipeline          → run full pipeline on transcript (WS fallback)
  POST /api/compliance-only   → run Compliance Goblin V3 only (approve and review button)
  WS   /ws/transcribe         → receives raw PCM chunks, encodes on-the-fly,
                                returns pipeline results when client sends "stop"

Pipeline order:
    Moonshine transcript → Field Extraction → Summary → Compliance

Run:
    python server.py
    or
    python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import gc
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import concurrent.futures

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingEncoderModelOutput,
)

from transcribe_streaming_experiment import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    OVERLAP_SAMPLES,
    MAX_TOKENS,
    encode_chunk,
)

# ── Pulse.AI Cortex agents — imported in pipeline order ──────────────────────
# Step 1: Field Extraction Gnome — extracts Veeva CRM fields from raw transcript
# Step 2: Summary Fairy          — generates call summary from raw transcript
# Step 3: Compliance Goblin V3   — redacts PI and flags AECP violations last
from fieldExtraction import FieldExtraction
from summaryField    import SummaryField
from complianceLayer import ComplianceLayer

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = r"C:\PulseAI\models\moonshine-streaming-medium"
STATIC    = Path(__file__).parent / "static"

# ─────────────────────────────────────────────────────────────────────────────
# Model — loaded once at startup, shared across all connections
# ─────────────────────────────────────────────────────────────────────────────

print("Loading model…")
_processor: AutoProcessor = AutoProcessor.from_pretrained(str(MODEL_DIR))
_model: AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq.from_pretrained(
    str(MODEL_DIR),
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
_model.eval()
print("Model ready.")

# Thread pool — max_workers=1 serialises all inference calls (model not thread-safe)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ─────────────────────────────────────────────────────────────────────────────
# Dummy call data
# ─────────────────────────────────────────────────────────────────────────────

DUMMY_CALLS: list[dict[str, Any]] = [
    {"id": "call-001", "hcp_name": "Dr. Sarah Patel",     "specialty": "Endocrinology",    "product": "Mounjaro",  "location": "North Houston Medical", "status": "pending"},
    {"id": "call-002", "hcp_name": "Dr. James Kim",       "specialty": "Cardiology",       "product": "Jardiance", "location": "Memorial Hermann",      "status": "done"},
    {"id": "call-003", "hcp_name": "Dr. Maria Rodriguez", "specialty": "Internal Medicine", "product": "Verzenio", "location": "UTHealth Houston",       "status": "pending"},
    {"id": "call-004", "hcp_name": "Dr. Chen Wei",        "specialty": "Oncology",         "product": "Verzenio", "location": "MD Anderson",            "status": "pending"},
]

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Pulse AI")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/calls")
async def get_calls():
    return JSONResponse(content=DUMMY_CALLS)


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoint — run full pipeline (used by WebSocket fallback)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/pipeline")
async def run_pipeline(body: dict):
    """
    Runs the full pipeline: Field Extraction → Summary → Compliance.
    Used as a fallback when the WebSocket pipeline has not already run.

    Request body:  {"transcript": "..."}
    Response:      {"fields": {...}, "summary": {...}, "compliance": {...}}
    """
    transcript = body.get("transcript", "")
    if not transcript:
        return JSONResponse(status_code=400, content={"error": "transcript is required"})

    extractor   = FieldExtraction()
    fields      = extractor.extract(transcript)

    summarizer  = SummaryField()
    summary     = summarizer.summarize(transcript)

    compliance  = ComplianceLayer()
    comp_result = compliance.analyze(transcript)

    return JSONResponse(content={
        "fields":     fields,
        "summary":    summary,
        "compliance": comp_result,
    })


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoint — compliance only (used by "Approve and review" button)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/compliance-only")
async def run_compliance_only(body: dict):
    """
    Runs only Compliance Goblin V3 on the transcript.
    Called when the rep taps "Approve and review" on the review screen.

    Fields and summary are already populated from the WebSocket pipeline
    that ran right after transcription — only compliance needs to run here.
    This makes the compliance check screen ~3x faster than running all three agents.

    Request body:  {"transcript": "..."}
    Response:      {"compliance": {...}}
    """
    transcript = body.get("transcript", "")
    if not transcript:
        return JSONResponse(status_code=400, content={"error": "transcript is required"})

    compliance  = ComplianceLayer()
    comp_result = compliance.analyze(transcript)

    return JSONResponse(content={"compliance": comp_result})


# ─────────────────────────────────────────────────────────────────────────────
# Sync helpers — run in thread pool to avoid blocking the event loop
# ─────────────────────────────────────────────────────────────────────────────

BROWSER_SAMPLE_RATE = 48_000


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    from math import gcd
    g = gcd(from_rate, to_rate)
    return resample_poly(audio, to_rate // g, from_rate // g).astype(np.float32)


def _sync_encode(chunk, overlap_buffer, is_first):
    return encode_chunk(chunk, overlap_buffer, is_first, _processor, _model)


def _sync_decode(accumulated_hidden, accumulated_masks):
    enc_hidden = torch.cat(accumulated_hidden, dim=1)
    enc_mask   = torch.cat(accumulated_masks,  dim=1)
    reconstructed = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=enc_hidden,
        attention_mask=enc_mask,
    )
    with torch.no_grad():
        ids = _model.generate(encoder_outputs=reconstructed, max_new_tokens=MAX_TOKENS)
    return _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _sync_run_pipeline(transcript: str) -> dict:
    """
    Runs the full Cortex agent pipeline synchronously in architecture order:
        Field Extraction → Summary → Compliance

    All three agents receive the raw Moonshine transcript directly.
    Called from the thread pool so it does not block the WebSocket event loop.
    """
    extractor   = FieldExtraction()
    fields      = extractor.extract(transcript)

    summarizer  = SummaryField()
    summary     = summarizer.summarize(transcript)

    compliance  = ComplianceLayer()
    comp_result = compliance.analyze(transcript)

    return {
        "fields":     fields,
        "summary":    summary,
        "compliance": comp_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket transcription endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    logging.info("WS accepted")

    loop = asyncio.get_running_loop()

    sample_rate        = BROWSER_SAMPLE_RATE
    resample_buf       = np.array([], dtype=np.float32)
    overlap_buffer     = np.array([], dtype=np.float32)
    accumulated_hidden: list[torch.Tensor] = []
    accumulated_masks:  list[torch.Tensor] = []
    chunk_index        = 0

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                text = msg["text"]

                if text.startswith("{"):
                    cfg         = json.loads(text)
                    sample_rate = int(cfg.get("sampleRate", BROWSER_SAMPLE_RATE))
                    logging.info(f"Config received: sampleRate={sample_rate}")
                    continue

                if text == "stop":
                    logging.info(f"STOP — chunks={chunk_index} buf={len(resample_buf)} hidden={len(accumulated_hidden)}")

                    if not accumulated_hidden:
                        logging.warning("STOP with no encoded chunks")
                        await websocket.send_json({"type": "transcript", "text": ""})
                        break

                    # Flush leftover samples
                    if len(resample_buf) > 0:
                        leftover_16k = (
                            _resample(resample_buf, sample_rate, SAMPLE_RATE)
                            if sample_rate != SAMPLE_RATE else resample_buf.copy()
                        )
                        resample_buf = np.array([], dtype=np.float32)
                        if len(leftover_16k) > 0:
                            is_first = (chunk_index == 0)
                            h, m = await loop.run_in_executor(_executor, _sync_encode, leftover_16k, overlap_buffer, is_first)
                            accumulated_hidden.append(h)
                            accumulated_masks.append(m)
                            chunk_index += 1
                            del leftover_16k
                            gc.collect()

                    await websocket.send_json({"type": "decoding"})
                    logging.info("Decoding…")

                    # Decode transcript from Moonshine encoder states
                    transcript = await loop.run_in_executor(_executor, _sync_decode, accumulated_hidden, accumulated_masks)
                    logging.info(f"Transcript: {len(transcript)} chars")

                    # Send transcript immediately so rep sees it
                    await websocket.send_json({"type": "transcript", "text": transcript})

                    # Run Field Extraction → Summary → Compliance pipeline
                    await websocket.send_json({"type": "status", "message": "Running field extraction..."})
                    logging.info("Starting pipeline: Field Extraction → Summary → Compliance")

                    pipeline_result = await loop.run_in_executor(_executor, _sync_run_pipeline, transcript)
                    logging.info("Pipeline complete")

                    # Send results in pipeline order
                    await websocket.send_json({"type": "fields",     "data": pipeline_result["fields"]})
                    await websocket.send_json({"type": "summary",    "data": pipeline_result["summary"]})
                    await websocket.send_json({"type": "compliance", "data": pipeline_result["compliance"]})
                    await websocket.send_json({"type": "done"})
                    await websocket.close()
                    break

            elif "bytes" in msg:
                raw_bytes: bytes = msg["bytes"]
                n_samples = len(raw_bytes) // 4
                if n_samples == 0:
                    continue

                frame    = np.frombuffer(raw_bytes, dtype="<f4").copy()
                frame_16k = _resample(frame, sample_rate, SAMPLE_RATE) if sample_rate != SAMPLE_RATE else frame
                del frame
                resample_buf = np.concatenate([resample_buf, frame_16k])
                del frame_16k

                while len(resample_buf) >= CHUNK_SAMPLES:
                    chunk        = resample_buf[:CHUNK_SAMPLES].copy()
                    resample_buf = resample_buf[CHUNK_SAMPLES:]
                    is_first     = (chunk_index == 0)

                    hidden, mask = await loop.run_in_executor(_executor, _sync_encode, chunk, overlap_buffer, is_first)
                    overlap_buffer = chunk[-OVERLAP_SAMPLES:].copy()
                    accumulated_hidden.append(hidden)
                    accumulated_masks.append(mask)
                    chunk_index += 1
                    del chunk
                    gc.collect()

                    await websocket.send_json({"type": "chunk_encoded", "index": chunk_index})

    except WebSocketDisconnect:
        logging.info("WS disconnected")
    except Exception as exc:
        logging.error(f"WS error:\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logging.info("WS handler exiting")
        del accumulated_hidden, accumulated_masks, overlap_buffer, resample_buf
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)