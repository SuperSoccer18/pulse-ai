"""
server.py  —  Pulse AI FastAPI backend
=======================================
Endpoints:
  GET  /          → serves static/index.html
  GET  /api/calls → dummy call list (JSON)
  WS   /ws/transcribe → receives raw PCM chunks, encodes on-the-fly,
                        returns {"transcript": "..."} when client sends "stop"

Audio contract with the browser:
  • Each binary WebSocket frame = one chunk of float32 samples, little-endian,
    already downsampled to 16 kHz mono by the server.
  • Browser sends raw float32 at its native sample rate (48 kHz typical) in
    frames of BROWSER_CHUNK_SAMPLES; server resamples each frame to 16 kHz.
  • After all chunks, browser sends the text message "stop" to trigger decode.
  • Server replies with JSON: {"type": "transcript", "text": "..."}
    or {"type": "error", "message": "..."}

Run:
    python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import gc
import json
import logging
import struct
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
    N_OVERLAP_FRAMES,
    MAX_TOKENS,
    encode_chunk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_APP_DIR  = Path(__file__).parent
MODEL_DIR = _APP_DIR / "models" / "moonshine-streaming-medium"
STATIC    = _APP_DIR / "static"

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

# Thread pool for running blocking torch operations without stalling the event loop.
# max_workers=1: the model is not thread-safe for concurrent inference, so we
# serialise all generate() / encode_chunk() calls through a single worker thread.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ─────────────────────────────────────────────────────────────────────────────
# Dummy call data (replace with CRM API call later)
# ─────────────────────────────────────────────────────────────────────────────

DUMMY_CALLS: list[dict[str, Any]] = [
    {
        "id": "call-001",
        "hcp_name": "Dr. Sarah Patel",
        "specialty": "Endocrinology",
        "product": "Mounjaro",
        "location": "North Houston Medical",
        "status": "pending",
    },
    {
        "id": "call-002",
        "hcp_name": "Dr. James Kim",
        "specialty": "Cardiology",
        "product": "Jardiance",
        "location": "Memorial Hermann",
        "status": "done",
    },
    {
        "id": "call-003",
        "hcp_name": "Dr. Maria Rodriguez",
        "specialty": "Internal Medicine",
        "product": "Verzenio",
        "location": "UTHealth Houston",
        "status": "pending",
    },
    {
        "id": "call-004",
        "hcp_name": "Dr. Chen Wei",
        "specialty": "Oncology",
        "product": "Verzenio",
        "location": "MD Anderson",
        "status": "pending",
    },
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
# WebSocket transcription endpoint
# ─────────────────────────────────────────────────────────────────────────────

# The browser AudioWorklet sends float32 PCM at the browser's native sample
# rate (48 000 Hz on most devices).  We resample each frame to 16 kHz and
# feed it into encode_chunk() in CHUNK_SAMPLES-sized windows.

BROWSER_SAMPLE_RATE = 48_000   # assumed; client sends actual rate in first msg

def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample mono float32 audio using integer-ratio polyphase filter."""
    from math import gcd
    g = gcd(from_rate, to_rate)
    up, down = to_rate // g, from_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def _sync_encode(chunk, overlap_buffer, is_first):
    """Blocking encode_chunk call — runs in the thread pool."""
    return encode_chunk(chunk, overlap_buffer, is_first, _processor, _model)


def _sync_decode(accumulated_hidden, accumulated_masks):
    """Blocking decode call — runs in the thread pool."""
    enc_hidden = torch.cat(accumulated_hidden, dim=1)
    enc_mask   = torch.cat(accumulated_masks,  dim=1)
    reconstructed = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=enc_hidden,
        attention_mask=enc_mask,
    )
    with torch.no_grad():
        ids = _model.generate(
            encoder_outputs=reconstructed,
            max_new_tokens=MAX_TOKENS,
        )
    return _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    logging.info("WS accepted")

    loop = asyncio.get_running_loop()   # get_running_loop() is correct in async context

    # Per-connection transcription state
    sample_rate      = BROWSER_SAMPLE_RATE   # updated by first message
    resample_buf     = np.array([], dtype=np.float32)  # leftover samples < CHUNK_SAMPLES
    overlap_buffer   = np.array([], dtype=np.float32)
    accumulated_hidden: list[torch.Tensor] = []
    accumulated_masks:  list[torch.Tensor] = []
    chunk_index      = 0

    try:
        while True:
            msg = await websocket.receive()

            # ── Text control messages ─────────────────────────────────────────
            if "text" in msg:
                text = msg["text"]

                if text.startswith("{"):
                    # JSON config frame sent before audio starts
                    cfg = json.loads(text)
                    sample_rate = int(cfg.get("sampleRate", BROWSER_SAMPLE_RATE))
                    logging.info(f"Config received: sampleRate={sample_rate}")
                    continue

                if text == "stop":
                    logging.info(f"STOP received — chunks_encoded={chunk_index}, resample_buf_len={len(resample_buf)}, accumulated_hidden={len(accumulated_hidden)}")
                    # No chunks received at all — nothing to decode
                    if not accumulated_hidden:
                        logging.warning("STOP with no encoded chunks — was recording too short?")
                        await websocket.send_json({"type": "transcript", "text": ""})
                        break

                    # Flush any leftover samples in resample_buf
                    if len(resample_buf) > 0:
                        leftover_16k = (
                            _resample(resample_buf, sample_rate, SAMPLE_RATE)
                            if sample_rate != SAMPLE_RATE
                            else resample_buf.copy()
                        )
                        resample_buf = np.array([], dtype=np.float32)

                        if len(leftover_16k) > 0:
                            is_first = (chunk_index == 0)
                            h, m = await loop.run_in_executor(
                                _executor,
                                _sync_encode,
                                leftover_16k, overlap_buffer, is_first,
                            )
                            accumulated_hidden.append(h)
                            accumulated_masks.append(m)
                            chunk_index += 1
                            del leftover_16k
                            gc.collect()

                    # Signal client that encoding is complete, decoding starting
                    await websocket.send_json({"type": "decoding"})
                    logging.info("Sent 'decoding', starting generate()…")

                    # Decode on thread pool — does NOT block the event loop
                    transcript = await loop.run_in_executor(
                        _executor,
                        _sync_decode,
                        accumulated_hidden, accumulated_masks,
                    )
                    logging.info(f"Decode complete: {len(transcript)} chars")

                    await websocket.send_json({"type": "transcript", "text": transcript})
                    await websocket.close()
                    break

            # ── Binary audio frames ───────────────────────────────────────────
            elif "bytes" in msg:
                raw_bytes: bytes = msg["bytes"]
                logging.debug(f"Binary frame received: {len(raw_bytes)} bytes ({len(raw_bytes)//4} float32 samples)")

                # Each frame is float32 little-endian PCM at browser sample rate
                n_samples = len(raw_bytes) // 4
                if n_samples == 0:
                    continue
                frame = np.frombuffer(raw_bytes, dtype="<f4").copy()   # float32 LE

                # Resample to 16 kHz if needed, then append to buffer
                if sample_rate != SAMPLE_RATE:
                    frame_16k = _resample(frame, sample_rate, SAMPLE_RATE)
                else:
                    frame_16k = frame

                del frame
                resample_buf = np.concatenate([resample_buf, frame_16k])
                del frame_16k

                # Process complete CHUNK_SAMPLES windows immediately
                while len(resample_buf) >= CHUNK_SAMPLES:
                    chunk = resample_buf[:CHUNK_SAMPLES].copy()
                    resample_buf = resample_buf[CHUNK_SAMPLES:]

                    is_first = (chunk_index == 0)
                    logging.info(f"Encoding chunk {chunk_index} (is_first={is_first}, samples={len(chunk)})")
                    hidden, mask = await loop.run_in_executor(
                        _executor,
                        _sync_encode,
                        chunk, overlap_buffer, is_first,
                    )
                    overlap_buffer = chunk[-OVERLAP_SAMPLES:].copy()
                    accumulated_hidden.append(hidden)
                    accumulated_masks.append(mask)
                    chunk_index += 1

                    del chunk
                    gc.collect()

                    # Notify client: chunk encoded, raw audio discarded
                    await websocket.send_json(
                        {"type": "chunk_encoded", "index": chunk_index}
                    )

    except WebSocketDisconnect:
        logging.info("WS disconnected by client")
    except Exception as exc:
        logging.error(f"Unhandled exception in ws_transcribe:\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        logging.info("WS handler exiting, releasing buffers")
        # Release all tensors and buffers for this connection
        del accumulated_hidden, accumulated_masks, overlap_buffer, resample_buf
        gc.collect()
