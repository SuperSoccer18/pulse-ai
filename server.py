"""
server.py  —  Pulse AI FastAPI backend
=======================================
Endpoints:
  GET  /                     → serves static/index.html
  GET  /api/calls             → live call list from Veeva CRM (JSON)
  POST /api/check-compliance  → runs compliance-gate only, no Veeva write —
                                used to show a checking/passed/held state
                                before the rep commits to submitting
  POST /api/submit-call       → runs compliance-gate on submitted fields; if
                                PROCEED, writes them to Veeva; if HOLD, returns
                                the violations for the review screen to display
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

import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
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

from cortexAgents import correct_transcript, extract_call_fields, check_compliance
from veeva_client import fetch_calls_for_rep, update_call_in_veeva

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
# App — call list is sourced live from Veeva CRM, see veeva_client.py
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Pulse AI")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC / "index.html"))


@app.get("/api/calls")
async def get_calls():
    try:
        calls = await fetch_calls_for_rep()
    except Exception as exc:
        logging.error(f"Failed to fetch calls from Veeva:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": f"Veeva fetch failed: {exc}"})
    return JSONResponse(content=calls)


# ─────────────────────────────────────────────────────────────────────────────
# Submit-call: compliance gate → Veeva write
# ─────────────────────────────────────────────────────────────────────────────
# Only the confirmed-writable Call2_vod__c fields established this session
# get sent to Veeva. products/next_steps/competitor_mentions/confidence stay
# UI-only — no Veeva field exists for the first three, and confidence is our
# own extraction metadata, not call content.

def _extraction_to_veeva_fields(extraction: dict) -> dict:
    """Map the extraction/edited-fields shape onto writable Call2_vod__c fields."""
    meta = extraction.get("call_metadata") or {}
    fields: dict = {}

    if meta.get("location"):
        fields["Territory_vod__c"] = meta["location"]

    call_datetime = meta.get("call_datetime")
    if call_datetime:
        # extract_call_fields() can return either a full ISO datetime
        # ("2026-07-15T10:00:00") or a date-only ISO string ("2026-07-15") —
        # see the CALL_DATETIME RESOLUTION rules added to the field-extractor
        # prompt. Branch on which we got.
        has_time = "T" in call_datetime
        fields["Call_Date_vod__c"] = call_datetime[:10]
        if has_time:
            # NOTE (unverified): Salesforce datetime fields typically expect
            # a timezone-qualified ISO string. Our extraction produces a
            # naive local ISO value with no offset — appending "Z" is a
            # best-effort guess, not a confirmed-correct format. Watch the
            # first live submit for a 400 specifically on this field.
            dt = call_datetime if call_datetime.endswith("Z") or "+" in call_datetime[10:] else call_datetime + "Z"
            fields["Call_Datetime_vod__c"] = dt

    if meta.get("engagement_method"):
        fields["Call_Channel_vod__c"] = meta["engagement_method"]
    if meta.get("virtual_engagement_tool"):
        fields["Remote_Meeting_Type_vod__c"] = meta["virtual_engagement_tool"]

    if extraction.get("summary"):
        fields["Chat_Summary_vod__c"] = extraction["summary"]

    # Mark the call finalized — matches veeva_client.py's _STATUS_MAP
    # (Submitted_vod -> "done"), so the next /api/calls read reflects this
    # submission as real persisted state, not just local UI state.
    fields["Status_vod__c"] = "Submitted_vod"

    return fields


@app.post("/api/check-compliance")
async def check_compliance_endpoint(payload: dict):
    """
    Compliance-only check — no Veeva write. Split out from /api/submit-call
    so the review screen can show a distinct "checking" step (spinner
    popup) before the rep commits to the actual write. On a clean result,
    the client calls /api/submit-call directly with a follow-up click
    rather than this endpoint re-checking or writing anything itself.
    """
    extraction = payload.get("extraction") or {}
    compliance = check_compliance(extraction)
    logging.info(f"[check-compliance] compliance={compliance!r}")

    status = "clean" if compliance.get("recommended_action") == "PROCEED" else "held"
    return JSONResponse(content={"status": status, "compliance": compliance})


@app.post("/api/submit-call")
async def submit_call(payload: dict):
    call_id    = payload.get("call_id")
    extraction = payload.get("extraction") or {}

    if not call_id:
        return JSONResponse(status_code=400, content={"status": "error", "error": "call_id is required"})

    compliance = check_compliance(extraction)
    logging.info(f"[submit] call_id={call_id} compliance={compliance!r}")

    if compliance.get("recommended_action") != "PROCEED":
        return JSONResponse(content={"status": "held", "compliance": compliance})

    veeva_fields = _extraction_to_veeva_fields(extraction)
    try:
        await update_call_in_veeva(call_id, veeva_fields)
    except Exception as exc:
        logging.error(f"Veeva write failed for {call_id}:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})

    return JSONResponse(content={"status": "submitted", "compliance": compliance})


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


# ─────────────────────────────────────────────────────────────────────────────
# Windowed decoding — tried and reverted (2026-07-17)
# ─────────────────────────────────────────────────────────────────────────────
# We attempted decoding the accumulated encoder states in fixed ~15s windows
# instead of one pass over the whole recording, to bound how long any single
# generate() call's target sequence could grow (autoregressive decode
# quality degrades on long sequences — observed as both a repetition loop
# and a hallucinated product extraction on ~50s recordings).
#
# Root cause found by reading modeling_moonshine_streaming.py: in
# MoonshineStreamingDecoder.forward(), the encoder-side positional embedding
# is computed from the LOCAL shape of whatever tensor is passed as
# encoder_hidden_states (torch.arange(encoder_hidden_states.shape[1])), not
# from that audio's true absolute position, and there is no exposed
# parameter to override this. Slicing accumulated_hidden into windows means
# every window after the first gets mislabeled as "starts at position 0"
# even though its real content is mid-recording. Confirmed numerically:
# pos_emb(0) vs pos_emb(750) differ by ~1.13 mean absolute magnitude, added
# directly into the decoder's cross-attention input.
#
# This fully explained the observed pattern (window 1 always clean, every
# window after it corrupted regardless of content, size, or generation
# settings — including an A/B test that ruled out repetition_penalty as the
# cause). What's NOT fully settled is why the mislabeling degrades output
# to this degree rather than just mildly — plausibly a distribution-shift
# effect on top of the positional bug: real mid-utterance audio content
# wearing a "this is the start of a recording" positional tag may be
# doubly out-of-distribution if the model was only ever trained on segments
# that are genuinely the start of an utterance. That would mean a truly
# independent per-window re-encode (correct local positions, but still
# mid-utterance acoustic content) might not fully fix this either — untested.
#
# Bookmarked as a possible future direction: true chunked transcription
# using VAD (voice activity detection) to find natural pause/silence
# boundaries for chunk splits, rather than fixed-duration windows — this
# would need each chunk to be independently re-encoded (not sliced from
# shared hidden states) and would still need to confirm whether the
# distribution-shift concern above applies to VAD-aligned chunks the same
# way it did to arbitrary-duration ones.
#
# For now: reverted to a single decode pass over the whole recording,
# keeping the two mitigations below as damage control for long-recording
# degradation.
# ─────────────────────────────────────────────────────────────────────────────

REPETITION_PENALTY = 1.3   # confirmed via A/B test not to be the cause of
                            # window-splitting corruption, but still useful
                            # against genuine same-phrase repetition loops.

# Frames-per-second for this model's encoder (SAMPLES_PER_FRAME/SAMPLE_RATE
# in transcribe_streaming_experiment.py: 320/16000 = 20ms/frame = 50/sec).
FRAMES_PER_SECOND = 50

# Damage control for runaway generation on long recordings: even with
# repetition guards, a long decode can degenerate into lexically-varied
# filler that dodges a 3-gram repeat ban ("I've. I need. I do. I and I
# am..."). MAX_TOKENS=448 as a flat ceiling lets a bad decode generate far
# more filler than any real recording of typical length would ever need
# (real speech: roughly 3-4 tokens/sec). Capping max_new_tokens
# proportional to the actual recording duration bounds the blast radius of
# a bad decode without affecting normal transcription — MIN_TOTAL_TOKENS
# keeps short recordings from being capped down to nothing.
TOKENS_PER_SECOND_CAP = 12   # generous — real speech is ~3-4 tokens/sec;
                              # this is a ceiling on garbage, not a target for speech
MIN_TOTAL_TOKENS = 32


def _max_tokens_for_recording(n_frames: int) -> int:
    """max_new_tokens proportional to recording duration, capped by the
    global MAX_TOKENS ceiling — see TOKENS_PER_SECOND_CAP above."""
    duration_seconds = n_frames / FRAMES_PER_SECOND
    return max(MIN_TOTAL_TOKENS, min(MAX_TOKENS, round(duration_seconds * TOKENS_PER_SECOND_CAP)))


def _sync_decode(accumulated_hidden, accumulated_masks):
    """Blocking decode call — runs in the thread pool."""
    enc_hidden = torch.cat(accumulated_hidden, dim=1)   # [1, T_total, 768]
    enc_mask   = torch.cat(accumulated_masks,  dim=1)   # [1, T_total]
    reconstructed = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=enc_hidden,
        attention_mask=enc_mask,
    )
    with torch.no_grad():
        ids = _model.generate(
            encoder_outputs=reconstructed,
            max_new_tokens=_max_tokens_for_recording(enc_hidden.shape[1]),
            no_repeat_ngram_size=3,
            repetition_penalty=REPETITION_PENALTY,
        )
    return _processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def _sync_pipeline(transcript: str) -> dict:
    """
    Runs the post-transcription Cortex pipeline synchronously: STT keyword
    correction, then structured field extraction. Called via run_in_executor
    so it does not block the event loop.

    NOTE: compliance (PI/AECP redaction) is not part of this stage anymore —
    it now runs as a submit-time gate right before the Veeva write, after the
    rep has reviewed/filled in fields. See the approve-btn handler in
    static/index.html for that seam. Always returns type "pipeline_complete";
    the old "pipeline_halted" message no longer exists.
    """
    correction = correct_transcript(transcript)
    corrected  = correction.get("corrected_transcript") or transcript
    logging.info(f"[pipeline] raw transcript ({len(transcript)} chars): {transcript!r}")
    logging.info(f"[pipeline] correction result: {correction!r}")
    logging.info(f"[pipeline] corrected transcript used for extraction: {corrected!r}")

    extraction = extract_call_fields(corrected)
    logging.info(f"[pipeline] extraction result: {extraction!r}")

    return {
        "type":       "pipeline_complete",
        "transcript": transcript,
        "correction": correction,
        "extraction": extraction,
    }


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
                    logging.info(f"Decode complete: {len(transcript)} chars — content: {transcript!r}")

                    # Run keyword-correction → field-extraction on the thread pool
                    await websocket.send_json({"type": "pipeline_running"})
                    result = await loop.run_in_executor(
                        _executor,
                        _sync_pipeline,
                        transcript,
                    )
                    logging.info(f"Pipeline complete: type={result['type']}")

                    await websocket.send_json(result)
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
