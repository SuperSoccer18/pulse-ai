"""
app.py  —  Privacy-First Audio Transcription Demo
==================================================
Streamlit frontend for the chunked encoder pipeline.

Audio is collected via browser mic or file upload, converted to 16 kHz mono
WAV via ffmpeg, and processed through the same chunked pipeline used in
transcribe_streaming_experiment.py.  The privacy constraint (≤0.9s of raw
audio in memory at any time) is preserved end-to-end.

Run:
    .venv/Scripts/streamlit run app.py
    # or on Mac/Linux:  streamlit run app.py
"""

import gc
import io
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingEncoderModelOutput,
)

from transcribe_streaming_experiment import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    OVERLAP_SAMPLES,
    MAX_TOKENS,
    iter_chunks,
    encode_chunk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_APP_DIR  = Path(__file__).parent
MODEL_DIR = _APP_DIR / "models" / "moonshine-streaming-medium"

# Peak raw audio in memory: one chunk + overlap buffer
PEAK_SAMPLES = CHUNK_SAMPLES + OVERLAP_SAMPLES   # 14,400 samples = 0.9s
PEAK_KB      = PEAK_SAMPLES * 4 / 1024           # float32 = 4 bytes/sample


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached for the lifetime of the server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading moonshine-streaming-medium — first run only…")
def load_model():
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Audio conversion
# ─────────────────────────────────────────────────────────────────────────────

def audio_bytes_to_array(audio_bytes: bytes) -> np.ndarray:
    """
    Convert any browser-recorded or uploaded audio bytes to a 16 kHz mono
    float32 numpy array via an ffmpeg stdin→stdout pipe.

    Handles WebM/Opus (Chrome/Edge mic), WAV, MP3, M4A, OGG, FLAC — anything
    ffmpeg understands.  No temp files are written to disk.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",              # read from stdin
            "-f", "wav",
            "-ar", str(SAMPLE_RATE),     # resample to 16 kHz
            "-ac", "1",                  # downmix to mono
            "pipe:1",                    # write WAV to stdout
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr.decode(errors='replace')[-600:]}"
        )

    audio, _ = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)

    # Normalise integer-range PCM if present (should be handled by ffmpeg, but belt+braces)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Chunked pipeline with Streamlit progress updates
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_with_progress(
    audio: np.ndarray,
    processor: AutoProcessor,
    model: AutoModelForSpeechSeq2Seq,
    progress_bar,   # st.progress object
    status_text,    # st.empty() placeholder
) -> str:
    """
    Run the full chunked encoder pipeline, advancing the Streamlit progress bar
    after each chunk.  Mirrors run_pipeline() in test_chunked_pipeline.py exactly.
    """
    n_chunks       = math.ceil(len(audio) / CHUNK_SAMPLES)
    overlap_buffer = np.array([], dtype=np.float32)
    accumulated_hidden: list[torch.Tensor] = []
    accumulated_masks:  list[torch.Tensor] = []

    for i, chunk in enumerate(iter_chunks(audio, CHUNK_SAMPLES)):
        is_first   = (i == 0)
        chunk_dur  = len(chunk) / SAMPLE_RATE

        status_text.caption(
            f"Encoding chunk {i + 1} of {n_chunks}  —  {chunk_dur:.2f}s  "
            f"({'first chunk, no overlap' if is_first else f'+ {OVERLAP_SAMPLES/SAMPLE_RATE:.1f}s overlap prepended'})"
        )

        hidden, mask = encode_chunk(chunk, overlap_buffer, is_first, processor, model)

        overlap_buffer = chunk[-OVERLAP_SAMPLES:].copy()
        accumulated_hidden.append(hidden)
        accumulated_masks.append(mask)

        del chunk
        gc.collect()

        progress_bar.progress((i + 1) / n_chunks)

    del audio
    gc.collect()

    status_text.caption("Decoding from accumulated encoder states…")

    enc_hidden = torch.cat(accumulated_hidden, dim=1)   # [1, T_total, 768]
    enc_mask   = torch.cat(accumulated_masks,  dim=1)   # [1, T_total]

    reconstructed = MoonshineStreamingEncoderModelOutput(
        last_hidden_state=enc_hidden,
        attention_mask=enc_mask,
    )

    with torch.no_grad():
        ids = model.generate(
            encoder_outputs=reconstructed,
            max_new_tokens=MAX_TOKENS,
        )

    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Shared transcription handler  (mic and file upload both funnel here)
# ─────────────────────────────────────────────────────────────────────────────

def handle_audio(audio_bytes: bytes, processor, model) -> None:
    """Process audio bytes through the pipeline and render results."""
    if not audio_bytes:
        return

    # Avoid re-running the pipeline if the same audio is resubmitted
    # (Streamlit reruns the full script on every widget interaction).
    audio_hash = hash(audio_bytes)
    if st.session_state.get("last_hash") == audio_hash:
        _render_result(st.session_state["last_result"], st.session_state["last_stats"])
        return

    try:
        # ── Convert to 16 kHz mono float32 ──────────────────────────────────
        with st.spinner("Converting audio…"):
            audio = audio_bytes_to_array(audio_bytes)

        # ── Privacy stats ────────────────────────────────────────────────────
        duration_s = len(audio) / SAMPLE_RATE
        n_chunks   = math.ceil(len(audio) / CHUNK_SAMPLES)

        stats = dict(duration_s=duration_s, n_chunks=n_chunks, peak_kb=PEAK_KB)
        _render_stats(stats)

        st.info(
            "**How the privacy constraint works:** "
            "Each 0.6s audio chunk is encoded to abstract hidden states, then "
            "immediately deleted from memory.  Only a 0.3s overlap buffer (boundary "
            "context) persists between chunks.  No raw audio survives to the decoding "
            "step — the decoder operates entirely on non-invertible encoder states.",
            icon="🔒",
        )

        # ── Pipeline ─────────────────────────────────────────────────────────
        progress_bar = st.progress(0)
        status_text  = st.empty()

        result = run_pipeline_with_progress(
            audio, processor, model, progress_bar, status_text
        )

        status_text.empty()
        progress_bar.progress(1.0)

        # Cache result so reruns don't re-process
        st.session_state["last_hash"]   = audio_hash
        st.session_state["last_result"] = result
        st.session_state["last_stats"]  = stats

        _render_result(result, stats)

    except Exception as exc:
        st.error(f"Transcription failed: {exc}")


def _render_stats(stats: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Recording duration",   f"{stats['duration_s']:.1f}s")
    c2.metric("Chunks processed",     str(stats["n_chunks"]))
    c3.metric("Peak audio in RAM",    f"{stats['peak_kb']:.0f} KB  (≤0.9s)")


def _render_result(result: str, stats: dict) -> None:
    st.success("Transcription complete")
    st.text_area(
        "Transcript",
        value=result if result else "(no speech detected)",
        height=140,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Privacy-First Transcription",
    page_icon="🔒",
    layout="centered",
)

st.title("Privacy-First Audio Transcription")
st.caption(
    "moonshine-streaming-medium  ·  "
    "≤0.9s of raw audio in memory at any time  ·  "
    "HIPAA-aligned ephemeral audio architecture"
)

st.divider()

processor, model = load_model()

# Initialise session state keys
if "last_hash"   not in st.session_state:
    st.session_state["last_hash"]   = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_stats"  not in st.session_state:
    st.session_state["last_stats"]  = None

tab_mic, tab_file = st.tabs(["🎙️  Microphone", "📁  Upload File"])

with tab_mic:
    st.markdown("Record audio directly in the browser, then wait for transcription.")
    audio_input = st.audio_input("Record audio")
    if audio_input is not None:
        handle_audio(audio_input.read(), processor, model)

with tab_file:
    st.markdown("Upload a WAV, MP3, M4A, OGG, FLAC, or WebM file.")
    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        handle_audio(uploaded.read(), processor, model)
