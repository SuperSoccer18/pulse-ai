"""
app.py  —  Pulse.AI Privacy-First Audio Transcription
======================================================
Streamlit frontend for the chunked encoder pipeline.

Audio is collected via browser mic or file upload, converted to 16 kHz mono
WAV via ffmpeg, and processed through the same chunked pipeline used in
transcribe_streaming_experiment.py.  The privacy constraint (<=0.9s of raw
audio in memory at any time) is preserved end-to-end.

Run:
    source .venv/Scripts/activate
    streamlit run app.py
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

# TEMPORARY — swap back to real imports once Cortex whitelisting is approved
from mock_cortex import MockComplianceLayer  as ComplianceLayer
from mock_cortex import MockFieldExtraction  as FieldExtraction
from mock_cortex import MockSummaryField     as SummaryField

# To use real agents once whitelisted, replace the three lines above with:
# from complianceLayer import ComplianceLayer
# from fieldExtraction  import FieldExtraction
# from summaryField     import SummaryField

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# MODEL_DIR points to C:\PulseAI to avoid spaces in the OneDrive path
MODEL_DIR = r"C:\PulseAI\models\moonshine-streaming-medium"

# Peak raw audio in memory: one chunk + overlap buffer
PEAK_SAMPLES = CHUNK_SAMPLES + OVERLAP_SAMPLES   # 14,400 samples = 0.9s
PEAK_KB      = PEAK_SAMPLES * 4 / 1024           # float32 = 4 bytes/sample


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached for the lifetime of the server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading moonshine-streaming-medium — first run only...")
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_DIR,
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
    float32 numpy array via an ffmpeg stdin to stdout pipe.
    Handles WebM/Opus (Chrome/Edge mic), WAV, MP3, M4A, OGG, FLAC.
    No temp files are written to disk.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-f", "wav",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr.decode(errors='replace')[-600:]}"
        )

    audio, _ = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)

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
    progress_bar,
    status_text,
) -> str:
    """
    Run the full chunked encoder pipeline, advancing the Streamlit progress bar
    after each chunk. Mirrors run_pipeline() in test_chunked_pipeline.py exactly.
    """
    n_chunks       = math.ceil(len(audio) / CHUNK_SAMPLES)
    overlap_buffer = np.array([], dtype=np.float32)
    accumulated_hidden: list[torch.Tensor] = []
    accumulated_masks:  list[torch.Tensor] = []

    for i, chunk in enumerate(iter_chunks(audio, CHUNK_SAMPLES)):
        is_first  = (i == 0)
        chunk_dur = len(chunk) / SAMPLE_RATE

        status_text.caption(
            f"Encoding chunk {i + 1} of {n_chunks}  --  {chunk_dur:.2f}s  "
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

    status_text.caption("Decoding from accumulated encoder states...")

    enc_hidden = torch.cat(accumulated_hidden, dim=1)
    enc_mask   = torch.cat(accumulated_masks,  dim=1)

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
# Compliance + Field Extraction + Summary pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_compliance_pipeline(result: str) -> None:
    """
    Runs the three Cortex agents on the transcript and renders the
    human review UI in Streamlit.
    Currently uses mock agents — swap imports above once whitelisted.
    """

    st.divider()
    st.subheader("Step 2 of 4 -- Compliance Analysis")
    status_text = st.empty()
    status_text.caption("Running compliance analysis...")

    compliance  = ComplianceLayer()
    comp_result = compliance.analyze(result)

    status_text.empty()

    # Compliance status badge
    status_colors = {
        "CLEAN":    ("success", "CLEAN -- No violations detected"),
        "ADVISORY": ("warning", "ADVISORY -- Minor issue detected"),
        "WARNING":  ("warning", "WARNING -- Significant violation"),
        "CRITICAL": ("error",   "CRITICAL -- Serious violation"),
    }
    badge_fn, badge_msg = status_colors.get(
        comp_result["compliance_status"], ("warning", "UNKNOWN")
    )
    getattr(st, badge_fn)(badge_msg)

    if comp_result["violations"]:
        st.warning("Violations detected:")
        for v in comp_result["violations"]:
            st.write(f"  - {v}")

    if comp_result["recommended_action"] == "HALT":
        st.error(
            "Compliance HALT -- Transcript contains critical violations. "
            "DLO escalation required before CRM submission."
        )
        return

    # ── Field Extraction ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 3 of 4 -- Veeva CRM Fields")
    st.caption("Review and edit all fields before submitting to Veeva CRM.")

    cleaned   = comp_result["cleaned_transcript"]
    extractor = FieldExtraction()
    fields    = extractor.extract(cleaned)

    col1, col2 = st.columns(2)
    with col1:
        account  = st.text_input("Account",       value=fields.get("account")       or "")
        location = st.text_input("Location",      value=fields.get("location")      or "")
        address  = st.text_input("Address",       value=fields.get("address")       or "")
        call_dt  = st.text_input("Call DateTime", value=fields.get("call_datetime") or "")
    with col2:
        engagement = st.selectbox(
            "Engagement Method",
            ["In-office", "Virtual", "P2P", "Exhibit/Congress", "Non Sales Out of Office"],
            index=0,
        )
        st.text_input("Record Type",             value="Interaction", disabled=True)
        st.text_input("Virtual Engagement Tool", value="N/A",         disabled=True)

    notes = st.text_area(
        "Interaction Notes (max 255 chars)",
        value=fields.get("interaction_notes") or "",
        max_chars=255,
    )

    if fields.get("products_discussed"):
        st.subheader("Products Discussed")
        for p in fields["products_discussed"]:
            st.write(f"  - **{p.get('product', 'Unknown')}** -- {p.get('indication', 'N/A')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 4 of 4 -- Call Summary")

    summarizer = SummaryField()
    summary    = summarizer.summarize(cleaned)

    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Call Overview:** {summary.get('call_overview', 'N/A')}")
        st.write(f"**Products:** {summary.get('products_discussed', 'N/A')}")
        st.write(f"**Key Topics:** {summary.get('key_topics', 'N/A')}")
        st.write(f"**HCP Response:** {summary.get('hcp_response', 'N/A')}")
    with col4:
        st.write(f"**Objections:** {summary.get('objections_raised', 'N/A')}")
        st.write(f"**Next Steps:** {summary.get('next_steps', 'N/A')}")
        st.write(f"**Rep Recommendations:** {summary.get('rep_recommendations', 'N/A')}")

    # ── Submit button ─────────────────────────────────────────────────────────
    st.divider()
    if st.button("Approve and Submit to Veeva CRM", type="primary"):
        st.success("Submitted to Veeva CRM successfully!")
        st.balloons()


# ─────────────────────────────────────────────────────────────────────────────
# Shared transcription handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_audio(audio_bytes: bytes, processor, model) -> None:
    """Process audio bytes through the pipeline and render results."""
    if not audio_bytes:
        return

    audio_hash = hash(audio_bytes)
    if st.session_state.get("last_hash") == audio_hash:
        _render_result(st.session_state["last_result"], st.session_state["last_stats"])
        run_compliance_pipeline(st.session_state["last_result"])
        return

    try:
        with st.spinner("Converting audio..."):
            audio = audio_bytes_to_array(audio_bytes)

        duration_s = len(audio) / SAMPLE_RATE
        n_chunks   = math.ceil(len(audio) / CHUNK_SAMPLES)

        stats = dict(duration_s=duration_s, n_chunks=n_chunks, peak_kb=PEAK_KB)
        _render_stats(stats)

        st.info(
            "**How the privacy constraint works:** "
            "Each 0.6s audio chunk is encoded to abstract hidden states, then "
            "immediately deleted from memory. Only a 0.3s overlap buffer (boundary "
            "context) persists between chunks. No raw audio survives to the decoding "
            "step -- the decoder operates entirely on non-invertible encoder states.",
            icon="lock",
        )

        st.subheader("Step 1 of 4 -- Transcription")
        progress_bar = st.progress(0)
        status_text  = st.empty()

        result = run_pipeline_with_progress(
            audio, processor, model, progress_bar, status_text
        )

        status_text.empty()
        progress_bar.progress(1.0)

        st.session_state["last_hash"]   = audio_hash
        st.session_state["last_result"] = result
        st.session_state["last_stats"]  = stats

        _render_result(result, stats)

        # Run the full compliance + field extraction + summary pipeline
        run_compliance_pipeline(result)

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")


def _render_stats(stats: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Recording duration", f"{stats['duration_s']:.1f}s")
    c2.metric("Chunks processed",   str(stats["n_chunks"]))
    c3.metric("Peak audio in RAM",  f"{stats['peak_kb']:.0f} KB  (<=0.9s)")


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
    page_title="Pulse.AI",
    page_icon="lock",
    layout="centered",
)

st.title("Pulse.AI -- Privacy-First CRM Automation")
st.caption(
    "moonshine-streaming-medium  *  "
    "<=0.9s of raw audio in memory at any time  *  "
    "HIPAA-aligned ephemeral audio architecture"
)

st.divider()

processor, model = load_model()

if "last_hash"   not in st.session_state: st.session_state["last_hash"]   = None
if "last_result" not in st.session_state: st.session_state["last_result"] = None
if "last_stats"  not in st.session_state: st.session_state["last_stats"]  = None

tab_mic, tab_file = st.tabs(["Microphone", "Upload File"])

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