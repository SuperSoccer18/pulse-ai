"""
transcribe_base.py — Local speech-to-text using Moonshine (ONNX)

Usage:
    python transcribe_base.py audio/TestAudio.m4a
    python transcribe_base.py audio/test.wav       # already 16kHz WAV
"""

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as rt
import soundfile as sf
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR       = Path("models/moonshine-base")
TOKENIZER_PATH  = Path("models/moonshine-base/tokenizer.json")
SAMPLE_RATE     = 16000
REPET_PENALTY   = 1.6    # was 1.3 — stronger penalty tames repetitive "I will..." patterns
NGRAM_SIZE      = 4      # block repeated n-grams of this size
TOKENS_PER_SEC  = 3      # real speech is ~2-3 tokens/sec; tight budget prevents hallucination after audio ends
MAX_TOKEN_CAP   = 60     # hard ceiling
ENTROPY_STOP    = 6.5    # entropy threshold per token (real speech ~2–4, guessing 6+)
ENTROPY_CONSEC  = 3      # stop only after this many consecutive high-entropy tokens
EOT_TOKEN       = 2


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def trim_silence(audio: np.ndarray, sr: int, frame_ms: int = 20, energy_threshold: float = 0.01, pad_ms: int = 200) -> np.ndarray:
    """Trim trailing silence by finding the last frame with RMS energy above threshold."""
    frame_len  = int(sr * frame_ms / 1000)
    pad_samples = int(sr * pad_ms / 1000)

    # Compute RMS per frame
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    rms_per_frame = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])

    # Find last active frame
    active = np.where(rms_per_frame > energy_threshold)[0]
    if len(active) == 0:
        return audio  # nothing to trim — return as-is

    last_active_sample = (active[-1] + 1) * frame_len + pad_samples
    trimmed = audio[:min(last_active_sample, len(audio))]
    print(f"Trimmed: {len(audio)/sr:.2f}s → {len(trimmed)/sr:.2f}s")
    return trimmed


def load_audio(path: str | Path) -> np.ndarray:
    """Load audio file, convert to 16kHz mono float32 in [-1, 1]."""
    path = Path(path)

    # Convert non-WAV formats (m4a, mp3, etc.) via ffmpeg
    if path.suffix.lower() != ".wav":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             # Speech-optimized filter chain:
             #   highpass: remove rumble/handling noise below 80Hz
             #   lowpass:  discard above 7kHz (not needed for speech, adds noise)
             #   afftdn:   FFT-based noise reduction
             #   loudnorm: EBU R128 loudness normalization for consistent level
             "-af", "highpass=f=80,lowpass=f=7000,afftdn=nf=-25,loudnorm",
             wav_path],
            check=True, capture_output=True,
        )
        path = Path(wav_path)

    audio, sr = sf.read(str(path), dtype="float32")

    # sf.read returns int16-range values for PCM-16 WAV files even with dtype=float32
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0

    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. "
                         "Pass a non-WAV file to auto-convert, or resample first.")

    # Normalize to RMS target 0.1 (more robust than peak for speech)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms * 0.1
        # Hard-clip to ±1 in case of extreme spikes
        audio = np.clip(audio, -1.0, 1.0)

    # Trim trailing silence: find last frame with meaningful energy
    audio = trim_silence(audio, sr)

    print(f"Audio: {len(audio)/sr:.2f}s, mean amplitude: {np.abs(audio).mean():.4f}")
    return audio


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def apply_repetition_penalty(logits: np.ndarray, token_ids: list[int], penalty: float) -> np.ndarray:
    for tok in set(token_ids):
        if tok != EOT_TOKEN:  # never penalize EOT — let the model stop naturally
            logits[tok] /= penalty
    return logits


def apply_ngram_blocking(logits: np.ndarray, token_ids: list[int], n: int) -> np.ndarray:
    """Zero out any token that would extend a previously seen n-gram."""
    if len(token_ids) < n:
        return logits
    context = tuple(token_ids[-(n - 1):])
    for i in range(len(token_ids) - n):
        if tuple(token_ids[i : i + n - 1]) == context:
            logits[token_ids[i + n - 1]] = -np.inf
    return logits


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray) -> str:
    # Load ONNX sessions
    preprocess = rt.InferenceSession(str(MODEL_DIR / "preprocess.onnx"))
    encode     = rt.InferenceSession(str(MODEL_DIR / "encode.onnx"))
    uncached   = rt.InferenceSession(str(MODEL_DIR / "uncached_decode.onnx"))
    cached     = rt.InferenceSession(str(MODEL_DIR / "cached_decode.onnx"))

    tokenizer  = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Token budget: ~3 tokens/sec + small buffer, capped at MAX_TOKEN_CAP
    duration_s = len(audio) / SAMPLE_RATE
    max_tokens = min(int(duration_s * TOKENS_PER_SEC) + 5, MAX_TOKEN_CAP)
    print(f"Max tokens: {max_tokens} (for {duration_s:.1f}s of audio)")

    audio_input = audio[np.newaxis, :]  # [1, samples]

    # 1. Preprocess → mel features
    features = preprocess.run(None, {"args_0": audio_input})[0]

    # 2. Encode
    seq_len = np.array([features.shape[1]], dtype=np.int32)
    encoded = encode.run(None, {"args_0": features, "args_1": seq_len})[0]

    # 3. First token (uncached)
    tokens   = np.array([[1]], dtype=np.int32)
    outputs  = uncached.run(None, {"args_0": tokens, "args_1": encoded, "args_2": seq_len})
    logits   = outputs[0]
    kv_cache = outputs[1:]

    next_logits = apply_repetition_penalty(logits[0, -1].copy(), [], REPET_PENALTY)
    next_token  = greedy(next_logits)
    token_ids        = [next_token]
    high_entropy_run = 0  # consecutive tokens above ENTROPY_STOP

    # 4. Autoregressive decode (cached)
    for _ in range(max_tokens):
        outputs = cached.run(None, {
            "args_0": np.array([[next_token]], dtype=np.int32),
            "args_1": encoded,
            "args_2": seq_len,
            **{f"args_{i+3}": kv_cache[i] for i in range(len(kv_cache))},
        })
        logits   = outputs[0]
        kv_cache = outputs[1:]

        next_logits = logits[0, -1].copy()
        next_logits = apply_repetition_penalty(next_logits, token_ids, REPET_PENALTY)
        next_logits = apply_ngram_blocking(next_logits, token_ids, NGRAM_SIZE)
        next_token  = greedy(next_logits)

        if next_token == EOT_TOKEN:
            print(f"EOT after {len(token_ids)} tokens")
            break

        # Entropy-based stop: high entropy = model is guessing beyond the audio
        probs = np.exp(next_logits - next_logits.max())
        probs /= probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        if entropy > ENTROPY_STOP:
            high_entropy_run += 1
            if high_entropy_run >= ENTROPY_CONSEC:
                print(f"High-entropy stop at token {len(token_ids)} (entropy={entropy:.2f}, run={high_entropy_run})")
                break
        else:
            high_entropy_run = 0

        token_ids.append(next_token)

    return tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "audio/TestAudio.m4a"
    audio = load_audio(audio_path)
    result = transcribe(audio)
    print("\nTranscription:\n", result)
