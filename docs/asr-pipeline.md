# ASR Pipeline Internals

Model: `moonshine-streaming-medium` (UsefulSensors), loaded via
`transformers`' `AutoProcessor`/`AutoModelForSpeechSeq2Seq`. Not checked
into git — see [setup.md](setup.md).

## Why chunked encoding

The core design goal (see `transcribe_streaming_experiment.py`'s module
docstring) is **privacy-first processing**: raw audio should never
accumulate in memory for more than one short window at a time. Concretely:

- Audio is processed in ≤0.9s at a time: a 0.6s chunk (`CHUNK_SAMPLES =
  9600` @ 16kHz) plus a 0.3s overlap buffer (`OVERLAP_SAMPLES = 4800`,
  `N_OVERLAP_FRAMES = 15`) carried from the previous chunk for boundary
  context.
- Each chunk is encoded (`encode_chunk()` in
  `transcribe_streaming_experiment.py`) and the raw waveform is deleted
  (`del ...; gc.collect()`) immediately — both before the encoder runs
  (deletes the pre-encode concatenated chunk+overlap) and immediately after
  (deletes the processed input tensor). These are called out in the code as
  "Privacy checkpoint 1/2".
- What survives across chunks is **encoder hidden states** — `[1, T, 768]`
  tensors, not invertible back to audio — plus a small overlap buffer of
  raw samples (the only raw audio that persists between iterations, and
  only for one chunk's worth).
- All accumulated hidden states are concatenated and decoded in a single
  `model.generate()` pass at the end (`_sync_decode()` in `server.py`).

This same logic is duplicated in three places, all importing shared
constants/functions from `transcribe_streaming_experiment.py`:
`server.py` (the live WS path), `test_chunked_pipeline.py` (edge-case
tests), and `benchmark_encode_chunk.py` (latency benchmarking).

The frontend surfaces this behavior directly to the rep — the "purge strip"
label on the recording screen (`static/index.html`'s
`resetChunkStrip()`/`handleWsMessage()`) ticks up per `chunk_encoded`
message and is *not* simulated: each message only arrives after
`server.py` has already run `del chunk; gc.collect()` on that chunk's raw
audio.

## Resampling

Browsers report `AudioContext.sampleRate` (typically 48000, but device
dependent) as the first WS message. `server.py`'s `_resample()` uses
`scipy.signal.resample_poly` with an integer up/down ratio (via `gcd`) to
convert each incoming frame to the model's required 16 kHz mono before
chunking.

## Decode-time token budgeting

`MAX_TOKENS = 448` is a flat ceiling from `transcribe_streaming_experiment.py`.
On top of that, `server.py` adds:

- `REPETITION_PENALTY = 1.3` and `no_repeat_ngram_size=3` — guards against
  literal repeated-phrase loops. Confirmed via A/B test (see comment above
  `REPETITION_PENALTY`) to **not** be the root cause of the window-splitting
  corruption described below — kept only as a defense against genuine
  repetition.
- `_max_tokens_for_recording()` — caps `max_new_tokens` proportional to
  actual recording duration (`TOKENS_PER_SECOND_CAP = 12`, generous vs. real
  speech's ~3-4 tokens/sec; `MIN_TOTAL_TOKENS = 32` floor for short
  recordings). This bounds how much filler a runaway decode can generate
  without affecting normal transcription — real speech at typical length
  never approaches the cap.

## Windowed decoding — tried and reverted (2026-07-17)

**Read this before attempting to re-introduce fixed-window decoding of long
recordings.** This is documented in detail as a comment block in
`server.py` right above `REPETITION_PENALTY`; this section is that
investigation summarized for the handoff doc, not new information.

### The problem being solved

Autoregressive decode quality degrades on long target sequences — observed
as both repetition loops and hallucinated product extraction on ~50s
recordings. The fix attempted: decode the accumulated encoder states in
fixed ~15s windows instead of one pass over the whole recording, to bound
how long any single `generate()` call's target sequence could grow.

### Root cause (confirmed)

Found by reading `modeling_moonshine_streaming.py`:
`MoonshineStreamingDecoder.forward()` computes the encoder-side positional
embedding from the **local shape** of whatever tensor is passed as
`encoder_hidden_states` — i.e. `torch.arange(encoder_hidden_states.shape[1])`
— not from that audio's true absolute position in the recording, and there
is no exposed parameter to override this.

Slicing accumulated hidden states into windows means every window after the
first gets mislabeled as "starts at position 0," even though its real
content is mid-recording. Confirmed numerically: `pos_emb(0)` vs.
`pos_emb(750)` differ by ~1.13 mean absolute magnitude, added directly into
the decoder's cross-attention input.

This fully explains the observed pattern: window 1 was always clean, every
window after it was corrupted regardless of content, size, or generation
settings — including the A/B test above that ruled out `repetition_penalty`
as the cause.

### What's *not* settled

Why the mislabeling degrades output to this degree rather than just mildly.
One plausible theory (untested): a distribution-shift effect on top of the
positional bug — real mid-utterance audio content wearing a "this is the
start of a recording" positional tag may be doubly out-of-distribution if
the model was only ever trained on segments that are genuinely the start of
an utterance.

### Current state

Reverted to a single decode pass over the whole recording
(`_sync_decode()` in `server.py`), relying on `REPETITION_PENALTY` and the
duration-proportional token cap as damage control for long-recording
degradation, rather than solving the underlying issue.

### Bookmarked future direction: VAD-triggered segment reset

The fix isn't windowing the *decode* step — it's resetting the *encoder
accumulator itself* at natural silence boundaries, so every decoded
sequence has a genuinely true, not just structurally-imposed, position-0
start. Concretely, alongside the existing chunked-encoding workflow
(unchanged: still 0.6s chunks, still purged immediately after encoding):

- Run a VAD model concurrently, evaluated per chunk.
- If VAD detects silence for a chunk, treat that as a segment boundary:
  trigger a `model.generate()` decode over the accumulated hidden states
  *up to but not including* that chunk, then reset the accumulator
  (`accumulated_hidden`, `accumulated_masks`) and drop the overlap buffer —
  the next chunk starts a fresh encoding sequence at true position 0, not a
  slice of a longer one.
- If VAD detects speech, accumulate as today and keep going.

This resolves the confirmed root cause directly: because each segment is a
real, independently-accumulated sequence rather than a slice of one,
`torch.arange(encoder_hidden_states.shape[1])` is correct by construction
for every segment, not just the first. It also resolves the open
distribution-shift question above without needing to test it separately —
under this scheme every decoded segment genuinely *is* the start of an
utterance (real silence preceded it), so there's no mislabeled content left
to be out-of-distribution. This supersedes the earlier idea (recorded in an
older revision of this doc) that a per-window *re-encode* from raw audio
would be needed to get correct positions — resetting the accumulator gets
correct positions for free, so the existing chunked-encoding path (and its
privacy-purge behavior) is otherwise unchanged.

Feasibility notes / open design questions for whoever picks this up:

- **VAD granularity.** A single silent/non-silent label per full 0.6s chunk
  is coarse relative to typical VAD frame sizes (~20-30ms for something
  like Silero VAD). Likely want VAD evaluated at finer sub-chunk resolution
  and only honor a split after a minimum run of consecutive silent frames —
  otherwise a chunk that's mostly speech but happens to end in trailing
  silence (or vice versa) could clip mid-word.
- **Debounce against over-splitting.** Brief pauses (breaths, "um," a beat
  between sentences) will read as VAD-silence far more often than we'd want
  segment resets. Needs a minimum-silence-duration threshold before a
  detected silence is allowed to trigger a reset, tuned against real call
  recordings rather than guessed.
- **Min/max segment bounds as safety nets.** A minimum accumulated duration
  before a silence event is allowed to trigger a reset (avoids a decode
  storm on a pause-heavy rep), and a maximum duration that forces a
  decode+reset even without confirmed silence (a fallback for a rep who
  talks continuously without pausing — otherwise that one long segment
  still hits the original degradation problem).
- **Overlap buffer must NOT survive a reset.** The existing
  `overlap_buffer` mechanism carries ~0.3s of raw audio across chunk
  boundaries *within* a segment for encoder context. At a VAD-triggered
  reset it should be dropped, not carried forward — the new segment's
  first chunk should be encoded exactly as if it were the very first chunk
  of the whole recording, and since the reset point is a real silence,
  nothing informative is lost by dropping it.
- **Thread-pool contention shape changes.** The single-worker
  `ThreadPoolExecutor` (see [architecture.md](architecture.md)) serializes
  every encode/decode call today, with one decode at the very end. Under
  this scheme decode calls interleave with encode calls throughout the
  recording — total compute shouldn't rise much, but whichever chunk lands
  right after a VAD-triggered decode queues behind it, which could surface
  as a small stutter in the "chunk encoded" UI at each split point. Worth
  watching in `benchmark_encode_chunk.py`-style benchmarking; may or may
  not warrant a second worker.
- **Cortex stages stay unchanged.** Correction and extraction
  (`cortexAgents.py`) should still run once over the full concatenated
  transcript, not per-segment — `field-extractor` needs whole-call context
  (e.g. resolving `call_datetime` from something said mid-call). This
  change is decode-granularity-only; it doesn't touch the pipeline stages
  after transcription.
- **Free side effect: progressive partial transcripts.** Since decode now
  happens throughout the recording instead of only at the end, each
  completed segment's transcript could be streamed to the frontend as it's
  produced (a new WS message type alongside `chunk_encoded`, e.g.
  `segment_transcribed`). Not required for the fix, but falls out of this
  design for free and is a natural UX upgrade.
- **Model choice for VAD.** Silero VAD is the natural default over energy-
  based approaches (e.g. WebRTC VAD) — better handles real-world noise
  (clinic environments, cross-talk), and has a small ONNX export that fits
  alongside the existing ONNX/torch stack already used in this repo. Worth
  checking whether it's cheap enough to run inline on the asyncio event
  loop rather than through the same executor that's already serializing
  encode/decode.
- **Validation approach.** Same pattern as `test_chunked_pipeline.py`/
  `benchmark_encode_chunk.py` — extend `audio/` with recordings that have
  natural pauses, confirm the previously-observed degradation (repetition
  loops, hallucinated extraction on long recordings) actually disappears,
  and separately stress-test a pause-free long monologue to confirm the
  max-duration fallback degrades gracefully rather than silently
  reintroducing the original bug.

No VAD implementation exists in this repo yet.

## Related scripts

- `transcribe_streaming_experiment.py` — the reference chunked pipeline
  (loads a WAV file, compares chunked vs. full-audio baseline transcription).
  Run directly for a quick sanity check against `audio/test.wav`.
- `test_chunked_pipeline.py` — edge-case tests (see [testing.md](testing.md)).
- `benchmark_encode_chunk.py` — latency benchmark across chunk sizes, to
  confirm real-time viability (mean/max encode time vs. chunk duration).
- `transcribe_base.py` — a separate, older, non-streaming pipeline against
  `models/moonshine-base` (ONNX). Not used by `server.py`; kept as a
  reference implementation with its own tuning (entropy-based early stop,
  silence trimming) that was never ported to the streaming path.
