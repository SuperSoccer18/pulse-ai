# API Reference

All endpoints are defined in `server.py`.

## `GET /`

Serves `static/index.html`. The entire frontend is this one file plus
`static/recorder-worklet.js` — no build step, no framework.

## `GET /api/calls`

Fetches the demo rep's active calls from Veeva (`fetch_calls_for_rep()` in
`veeva_client.py`) and returns them as JSON for the call list screen.

Response shape (one entry per HCP, see [veeva-integration.md](veeva-integration.md)
for the one-row-per-HCP collapsing logic):

```json
[
  {
    "id": "a04...",
    "hcp_name": "Mahnaz Reyner",
    "specialty": "Dermatology",
    "location": "...",
    "product": "Trulicity",
    "status": "pending"
  }
]
```

`status` is `"pending"` or `"done"` — collapsed from Veeva's
`Status_vod__c` picklist (`Planned_vod`/`Saved_vod` → pending,
`Submitted_vod` → done; `Cancelled_vod` excluded).

On Veeva fetch failure: `500` with `{"error": "Veeva fetch failed: ..."}`.

## `POST /api/check-compliance`

Runs the compliance-gate agent only — **does not write to Veeva**. Exists
so the review screen can show a "checking…" spinner state before the rep
commits to the real submit. On a clean result, the frontend calls
`/api/submit-call` directly on a follow-up click; this endpoint does not
chain into a write itself.

Request:
```json
{ "extraction": { /* call_metadata, summary, products, ... */ } }
```

Response:
```json
{ "status": "clean" | "held", "compliance": { "overall_status": "...", "field_violations": [...], "recommended_action": "PROCEED" | "HOLD" } }
```

## `POST /api/submit-call`

Runs the same compliance gate, and if `PROCEED`, writes the mapped fields to
Veeva via `update_call_in_veeva()`. If `HOLD`, returns the violations without
writing anything — same shape the review screen already knows how to render
from `/api/check-compliance`.

Request:
```json
{ "call_id": "a04...", "extraction": { /* ... */ } }
```

Responses:
- `400` `{"status": "error", "error": "call_id is required"}` — missing call_id
- `200` `{"status": "held", "compliance": {...}}` — compliance gate failed, nothing written
- `500` `{"status": "error", "error": "..."}` — Veeva write failed
- `200` `{"status": "submitted", "compliance": {...}}` — written successfully

See `server.py`'s `_extraction_to_veeva_fields()` for the extraction→Veeva
field mapping, and [veeva-integration.md](veeva-integration.md) for which
fields are actually writable.

## `WS /ws/transcribe`

Streams raw audio in, returns a transcript + extracted fields when the
client signals done. One WebSocket connection = one recording session; all
state is connection-scoped (see [architecture.md](architecture.md)).

### Client → server

1. **First message, JSON text**: `{"sampleRate": 48000}` — the browser's
   `AudioContext.sampleRate` (varies by device). Must be sent before any
   binary frames; the server resamples every subsequent frame from this
   rate to 16 kHz internally.
2. **Binary frames**: raw float32 PCM, little-endian, mono, at the sample
   rate declared above. Framed by `recorder-worklet.js` in fixed
   `FRAME_SAMPLES = 4800` (0.1s @ 48kHz) chunks — the server does not care
   about frame boundaries, it just concatenates and re-chunks into its own
   `CHUNK_SAMPLES` (9600 samples = 0.6s @ 16kHz) windows.
3. **Text message `"stop"`**: signals end of recording. Flushes any
   leftover buffered samples, then triggers decode + the Cortex pipeline.

### Server → client

| Message | When | Payload |
|---|---|---|
| `{"type": "chunk_encoded", "index": N}` | After each `CHUNK_SAMPLES` window is encoded and its raw audio discarded | running chunk count |
| `{"type": "decoding"}` | After `"stop"` is received and all buffered audio is flushed/encoded | — |
| `{"type": "pipeline_running"}` | After decode finishes, before the Cortex correction/extraction calls | — |
| `{"type": "pipeline_complete", "transcript": "...", "correction": {...}, "extraction": {...}}` | Final message before the server closes the socket | full pipeline result |
| `{"type": "transcript", "text": ""}` | Sent instead of the above if `"stop"` arrives with zero encoded chunks (recording too short) | — |
| `{"type": "error", "message": "..."}` | Unhandled exception in the handler | — |

Note the docstring at the top of `server.py` describes an older
`{"type": "transcript", ...}` success shape — that's stale for the normal
path; `pipeline_complete` is what actually ships now (the pipeline no
longer has a "halted" state either — see the note in `_sync_pipeline()`'s
docstring). The `{"type": "transcript", "text": ""}` empty-recording case is
the one place the old shape is still real.

The server closes the socket itself after sending `pipeline_complete`, or on
an unhandled exception. `WebSocketDisconnect` from the client is handled
silently (rep closed the tab / aborted mid-recording).
