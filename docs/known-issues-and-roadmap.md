# Known Issues, Open Questions & Roadmap

## Known gaps / demo scaffolding to replace before any real usage

- **Single hardcoded rep** (`ARTHUR_ID`) — no login, session, or multi-rep
  support anywhere in the app.
- **No Veeva token caching** — a fresh OAuth token is fetched on every
  Veeva call. Explicitly noted as fine for demo volume, not production
  traffic.
- **Placeholder product data** (`_PRODUCT_PLACEHOLDER`) — stand-in for the
  real product source (`Call2_Detail_vod__c`), which is locked for this
  integration user's permission set.
- **Products/next-steps/competitor-mentions never persist to Veeva** — no
  writable field exists for them; they're UI-only. If a real Veeva field
  ever opens up for these, `_extraction_to_veeva_fields()` needs updating.
- **No auth on the FastAPI app itself** — anyone who can reach the server
  can hit any endpoint or the WebSocket.
- **Single-worker thread pool serializes all model calls across all
  connections** — two reps recording simultaneously would queue behind
  each other rather than run in parallel. See
  [architecture.md](architecture.md).
- **No deployment target** — no Dockerfile, CI, or infra config exists;
  local uvicorn only.
- **Duplicate HCP rows possible** after re-recording a submitted call
  (Veeva locks `Status_vod__c` post-submit) — see
  [veeva-integration.md](veeva-integration.md).

## Environment status

- **Veeva: sandbox only, no production integration plan yet.** Treat the
  current rep/product scaffolding as demo-only, not a base to
  incrementally generalize — a real integration will likely need to
  revisit rep identity, product sourcing, and token handling from scratch.

## Roadmap / next steps

### Mobile deployment (iPad / iOS)

The longer-term direction for this project is a native iPad app rather than
a browser-facing web app talking to a remote server. This is a substantial
architectural shift, not an incremental change — it collapses the current
browser ↔ WebSocket ↔ server split into a single on-device process, and
raises the project's privacy posture from a single tier (bounded chunked
exposure) to two:

- **Tier 1 (existing, unchanged in kind):** the current chunked-encoding
  approach — and the bookmarked VAD-triggered segment reset, see
  [asr-pipeline.md](asr-pipeline.md) — already bounds how much raw audio
  exists in memory at any moment. This property carries over to mobile
  unchanged; it just runs on-device instead of on a server.
- **Tier 2 (new):** lower-level memory hardening (mlock, disabled crash
  dumps, explicit zeroization) as a first line of defense underneath Tier
  1, since Tier 1's Python-level `del`/`gc.collect()` calls have no
  authority over what the OS itself does with that memory (crash dumps,
  swap, memory compression, background-state snapshots).

Key findings from early scoping, each a real constraint rather than a
detail to gloss over:

- **On-device inference alone is the single biggest privacy win available**
  — it removes network transmission of raw audio entirely, independent of
  any memory-hardening work layered on top.
- **iOS does not allow a third-party app to spawn its own isolated OS
  process.** Tier 2 has to be same-process isolation (a Rust/C module via
  FFI, or careful native memory handling) rather than true process
  isolation — a real, permanent weakening of the isolation guarantee
  compared to a desktop/Linux version of this design.
- **Hardware-accelerated inference (GPU/Apple Neural Engine via CoreML)
  is opaque to Tier 2.** Once inference runs on the ANE/GPU, there's no
  hook to mlock or zeroize those buffers — Apple's runtime owns that
  memory. CPU-only CoreML compute units keep memory reachable at a real
  performance cost. This is a genuine three-way tradeoff (PyTorch-as-is /
  CoreML CPU-only / CoreML GPU-ANE), not a solved question.
- **App-switcher screen-snapshot redaction is a separate, UI-level fix**
  (blank sensitive views before backgrounding) — orthogonal to memory
  hardening, still required regardless of how Tier 2 lands.
- **Validation plan:** deliberate crash/OOM/memory-pressure induction,
  then forensic inspection for recoverable audio artifacts — comparing a
  CoreML `.cpuOnly` build against `.all` (GPU/ANE) rather than assuming
  the answer. Likely needs a jailbroken test device or Xcode-level tooling
  (Memory Graph Debugger, Instruments), since a stock iOS device doesn't
  expose raw physical memory for inspection.

Not yet decided: overall timeline, whether to prototype the CoreML
conversion spike before or after committing to the VAD-based chunking
work, and how the Cortex/Veeva integration ports to a mobile HTTP client
(`URLSession`) and Keychain-based auth storage — noted here as an open
integration task, not a privacy question.

### Quality-of-life improvements

Two directions, each already anchored in an existing gap or existing
capability rather than being a net-new idea — split by risk tier, since
they carry very different compliance weight.

**Automatic task tracking.** `next_steps` comes back from `field-extractor`
today as a single free-text blob, and is already confirmed **UI-only — no
Veeva field exists to persist it** (`Call2_Detail_vod__c` is locked, same
limitation blocking products/competitor-mentions — see
[veeva-integration.md](veeva-integration.md)). To make this trackable:

- Extend `field-extractor`'s schema so `next_steps` returns structured
  items (action, rough timeframe, target) instead of prose — a tracked task
  can't be built out of an unstructured paragraph.
- Decide where tracked items actually live: check whether Veeva has a
  writable home for this on a *different* object (e.g. a native
  `Task_vod__c`/Activity object, if the org has one) before assuming the
  `Call2_Detail_vod__c` lockout applies there too — it may not. If Veeva
  genuinely has no path, an in-app follow-up list becomes the source of
  truth instead.
- Surface it as a "follow-ups due" view aggregating next-steps items across
  a rep's calls, reusing the call-list screen's existing pending/done
  visual language rather than inventing new UI.

**Document drafting.** Two features hiding under one label, worth keeping
distinct:

- *Internal* — a polished, exportable version of the call summary for the
  rep's own records (e.g. a PDF, alongside the review screen's existing
  "download audit log" button), and a **pre-call briefing** summarizing an
  HCP's prior call history (`Chat_Summary_vod__c` entries Veeva already
  holds) before a visit. Both reuse existing summarization/read
  capabilities and add no new outbound-content compliance surface.
- *Outbound* — auto-drafting content an HCP actually sees (a follow-up
  email, a leave-behind note). Materially higher risk: pharma HCP
  communication is typically constrained to pre-approved (MLR-reviewed)
  content, not freely LLM-generated text. If pursued, this needs its own
  compliance gate checking *generated outbound content* against an
  approved-content library — a different and larger scope than the
  existing `compliance-gate` agent, which only checks extracted call
  fields for PI/AECP violations. Not a small add-on to the existing gate.

### Everything else

- [ ] TODO
