# Pulse AI — Handoff Documentation

Technical handoff for the Pulse.AI POC: a browser-based call-note recorder for
pharma sales reps. Audio is transcribed locally, run through three Cortex
LLM agents (keyword correction → field extraction → compliance gate), and
clean submissions are written back to Veeva CRM.

Written for an engineer picking up this codebase — it assumes you can read
the code, and focuses on decisions, gotchas, and context that aren't visible
from the code alone (or that are visible but easy to miss).

**Status: POC / demo.** Single hardcoded rep, sandbox Veeva org, no auth on
the FastAPI app itself, no token caching, no production deployment target
yet. See [known-issues-and-roadmap.md](known-issues-and-roadmap.md).

## Contents

| Doc | Covers |
|---|---|
| [architecture.md](architecture.md) | End-to-end request flow, component map, where state lives |
| [api-reference.md](api-reference.md) | REST endpoints and the `/ws/transcribe` WebSocket protocol |
| [asr-pipeline.md](asr-pipeline.md) | Chunked streaming encoder, decode strategy, and the windowed-decoding postmortem |
| [cortex-agents.md](cortex-agents.md) | The three Cortex agents, response envelope gotchas, ownership |
| [veeva-integration.md](veeva-integration.md) | Auth, SOQL queries, field-writability limits, sandbox status |
| [setup.md](setup.md) | Local environment setup, model download, env vars, run command |
| [testing.md](testing.md) | What each test/benchmark script actually verifies and how to run it |
| [known-issues-and-roadmap.md](known-issues-and-roadmap.md) | Open questions, unverified assumptions, next steps |

## Fastest path to productive

1. Read [architecture.md](architecture.md) for the shape of the system.
2. Read [asr-pipeline.md](asr-pipeline.md)'s "windowed decoding" section — it's
   the single most valuable piece of tribal knowledge in this repo and it's
   easy to accidentally rediscover the hard way if you don't read it first.
3. Skim [setup.md](setup.md) and get the app running locally.
4. [cortex-agents.md](cortex-agents.md) and [veeva-integration.md](veeva-integration.md)
   before touching either integration.
