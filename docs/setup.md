# Setup

## Prerequisites

- Python 3.13 (matches the checked-in `.venv`'s `pyvenv.cfg` / installed
  `pip3.13.exe`)
- A microphone-capable browser (Chrome/Edge recommended — uses
  `AudioWorklet` + `getUserMedia`)
- Access to: the Cortex endpoint this org uses, and a Veeva sandbox org
  with client-credentials OAuth configured (see
  [veeva-integration.md](veeva-integration.md))

## 1. Python environment

A `.venv/` already exists in the repo (gitignored). If setting up fresh:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
pip install -r requirements_server.txt
```

`requirements_server.txt` documents which packages are server-specific
(`fastapi`, `python-multipart`, `websockets`, `scipy`) vs. assumed
already present from the base environment (`uvicorn`, `torch`,
`transformers`, `numpy`, `soundfile`) — read its header comment if
`pip install` order/state ever gets confusing.

## 2. Download the ASR model

**Not checked into git** (`models/` is gitignored — model files are large).
Per the repo README:

> Download moonshine-streaming-medium model files here:
> https://huggingface.co/UsefulSensors/moonshine-streaming-medium/tree/main

Place the downloaded files under `models/moonshine-streaming-medium/` so
the layout matches what's already referenced in code:

```
models/moonshine-streaming-medium/
  config.json
  generation_config.json
  model.safetensors
  preprocessor_config.json
  processor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
```

(`models/moonshine-base/` — the older ONNX model used only by
`transcribe_base.py` — is a separate download if you need that path too;
not required for the main server.)

## 3. Environment variables

A `.env` file at the repo root is required (gitignored, not included here).
Variable names referenced across the codebase:

| Variable | Used by | Purpose |
|---|---|---|
| `CORTEX_BASE` | `cortexAgents.py`, `testAgents.py` | Base URL for the Cortex API |
| `VEEVA_CLIENT_ID` | `veeva_client.py`, `veeva_integration_test.py` | Veeva OAuth client-credentials |
| `VEEVA_CLIENT_SECRET` | `veeva_client.py`, `testAgents.py`, `veeva_integration_test.py` | Veeva OAuth client-credentials |
| `VEEVA_LOGIN_URL` | `veeva_client.py`, `testAgents.py`, `veeva_integration_test.py` | Veeva OAuth token endpoint |
| `EMAIL` | `testAgents.py` | Owner email for ad-hoc Cortex model creation |

Cortex auth itself (Azure AD) is handled internally by `LIGHTClient` — no
separate Cortex credential env vars are read directly in this repo's code;
`LIGHTClient` presumably sources its own from wherever it's configured (not
something this repo controls — see `.venv/Lib/site-packages/light_client`
if you need to trace that further).

## 4. Run the server

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Model load happens at import time (`server.py` module scope) — expect a
delay on first request/reload while `AutoModelForSpeechSeq2Seq` loads.
Watch stdout for `Loading model…` / `Model ready.`.

Then open `http://localhost:8000/` in a browser and grant microphone
permission when prompted.

## Notes

- No Dockerfile, CI config, or deployment manifests exist in this repo —
  running locally via uvicorn is the only supported path today.
- `--reload` will re-trigger the model load on every code change (it's
  module-level, not behind a lazy-load guard) — expect a multi-second pause
  after each save.
