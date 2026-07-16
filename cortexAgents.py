"""
cortexAgents.py — Cortex agent calls for the post-transcription pipeline
==========================================================================
Three Cortex agents, called via LIGHTClient (handles Azure AD auth internally
— see .venv/Lib/site-packages/light_client). Each agent's system prompt
(with its own {question} template slot) lives server-side on Cortex; we
only ever POST the raw question text as `q`. No prompts are embedded here.

Pipeline position:
    Moonshine transcript → correct_transcript() → extract_call_fields()
        → review UI (rep edits) → check_compliance() → Veeva write

Agents:
  lilly-keyword-recognizer — fixes STT mishearings of Lilly drug/medical terms
  field-extractor          — extracts structured Veeva call-report fields
  compliance-gate          — submit-time PI/AECP gate on the structured fields

Note: the JSON envelope Cortex wraps the agent's answer in is confirmed —
see _ask_cortex()'s docstring for the exact shape and the two gotchas found
in live testing (message-key nesting, markdown code fences).
"""

import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from light_client import LIGHTClient

load_dotenv()

CORTEX_BASE = os.getenv("CORTEX_BASE")

# Module-level singleton — LIGHTClient caches/reuses the authenticated
# session internally, so we don't want to re-auth on every call.
_client = LIGHTClient()


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing ```json or ``` markdown fence, if present.
    Seen on field-extractor's output but not lilly-keyword-recognizer's in
    the same session — model/prompt-dependent, so always safe to try."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    return stripped


def _ask_cortex(model_name: str, question: str) -> dict:
    """POST to a Cortex agent and parse its JSON answer.

    timeout is required — without it a hung/unreachable Cortex endpoint
    blocks forever on server.py's single-worker thread pool, which also
    starves every unrelated job (encoding, decoding) queued behind it.

    Confirmed response envelope (2026-07-14 live run): the agent's actual
    JSON answer is a STRING under the "message" key, alongside unrelated
    metadata (source_metadata, steps, llm_model, ...). E.g.:
        {"message": "{\"corrected_transcript\": ...}", "steps": [...], ...}
    "message" must be checked before the raw body itself — raw is already a
    dict, so an isinstance(dict) check against it would match immediately
    and silently return the metadata envelope instead of the real payload.

    Also confirmed: "message" is sometimes wrapped in a markdown ```json
    fence (model-dependent — seen on field-extractor, not on
    lilly-keyword-recognizer in the same session). Strip fences before
    parsing, or json.loads() throws and silently falls through to the
    envelope-returning fallback below with no visible error.
    """
    resp = _client.post(f"{CORTEX_BASE}/model/ask/{model_name}", data={"q": question}, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    logging.debug(f"{model_name}: raw Cortex response = {raw!r}")

    if isinstance(raw, dict) and isinstance(raw.get("message"), str):
        try:
            return json.loads(_strip_code_fence(raw["message"]))
        except json.JSONDecodeError:
            pass

    # Fallback for other possible envelope shapes, in case a different
    # agent/route wraps its answer differently than the one confirmed above.
    candidates = [raw.get(k) for k in ("response", "answer", "output", "result")] if isinstance(raw, dict) else []
    candidates.append(raw)
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str):
            try:
                return json.loads(_strip_code_fence(candidate))
            except (json.JSONDecodeError, TypeError):
                continue
    raise ValueError(f"Could not parse Cortex response shape from {model_name}: {raw!r}")


def correct_transcript(transcript: str) -> dict:
    """
    Runs lilly-keyword-recognizer on the raw transcript.

    Returns
    -------
    dict
        {"corrected_transcript", "changes", "flagged_uncertain"} on success.
        Falls back to the original transcript (pipeline degrades, not crashes)
        with an "error" key set on failure.
    """
    try:
        return _ask_cortex("lilly-keyword-recognizer", transcript)
    except Exception as e:
        logging.error(f"lilly-keyword-recognizer failed: {e}")
        return {
            "corrected_transcript": transcript,
            "changes": [],
            "flagged_uncertain": [],
            "error": str(e),
        }


def extract_call_fields(corrected_transcript: str) -> dict:
    """
    Runs field-extractor on the corrected transcript.

    The agent has no inherent notion of "today" — it only sees whatever text
    we send as `q`. We prepend a CURRENT DATE AND TIME line (server's own
    clock, at the moment the rep stopped recording) so the agent's
    CALL_DATETIME RESOLUTION logic can turn relative phrases like "this
    morning around 10 a.m." into a real ISO 8601 datetime instead of either
    parroting the phrase back or giving up with null.

    Returns
    -------
    dict
        call_metadata/summary/products/next_steps/competitor_mentions/confidence
        on success. Falls back to an all-null skeleton on failure, with an
        "error" key set.
    """
    now_iso = datetime.now().isoformat(timespec="seconds")
    question = f"CURRENT DATE AND TIME: {now_iso}\nTRANSCRIPT:\n{corrected_transcript}"
    try:
        return _ask_cortex("field-extractor", question)
    except Exception as e:
        logging.error(f"field-extractor failed: {e}")
        return {
            "call_metadata": {
                "call_datetime": None,
                "engagement_method": "Face_to_face_vod",
                "virtual_engagement_tool": None,
                "location": None,
            },
            "summary": "",
            "products": [],
            "next_steps": None,
            "competitor_mentions": [],
            "confidence": 0.0,
            "error": str(e),
        }


def check_compliance(extraction: dict) -> dict:
    """
    Runs compliance-gate on the (possibly rep-edited) extraction fields.

    Evaluates the structured fields, not the raw transcript — transcript-level
    PI/AECP scanning is a separate, not-yet-built concern (see
    complianceLayer.py, left untouched as that future seam). This is the
    submit-time gate: it runs after the rep has reviewed/filled in fields,
    right before the Veeva write.

    Returns
    -------
    dict
        {"overall_status", "field_violations", "violation_summary",
        "recommended_action"} on success. On failure, FAILS CLOSED — returns
        NON_COMPLIANT/HOLD rather than a permissive default, since a broken
        compliance check must never be silently equivalent to "passed". An
        "error" key distinguishes this from a genuine compliance hold.
    """
    question = json.dumps(extraction)
    try:
        return _ask_cortex("compliance-gate", question)
    except Exception as e:
        logging.error(f"compliance-gate failed: {e}")
        return {
            "overall_status": "NON_COMPLIANT",
            "field_violations": [],
            "violation_summary": {"total_violations": 0, "pi_violations": 0, "aecp_violations": 0},
            "recommended_action": "HOLD",
            "error": str(e),
        }
