"""
complianceLayer.py  —  Pulse.AI Compliance Layer
=================================================
Receives the raw transcript from the Moonshine ephemeral audio pipeline
and sends it to Compliance Goblin V3 on Cortex for PI and AECP
redaction analysis.

Pipeline position:
    Moonshine transcript → fieldExtraction.py + summaryField.py → complianceLayer.py

Usage:
    from complianceLayer import ComplianceLayer
    compliance = ComplianceLayer()
    result = compliance.analyze(transcript_text)

Requirements:
    pip install light-client python-dotenv
"""

import json
import logging
from datetime import datetime
from light_client import LIGHTClient
import os
from dotenv import load_dotenv

load_dotenv()

client = LIGHTClient()
CORTEX_BASE_URL = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ComplianceLayer] %(message)s"
)
log = logging.getLogger(__name__)


class ComplianceLayer:
    """
    Sends raw transcript from Moonshine to Compliance Goblin V3 on Cortex.
    Runs last in the pipeline after Field Extraction and Summary Fairy.
    Returns compliance status and redacted transcript for audit records.
    """

    def __init__(self):
        self.chunks_analyzed = 0
        self.session_halted  = False
        self.session_start   = datetime.utcnow()
        log.info("ComplianceLayer initialized")


    def analyze(self, transcript: str) -> dict:
        """
        Sends raw transcript to Compliance Goblin V3 and returns a result dict.

        Runs after Field Extraction and Summary Fairy in the pipeline.
        The compliance result is used for audit records and DLO escalation.

        Parameters
        ----------
        transcript : str
            Raw decoded text from the Moonshine ephemeral audio pipeline.

        Returns
        -------
        dict
            compliance_status, dlo_escalation, cleaned_transcript,
            violations, recommended_action, raw_response
        """

        if self.session_halted:
            log.warning("Session halted — skipping compliance analysis")
            return self._halted_result()

        self.chunks_analyzed += 1
        chunk_id = f"CHUNK-{self.chunks_analyzed:04d}"

        log.info(f"{chunk_id}: Sending to Compliance Goblin V3")
        log.info(f"{chunk_id}: Preview: {transcript[:80]}...")

        raw_response = self._call_api(transcript, chunk_id)

        if raw_response is None:
            log.error(f"{chunk_id}: No response received from Compliance Goblin V3")
            return self._error_result(transcript, chunk_id)

        result = self._parse(raw_response, chunk_id)
        self._handle(chunk_id, result)
        return result


    def _call_api(self, transcript: str, chunk_id: str) -> str | None:
        """
        POSTs transcript to Compliance Goblin V3 using Lilly Light Client.
        Extracts message_fragment from streaming JSON response lines.
        Strips markdown code fences before returning.
        """

        log.info(f"{chunk_id}: POST {CORTEX_BASE_URL}/model/ask/compliance-goblin-v2")

        try:
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/compliance-goblin-v2",
                data={
                    "q":                transcript,
                    "stream":           "true",
                    "no_summary":       "false",
                    "background_job":   "false",
                    "workflow_timeout": "1",
                },
                headers=client.get_auth_header(),
            )

            response.raise_for_status()

            print(f"\n{'='*60}")
            print(f"COMPLIANCE GOBLIN V3 — {chunk_id}")
            print(f"{'='*60}")

            # Extract message_fragment from streaming JSON response lines
            message_content = ""
            for line in response.text.strip().split("\n"):
                try:
                    parsed = json.loads(line)
                    if parsed.get("type") == "message" and parsed.get("message_fragment"):
                        message_content += parsed["message_fragment"]
                except Exception:
                    pass

            # Strip markdown code fences if agent wraps JSON in ```json ... ```
            message_content = message_content.strip()
            if message_content.startswith("```"):
                message_content = message_content.split("```")[1]
                if message_content.startswith("json"):
                    message_content = message_content[4:]
            message_content = message_content.strip()

            print(message_content if message_content else response.text)
            print(f"{'='*60}\n")

            log.info(f"{chunk_id}: Response received from Compliance Goblin V3")
            return message_content if message_content else response.text

        except Exception as e:
            log.error(f"{chunk_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, chunk_id: str) -> dict:
        """
        Parses Compliance Goblin V3 JSON response into structured fields.
        Falls back to string parsing if JSON parsing fails.
        """

        try:
            data = json.loads(raw)
            return {
                "chunk_id":           chunk_id,
                "compliance_status":  data.get("compliance_status",      "WARNING"),
                "dlo_escalation":     data.get("dlo_escalation_required", False),
                "cleaned_transcript": data.get("cleaned_transcript",      ""),
                "violations":         data.get("violations_detected",     []),
                "recommended_action": data.get("recommended_action",      "REVIEW"),
                "raw_response":       raw,
                "timestamp":          datetime.utcnow().isoformat(),
            }

        except json.JSONDecodeError:
            log.warning(f"{chunk_id}: JSON parse failed — falling back to string parsing")
            upper = raw.upper()

            if "CRITICAL" in upper:   status = "CRITICAL"
            elif "WARNING" in upper:  status = "WARNING"
            elif "ADVISORY" in upper: status = "ADVISORY"
            elif "CLEAN" in upper:    status = "CLEAN"
            else:                     status = "WARNING"

            if "HALT" in upper:       action = "HALT"
            elif "REVIEW" in upper:   action = "REVIEW"
            else:                     action = "PROCEED"

            return {
                "chunk_id":           chunk_id,
                "compliance_status":  status,
                "dlo_escalation":     "DLO ESCALATION REQUIRED: YES" in upper,
                "cleaned_transcript": "",
                "violations":         [],
                "recommended_action": action,
                "raw_response":       raw,
                "timestamp":          datetime.utcnow().isoformat(),
            }


    def _handle(self, chunk_id: str, result: dict) -> None:
        """Updates session state and logs the compliance outcome."""

        status = result["compliance_status"]

        if status == "CLEAN":
            log.info(f"{chunk_id}: CLEAN — no violations detected")
        elif status == "ADVISORY":
            log.info(f"{chunk_id}: ADVISORY — {result['violations']}")
        elif status == "WARNING":
            log.warning(f"{chunk_id}: WARNING — {result['violations']}")
        elif status == "CRITICAL":
            log.error(f"{chunk_id}: CRITICAL — session halted")
            log.error(f"{chunk_id}: Violations: {result['violations']}")
            self.session_halted = True
            if result["dlo_escalation"]:
                log.error(f"{chunk_id}: DLO ESCALATION REQUIRED")


    def get_session_summary(self) -> dict:
        """Returns a summary of all compliance activity this session."""
        duration = (datetime.utcnow() - self.session_start).seconds
        return {
            "chunks_analyzed":    self.chunks_analyzed,
            "session_halted":     self.session_halted,
            "session_duration_s": duration,
        }


    def _error_result(self, transcript: str, chunk_id: str) -> dict:
        """Returned when the Cortex API call fails — defaults to WARNING."""
        return {
            "chunk_id":           chunk_id,
            "compliance_status":  "WARNING",
            "dlo_escalation":     False,
            "cleaned_transcript": transcript,
            "violations":         ["Cortex API call failed — manual review required"],
            "recommended_action": "REVIEW",
            "raw_response":       "",
            "timestamp":          datetime.utcnow().isoformat(),
        }

    def _halted_result(self) -> dict:
        """Returned when a prior CRITICAL violation halted the session."""
        return {
            "chunk_id":           "HALTED",
            "compliance_status":  "CRITICAL",
            "dlo_escalation":     False,
            "cleaned_transcript": "",
            "violations":         ["Session halted by prior CRITICAL violation"],
            "recommended_action": "HALT",
            "raw_response":       "",
            "timestamp":          datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Compliance Layer Standalone Test ===\n")

    if len(sys.argv) > 1:
        transcript_text = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
        log.info(f"Loaded: {sys.argv[1]}")
    else:
        transcript_text = (
            "Visited the office today and discussed ALIMTA for Malignant "
            "Pleural Mesothelioma with Dr. Sarah Mitchell. She mentioned that "
            "her patient James Cooper, a 67 year old male with stage 3 "
            "mesothelioma, has been on ALIMTA for three cycles. His most recent "
            "CT scan showed stable disease. Dr. Mitchell wants additional patient "
            "support resources for this patient."
        )
        log.info("Using built-in PI-Red test transcript")

    compliance = ComplianceLayer()
    result = compliance.analyze(transcript_text)

    print("\n=== Compliance Result ===")
    print(f"Status:             {result['compliance_status']}")
    print(f"DLO Escalation:     {result['dlo_escalation']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print(f"Violations:         {len(result['violations'])}")
    for v in result["violations"]:
        print(f"  - {v}")
    print(f"\nCleaned Transcript:\n{result['cleaned_transcript'] or '(none returned)'}")