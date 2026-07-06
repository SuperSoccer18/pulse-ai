"""
complianceLayer.py  —  Pulse.AI Compliance Layer
=================================================
Receives transcript text from the audio transcription pipeline
and sends it to Compliance Goblin V3 on Cortex for PI and AECP
redaction analysis.

Pipeline position:
    Audio transcript → complianceLayer.py → fieldExtraction.py

Usage:
    from complianceLayer import ComplianceLayer
    compliance = ComplianceLayer()
    result = compliance.analyze(transcript_text)

Requirements:
    pip install light-client python-dotenv
"""

# logging prints timestamped status messages for each pipeline step
import logging

# datetime records when compliance events happen during the session
from datetime import datetime

# LIGHTClient handles Lilly authentication automatically
from light_client import LIGHTClient

# os and dotenv load environment variables from the .env file
import os
from dotenv import load_dotenv

# Load .env file so CORTEX_BASE_URL and EMAIL are available
load_dotenv()

# Initialize the Light Client — handles Lilly auth automatically
client = LIGHTClient()

# CORTEX_BASE_URL loaded from .env — falls back to dev environment
CORTEX_BASE_URL = os.getenv("CORTEX_BASE_URL", "https://api.dev.cortex.lilly.com")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ComplianceLayer] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ComplianceLayer class
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceLayer:
    """
    Sends transcript text to Compliance Goblin V3 on Cortex.
    Returns a cleaned redacted transcript and compliance status
    for downstream processing by fieldExtraction.py.
    """

    def __init__(self):
        # chunks_analyzed counts how many transcripts have been sent this session
        self.chunks_analyzed = 0

        # session_halted becomes True when CRITICAL is detected
        # No further transcripts are sent after a session halt
        self.session_halted = False

        # cleaned_chunks stores the redacted transcript from each analysis
        # Joined at the end to form the complete safe transcript for CRM
        self.cleaned_chunks = []

        # session_start records when this ComplianceLayer was initialized
        self.session_start = datetime.utcnow()

        log.info("ComplianceLayer initialized")


    def analyze(self, transcript: str) -> dict:
        """
        Sends transcript to Compliance Goblin V3 and returns a result dict.

        Called by the transcription pipeline after Moonshine decodes audio.
        The cleaned_transcript in the result is passed to fieldExtraction.py.

        Parameters
        ----------
        transcript : str
            Decoded text from the Moonshine pipeline — no raw audio

        Returns
        -------
        dict
            compliance_status, dlo_escalation, cleaned_transcript,
            violations, recommended_action, raw_response
        """

        # Skip analysis if a prior CRITICAL halted the session
        if self.session_halted:
            log.warning("Session halted — skipping compliance analysis")
            return self._halted_result()

        # Increment chunk counter and build zero-padded chunk ID
        self.chunks_analyzed += 1
        chunk_id = f"CHUNK-{self.chunks_analyzed:04d}"

        log.info(f"{chunk_id}: Sending to Compliance Goblin V3")
        log.info(f"{chunk_id}: Preview: {transcript[:80]}...")

        # POST transcript to Compliance Goblin V3 via Light Client
        raw_response = self._call_api(transcript, chunk_id)

        # Return error result if API call failed
        if raw_response is None:
            log.error(f"{chunk_id}: No response received from Compliance Goblin V3")
            return self._error_result(transcript, chunk_id)

        # Parse the response into structured compliance fields
        result = self._parse(raw_response, chunk_id)

        # Update session state and log the outcome
        self._handle(chunk_id, result)

        return result


    def _call_api(self, transcript: str, chunk_id: str) -> str | None:
        """
        POSTs transcript to Compliance Goblin V3 using Lilly Light Client.
        Light Client handles all Lilly authentication automatically.
        """

        log.info(f"{chunk_id}: POST {CORTEX_BASE_URL}/model/ask/compliance-goblin-v3")

        try:
            # client.post uses Light Client auth — no Bearer token needed
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/compliance-goblin-v3",
                data={"q": transcript},
            )

            # Raise exception for 4xx/5xx responses
            response.raise_for_status()

            # Print streamed output so rep sees compliance analysis in real time
            print(f"\n{'='*60}")
            print(f"COMPLIANCE GOBLIN V3 — {chunk_id}")
            print(f"{'='*60}")
            print(response.text)
            print(f"{'='*60}\n")

            log.info(f"{chunk_id}: Response received from Compliance Goblin V3")

            # Return full response text for parsing
            return response.text

        except Exception as e:
            log.error(f"{chunk_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, chunk_id: str) -> dict:
        """
        Parses raw Compliance Goblin V3 response text into structured fields.
        Extracts compliance_status, dlo_escalation, cleaned_transcript,
        violations, and recommended_action.
        """

        upper = raw.upper()

        # Parse compliance status — check in severity order so CRITICAL wins
        if "CRITICAL" in upper:
            status = "CRITICAL"
        elif "WARNING" in upper:
            status = "WARNING"
        elif "ADVISORY" in upper:
            status = "ADVISORY"
        elif "CLEAN" in upper:
            status = "CLEAN"
        else:
            # Default to WARNING if format is unexpected — never silently pass
            status = "WARNING"
            log.warning(f"{chunk_id}: Could not parse status — defaulting to WARNING")

        # Parse DLO escalation — True if YES appears after the label
        dlo = (
            "DLO ESCALATION REQUIRED: YES" in upper or
            'DLO ESCALATION REQUIRED: "YES"' in upper
        )

        # Parse recommended action — most restrictive wins
        if "HALT" in upper:
            action = "HALT"
        elif "REVIEW" in upper:
            action = "REVIEW"
        else:
            action = "PROCEED"

        # Extract cleaned transcript between section labels
        cleaned = ""
        if "CLEANED TRANSCRIPT:" in raw:
            start = raw.find("CLEANED TRANSCRIPT:") + len("CLEANED TRANSCRIPT:")
            end   = raw.find("REDACTION SUMMARY:")
            if end > start:
                cleaned = raw[start:end].strip().strip('"')

        # Extract violation lines — each uses pipe separator
        violations = []
        if "VIOLATIONS DETECTED:" in raw:
            start = raw.find("VIOLATIONS DETECTED:") + len("VIOLATIONS DETECTED:")
            end   = raw.find("RECOMMENDED ACTION:")
            if end > start:
                for vline in raw[start:end].strip().split("\n"):
                    vline = vline.strip()
                    if vline and vline != "None" and "|" in vline:
                        violations.append(vline.lstrip("- ").strip())

        return {
            "chunk_id":           chunk_id,
            "compliance_status":  status,
            "dlo_escalation":     dlo,
            "cleaned_transcript": cleaned,
            "violations":         violations,
            "recommended_action": action,
            "raw_response":       raw,
            "timestamp":          datetime.utcnow().isoformat(),
        }


    def _handle(self, chunk_id: str, result: dict) -> None:
        """Updates session state and logs the compliance outcome."""

        status = result["compliance_status"]

        # Store cleaned transcript chunk for later session summary
        if result["cleaned_transcript"]:
            self.cleaned_chunks.append(result["cleaned_transcript"])

        if status == "CLEAN":
            log.info(f"{chunk_id}: CLEAN — safe for field extraction")
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
            "cleaned_transcript": " ".join(self.cleaned_chunks),
        }


    # ── Fallback result constructors ──────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Compliance Layer Standalone Test ===\n")

    # Load transcript from file argument or use built-in PI-Red test
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