"""
complianceLayer.py  —  Pulse.AI Compliance Layer
=================================================
Receives transcript text from the Moonshine transcription pipeline
and sends it to Compliance Goblin V3 on Cortex for PI and AECP
redaction analysis.

Pipeline position:
    Moonshine transcript → complianceLayer.py → fieldExtraction.py

Usage:
    from complianceLayer import ComplianceLayer
    compliance = ComplianceLayer()
    result = compliance.analyze(transcript_text)
"""

# requests is the HTTP library used to call the Cortex streaming API
import requests

# logging prints timestamped status messages for each pipeline step
import logging

# datetime records when compliance events happen during the session
from datetime import datetime

# sys lets us exit with a non-zero code on critical startup failures
import sys

# get_bearer_token obtains the Azure AD OAuth2 Bearer token for Cortex
# It uses Tenant ID, Client ID, and Secret Value from cortexAuthenticator.py
from cortexAuthenticator import get_bearer_token

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ComplianceLayer] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cortex API Configuration — Compliance Goblin V3
# ─────────────────────────────────────────────────────────────────────────────

# CORTEX_BASE_URL is the root URL for Lilly's Cortex platform
# All agent endpoints are appended to this base
CORTEX_BASE_URL = "https://gateway-intranet.apim-dev.lilly.com"

# COMPLIANCE_GOBLIN_ENDPOINT is the path for the Compliance Goblin V3 agent
# POST requests here ask the agent to analyze a transcript for PI and AECP
COMPLIANCE_GOBLIN_ENDPOINT = "/api/model/ask/compliance-goblin-v3"

# COMPLIANCE_GOBLIN_URL is the full URL used in every POST request
COMPLIANCE_GOBLIN_URL = CORTEX_BASE_URL + COMPLIANCE_GOBLIN_ENDPOINT

# CORTEX_PARAMS are the query string parameters for the Compliance Goblin API
# stream: "true"           — stream the response line by line for real-time output
# no_summary: "false"      — include the agent summary in the response
# background_job: "false"  — process immediately, not as an async background job
# workflow_timeout: "1"    — allow up to 1 minute for the agent to complete
CORTEX_PARAMS = {
    "stream": "true",
    "no_summary": "false",
    "background_job": "false",
    "workflow_timeout": "1",
}

# REQUEST_TIMEOUT_SECONDS is how long we wait for Cortex to start responding
# before raising a Timeout error — set higher than workflow_timeout
REQUEST_TIMEOUT_SECONDS = 90

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

        # POST transcript to Compliance Goblin V3 via streaming API
        raw_response = self._call_api(transcript, chunk_id)

        # Return error result if API call failed
        if raw_response is None:
            log.error(f"{chunk_id}: No response received from Compliance Goblin V3")
            return self._error_result(transcript, chunk_id)

        # Parse the streaming response into structured compliance fields
        result = self._parse(raw_response, chunk_id)

        # Update session state and log the outcome
        self._handle(chunk_id, result)

        return result

    def _call_api(self, transcript: str, chunk_id: str) -> str | None:
        """
        POSTs transcript to Compliance Goblin V3 using the Cortex streaming API.

        Uses the exact API pattern from the Pulse.AI specification:
            POST /api/model/ask/compliance-goblin-v3
            params: stream=true, no_summary=false,
                    background_job=false, workflow_timeout=1
            form_data: q=<transcript>
            headers: accept: application/json, Authorization: Bearer <token>

        Streams response line by line and returns full accumulated text.
        """

        # Build form data — "q" is the field name Compliance Goblin V3 expects
        form_data = {"q": transcript}

        # Obtain a valid Bearer token from Azure AD via cortex_auth.py
        # get_bearer_token() caches the token and auto-refreshes before expiry
        try:
            bearer_token = get_bearer_token()
        except Exception as e:
            log.error(f"{chunk_id}: Failed to obtain Bearer token — {e}")
            return None

        # Build the Authorization header using the Bearer token
        # This is the token obtained by exchanging Tenant ID, Client ID,
        # and Secret Value with Azure AD — NOT the Secret ID
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        }

        log.info(f"{chunk_id}: POST {COMPLIANCE_GOBLIN_URL}")

        try:
            # Send streaming POST request to Compliance Goblin V3
            # stream=True tells requests to read the body line by line
            # rather than downloading the entire response at once
            response = requests.post(
                COMPLIANCE_GOBLIN_URL,       # Compliance Goblin V3 endpoint
                params=CORTEX_PARAMS,        # stream=true, workflow_timeout=1 etc.
                data=form_data,              # transcript in form field q
                headers=headers,             # accept + Bearer token
                stream=True,                 # enable line-by-line streaming
                timeout=REQUEST_TIMEOUT_SECONDS,  # abort if no response in 90s
            )

            # Raise an exception for 4xx/5xx HTTP errors (e.g. 401 Unauthorized)
            response.raise_for_status()

            log.info(f"{chunk_id}: Streaming Compliance Goblin V3 response")

            # Print visual separator so streamed output is easy to read
            print(f"\n{'='*60}")
            print(f"COMPLIANCE GOBLIN V3 — {chunk_id}")
            print(f"{'='*60}")

            # accumulated_lines stores each decoded line as it arrives
            accumulated_lines = []

            # iter_lines() reads the streaming response body one line at a time
            # Each line arrives as bytes — decoded to UTF-8 string
            for line in response.iter_lines():
                if line:
                    # Decode bytes to string for display and accumulation
                    decoded = line.decode("utf-8")
                    print(decoded)               # real-time display
                    accumulated_lines.append(decoded)

            print(f"{'='*60}\n")

            log.info(f"{chunk_id}: Received {len(accumulated_lines)} lines")

            # Join all lines into one complete response string for parsing
            return "\n".join(accumulated_lines)

        except requests.exceptions.ConnectionError as e:
            log.error(f"{chunk_id}: Connection error — check VPN/network: {e}")
            return None
        except requests.exceptions.Timeout:
            log.error(f"{chunk_id}: Timed out after {REQUEST_TIMEOUT_SECONDS}s")
            return None
        except requests.exceptions.HTTPError as e:
            log.error(f"{chunk_id}: HTTP {response.status_code} — {e}")
            if response.status_code == 401:
                log.error(f"{chunk_id}: 401 Unauthorized — check Azure AD credentials")
            return None
        except requests.exceptions.RequestException as e:
            log.error(f"{chunk_id}: Request error — {e}")
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
            status = "WARNING"    # default to WARNING if format is unexpected
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
            "cleaned_transcript": transcript,     # return original unredacted
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

