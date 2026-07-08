"""
summaryField.py  —  Pulse.AI Summary Layer
==========================================
Receives the cleaned transcript from complianceLayer.py and
sends it to Summary Fairy on Cortex to generate a professional
call summary for internal records and coaching.

Pipeline position:
    complianceLayer.py → summaryField.py
    (runs in parallel with fieldExtraction.py)

Usage:
    from summaryField import SummaryField
    summarizer = SummaryField()
    result = summarizer.summarize(cleaned_transcript)

Requirements:
    pip install light-client python-dotenv
"""

# logging prints timestamped status messages for each pipeline step
import logging

# datetime records when summary events happen during the session
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
CORTEX_BASE_URL = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SummaryField] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SummaryField class
# ─────────────────────────────────────────────────────────────────────────────

class SummaryField:
    """
    Sends the cleaned transcript from Compliance Goblin to Summary Fairy
    on Cortex and returns a structured call summary for internal records
    and rep coaching.

    Receives the cleaned_transcript from complianceLayer.py result dict.
    Runs in parallel with fieldExtraction.py in the full pipeline.
    """

    def __init__(self):
        # summaries_run counts how many summaries have been generated
        self.summaries_run = 0

        # session_start records when this SummaryField was initialized
        self.session_start = datetime.utcnow()

        log.info("SummaryField initialized")


    def summarize(self, cleaned_transcript: str) -> dict:
        """
        Sends the cleaned transcript to Summary Fairy and returns a
        structured call summary dict.

        Parameters
        ----------
        cleaned_transcript : str
            The redacted transcript from complianceLayer.py result dict.
            All PI and AECP content has already been removed by Compliance Goblin.

        Returns
        -------
        dict
            call_overview, products_discussed, key_topics, hcp_response,
            objections_raised, next_steps, rep_recommendations, raw_response
        """

        # Validate that a cleaned transcript was provided
        if not cleaned_transcript or not cleaned_transcript.strip():
            log.error("Empty or null cleaned transcript received — skipping summary")
            return self._empty_result()

        # Increment summary counter and build zero-padded summary ID
        self.summaries_run += 1
        summary_id = f"SUMMARY-{self.summaries_run:04d}"

        log.info(f"{summary_id}: Sending cleaned transcript to Summary Fairy")
        log.info(f"{summary_id}: Preview: {cleaned_transcript[:80]}...")

        # POST cleaned transcript to Summary Fairy via Light Client
        raw_response = self._call_api(cleaned_transcript, summary_id)

        # Return empty result if API call failed
        if raw_response is None:
            log.error(f"{summary_id}: No response from Summary Fairy")
            return self._error_result(summary_id)

        # Parse the response into structured summary fields
        result = self._parse(raw_response, summary_id)

        log.info(f"{summary_id}: Summary generated successfully")

        return result


    def _call_api(self, cleaned_transcript: str, summary_id: str) -> str | None:
        """
        POSTs the cleaned transcript to Summary Fairy using Lilly Light Client.
        get_auth_header() provides the personal Lilly token accepted by Cortex.
        """

        log.info(f"{summary_id}: POST {CORTEX_BASE_URL}/model/ask/summary-fairy")

        try:
            # get_auth_header() returns the personal Lilly Bearer token
            # This is the token method confirmed working with Cortex
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/summary-fairy",
                data={
                    "q":                cleaned_transcript,
                    "stream":           "true",
                    "no_summary":       "false",
                    "background_job":   "false",
                    "workflow_timeout": "1",
                },
                headers=client.get_auth_header(),
            )

            # Raise exception for 4xx/5xx responses
            response.raise_for_status()

            # Print response so rep sees summary in real time
            print(f"\n{'='*60}")
            print(f"SUMMARY FAIRY — {summary_id}")
            print(f"{'='*60}")
            print(response.text)
            print(f"{'='*60}\n")

            log.info(f"{summary_id}: Response received from Summary Fairy")

            # Return full response text for parsing
            return response.text

        except Exception as e:
            log.error(f"{summary_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, summary_id: str) -> dict:
        """
        Parses the raw Summary Fairy response into structured summary fields.

        Expected response format:
            CALL OVERVIEW: <one sentence description>
            PRODUCTS DISCUSSED: <product list>
            KEY TOPICS: <main discussion points>
            HCP RESPONSE: <receptivity and interest level>
            OBJECTIONS RAISED: <concerns or pushback, or N/A>
            NEXT STEPS: <agreed follow up actions and timing>
            REP RECOMMENDATIONS: <suggested actions for the rep>
        """

        # Helper to extract a multi-line field value after a section label
        def extract_field(label: str) -> str:
            if label not in raw:
                return "N/A"
            start = raw.find(label) + len(label)
            remaining = raw[start:]
            lines = remaining.split("\n")
            collected = []
            known_labels = [
                "CALL OVERVIEW:", "PRODUCTS DISCUSSED:", "KEY TOPICS:",
                "HCP RESPONSE:", "OBJECTIONS RAISED:", "NEXT STEPS:",
                "REP RECOMMENDATIONS:",
            ]
            for line in lines:
                stripped = line.strip()
                # Stop when we hit the next field label
                if any(stripped.startswith(lbl) for lbl in known_labels if lbl != label):
                    break
                collected.append(stripped)
            value = " ".join(c for c in collected if c).strip()
            return value if value else "N/A"

        # Extract each summary field from the raw response
        call_overview       = extract_field("CALL OVERVIEW:")
        products_discussed  = extract_field("PRODUCTS DISCUSSED:")
        key_topics          = extract_field("KEY TOPICS:")
        hcp_response        = extract_field("HCP RESPONSE:")
        objections_raised   = extract_field("OBJECTIONS RAISED:")
        next_steps          = extract_field("NEXT STEPS:")
        rep_recommendations = extract_field("REP RECOMMENDATIONS:")

        return {
            "summary_id":          summary_id,
            "call_overview":       call_overview,
            "products_discussed":  products_discussed,
            "key_topics":          key_topics,
            "hcp_response":        hcp_response,
            "objections_raised":   objections_raised,
            "next_steps":          next_steps,
            "rep_recommendations": rep_recommendations,
            "raw_response":        raw,
            "timestamp":           datetime.utcnow().isoformat(),
        }


    # ── Fallback result constructors ──────────────────────────────────────────

    def _empty_result(self) -> dict:
        """Returned when cleaned transcript is empty or None."""
        return {
            "summary_id":          "EMPTY",
            "call_overview":       "N/A",
            "products_discussed":  "N/A",
            "key_topics":          "N/A",
            "hcp_response":        "N/A",
            "objections_raised":   "N/A",
            "next_steps":          "N/A",
            "rep_recommendations": "N/A",
            "raw_response":        "",
            "timestamp":           datetime.utcnow().isoformat(),
        }

    def _error_result(self, summary_id: str) -> dict:
        """Returned when the Cortex API call fails."""
        result = self._empty_result()
        result["summary_id"]    = summary_id
        result["call_overview"] = "Summary generation failed — manual summary required"
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Summary Field Standalone Test ===\n")

    if len(sys.argv) > 1:
        cleaned_text = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
        log.info(f"Loaded: {sys.argv[1]}")
    else:
        cleaned_text = (
            "Visited Chicago Oncology Associates at 676 North St. Clair Street "
            "Suite 1200 Chicago IL 60611 on June 17 2026 at 10:30 AM. "
            "Discussed VERZENIO for HR+/HER2- mBC in combination with an "
            "aromatase inhibitor. Reviewed MONARCH 3 efficacy data including "
            "progression-free survival and overall survival benefit data. "
            "Covered full fair balance on diarrhea, neutropenia, and fatigue. "
            "Physician showed strong interest in prescribing for appropriate "
            "patients and asked for additional patient case studies. Prior "
            "authorization requirements raised as a barrier for some patients. "
            "Agreed to follow up in two weeks with patient support materials."
        )
        log.info("Using built-in TC-FEG-01 clean test transcript")

    summarizer = SummaryField()
    result = summarizer.summarize(cleaned_text)

    print("\n=== Summary Result ===")
    print(f"Call Overview:       {result['call_overview']}")
    print(f"Products Discussed:  {result['products_discussed']}")
    print(f"Key Topics:          {result['key_topics']}")
    print(f"HCP Response:        {result['hcp_response']}")
    print(f"Objections Raised:   {result['objections_raised']}")
    print(f"Next Steps:          {result['next_steps']}")
    print(f"Rep Recommendations: {result['rep_recommendations']}")