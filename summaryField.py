"""
summaryField.py  —  Pulse.AI Summary Layer
==========================================
Receives the raw transcript from the Moonshine ephemeral audio pipeline
and sends it to Summary Fairy on Cortex to generate a professional
call summary for internal records and coaching.

Pipeline position:
    Moonshine transcript → fieldExtraction.py → summaryField.py → complianceLayer.py

Usage:
    from summaryField import SummaryField
    summarizer = SummaryField()
    result = summarizer.summarize(transcript)

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
    format="%(asctime)s [SummaryField] %(message)s"
)
log = logging.getLogger(__name__)


class SummaryField:
    """
    Sends the raw transcript from Moonshine to Summary Fairy on Cortex.
    Runs second in the agent pipeline after Field Extraction
    and before Compliance Goblin.
    Returns a structured call summary for internal records and coaching.
    """

    def __init__(self):
        self.summaries_run = 0
        self.session_start = datetime.utcnow()
        log.info("SummaryField initialized")


    def summarize(self, transcript: str) -> dict:
        """
        Sends raw transcript to Summary Fairy and returns structured call summary.

        Receives the raw decoded transcript from the Moonshine ephemeral audio
        pipeline. Runs after Field Extraction and before Compliance Goblin.

        Parameters
        ----------
        transcript : str
            Raw decoded text from the Moonshine ephemeral audio pipeline.

        Returns
        -------
        dict
            call_overview, products_discussed, key_topics, hcp_response,
            objections_raised, next_steps, rep_recommendations, raw_response
        """

        if not transcript or not transcript.strip():
            log.error("Empty or null transcript received — skipping summary")
            return self._empty_result()

        self.summaries_run += 1
        summary_id = f"SUMMARY-{self.summaries_run:04d}"

        log.info(f"{summary_id}: Sending to Summary Fairy")
        log.info(f"{summary_id}: Preview: {transcript[:80]}...")

        raw_response = self._call_api(transcript, summary_id)

        if raw_response is None:
            log.error(f"{summary_id}: No response from Summary Fairy")
            return self._error_result(summary_id)

        result = self._parse(raw_response, summary_id)
        log.info(f"{summary_id}: Summary generated successfully")
        return result


    def _call_api(self, transcript: str, summary_id: str) -> str | None:
        """
        POSTs transcript to Summary Fairy using Lilly Light Client.
        Extracts message_fragment from streaming JSON response lines.
        Strips markdown code fences before returning.
        """

        log.info(f"{summary_id}: POST {CORTEX_BASE_URL}/model/ask/summary-fairy")

        try:
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/summary-fairy",
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
            print(f"SUMMARY FAIRY — {summary_id}")
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

            log.info(f"{summary_id}: Response received from Summary Fairy")
            return message_content if message_content else response.text

        except Exception as e:
            log.error(f"{summary_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, summary_id: str) -> dict:
        """
        Parses Summary Fairy JSON response into structured summary fields.
        Falls back to empty result if JSON parsing fails.
        """

        try:
            data = json.loads(raw)
            return {
                "summary_id":          summary_id,
                "call_overview":       data.get("call_overview",       "N/A"),
                "products_discussed":  data.get("products_discussed",  "N/A"),
                "key_topics":          data.get("key_topics",          "N/A"),
                "hcp_response":        data.get("hcp_response",        "N/A"),
                "objections_raised":   data.get("objections_raised",   "N/A"),
                "next_steps":          data.get("next_steps",          "N/A"),
                "rep_recommendations": data.get("rep_recommendations", "N/A"),
                "raw_response":        raw,
                "timestamp":           datetime.utcnow().isoformat(),
            }

        except json.JSONDecodeError:
            log.warning(f"{summary_id}: JSON parse failed — returning empty result")
            return self._error_result(summary_id)


    def _empty_result(self) -> dict:
        """Returned when transcript is empty or None."""
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
        """Returned when the Cortex API call or JSON parsing fails."""
        result = self._empty_result()
        result["summary_id"]    = summary_id
        result["call_overview"] = "Summary generation failed — manual summary required"
        return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Summary Field Standalone Test ===\n")

    if len(sys.argv) > 1:
        transcript = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
        log.info(f"Loaded: {sys.argv[1]}")
    else:
        transcript = (
            "Visited Chicago Oncology Associates at 676 North St. Clair Street "
            "Suite 1200 Chicago IL 60611 on June 17 2026 at 10:30 AM. "
            "Discussed VERZENIO for HR+/HER2- mBC in combination with an "
            "aromatase inhibitor. Reviewed MONARCH 3 efficacy data. "
            "Physician showed strong interest in prescribing for appropriate "
            "patients. Prior authorization noted as a barrier. "
            "Agreed to follow up in two weeks with patient support materials."
        )
        log.info("Using built-in TC-FEG-01 test transcript")

    summarizer = SummaryField()
    result     = summarizer.summarize(transcript)

    print("\n=== Summary Result ===")
    print(f"Call Overview:       {result['call_overview']}")
    print(f"Products Discussed:  {result['products_discussed']}")
    print(f"Key Topics:          {result['key_topics']}")
    print(f"HCP Response:        {result['hcp_response']}")
    print(f"Objections Raised:   {result['objections_raised']}")
    print(f"Next Steps:          {result['next_steps']}")
    print(f"Rep Recommendations: {result['rep_recommendations']}")