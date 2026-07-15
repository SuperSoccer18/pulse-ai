"""
fieldExtraction.py  —  Pulse.AI Field Extraction Layer
=======================================================
Receives the raw transcript from the Moonshine ephemeral audio pipeline
and sends it to Field Extraction Gnome on Cortex to extract
structured Veeva CRM field values.

Pipeline position:
    Moonshine transcript → fieldExtraction.py → summaryField.py → complianceLayer.py

Usage:
    from fieldExtraction import FieldExtraction
    extractor = FieldExtraction()
    result = extractor.extract(transcript)

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
    format="%(asctime)s [FieldExtraction] %(message)s"
)
log = logging.getLogger(__name__)


class FieldExtraction:
    """
    Sends the raw transcript from Moonshine to Field Extraction Gnome on Cortex.
    Runs first in the agent pipeline before Summary Fairy and Compliance Goblin.
    Returns structured Veeva CRM field values ready for human review.
    """

    def __init__(self):
        self.extractions_run = 0
        self.session_start   = datetime.utcnow()
        log.info("FieldExtraction initialized")


    def extract(self, transcript: str) -> dict:
        """
        Sends raw transcript to Field Extraction Gnome and returns Veeva CRM fields.

        Receives the raw decoded transcript from the Moonshine ephemeral audio
        pipeline. Runs before Summary Fairy and Compliance Goblin in the pipeline.

        Parameters
        ----------
        transcript : str
            Raw decoded text from the Moonshine ephemeral audio pipeline.

        Returns
        -------
        dict
            account, location, address, call_datetime, record_type,
            engagement_method, virtual_engagement_tool, interaction_notes,
            products_discussed, fields_extracted, fields_null,
            recommended_action, raw_response
        """

        if not transcript or not transcript.strip():
            log.error("Empty or null transcript received — skipping extraction")
            return self._empty_result()

        self.extractions_run += 1
        extraction_id = f"EXTRACT-{self.extractions_run:04d}"

        log.info(f"{extraction_id}: Sending to Field Extraction Gnome")
        log.info(f"{extraction_id}: Preview: {transcript[:80]}...")

        raw_response = self._call_api(transcript, extraction_id)

        if raw_response is None:
            log.error(f"{extraction_id}: No response from Field Extraction Gnome")
            return self._error_result(extraction_id)

        result = self._parse(raw_response, extraction_id)

        log.info(
            f"{extraction_id}: Extracted {result['fields_extracted']} fields "
            f"({result['fields_null']} null) — {result['recommended_action']}"
        )

        return result


    def _call_api(self, transcript: str, extraction_id: str) -> str | None:
        """
        POSTs transcript to Field Extraction Gnome using Lilly Light Client.
        Extracts message_fragment from streaming JSON response lines.
        Strips markdown code fences before returning.
        """

        log.info(f"{extraction_id}: POST {CORTEX_BASE_URL}/model/ask/field-extraction-gnome")

        try:
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/field-extraction-gnome",
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
            print(f"FIELD EXTRACTION GNOME — {extraction_id}")
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

            log.info(f"{extraction_id}: Response received from Field Extraction Gnome")
            return message_content if message_content else response.text

        except Exception as e:
            log.error(f"{extraction_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, extraction_id: str) -> dict:
        """
        Parses Field Extraction Gnome JSON response into structured Veeva CRM fields.
        Falls back to empty result if JSON parsing fails.
        """

        try:
            data = json.loads(raw)

            products_discussed = data.get("products_discussed", [])

            interaction_notes = data.get("interaction_notes")
            if interaction_notes and len(interaction_notes) > 255:
                interaction_notes = interaction_notes[:252] + "..."

            all_fields = [
                data.get("account"),
                data.get("location"),
                data.get("address"),
                data.get("call_datetime"),
                interaction_notes,
            ]
            fields_extracted = sum(
                1 for f in all_fields if f is not None and f != "null"
            ) + len(products_discussed) + 3
            fields_null = sum(
                1 for f in all_fields if f is None or f == "null"
            )

            return {
                "extraction_id":           extraction_id,
                "account":                 data.get("account"),
                "location":                data.get("location"),
                "address":                 data.get("address"),
                "call_datetime":           data.get("call_datetime"),
                "record_type":             data.get("record_type",             "Interaction"),
                "engagement_method":       data.get("engagement_method",       "In-office"),
                "virtual_engagement_tool": data.get("virtual_engagement_tool", "N/A"),
                "interaction_notes":       interaction_notes,
                "products_discussed":      products_discussed,
                "fields_extracted":        fields_extracted,
                "fields_null":             fields_null,
                "recommended_action":      data.get("recommended_action",      "REVIEW"),
                "raw_response":            raw,
                "timestamp":               datetime.utcnow().isoformat(),
            }

        except json.JSONDecodeError:
            log.warning(f"{extraction_id}: JSON parse failed — returning error result")
            return self._error_result(extraction_id)


    def _empty_result(self) -> dict:
        """Returned when transcript is empty or None."""
        return {
            "extraction_id":           "EMPTY",
            "account":                 None,
            "location":                None,
            "address":                 None,
            "call_datetime":           None,
            "record_type":             "Interaction",
            "engagement_method":       "In-office",
            "virtual_engagement_tool": "N/A",
            "interaction_notes":       None,
            "products_discussed":      [],
            "fields_extracted":        0,
            "fields_null":             5,
            "recommended_action":      "REVIEW",
            "raw_response":            "",
            "timestamp":               datetime.utcnow().isoformat(),
        }

    def _error_result(self, extraction_id: str) -> dict:
        """Returned when the Cortex API call or JSON parsing fails."""
        result = self._empty_result()
        result["extraction_id"]      = extraction_id
        result["recommended_action"] = "REVIEW"
        result["interaction_notes"]  = "Field extraction failed — manual entry required"
        return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Field Extraction Standalone Test ===\n")

    if len(sys.argv) > 1:
        transcript = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
        log.info(f"Loaded: {sys.argv[1]}")
    else:
        transcript = (
            "Visited Chicago Oncology Associates at 676 North St. Clair Street "
            "Suite 1200 Chicago IL 60611 on June 17 2026 at 10:30 AM. "
            "Discussed VERZENIO for HR+/HER2- mBC in combination with an "
            "aromatase inhibitor. Reviewed MONARCH 3 efficacy data and covered "
            "full fair balance on diarrhea, neutropenia, and fatigue. Physician "
            "showed interest in prescribing for appropriate patients. Agreed to "
            "follow up in two weeks with patient support materials."
        )
        log.info("Using built-in TC-FEG-01 test transcript")

    extractor = FieldExtraction()
    result    = extractor.extract(transcript)

    print("\n=== Field Extraction Result ===")
    print(f"Account:          {result['account']}")
    print(f"Location:         {result['location']}")
    print(f"Address:          {result['address']}")
    print(f"Call DateTime:    {result['call_datetime']}")
    print(f"Record Type:      {result['record_type']}")
    print(f"Engagement:       {result['engagement_method']}")
    print(f"Virtual Tool:     {result['virtual_engagement_tool']}")
    print(f"Notes:            {result['interaction_notes']}")
    print(f"Products:         {result['products_discussed']}")
    print(f"Fields Extracted: {result['fields_extracted']}")
    print(f"Fields Null:      {result['fields_null']}")
    print(f"Action:           {result['recommended_action']}")