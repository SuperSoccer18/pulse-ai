"""
fieldExtraction.py  —  Pulse.AI Field Extraction Layer
=======================================================
Receives the cleaned transcript from complianceLayer.py and
sends it to Field Extraction Gnome on Cortex to extract
structured Veeva CRM field values.

Pipeline position:
    complianceLayer.py → fieldExtraction.py → summaryField.py

Usage:
    from fieldExtraction import FieldExtraction
    extractor = FieldExtraction()
    result = extractor.extract(cleaned_transcript)

Requirements:
    pip install light-client python-dotenv
"""

# logging prints timestamped status messages for each pipeline step
import logging

# datetime records when extraction events happen during the session
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
    format="%(asctime)s [FieldExtraction] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FieldExtraction class
# ─────────────────────────────────────────────────────────────────────────────

class FieldExtraction:
    """
    Sends the cleaned transcript from Compliance Goblin to Field Extraction
    Gnome on Cortex and returns structured Veeva CRM field values.

    Receives the cleaned_transcript from complianceLayer.py result dict.
    Passes extracted fields to the human review step before CRM submission.
    """

    def __init__(self):
        # extractions_run counts how many field extractions have been performed
        self.extractions_run = 0

        # session_start records when this FieldExtraction was initialized
        self.session_start = datetime.utcnow()

        log.info("FieldExtraction initialized")


    def extract(self, cleaned_transcript: str) -> dict:
        """
        Sends the cleaned transcript to Field Extraction Gnome and returns
        structured Veeva CRM field values ready for human review.

        Parameters
        ----------
        cleaned_transcript : str
            The redacted transcript from complianceLayer.py result dict.
            All PI and AECP content has already been removed by Compliance Goblin.

        Returns
        -------
        dict
            account, location, address, call_datetime, record_type,
            engagement_method, virtual_engagement_tool, interaction_notes,
            products_discussed, fields_extracted, fields_null,
            recommended_action, raw_response
        """

        # Validate that a cleaned transcript was actually provided
        if not cleaned_transcript or not cleaned_transcript.strip():
            log.error("Empty or null cleaned transcript received — skipping extraction")
            return self._empty_result()

        # Increment extraction counter and build zero-padded extraction ID
        self.extractions_run += 1
        extraction_id = f"EXTRACT-{self.extractions_run:04d}"

        log.info(f"{extraction_id}: Sending cleaned transcript to Field Extraction Gnome")
        log.info(f"{extraction_id}: Preview: {cleaned_transcript[:80]}...")

        # POST cleaned transcript to Field Extraction Gnome via Light Client
        raw_response = self._call_api(cleaned_transcript, extraction_id)

        # Return empty result if API call failed
        if raw_response is None:
            log.error(f"{extraction_id}: No response from Field Extraction Gnome")
            return self._error_result(extraction_id)

        # Parse the response into structured Veeva CRM fields
        result = self._parse(raw_response, extraction_id)

        log.info(
            f"{extraction_id}: Extracted {result['fields_extracted']} fields "
            f"({result['fields_null']} null) — {result['recommended_action']}"
        )

        return result


    def _call_api(self, cleaned_transcript: str, extraction_id: str) -> str | None:
        """
        POSTs the cleaned transcript to Field Extraction Gnome using
        Lilly Light Client. get_auth_header() provides the personal
        Lilly token accepted by Cortex.
        """

        log.info(f"{extraction_id}: POST {CORTEX_BASE_URL}/model/ask/field-extraction-gnome")

        try:
            # get_auth_header() returns the personal Lilly Bearer token
            # This is the token method confirmed working with Cortex
            response = client.post(
                f"{CORTEX_BASE_URL}/model/ask/field-extraction-gnome",
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

            # Print response so rep sees field extraction in real time
            print(f"\n{'='*60}")
            print(f"FIELD EXTRACTION GNOME — {extraction_id}")
            print(f"{'='*60}")
            print(response.text)
            print(f"{'='*60}\n")

            log.info(f"{extraction_id}: Response received from Field Extraction Gnome")

            # Return full response text for parsing
            return response.text

        except Exception as e:
            log.error(f"{extraction_id}: API call failed — {e}")
            return None


    def _parse(self, raw: str, extraction_id: str) -> dict:
        """
        Parses the raw Field Extraction Gnome response into structured
        Veeva CRM fields matching the output schema.

        Expected response format:
            ACCOUNT: <value or null>
            LOCATION: <value or null>
            ADDRESS: <value or null>
            CALL DATETIME: <ISO datetime or null>
            RECORD TYPE: Interaction
            ENGAGEMENT METHOD: <value>
            VIRTUAL ENGAGEMENT TOOL: N/A
            INTERACTION NOTES: <text max 255 chars>
            PRODUCTS DISCUSSED:
            - Product: <name>
              Indication: <indication>
            FIELDS EXTRACTED: <number>
            FIELDS NULL: <number>
            RECOMMENDED ACTION: PROCEED or REVIEW or HALT
        """

        # Helper to extract a single-line field value after a label
        def extract_field(label: str) -> str | None:
            if label not in raw:
                return None
            start = raw.find(label) + len(label)
            end = raw.find("\n", start)
            value = raw[start:end if end != -1 else len(raw)].strip().strip('"')
            return None if value.lower() in ("null", "none", "") else value

        # Extract each Veeva CRM field from the response text
        account           = extract_field("ACCOUNT:")
        location          = extract_field("LOCATION:")
        address           = extract_field("ADDRESS:")
        call_datetime     = extract_field("CALL DATETIME:")
        record_type       = extract_field("RECORD TYPE:") or "Interaction"
        engagement_method = extract_field("ENGAGEMENT METHOD:") or "In-office"
        virtual_tool      = "N/A"   # always fixed to N/A per Veeva mapping
        interaction_notes = extract_field("INTERACTION NOTES:")

        # Trim interaction notes to 255 character Veeva field limit
        if interaction_notes and len(interaction_notes) > 255:
            interaction_notes = interaction_notes[:252] + "..."

        # Parse the PRODUCTS DISCUSSED section into a list of dicts
        products_discussed = []
        if "PRODUCTS DISCUSSED:" in raw:
            start = raw.find("PRODUCTS DISCUSSED:") + len("PRODUCTS DISCUSSED:")
            end_markers = ["FIELDS EXTRACTED:", "FIELDS NULL:", "RECOMMENDED ACTION:"]
            end = len(raw)
            for marker in end_markers:
                pos = raw.find(marker, start)
                if pos != -1 and pos < end:
                    end = pos
            products_block = raw[start:end].strip()
            current_product = None
            for line in products_block.split("\n"):
                line = line.strip()
                if line.startswith("- Product:"):
                    if current_product:
                        products_discussed.append(current_product)
                    current_product = {
                        "product":    line.replace("- Product:", "").strip(),
                        "indication": None,
                    }
                elif line.startswith("Indication:") and current_product:
                    current_product["indication"] = line.replace("Indication:", "").strip()
            if current_product:
                products_discussed.append(current_product)

        # Count extracted vs null fields
        all_fields = [account, location, address, call_datetime, interaction_notes]
        fields_extracted = sum(1 for f in all_fields if f is not None) + len(products_discussed) + 3
        fields_null      = sum(1 for f in all_fields if f is None)

        # Parse recommended action
        upper = raw.upper()
        if "HALT" in upper:
            action = "HALT"
        elif "REVIEW" in upper:
            action = "REVIEW"
        else:
            action = "PROCEED"

        return {
            "extraction_id":           extraction_id,
            "account":                 account,
            "location":                location,
            "address":                 address,
            "call_datetime":           call_datetime,
            "record_type":             record_type,
            "engagement_method":       engagement_method,
            "virtual_engagement_tool": virtual_tool,
            "interaction_notes":       interaction_notes,
            "products_discussed":      products_discussed,
            "fields_extracted":        fields_extracted,
            "fields_null":             fields_null,
            "recommended_action":      action,
            "raw_response":            raw,
            "timestamp":               datetime.utcnow().isoformat(),
        }


    # ── Fallback result constructors ──────────────────────────────────────────

    def _empty_result(self) -> dict:
        """Returned when cleaned transcript is empty or None."""
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
        """Returned when the Cortex API call fails."""
        result = self._empty_result()
        result["extraction_id"]      = extraction_id
        result["recommended_action"] = "REVIEW"
        result["interaction_notes"]  = "Field extraction failed — manual entry required"
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("\n=== Pulse.AI — Field Extraction Standalone Test ===\n")

    if len(sys.argv) > 1:
        cleaned_text = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
        log.info(f"Loaded: {sys.argv[1]}")
    else:
        cleaned_text = (
            "Visited Chicago Oncology Associates at 676 North St. Clair Street "
            "Suite 1200 Chicago IL 60611 on June 17 2026 at 10:30 AM. "
            "Discussed VERZENIO for HR+/HER2- mBC in combination with an "
            "aromatase inhibitor. Reviewed MONARCH 3 efficacy data and covered "
            "full fair balance on diarrhea, neutropenia, and fatigue. Physician "
            "showed interest in prescribing for appropriate patients. Agreed to "
            "follow up in two weeks with patient support materials."
        )
        log.info("Using built-in TC-FEG-01 clean test transcript")

    extractor = FieldExtraction()
    result = extractor.extract(cleaned_text)

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