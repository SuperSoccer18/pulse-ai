"""
mock_cortex.py — Pulse.AI Mock Cortex Agents
=============================================
Temporary mock replacements for the three Cortex agents.
Use while waiting for API whitelisting from Sneha.

To use: swap imports in app.py to use these classes.
To revert: swap back to real imports once whitelisted.
"""

class MockComplianceLayer:
    def analyze(self, transcript: str) -> dict:
        return {
            "compliance_status":  "CLEAN",
            "dlo_escalation":     False,
            "cleaned_transcript": transcript,
            "violations":         [],
            "recommended_action": "PROCEED",
            "raw_response":       "COMPLIANCE STATUS: CLEAN",
            "chunk_id":           "CHUNK-0001",
        }

class MockFieldExtraction:
    def extract(self, cleaned: str) -> dict:
        return {
            "account":                 "Chicago Oncology Associates",
            "location":                "Chicago Oncology Associates",
            "address":                 "676 North St. Clair Street Suite 1200 Chicago IL 60611",
            "call_datetime":           "2026-06-17T10:30:00",
            "record_type":             "Interaction",
            "engagement_method":       "In-office",
            "virtual_engagement_tool": "N/A",
            "interaction_notes":       "Discussed VERZENIO for HR+/HER2- mBC. Physician interested. Follow up in two weeks.",
            "products_discussed":      [
                {"product": "VERZENIO", "indication": "HR+/HER2- mBC in combination with an aromatase"}
            ],
            "fields_extracted":        9,
            "fields_null":             0,
            "recommended_action":      "PROCEED",
        }

class MockSummaryField:
    def summarize(self, cleaned: str) -> dict:
        return {
            "call_overview":       "Sales visit to discuss VERZENIO for HR+/HER2- mBC.",
            "products_discussed":  "VERZENIO",
            "key_topics":          "MONARCH 3 efficacy data, fair balance on diarrhea and fatigue.",
            "hcp_response":        "Strong interest in prescribing for appropriate patients.",
            "objections_raised":   "Prior authorization barriers noted.",
            "next_steps":          "Follow up in two weeks with patient support materials.",
            "rep_recommendations": "Bring copay assistance program details to follow up visit.",
        }