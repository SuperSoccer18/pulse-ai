"""
testAgents.py
Quick test script using Lilly Light Client authentication.
Run: python testAgents.py
"""

from light_client import LIGHTClient
import os
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
CORTEX_BASE = os.getenv("CORTEX_BASE_URL", "https://api.dev.cortex.lilly.com")

# Light Client handles Lilly authentication automatically
client = LIGHTClient()

# ── Clean test transcript ─────────────────────────────────────────────────
CLEAN_TRANSCRIPT = (
    "Visited the office today and discussed VERZENIO for HR+/HER2- mBC "
    "in combination with an aromatase inhibitor. Reviewed MONARCH 3 "
    "efficacy data and covered full fair balance on diarrhea, neutropenia, "
    "and fatigue. Physician showed interest in prescribing for appropriate "
    "patients. Agreed to follow up in two weeks with patient support materials."
)

# ── Test 1: Compliance Goblin V3 ─────────────────────────────────────────
print("=" * 50)
print("TEST 1 — Compliance Goblin V3")
print("=" * 50)

response = client.post(
    f"{CORTEX_BASE}/model/ask/compliance-goblin-v3",
    data={
        "q":                CLEAN_TRANSCRIPT,
        "stream":           "true",
        "no_summary":       "false",
        "background_job":   "false",
        "workflow_timeout": "1",
    }
)
print(f"Status: {response.status_code}")
print(response.text[:500])
print()

# ── Test 2: Field Extraction Gnome ────────────────────────────────────────
print("=" * 50)
print("TEST 2 — Field Extraction Gnome")
print("=" * 50)

response = client.post(
    f"{CORTEX_BASE}/model/ask/field-extraction-gnome",
    data={
        "q":                CLEAN_TRANSCRIPT,
        "stream":           "true",
        "no_summary":       "false",
        "background_job":   "false",
        "workflow_timeout": "1",
    }
)
print(f"Status: {response.status_code}")
print(response.text[:500])
print()

# ── Test 3: Summary Fairy ─────────────────────────────────────────────────
print("=" * 50)
print("TEST 3 — Summary Fairy")
print("=" * 50)

response = client.post(
    f"{CORTEX_BASE}/model/ask/summary-fairy",
    data={
        "q":                CLEAN_TRANSCRIPT,
        "stream":           "true",
        "no_summary":       "false",
        "background_job":   "false",
        "workflow_timeout": "1",
    }
)
print(f"Status: {response.status_code}")
print(response.text[:500])
print()