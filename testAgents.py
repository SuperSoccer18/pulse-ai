"""
testAgents.py
Test script for all three Cortex agents in pipeline order:
    Field Extraction → Summary → Compliance

Run: python testAgents.py
"""

import requests
from light_client import LIGHTClient
import os
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
# Pipeline order: Field Extraction → Summary → Compliance
CORTEX_BASE = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")

# ── Auth ──────────────────────────────────────────────────────────────────────
client      = LIGHTClient()
auth_header = client.get_auth_header()
print(f"Auth header obtained: {str(auth_header)[:50]}...\n")

HEADERS = {
    "accept": "application/json",
    **auth_header,
}

PARAMS = {
    "stream":           "true",
    "no_summary":       "false",
    "background_job":   "false",
    "workflow_timeout": "1",
}

# ── Raw test transcript from Moonshine ────────────────────────────────────────
# Simulates the raw output from the ephemeral audio transcription step
RAW_TRANSCRIPT = (
    "Visited the office today and discussed VERZENIO for HR+/HER2- mBC "
    "in combination with an aromatase inhibitor. Reviewed MONARCH 3 "
    "efficacy data and covered full fair balance on diarrhea, neutropenia, "
    "and fatigue. Physician showed interest in prescribing for appropriate "
    "patients. Agreed to follow up in two weeks with patient support materials."
)

# ── Step 1: Field Extraction Gnome ────────────────────────────────────────────
print("=" * 50)
print("STEP 1 — Field Extraction Gnome")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/field-extraction-gnome",
    params=PARAMS,
    data={"q": RAW_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()

# ── Step 2: Summary Fairy ─────────────────────────────────────────────────────
print("=" * 50)
print("STEP 2 — Summary Fairy")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/summary-fairy",
    params=PARAMS,
    data={"q": RAW_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()

# ── Step 3: Compliance Goblin V3 ──────────────────────────────────────────────
print("=" * 50)
print("STEP 3 — Compliance Goblin V3")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/compliance-goblin-v2",
    params=PARAMS,
    data={"q": RAW_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()