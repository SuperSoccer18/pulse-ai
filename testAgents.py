"""
testAgents.py
Test script using real Lilly Light Client authentication.
Run: python testAgents.py
"""

from light_client import LIGHTClient
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
CORTEX_BASE = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")

# ── Get light_auth token from Light Client ────────────────────────────────
client     = LIGHTClient()
auth_header = client.get_auth_header()
print(f"Auth header obtained: {str(auth_header)[:50]}...\n")

HEADERS = {
    "accept": "application/json",
    **auth_header,    # spreads the auth header directly into headers
}

PARAMS = {
    "stream":           "true",
    "no_summary":       "false",
    "background_job":   "false",
    "workflow_timeout": "1",
}

# ── Clean test transcript ─────────────────────────────────────────────────
CLEAN_TRANSCRIPT = (
    "Visited the office today and discussed VERZENIO for HR+/HER2- mBC "
    "in combination with an aromatase inhibitor. Reviewed MONARCH 3 "
    "efficacy data and covered full fair balance on diarrhea, neutropenia, "
    "and fatigue. Physician showed interest in prescribing for appropriate "
    "patients. Agreed to follow up in two weeks with patient support materials."
)

# ── Test 1: Compliance Goblin V3 ──────────────────────────────────────────
print("=" * 50)
print("TEST 1 — Compliance Goblin V3")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/compliance-goblin-v2",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()

# ── Test 2: Field Extraction Gnome ────────────────────────────────────────
print("=" * 50)
print("TEST 2 — Field Extraction Gnome")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/field-extraction-gnome",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()

# ── Test 3: Summary Fairy ─────────────────────────────────────────────────
print("=" * 50)
print("TEST 3 — Summary Fairy")
print("=" * 50)

response = requests.post(
    f"{CORTEX_BASE}/model/ask/summary-fairy",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    stream=True,
    timeout=90,
)
print(f"Status: {response.status_code}")
for line in response.iter_lines():
    if line:
        print(line.decode("utf-8"))
print()