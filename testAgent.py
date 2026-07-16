"""
testAgent.py
Quick test script for all three Cortex agents.
Run: python testAgent.py
"""

from cortexAuthenticator import get_bearer_token
import requests
#def get_bearer_token():
    #return "MOCK_TOKEN_FOR_TESTING"


# ── Config ────────────────────────────────────────────────────────────────
BASE_URL = "https://gateway-intranet.apim.lilly.com/cortex"
PARAMS   = {
    "stream":          "false",
    "no_summary":      "false",
    "background_job":  "false",
    "workflow_timeout": "1",
}

# ── Get token once — reused for all three calls ───────────────────────────
print("Getting Bearer token...")
token = get_bearer_token()
print(f"Token obtained: {token[:20]}...\n")

HEADERS = {
    "accept":        "application/json",
    "Authorization": f"Bearer {token}",
}

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

response = requests.post(
    BASE_URL + "/cortex/model/ask/compliance-goblin-v3",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    timeout=90,
)
print(f"Status: {response.status_code}")
print(response.text[:500])
print()

# ── Test 2: Field Extraction Gnome ────────────────────────────────────────
print("=" * 50)
print("TEST 2 — Field Extraction Gnome")
print("=" * 50)

response = requests.post(
    BASE_URL + "/cortex/model/ask/field-extraction-gnome",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    timeout=90,
)
print(f"Status: {response.status_code}")
print(response.text[:500])
print()

# ── Test 3: Summary Fairy ─────────────────────────────────────────────────
print("=" * 50)
print("TEST 3 — Summary Fairy")
print("=" * 50)

response = requests.post(
    BASE_URL + "/cortex/model/ask/summary-fairy",
    params=PARAMS,
    data={"q": CLEAN_TRANSCRIPT},
    headers=HEADERS,
    timeout=90,
)
print(f"Status: {response.status_code}")
print(response.text[:500])