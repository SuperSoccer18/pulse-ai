"""
veeva_client.py — Veeva CRM read/write access for the Pulse AI backend
==================================================================
Owns the Veeva API logic used by server.py's /api/calls (read) and
/api/submit-call (write) endpoints. Auth + SOQL/PATCH helpers mirror the
patterns validated in veeva_integration_test.py.

Known limitations (see veeva_integration_test.py exploration history):
  • Product/indication/next-steps/competitor data has no writable home on
    Call2_vod__c or Call2_Detail_vod__c (the latter is read/write-locked for
    our integration user's permission set) — these fields stay UI-only,
    never persisted to Veeva. update_call_in_veeva() only ever receives the
    confirmed-writable subset; see server.py's _extraction_to_veeva_fields().
  • No token caching: a fresh token is fetched on every call. Fine for a
    demo's request volume; would need caching for production traffic.
"""

import os
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("VEEVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("VEEVA_CLIENT_SECRET")
LOGIN_URL = os.getenv("VEEVA_LOGIN_URL")

# Demo rep — Arthur Stephenson, resolved in veeva_integration_test.py
ARTHUR_ID = "005G0000001AxUMIA0"

# Veeva Status_vod__c picklist collapsed to the UI's pending/done binary.
# Cancelled_vod calls are excluded via the SOQL WHERE clause, not this map.
_STATUS_MAP = {
    "Submitted_vod": "done",
    "Saved_vod": "pending",
    "Planned_vod": "pending",
}

# Placeholder product per HCP account, mirroring the products seeded onto
# these 6 dummy calls in veeva_integration_test.py's DUMMY_CALLS. Stand-in
# only — Call2_Detail_vod__c (the real source of product data) is locked.
_PRODUCT_PLACEHOLDER = {
    "0010f000028Na09AAC": "Trulicity",   # Mahnaz Reyner
    "0010f000028Na0EAAS": "Jardiance",   # Osie Hennagin
    "0010f000028Na0JAAS": "Jardiance",   # Tarek Liff
    "0010f000028NZzVAAW": "Trulicity",   # Kenesha Vient
    "0010f000028NZzaAAG": "Jardiance",   # Tae Wildermuth
    "0010f000028NZzzAAG": "Trulicity",   # Rafiga Cicen
}


async def get_veeva_token() -> tuple[str, str]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            LOGIN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data["instance_url"]


async def query_veeva(token: str, instance_url: str, soql: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{instance_url}/services/data/v60.0/query",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": soql},
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_calls_for_rep() -> list[dict]:
    """
    Fetch Arthur Stephenson's active calls from Veeva and map them to the
    shape static/index.html expects: id, hcp_name, specialty, location,
    product, status.

    Once submitted, Veeva locks a call record (Status_vod__c can never move
    off Submitted_vod again — confirmed live via a 400
    FIELD_CUSTOM_VALIDATION_EXCEPTION when attempting to reset one for
    re-testing). Re-recording for the same HCP means creating a brand new
    Call2_vod__c row rather than reusing the old one, which can leave an HCP
    with both a Submitted_vod (done) row and a fresh Planned_vod (pending)
    row at once. We keep querying Submitted_vod calls too (not just
    excluding them) so the "Documented" stat and done-card styling stay
    accurate on a fresh page load, not just right after a submit in the
    same browser session — but only surface ONE row per HCP: prefer an
    actionable pending call over a done one, since a done call has nothing
    left for the rep to do.
    """
    token, instance_url = await get_veeva_token()

    soql = f"""
    SELECT Id, Call_Date_vod__c, Status_vod__c, Territory_vod__c,
           Account_vod__c, Account_vod__r.Name, Account_vod__r.Specialty_1_vod__c
    FROM Call2_vod__c
    WHERE OwnerId = '{ARTHUR_ID}'
    AND Status_vod__c != 'Cancelled_vod'
    ORDER BY Call_Date_vod__c ASC
    LIMIT 20
    """
    result = await query_veeva(token, instance_url, soql)

    by_account: dict[str, dict] = {}
    for r in result.get("records", []):
        acct = r.get("Account_vod__r") or {}
        account_id = r.get("Account_vod__c")
        call = {
            "id": r["Id"],
            "hcp_name": acct.get("Name"),
            "specialty": acct.get("Specialty_1_vod__c"),
            "location": r.get("Territory_vod__c"),
            "product": _PRODUCT_PLACEHOLDER.get(account_id, "—"),
            "status": _STATUS_MAP.get(r.get("Status_vod__c"), "pending"),
        }
        existing = by_account.get(account_id)
        # Prefer a pending call over a done one for the same HCP — that's
        # the one the rep can actually act on. If both are the same status
        # (e.g. two pending, shouldn't normally happen), keep the first seen.
        if existing is None or (existing["status"] == "done" and call["status"] == "pending"):
            by_account[account_id] = call

    return list(by_account.values())


async def update_call_in_veeva(call_id: str, fields: dict) -> None:
    """
    PATCH a Call2_vod__c record with the given fields. Mirrors the pattern
    validated in veeva_integration_test.py's update_call(), moved here since
    this is the real write path used by server.py's /api/submit-call.

    `fields` is open-ended — caller (server.py) is responsible for only
    including confirmed-writable Call2_vod__c fields. See server.py's
    _extraction_to_veeva_fields() for the extraction → Veeva field mapping.
    """
    token, instance_url = await get_veeva_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.patch(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_vod__c/{call_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=fields,
        )
        resp.raise_for_status()
