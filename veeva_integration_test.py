import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("VEEVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("VEEVA_CLIENT_SECRET")
LOGIN_URL = os.getenv("VEEVA_LOGIN_URL")

ARTHUR_ID = "005G0000001AxUMIA0"

# Dummy calls: (hcp_account_id, hcp_name, specialty, product_id, product_name, date, location)
DUMMY_CALLS = [
    ("0010f000028Na09AAC", "Mahnaz Reyner",   "Dermatology",             "a000f00000Wl2yhAAB", "Trulicity", "2026-07-10", "Houston Methodist Hospital"),
    ("0010f000028Na0EAAS", "Osie Hennagin",   "Dermatology",             "a000f00000Wl3INAAZ", "Jardiance", "2026-07-11", "Texas Medical Center"),
    ("0010f000028Na0JAAS", "Tarek Liff",      "Dermatology",             "a000f00000Wl3IOAAZ", "Jardiance", "2026-07-12", "Memorial Hermann Clinic"),
    ("0010f000028NZzVAAW", "Kenesha Vient",   "Pharmaceutical Medicine", "a000f00000Wl2yhAAB", "Trulicity", "2026-07-13", "St. Luke's Health - Baylor"),
    ("0010f000028NZzaAAG", "Tae Wildermuth",  "Pharmaceutical Medicine", "a000f00000Wl3INAAZ", "Jardiance", "2026-07-14", "UTHealth Medical Plaza"),
    ("0010f000028NZzzAAG", "Rafiga Cicen",    "Nutrition",               "a000f00000Wl2yhAAB", "Trulicity", "2026-07-15", "Baylor Scott & White Clinic"),
]

# call IDs created in the previous run — keyed by HCP account ID
EXISTING_CALL_IDS = {
    "0010f000028Na09AAC": "a04RT000007KkqfYAC",   # Mahnaz Reyner
    "0010f000028Na0EAAS": "a04RT000007KksHYAS",   # Osie Hennagin
    "0010f000028Na0JAAS": "a04RT000007Ka9nYAC",   # Tarek Liff
    "0010f000028NZzVAAW": "a04RT000007KkttYAC",   # Kenesha Vient
    "0010f000028NZzaAAG": "a04RT000007KkvVYAS",   # Tae Wildermuth
    "0010f000028NZzzAAG": "a04RT000007KkkDYAS",   # Rafiga Cicen (repurposed orphan)
}

# get access token
async def get_veeva_token():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            LOGIN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            },
        )
        if resp.status_code != 200:
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            resp.raise_for_status()
        data = resp.json()
        return data["access_token"], data["instance_url"]

# query veeva with soql
async def query_veeva(token: str, instance_url: str, soql: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{instance_url}/services/data/v60.0/query",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": soql},
        )
        if resp.status_code != 200:
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            resp.raise_for_status()
        return resp.json()

# create a single Call2_vod__c record, return its new Id
async def create_call(token: str, instance_url: str, account_id: str, call_date: str) -> str:
    payload = {
        "OwnerId":              ARTHUR_ID,
        "Account_vod__c":       account_id,
        "Call_Date_vod__c":     call_date,
        "Status_vod__c":        "Planned_vod",
        "Call_Type_vod__c":     "Detail Only",
        "Call_Channel_vod__c":  "Face_to_face_vod",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_vod__c",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code not in (200, 201):
            print("  Create call failed:", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()["id"]

# create a Call2_Detail_vod__c child record linking the call to a product
async def create_call_detail(token: str, instance_url: str, call_id: str, product_id: str, product_name: str):
    payload = {
        "Call2_vod__c":     call_id,
        "Product_vod__c":   product_id,
        "Product_Name__c":  product_name,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_Detail_vod__c",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code not in (200, 201):
            print("  Create detail failed:", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()["id"]

# delete a call record by ID
async def delete_call(token: str, instance_url: str, call_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_vod__c/{call_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 204:
            print(f"  Delete failed for {call_id}:", resp.status_code, resp.text)
            resp.raise_for_status()

# patch fields on an existing call record
async def update_call(token: str, instance_url: str, call_id: str, fields: dict):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.patch(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_vod__c/{call_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=fields,
        )
        if resp.status_code != 204:
            print(f"  Update failed for {call_id}:", resp.status_code, resp.text)
            resp.raise_for_status()

# read back Arthur's calls and map each to our application dict shape
async def fetch_calls(token: str, instance_url: str) -> list[dict]:
    soql = """
    SELECT Id, Call_Date_vod__c, Status_vod__c, Call_Type_vod__c,
           Territory_vod__c,
           Account_vod__c, Account_vod__r.Name, Account_vod__r.Specialty_1_vod__c
    FROM Call2_vod__c
    WHERE OwnerId = '005G0000001AxUMIA0'
    ORDER BY Call_Date_vod__c DESC
    LIMIT 20
    """
    result = await query_veeva(token, instance_url, soql)
    calls = []
    for r in result.get("records", []):
        acct = r.get("Account_vod__r") or {}
        calls.append({
            "id":        r["Id"],
            "hcp_name":  acct.get("Name"),
            "specialty": acct.get("Specialty_1_vod__c"),
            "location":  r.get("Territory_vod__c"),
            "status":    r.get("Status_vod__c"),
            "date":      r.get("Call_Date_vod__c"),
        })
    return calls

async def main():
    token, instance_url = await get_veeva_token()

    # --- Full field list for Call2_Detail_vod__c ---
    print("\n" + "="*60)
    print("  Call2_Detail_vod__c  —  ALL fields (create/update permission shown)")
    print("="*60)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{instance_url}/services/data/v60.0/sobjects/Call2_Detail_vod__c/describe",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        fields = resp.json()["fields"]
    print(f"{'Field API Name':<45} {'Label':<40} {'Type':<12} {'Create':<8} {'Update':<8} Picklist values")
    print(f"{'-'*45} {'-'*40} {'-'*12} {'-'*8} {'-'*8} ---------------")
    for f in fields:
        picklist = ", ".join(v["value"] for v in f["picklistValues"] if v["active"]) if f["picklistValues"] else ""
        print(f"{f['name']:<45} {f['label']:<40} {f['type']:<12} {str(f['createable']):<8} {str(f['updateable']):<8} {picklist}")

if __name__ == "__main__":
    asyncio.run(main())