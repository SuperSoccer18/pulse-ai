import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("VEEVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("VEEVA_CLIENT_SECRET")
LOGIN_URL = os.getenv("VEEVA_LOGIN_URL")

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

# check user identity
async def get_identity(token: str, instance_url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{instance_url}/services/oauth2/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        print(resp.status_code)
        print(resp.json())

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

# get object description
async def describe_object(token: str, instance_url: str, object_name: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{instance_url}/services/data/v60.0/sobjects/{object_name}/describe",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            resp.raise_for_status()
        data = resp.json()
        for f in data["fields"]:
            print(f"{f['name']:40} {f['label']}")

async def check_object_permissions(token: str, instance_url: str, object_name: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{instance_url}/services/data/v60.0/sobjects/{object_name}/describe",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            resp.raise_for_status()
        data = resp.json()
        print(f"--- {object_name} permissions for pulse_ai_acc ---")
        print("Createable:", data["createable"])
        print("Updateable:", data["updateable"])
        print("Deletable:", data["deletable"])

async def main():
    token, instance_url = await get_veeva_token()

    soql = """
    SELECT Id, Status_vod__c, Call_Date_vod__c,
        Account_vod__r.Name,
        Account_vod__r.Specialty_1_vod__c,
        Account_vod__r.BillingCity,
        Account_vod__r.BillingState
    FROM Call2_vod__c
    WHERE OwnerId = '005G0000001AxeCIAS'
    ORDER BY Call_Date_vod__c DESC
    LIMIT 10
    """
    result = await query_veeva(token, instance_url, soql)
    print(result)



if __name__ == "__main__":
    asyncio.run(main())