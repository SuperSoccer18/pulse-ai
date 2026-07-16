import os
from light_client import LIGHTClient
from dotenv import load_dotenv

load_dotenv()

CLIENT_SECRET = os.getenv("VEEVA_CLIENT_SECRET")
LOGIN_URL = os.getenv("VEEVA_LOGIN_URL")

client = LIGHTClient()
CORTEX_BASE = os.getenv('CORTEX_BASE')
YOUR_EMAIL = os.getenv('EMAIL')

# model creation
model_config = {
    "name": "elliot-cortex-goat",
    "auth": {"owners": [YOUR_EMAIL], "private": True},
    "displayName": "test",
    "model_description": "pls work bro",
    "chain": [{"chain_class": "model-only-chain", "model_iteration": 1, "order": 1, "chain_params": {}}],
    "model_versions": [{"model_class": "lilly-openai", "model_iteration": 7, "priority": 100}],
    "temperature": 0
}

def create_model():
    response = client.post(f"{CORTEX_BASE}/model", json=model_config)
    if response.status_code == 200:
        print("Model created:", response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")

# inference
if __name__ == "__main__":
    model_name = 'lilly-keyword-recognizer'
    r = client.post(f'{CORTEX_BASE}/model/ask/{model_name}', data={'q': 'monjario truelicity'})
    print(r.status_code)
    print(r.json())