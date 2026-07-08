"""
cortexAuthenticator.py — Pulse.AI Cortex Authentication
Using Lilly's Light Client instead of raw Azure AD tokens
"""

from light_client import LIGHTClient
import os
from dotenv import load_dotenv
load_dotenv()

# Cortex base URL loaded from .env
CORTEX_BASE_URL = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")

# Initialize the Light Client — handles Lilly auth automatically
client = LIGHTClient()


def get_client():
    """Returns the authenticated Light Client instance."""
    return client


def get_auth_header() -> dict:
    """
    Returns the Authorization header dict from the Light Client.
    Use this instead of a raw Bearer token.
    """
    return client.get_auth_header()


def get_bearer_token() -> str:
    """
    Returns the Bearer token string from the Light Client.
    Kept for backwards compatibility with complianceLayer.py,
    fieldExtraction.py and summaryField.py.
    """
    auth_header = client.get_auth_header()
    # Extract just the token string from the header dict
    # auth_header looks like: {'Authorization': 'Bearer eyJ...'}
    return auth_header["Authorization"].replace("Bearer ", "")