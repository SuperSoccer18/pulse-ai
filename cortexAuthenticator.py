"""
cortexAuthenticator.py — Pulse.AI Cortex Authentication
Using Lilly's Light Client instead of raw Azure AD tokens
"""

from light_client import LIGHTClient
import os
from dotenv import load_dotenv
load_dotenv()

# Cortex dev environment base URL
CORTEX_BASE_URL = os.getenv("CORTEX_BASE_URL", "https://api.dev.cortex.lilly.com")

# Initialize the Light Client — handles auth automatically using your Lilly credentials
client = LIGHTClient()

def get_client():
    """Returns the authenticated Light Client instance."""
    return client