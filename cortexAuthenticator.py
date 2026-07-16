"""
cortexAuthenticator.py —  Pulse.AI Azure AD Authentication
=====================================================
Obtains a Bearer access token from Azure AD using the
Client Credentials flow and the four values provided:
    - Tenant ID
    - Client ID
    - Secret Value
    - Secret ID  (not used in the token request — label only)
 
Usage:
    from cortexAuthenticator import get_bearer_token
    token = get_bearer_token()
"""

import requests
 
# logging prints timestamped messages so we can track auth events
import logging
 
# datetime is used to track when the token was obtained and when it expires
from datetime import datetime, timedelta
 
log = logging.getLogger(__name__)
 
from dotenv import load_dotenv
import os
load_dotenv()
 
TENANT_ID    = os.getenv("TENANT_ID")
CLIENT_ID    = os.getenv("CLIENT_ID")
SECRET_VALUE = os.getenv("SECRET_VALUE")
#SECRET_ID    = os.getenv("SECRET_ID")
 
# SCOPE — must include .default suffix for client credentials flow
SCOPE = os.getenv("SCOPE", "api://Cortex.lilly.com/.default")
 
CORTEX_BASE_URL  = os.getenv("CORTEX_BASE_URL", "https://gateway-intranet.apim.lilly.com/cortex")
 
# TOKEN_URL is the Azure AD OAuth2 token endpoint for this tenant
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
 
# ─────────────────────────────────────────────────────────────────────────────
# Token cache — avoids requesting a new token on every API call
# ─────────────────────────────────────────────────────────────────────────────
 
# _cached_token stores the last successfully obtained Bearer token string
_cached_token    = None
 
# _token_expiry stores when the cached token expires as a datetime object
# Tokens are refreshed 60 seconds before expiry to avoid edge case failures
_token_expiry    = None
 
# _REFRESH_BUFFER_SECONDS how many seconds before expiry to pre-refresh the token
_REFRESH_BUFFER_SECONDS = 60
 
def get_bearer_token() -> str:
    """
    Returns a valid Bearer access token for the Cortex API.
 
    Uses a cached token if it is still valid.
    Requests a new token from Azure AD if the cache is empty or expired.
 
    Returns
    -------
    str
        The Bearer token string to include in Authorization header
    """
 
    # Access the module-level cache variables
    global _cached_token, _token_expiry
 
    # Check if we have a cached token that is still valid
    # datetime.utcnow() gets the current UTC time for comparison
    # timedelta converts the buffer seconds into a time delta for subtraction
    if (
        _cached_token is not None and
        _token_expiry is not None and
        datetime.utcnow() < _token_expiry - timedelta(seconds=_REFRESH_BUFFER_SECONDS)
    ):
        # Cached token is still valid — return it without making a network call
        log.debug("Using cached Bearer token")
        return _cached_token
 
    # No valid cached token — request a new one from Azure AD
    log.info("Requesting new Bearer token from Azure AD")
 
    # try block attempts the token request — jumps to except on any error
    try:
        # POST to the Azure AD token endpoint with client credentials
        # grant_type: client_credentials means the app authenticates as itself
        # not on behalf of a user — this is the correct flow for backend services
        response = requests.post(
            TOKEN_URL,                      # Azure AD token endpoint for this tenant
            data={
                "grant_type":    "client_credentials",  # app-to-app auth flow
                "client_id":     CLIENT_ID,              # app identity in Azure AD
                "client_secret": SECRET_VALUE,           # app password — Secret Value
                "scope":         SCOPE,                  # resource being accessed
            },
            timeout=30,    # abort if Azure AD does not respond within 30 seconds
        )
 
        # raise_for_status raises an exception for 4xx/5xx HTTP errors
        # This catches invalid credentials, wrong tenant, etc.
        # Add this temporarily to see the exact Azure AD error
        print("Azure AD error details:", response.json())
        response.raise_for_status()
 
        # Parse the JSON response body from Azure AD
        token_data = response.json()
 
        # Extract the access token string from the response
        # This is the Bearer token we will send to Cortex
        _cached_token = token_data["access_token"]
 
        # Extract how many seconds until the token expires
        # Azure AD typically returns 3600 (1 hour) for client credential tokens
        expires_in = int(token_data.get("expires_in", 3600))
 
        # Calculate the absolute expiry datetime from now + expires_in seconds
        _token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
 
        # Log success and when the token expires
        log.info(f"Bearer token obtained — expires at {_token_expiry.isoformat()} UTC")
 
        # Return the freshly obtained Bearer token
        return _cached_token
 
    # Catch HTTP errors from raise_for_status() — wrong credentials etc.
    except requests.exceptions.HTTPError as e:
        log.error(f"Azure AD token request failed: HTTP {response.status_code} — {e}")
        raise
 
    # Catch network errors — Azure AD unreachable, DNS failure, VPN issue
    except requests.exceptions.RequestException as e:
        log.error(f"Azure AD token request failed: {e}")
        raise