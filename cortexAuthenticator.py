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
    from cortex_auth import get_bearer_token
    token = get_bearer_token()
"""

# requests is the HTTP library used to POST to the Azure AD token endpoint
import requests

# logging prints timestamped messages so we can track auth events
import logging

# datetime is used to track when the token was obtained and when it expires
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Azure AD Credentials
# Replace the placeholder strings with your actual values from Azure Portal
# ─────────────────────────────────────────────────────────────────────────────

# TENANT_ID identifies Lilly's Azure Active Directory directory
# Found in Azure Portal > Azure Active Directory > Overview
TENANT_ID    = "18a59a81-eea8-4c30-948a-d8824cdc2580" # YOUR_TENANT_ID_HERE

# CLIENT_ID identifies the registered application in Azure AD
# Found in Azure Portal > App Registrations > your app > Overview
CLIENT_ID    = "43643ea2-75eb-4910-b6fb-1e9544dddca2" # YOUR_CLIENT_ID_HERE

# SECRET_VALUE is the actual password used to authenticate the application
# Found in Azure Portal > App Registrations > your app > Certificates & Secrets
# USE the Secret Value column — NOT the Secret ID column
SECRET_VALUE = "TgD8Q~5CgRZO9OW8h8UI3EnIXyz2MOKXhymL3c-A"

# SECRET_ID is just a label for managing secrets in Azure Portal
# It is NOT used in the token request — only SECRET_VALUE is needed
# SECRET_ID = "b86e637f-ffd6-49fd-8b1a-5c9464b8f817"  # not needed for token request

# SCOPE tells Azure which resource we want to access
# The .default suffix requests all permissions granted to the app
# Confirm the exact scope value with Kuntal for Lilly's Cortex instance
SCOPE = "https://cortex.lilly.com/.default"

# TOKEN_URL is the Azure AD OAuth2 token endpoint for this tenant
# Constructed from the tenant ID — unique per organization
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

