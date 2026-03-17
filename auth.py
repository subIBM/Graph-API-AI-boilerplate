"""
Thread-safe Microsoft Graph API authentication using MSAL.

Responsibilities:
    - Manage a single PublicClientApplication instance (singleton)
    - Serialize / deserialize the token cache to disk with a threading lock
    - Provide get_access_token() and get_headers() for all Graph API calls

Usage:
    from auth import get_headers
    response = requests.get(url, headers=get_headers())
"""

import os
import logging
import threading

import requests
from dotenv import load_dotenv
from msal import PublicClientApplication, SerializableTokenCache


load_dotenv()

logger = logging.getLogger(__name__)


CLIENT_ID  : str = os.environ["CLIENT_ID"]
AUTHORITY  : str = os.environ["AUTHORITY"]
SCOPES     : list[str] = [os.environ["SCOPES"]]          # e.g. ["Mail.Read"]
CACHE_FILE : str = os.environ.get("CACHE_FILE", "token_cache.bin")


_cache_lock : threading.Lock = threading.Lock()
_token_cache: SerializableTokenCache = SerializableTokenCache()

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            _token_cache.deserialize(fh.read())
        logger.debug("Token cache loaded from %s", CACHE_FILE)
    except Exception as exc:
        logger.warning("Could not load token cache (%s) — starting fresh: %s", CACHE_FILE, exc)


_app: PublicClientApplication = PublicClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    token_cache=_token_cache,
)



def _persist_cache() -> None:
    """Write the in-memory token cache to disk if it has changed."""
    if _token_cache.has_state_changed:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as fh:
                fh.write(_token_cache.serialize())
            logger.debug("Token cache persisted to %s", CACHE_FILE)
        except OSError as exc:
            logger.error("Failed to persist token cache: %s", exc)



def get_access_token() -> str:
    """
    Return a valid access token.

    Strategy:
      1. Try silent acquisition (uses cached token / refresh token).
      2. Fall back to interactive browser login if silent fails.

    Thread-safe — only one thread acquires / refreshes at a time.

    Returns:
        str: A valid Bearer access token.

    Raises:
        RuntimeError: If token acquisition fails completely.
    """
    with _cache_lock:
        accounts = _app.get_accounts()

        # --- Silent path (happy path) ---
        if accounts:
            result = _app.acquire_token_silent(SCOPES, account=accounts[0])
            if result and "access_token" in result:
                _persist_cache()
                logger.debug("Token acquired silently for %s", accounts[0].get("username"))
                return result["access_token"]

        # --- Interactive fallback ---
        logger.info("No cached token available — launching interactive login…")
        result = _app.acquire_token_interactive(scopes=SCOPES)

        if not result or "access_token" not in result:
            error_desc = result.get("error_description", "Unknown error") if result else "No response"
            raise RuntimeError(f"Token acquisition failed: {error_desc}")

        _persist_cache()
        logger.info("Interactive login successful.")
        return result["access_token"]


def get_headers() -> dict[str, str]:
    """
    Return HTTP headers ready for Microsoft Graph API requests.

    Returns:
        dict: Headers containing Authorization (Bearer token) and Content-Type.

    Example:
        >>> requests.get("https://graph.microsoft.com/v1.0/me", headers=get_headers())
    """
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }


def verify_connection() -> bool:
    """
    Smoke-test the credentials by calling /me on Graph API.

    Returns:
        bool: True if the token is valid and the Graph API is reachable.
    """
    try:
        resp = requests.get(
            "https://graph.microsoft.com/v1.0/me",
            headers=get_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            display_name = resp.json().get("displayName", "unknown")
            logger.info("Graph API connection verified — signed in as: %s", display_name)
            return True

        logger.warning("Graph API returned %s during verification.", resp.status_code)
        return False

    except requests.RequestException as exc:
        logger.error("Graph API connection check failed: %s", exc)
        return False