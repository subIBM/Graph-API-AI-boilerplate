"""
alert_tool.py
-------------
All executable tool functions and their Groq-compatible JSON schemas
for the Alert Agent.
"""

from __future__ import annotations

import msal
import json
import logging
import os
import pickle
import re
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXCELS_FOLDER: str = os.getenv("EXCELS_FOLDER", "Excels")
MASTER_PATH: str   = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

ALERT_RECIPIENT: str = os.getenv("ALERT_RECIPIENT_EMAIL", "")
ALERT_SENDER: str    = os.getenv("ALERT_SENDER_EMAIL", "")

GRAPH_CLIENT_ID:     str = os.getenv("GRAPH_CLIENT_ID", "")
GRAPH_CLIENT_SECRET: str = os.getenv("GRAPH_CLIENT_SECRET", "")
GRAPH_TENANT_ID:     str = os.getenv("GRAPH_TENANT_ID", "")

# ---------------------------------------------------------------------------
# Graph authentication — Interactive Browser (delegated, one-time login)
# ---------------------------------------------------------------------------

TOKEN_CACHE_FILE = "graph_token_cache.pkl"


def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, "rb") as f:
            cache.deserialize(pickle.load(f))
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        with open(TOKEN_CACHE_FILE, "wb") as f:
            pickle.dump(cache.serialize(), f)


def _get_graph_token() -> str:
    """
    Acquire Microsoft Graph token using Interactive Browser Flow.
    Token is cached to disk — browser login only appears once.
    """
    cache = _load_cache()

    app = msal.PublicClientApplication(
        client_id=GRAPH_CLIENT_ID,
        authority="https://login.microsoftonline.com/common",
        token_cache=cache,
    )

    SCOPES = ["https://graph.microsoft.com/Mail.Send"]

    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache)
            return result["access_token"]

    result = app.acquire_token_interactive(scopes=SCOPES)

    if "error" in result:
        raise RuntimeError(
            f"Token error: {result['error']} — {result.get('error_description')}"
        )
    if "access_token" not in result:
        raise RuntimeError(f"Graph token acquisition failed: {result}")

    _save_cache(cache)
    return result["access_token"]


# ---------------------------------------------------------------------------
# Patch-window parser — returns end datetime only
# ---------------------------------------------------------------------------

def _parse_patch_window_end(
    patch_window: str, reference_date: datetime | None = None
) -> datetime | None:

    if not patch_window or pd.isna(patch_window):
        return None

    pw  = str(patch_window).strip()
    ref = reference_date or datetime.now()

    m = re.match(
        r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*[-–to]+\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
        pw, re.IGNORECASE,
    )
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
        except ValueError:
            pass

    m = re.match(
        r"\d{1,2}-[A-Za-z]+-\d{4}\s+\d{2}:\d{2}\s*(?:to|-)\s*(\d{1,2}-[A-Za-z]+-\d{4}\s+\d{2}:\d{2})",
        pw, re.IGNORECASE,
    )
    if m:
        try:
            return datetime.strptime(m.group(1), "%d-%b-%Y %H:%M")
        except ValueError:
            pass

    m = re.match(r"\d{2}:\d{2}\s*[-–]\s*(\d{2}:\d{2})", pw)
    if m:
        try:
            end_time = m.group(1)
            return ref.replace(
                hour=int(end_time[:2]), minute=int(end_time[3:]),
                second=0, microsecond=0,
            )
        except ValueError:
            pass

    m = re.match(
        r"([A-Za-z]+)-(\d{2}:\d{2})(?::\d{2})?\s*to\s*(\d{2}:\d{2})(?::\d{2})?",
        pw, re.IGNORECASE,
    )
    if m:
        try:
            day_name  = m.group(1).lower()
            start_str = m.group(2)
            end_str   = m.group(3)

            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            }
            if day_name not in day_map:
                return None

            target_weekday = day_map[day_name]
            days_diff      = (ref.weekday() - target_weekday) % 7
            window_date    = ref - timedelta(days=days_diff)

            start = window_date.replace(
                hour=int(start_str[:2]), minute=int(start_str[3:]),
                second=0, microsecond=0,
            )
            end = window_date.replace(
                hour=int(end_str[:2]), minute=int(end_str[3:]),
                second=0, microsecond=0,
            )
            if end < start:
                end += timedelta(days=1)
            return end

        except ValueError:
            pass

    return None


def _format_patch_window(patch_window: str) -> str:
    """
    Convert raw patch window string into a clean readable format.
    e.g. 'Sunday-04:00:00 to 09:00:00'  →  'Sunday 04:00 – 09:00'
    """
    if not patch_window or pd.isna(patch_window):
        return str(patch_window)

    pw = str(patch_window).strip()

    m = re.match(
        r"([A-Za-z]+)-(\d{2}:\d{2})(?::\d{2})?\s*to\s*(\d{2}:\d{2})(?::\d{2})?",
        pw, re.IGNORECASE,
    )
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)} – {m.group(3)}"

    m = re.match(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*[-–to]+\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
        pw, re.IGNORECASE,
    )
    if m:
        return f"{m.group(1)} – {m.group(2)}"

    return pw


def _is_empty(value) -> bool:
    """Return True if value is None, NaN, or blank string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in ("", "nan", "none")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_lyric_alert_summary() -> str:
    """
    Read master Excel and return three categorised lists for Lyric servers:
      - unreachable : Error is set (connection failed, no boot time)
      - failed      : Validation Status is 'Failed' (boot outside window)
      - pending     : Implementation Status is 'Pending' (not yet patched)

    alert_required is True if ANY of the three lists is non-empty.
    """
    if not os.path.exists(MASTER_PATH):
        return json.dumps({"error": f"Master Excel not found at {MASTER_PATH}"})

    df = pd.read_excel(MASTER_PATH)
    df.columns = df.columns.str.strip()

    # Only Lyric servers
    lyric_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]

    now         = datetime.now()
    window_ends = []

    for _, row in lyric_df.iterrows():
        end_dt = _parse_patch_window_end(row.get("Patch Window"), reference_date=now)
        if end_dt:
            window_ends.append(end_dt)

    latest_end = max(window_ends) if window_ends else None

    unreachable: list[dict] = []
    failed:      list[dict] = []
    pending:     list[dict] = []

    for _, row in lyric_df.iterrows():
        error             = row.get("Error")
        val_status        = str(row.get("Application Team Validation Status", "")).strip().lower()
        impl_status       = str(row.get("Implementation Status", "")).strip().lower()
        boot_time         = row.get("Boot Time")
        patch_window_fmt  = _format_patch_window(str(row.get("Patch Window", "")))
        reboot_required   = str(row.get("Reboot Required", "")).strip()
        server_name       = str(row.get("Server Name", "")).strip()

        has_error = not _is_empty(error)

        if has_error:
            # Could not connect — no boot time
            unreachable.append({
                "server_name":     server_name,
                "patch_window":    patch_window_fmt,
                "reboot_required": reboot_required,
            })

        elif val_status == "failed":
            # Boot time retrieved but falls outside patch window
            boot_str = str(boot_time).strip() if not _is_empty(boot_time) else "N/A"
            failed.append({
                "server_name":     server_name,
                "patch_window":    patch_window_fmt,
                "boot_time":       boot_str,
                "reboot_required": reboot_required,
            })

        elif impl_status == "pending":
            # Not yet patched — ask patching team for status update
            pending.append({
                "server_name":     server_name,
                "patch_window":    patch_window_fmt,
                "reboot_required": reboot_required,
            })

    return json.dumps({
        "alert_required":      bool(unreachable or failed or pending),
        "total_lyric_servers": len(lyric_df),
        "unreachable":         unreachable,
        "failed":              failed,
        "pending":             pending,
        "latest_window_end":   latest_end.isoformat() if latest_end else None,
    })


# ---------------------------------------------------------------------------
# Email sender via Microsoft Graph
# ---------------------------------------------------------------------------

def send_alert_email(subject: str, html_body: str) -> str:

    if not ALERT_RECIPIENT:
        return json.dumps({"error": "ALERT_RECIPIENT_EMAIL not configured"})

    try:
        token = _get_graph_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

        payload = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content":     html_body,
                },
                "toRecipients": [
                    {"emailAddress": {"address": ALERT_RECIPIENT}}
                ],
            },
            "saveToSentItems": True,
        }

        resp = requests.post(
            "https://graph.microsoft.com/v1.0/me/sendMail",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if resp.status_code == 202:
            logger.info("Alert email sent to %s", ALERT_RECIPIENT)
            return json.dumps({"status": "sent", "recipient": ALERT_RECIPIENT})

        logger.error("Graph sendMail failed: %s %s", resp.status_code, resp.text)
        return json.dumps({
            "error":  f"sendMail returned {resp.status_code}",
            "detail": resp.text,
        })

    except Exception as exc:
        logger.error("send_alert_email failed: %s", exc)
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# TOOL registries
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "get_lyric_alert_summary": get_lyric_alert_summary,
    "send_alert_email":        send_alert_email,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_lyric_alert_summary",
            "description": (
                "Read the master Excel and return alert summary for Lyric servers. "
                "Returns three lists — 'unreachable' (connection errors), "
                "'failed' (boot time outside patch window), and "
                "'pending' (Implementation Status is Pending, not yet patched). "
                "alert_required is true if any list is non-empty."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "send_alert_email",
            "description": "Send HTML alert email to the patching team.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "subject":   {"type": "string"},
                    "html_body": {"type": "string"},
                },
                "required": ["subject", "html_body"],
            },
        },
    },
]
