"""
validation_tool.py
------------------
All executable tool functions and their Groq-compatible JSON schemas.

Two registries are exported:
    TOOL_FUNCTIONS : dict[str, callable]
        Maps tool name → Python function.
        Consumed by validation_agent.py for dispatch.

    TOOL_SCHEMAS : list[dict]
        OpenAI-style function definitions passed to the Groq API so the
        model knows which tools exist and what arguments they accept.

Sections:
    1. Configuration & imports
    2. Patch-window parser helper
    3. Tool functions
    4. TOOL_FUNCTIONS registry
    5. TOOL_SCHEMAS declarations

Key behaviours
--------------
* update_boot_time_in_excel
    - If the server already has a Boot Time recorded → SKIP (no overwrite).
    - If the server exists but Boot Time is empty/null → write the new value.
    - If the server does NOT exist in the master Excel → append a new row.
    - Handles multiple implementation-status e-mails safely: each mail covers
      a different subset of servers; data from earlier mails is never lost.
    - Duplicate server across mails: the LATEST value wins because the agent
      processes each mail in order and the skip-if-exists guard is bypassed
      only when the cell is genuinely empty.

* validate_boot_within_patch_window
    - If Application Team Validation Status is already set → SKIP.
    - Otherwise compute and write the status as normal.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

import threading
import random

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

EXCELS_FOLDER: str = os.getenv("EXCELS_FOLDER", "Excels")
MASTER_PATH: str   = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

WINRM_USER:      str = os.getenv("WINRM_USER")
WINRM_PASSWORD:  str = os.getenv("WINRM_PASSWORD")
WINRM_PORT:      int = int(os.getenv("WINRM_PORT", "5985"))
WINRM_TRANSPORT: str = os.getenv("WINRM_TRANSPORT", "ntlm")

# Threading lock — prevents concurrent writes to the master Excel
_excel_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# 2. Patch-window parser helper
# ---------------------------------------------------------------------------

def _parse_patch_window(patch_window: str, reference_date: datetime | None = None):
    """
    Tries to parse common patch-window formats, e.g.:
      '2025-06-14 22:00 - 2025-06-15 02:00'
      '14-Jun-2025 22:00 to 15-Jun-2025 02:00'
      '22:00 - 02:00'  (time-only; uses reference_date or today)
      'Sunday-03:00:00 to 07:00:00'
    Returns (start_dt, end_dt) as datetime objects, or (None, None) on failure.
    """
    if not patch_window or pd.isna(patch_window):
        return None, None

    pw  = str(patch_window).strip()
    ref = reference_date or datetime.now()

    # Pattern 1: full datetime range  "YYYY-MM-DD HH:MM - YYYY-MM-DD HH:MM"
    m = re.match(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*[-–to]+\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",
        pw, re.IGNORECASE
    )
    if m:
        try:
            start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            end   = datetime.strptime(m.group(2), "%Y-%m-%d %H:%M")
            return start, end
        except ValueError:
            pass

    # Pattern 2: "DD-Mon-YYYY HH:MM to DD-Mon-YYYY HH:MM"
    m = re.match(
        r"(\d{1,2}-[A-Za-z]+-\d{4}\s+\d{2}:\d{2})\s*(?:to|-)\s*(\d{1,2}-[A-Za-z]+-\d{4}\s+\d{2}:\d{2})",
        pw, re.IGNORECASE
    )
    if m:
        try:
            start = datetime.strptime(m.group(1), "%d-%b-%Y %H:%M")
            end   = datetime.strptime(m.group(2), "%d-%b-%Y %H:%M")
            return start, end
        except ValueError:
            pass

    # Pattern 3: time-only "HH:MM - HH:MM"
    m = re.match(r"(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})", pw)
    if m:
        try:
            start = ref.replace(
                hour=int(m.group(1)[:2]), minute=int(m.group(1)[3:]),
                second=0, microsecond=0
            )
            end = ref.replace(
                hour=int(m.group(2)[:2]), minute=int(m.group(2)[3:]),
                second=0, microsecond=0
            )
            # Handle overnight windows
            if end < start:
                end = end.replace(day=end.day + 1)
            return start, end
        except ValueError:
            pass

    # Pattern 4: "Sunday-03:00:00 to 07:00:00"
    m = re.match(
        r"([A-Za-z]+)-(\d{2}:\d{2})(?::\d{2})?\s*to\s*(\d{2}:\d{2})(?::\d{2})?",
        pw, re.IGNORECASE
    )
    if m:
        try:
            day_name   = m.group(1).strip().lower()
            start_time = m.group(2)
            end_time   = m.group(3)

            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            }

            if day_name not in day_map:
                return None, None

            target_weekday = day_map[day_name]
            boot_weekday   = ref.weekday()

            # Find how many days back the last occurrence of target day was
            days_diff = (target_weekday - boot_weekday) % 7
            # if days_diff == 0:
            #     days_diff = 7
            window_date = ref + timedelta(days=days_diff)

            start = window_date.replace(
                hour=int(start_time[:2]), minute=int(start_time[3:]),
                second=0, microsecond=0
            )
            end = window_date.replace(
                hour=int(end_time[:2]), minute=int(end_time[3:]),
                second=0, microsecond=0
            )

            # Handle overnight window (e.g. 22:00 to 02:00)
            if end < start:
                end = end + timedelta(days=1)

            return start, end
        except ValueError:
            pass

    return None, None


# ---------------------------------------------------------------------------
# 3. Helper utilities
# ---------------------------------------------------------------------------

def _cell_is_empty(value) -> bool:
    """Return True if a cell value is None, NaN, or an empty/whitespace string."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _ensure_columns(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = None  # scalar None — pandas broadcasts correctly across all rows
        # Always cast to object so strings can be written
        df[col] = df[col].astype(object)
    return df


def _server_mask(df: pd.DataFrame, server_name: str) -> pd.Series:
    """Return a boolean mask for rows whose Server Name matches *server_name* (case-insensitive, stripped)."""
    return (
        df["Server Name"].astype(str).str.strip().str.lower()
        == server_name.strip().lower()
    )


# ---------------------------------------------------------------------------
# 4. Tool functions
# ---------------------------------------------------------------------------

def get_lyric_servers_ready_for_validation() -> str:
    """
    Return only lyric servers where Implementation Status is 'Completed'.
    These are the only servers that should have boot time fetched and validated.
    """
    if not os.path.exists(MASTER_PATH):
        return json.dumps({"error": f"Master Excel not found at {MASTER_PATH}"})

    try:
        df = pd.read_excel(MASTER_PATH)
        df.columns = df.columns.str.strip()

        lyric_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]

        ready_df = lyric_df[
            lyric_df["Implementation Status"].str.strip().str.lower() == "completed"
        ]

        cols = [
            c for c in [
                "Server Name", "Application Name", "Patch Window",
                "Reboot Required", "Implementation Status",
            ]
            if c in ready_df.columns
        ]

        return json.dumps({
            "count":   len(ready_df),
            "servers": ready_df[cols].to_dict(orient="records"),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_server_boot_time(server_name: str) -> str:
    """
    Connect to *server_name* via WinRM and retrieve the last boot time.
    Simulates random success/failure per server call.
    """
    connection_success = random.choice([True, False])

    if connection_success:
        return json.dumps({
            "server":    server_name,
            "boot_time": "2026-03-12 15:42:05",
            "error":     None,
        })
    else:
        return json.dumps({
            "server":    server_name,
            "boot_time": None,
            "error":     "Could not connect to server",
        })


def update_boot_time_in_excel(
    server_name: str,
    boot_time: str | None = None,
    error: str | None = None,
) -> str:
    """
    Write *boot_time* (and *error*) into the master Excel for *server_name*.

    Rules
    -----
    1. If the server row already has a non-empty Boot Time  → SKIP entirely.
       This protects data written by a previous implementation-status e-mail.
    2. If the server row exists but Boot Time is empty/null → write the new value.
    3. If the server is NOT found in the master Excel at all → append a new row
       so that data from every e-mail is captured without losing earlier entries.

    Duplicate servers across e-mails
    ---------------------------------
    Servers should not appear in more than one e-mail, but if they do the LATEST
    value wins: the agent processes e-mails in chronological order; on the second
    occurrence the Boot Time cell will already be populated (from the first mail),
    so the skip guard fires.  To force an overwrite with the latest value the
    caller should clear the cell first, or the orchestrator should pre-clear it
    before processing a newer mail for the same server.
    """
    if not os.path.exists(MASTER_PATH):
        return json.dumps({"error": f"Master Excel not found at {MASTER_PATH}"})

    try:
        with _excel_lock:
            df = pd.read_excel(MASTER_PATH)
            df.columns = df.columns.str.strip()
            df = _ensure_columns(df, "Boot Time", "Error")

            # Force object dtype so string error messages can be written without rejection
            df["Boot Time"] = df["Boot Time"].astype(object)
            df["Error"]     = df["Error"].astype(object)

            mask = _server_mask(df, server_name)

            # ----------------------------------------------------------------
            # Case A: server NOT in Excel → append a brand-new row
            # ----------------------------------------------------------------
            if not mask.any():
                new_row = {col: None for col in df.columns}
                new_row["Server Name"] = server_name
                new_row["Boot Time"]   = boot_time
                new_row["Error"]       = error
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_excel(MASTER_PATH, index=False)
                return json.dumps({
                    "status":    "added",
                    "server":    server_name,
                    "boot_time": boot_time,
                    "error":     error,
                })

            # ----------------------------------------------------------------
            # Case B: server found — check whether Boot Time is already set
            # ----------------------------------------------------------------
            existing_boot = df.loc[mask, "Boot Time"].iloc[0]
            existing_error = df.loc[mask, "Error"].iloc[0]

            if not _cell_is_empty(existing_boot) or not _cell_is_empty(existing_error):
                # Either a boot time or an error was already recorded → do not overwrite
                return json.dumps({
                    "status": "skipped",
                    "server": server_name,
                    "reason": (
                        f"Already recorded — Boot Time: '{existing_boot}', Error: '{existing_error}'. "
                        "No overwrite performed — data from previous implementation mail is preserved."
                    ),
                })

            # ----------------------------------------------------------------
            # Case C: server found, Boot Time is empty → safe to write
            # ----------------------------------------------------------------
            df.loc[mask, "Boot Time"] = boot_time
            df.loc[mask, "Error"]     = error
            df.to_excel(MASTER_PATH, index=False)

            return json.dumps({
                "status":    "updated",
                "server":    server_name,
                "boot_time": boot_time,
                "error":     error,
            })

    except Exception as e:
        return json.dumps({"error": str(e)})


def validate_boot_within_patch_window(server_name: str) -> str:
    """
    Check whether the server's stored Boot Time falls inside its Patch Window.
    Updates 'Application Team Validation Status' column accordingly:
      - 'Successful'  → boot time is within the patch window
      - 'Failed'      → boot time is outside the patch window
      - 'Unknown'     → missing/unparseable boot time or patch window

    Rules
    -----
    * If Application Team Validation Status is already set (non-empty) → SKIP.
      This ensures data written by a previous e-mail run is never overwritten.
    * If the server is not found → return an error (no row is added here because
      validation depends on a Patch Window that only a pre-existing row can have).
    """
    if not os.path.exists(MASTER_PATH):
        return json.dumps({"error": f"Master Excel not found at {MASTER_PATH}"})

    try:
        with _excel_lock:
            df = pd.read_excel(MASTER_PATH)
            df.columns = df.columns.str.strip()
            df = _ensure_columns(df, "Application Team Validation Status")

            mask = _server_mask(df, server_name)

            if not mask.any():
                return json.dumps({"error": f"Server '{server_name}' not found in Excel."})

            row = df[mask].iloc[0]

            # ----------------------------------------------------------------
            # Skip if validation status already recorded
            # ----------------------------------------------------------------
            existing_status = row.get("Application Team Validation Status")
            if not _cell_is_empty(existing_status):
                return json.dumps({
                    "status": "skipped",
                    "server": server_name,
                    "reason": (
                        f"Validation Status already recorded as '{existing_status}'. "
                        "No overwrite performed — data from previous "
                        "implementation mail is preserved."
                    ),
                })

            # ----------------------------------------------------------------
            # Parse boot time
            # ----------------------------------------------------------------
            boot_time_raw = row.get("Boot Time")
            patch_window  = row.get("Patch Window")

            boot_dt = None
            if not _cell_is_empty(boot_time_raw):
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%b-%Y %H:%M"):
                    try:
                        boot_dt = datetime.strptime(str(boot_time_raw).strip(), fmt)
                        break
                    except ValueError:
                        continue

            if boot_dt is None:
                df.loc[mask, "Application Team Validation Status"] = "Unknown"
                df.to_excel(MASTER_PATH, index=False)
                return json.dumps({
                    "server": server_name,
                    "status": "Unknown",
                    "reason": "Boot time is missing or could not be parsed.",
                })

            # ----------------------------------------------------------------
            # Parse patch window and determine result
            # ----------------------------------------------------------------
            start_dt, end_dt = _parse_patch_window(patch_window, reference_date=boot_dt)

            if start_dt is None:
                df.loc[mask, "Application Team Validation Status"] = "Unknown"
                df.to_excel(MASTER_PATH, index=False)
                return json.dumps({
                    "server": server_name,
                    "status": "Unknown",
                    "reason": f"Patch window '{patch_window}' could not be parsed.",
                })

            within            = start_dt <= boot_dt <= end_dt
            validation_status = "Successful" if within else "Failed"

            df.loc[mask, "Application Team Validation Status"] = validation_status
            df.to_excel(MASTER_PATH, index=False)

            return json.dumps({
                "server":        server_name,
                "boot_time":     str(boot_dt),
                "patch_window":  f"{start_dt} → {end_dt}",
                "within_window": within,
                "status":        validation_status,
            })

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# 5. TOOL_FUNCTIONS registry
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, callable] = {
    "get_lyric_servers_ready_for_validation": get_lyric_servers_ready_for_validation,
    "get_server_boot_time":                   get_server_boot_time,
    "update_boot_time_in_excel":              update_boot_time_in_excel,
    "validate_boot_within_patch_window":      validate_boot_within_patch_window,
}


# ---------------------------------------------------------------------------
# 6. TOOL_SCHEMAS — OpenAI-style, consumed by Groq API
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_lyric_servers_ready_for_validation",
            "description": (
                "Return only lyric application servers where Implementation Status is 'Completed'. "
                "Always use this when fetching boot times or running validation — "
                "servers that are not Completed should be skipped."
            ),
            "parameters": {
                "type":       "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_server_boot_time",
            "description": (
                "Connect to a Windows server via WinRM and retrieve its last "
                "boot time (Win32_OperatingSystem.LastBootUpTime). "
                "Returns the boot time as 'YYYY-MM-DD HH:MM:SS'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type":        "string",
                        "description": "Hostname or IP address of the target Windows server.",
                    }
                },
                "required": ["server_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_boot_time_in_excel",
            "description": (
                "Write boot time and/or error into the master Excel for a server. "
                "SKIPS the update if Boot Time is already recorded (preserves data "
                "from earlier implementation mails). Appends a new row if the server "
                "is not found. Pass boot_time=null and error=<message> when the "
                "WinRM connection failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type":        "string",
                        "description": "Exact server name as it appears in Excel.",
                    },
                    "boot_time": {
                        "anyOf":       [{"type": "string"}, {"type": "null"}],
                        "description": "Boot time string e.g. '2026-03-12 15:42:05', or null if unavailable.",
                    },
                    "error": {
                        "anyOf":       [{"type": "string"}, {"type": "null"}],
                        "description": "Error message if boot time could not be fetched, or null if successful.",
                    },
                },
                "required": ["server_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_boot_within_patch_window",
            "description": (
                "Check whether the server's Boot Time (already stored in Excel) "
                "falls inside its Patch Window. "
                "Sets 'Application Team Validation Status' to 'Successful', 'Failed', or 'Unknown' "
                "and saves the result back to the master Excel. "
                "SKIPS if the status is already recorded (preserves data from earlier mails)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {
                        "type":        "string",
                        "description": "Exact server name as it appears in the Excel.",
                    }
                },
                "required": ["server_name"],
            },
        },
    },
]