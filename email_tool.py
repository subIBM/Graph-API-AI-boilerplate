"""

-------------
All executable tool functions and their Groq-compatible JSON schemas.

Two registries are exported:
    TOOL_REGISTRY : dict[str, callable]
        Maps tool name → Python function.
        Consumed by email_agent.py for dispatch.

    TOOL_SCHEMAS : list[dict]
        OpenAI-style function definitions passed to the Groq API so the
        model knows which tools exist and what arguments they accept.

Sections:
    1. Configuration & shared state
    2. Excel helpers  (load, build master, file utilities)
    3. Excel query tools   (filter, stats, unique values, row count …)
    4. Mail tools          (fetch latest, search by subject)
    5. TOOL_REGISTRY
    6. TOOL_SCHEMAS
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from auth import get_headers


load_dotenv()
logger = logging.getLogger(__name__)


EXCELS_FOLDER   : str       = os.environ.get("EXCELS_FOLDER", "Excels")
FOLDER_NAME     : str       = os.environ["FOLDER_NAME"]          # Outlook subfolder
EXCEL_EXTENSIONS: set[str]  = {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"}

IMPORTANT_COLUMNS: list[str] = [
    "Server Name",
    "Application Name",
    "Patch Window",
    "Reboot Required",
    "Implementation Status",
]

# Sub-folder priority: higher number wins on deduplication
_SUBFOLDER_PRIORITY: dict[str, int] = {
    "Maintenance":        1,
    "Rescheduled":        2,
    "ImplementationStatus": 3,
}

# Ensure directory structure exists
for _sub in _SUBFOLDER_PRIORITY:
    os.makedirs(os.path.join(EXCELS_FOLDER, _sub), exist_ok=True)

# Threading lock — prevents concurrent writes to the master Excel
_excel_lock: threading.Lock = threading.Lock()



def _get_latest_file(folder: str) -> Path | None:
    """Return the most-recently-modified Excel/CSV file in *folder*, or None."""
    candidates = [
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in EXCEL_EXTENSIONS
    ]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def _read_file(path: Path) -> pd.DataFrame:
    """Read an Excel or CSV file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def build_master_excel(default_impl_status: str = "Pending") -> pd.DataFrame | None:
    """
    Merge the latest file from each sub-folder into a single master Excel.

    Deduplication:
        Rows are deduplicated on 'Server Name'.
        The sub-folder with the highest priority wins (ImplementationStatus > Rescheduled > Maintenance).

    The resulting file is written atomically (temp-file + rename) to avoid
    partial reads if another thread loads while we are writing.

    Args:
        default_impl_status: Value to fill when 'Implementation Status' is absent.

    Returns:
        The merged DataFrame, or None if no source files exist.
    """
    dfs: list[pd.DataFrame] = []

    for folder_name, priority in _SUBFOLDER_PRIORITY.items():
        folder_path = os.path.join(EXCELS_FOLDER, folder_name)
        latest_file = _get_latest_file(folder_path)

        if not latest_file:
            logger.debug("No file found in %s — skipping.", folder_path)
            continue

        try:
            df = _read_file(latest_file)
            df.columns = df.columns.str.strip()
            df["_source_folder"] = folder_name
            df["_source_file"]   = latest_file.name
            df["_priority"]      = priority
            dfs.append(df)
            logger.debug("Loaded %d rows from %s", len(df), latest_file)
        except Exception as exc:
            logger.error("Failed to read %s: %s", latest_file, exc)

    if not dfs:
        logger.warning("build_master_excel: no source files found — nothing to merge.")
        return None

    master_df = pd.concat(dfs, ignore_index=True)

    # Ensure required columns exist
    for col in IMPORTANT_COLUMNS:
        if col not in master_df.columns:
            master_df[col] = None

    # Deduplicate: keep the row with the highest priority
    master_df.sort_values("_priority", inplace=True)
    master_df.drop_duplicates(subset=["Server Name"], keep="last", inplace=True)
    master_df["Implementation Status"].fillna(default_impl_status, inplace=True)

    # Drop internal helper columns before saving
    master_df.drop(columns=["_priority"], inplace=True)

    # Atomic write: write to a temp file then rename
    master_path = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")
    # tmp_path    = master_path + ".tmp"
    with _excel_lock:
        master_df.to_excel(master_path, index=False)
        print(f"Master Excel updated: {master_path}")
 

    # with _excel_lock:
    #     master_df.to_excel(tmp_path, index=False)
    #     os.replace(tmp_path, master_path)

    logger.info("Master Excel rebuilt — %d unique servers → %s", len(master_df), master_path)
    return master_df


def load_excel() -> pd.DataFrame | None:
    """
    Load the master Excel.  If it does not exist yet, build it first.

    Returns:
        DataFrame or None if no source data is available at all.
    """
    master_path = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

    if not os.path.exists(master_path):
        logger.info("Master Excel not found — building now…")
        return build_master_excel()

    try:
        with _excel_lock:
            df = pd.read_excel(master_path)
        logger.debug("Master Excel loaded — %d rows.", len(df))
        return df
    except Exception as exc:
        logger.error("Failed to load master Excel: %s", exc)
        return None


def delete_stale_files(days: int = 7) -> int:
    """
    Delete files older than *days* from the Excels folder (non-recursive).

    Returns:
        Number of files deleted.
    """
    cutoff   = datetime.now() - timedelta(days=days)
    deleted  = 0

    for file_path in Path(EXCELS_FOLDER).iterdir():
        if file_path.is_file():
            modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            if modified < cutoff:
                file_path.unlink()
                logger.info("Deleted stale file: %s", file_path.name)
                deleted += 1

    return deleted




def filter_by_application_name(keyword: str) -> str:
    """
    Filter rows where 'Application Name' contains *keyword* (case-insensitive).

    Args:
        keyword: Partial or full application name to search for.

    Returns:
        JSON string with count and matching rows.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    mask     = df["Application Name"].str.contains(re.escape(keyword), case=False, na=False)
    filtered = df[mask][IMPORTANT_COLUMNS]

    return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


def get_column_names() -> str:
    """
    Return all column names present in the master Excel.

    Returns:
        JSON string listing column names.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    return json.dumps({"columns": list(df.columns)})


def get_summary_stats(column_name: str) -> str:
    """
    Return descriptive statistics for a numeric column.

    Args:
        column_name: Name of the column to describe.

    Returns:
        JSON string of pandas describe() output.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})

    return json.dumps(df[column_name].describe().to_dict())


def get_unique_values(column_name: str) -> str:
    """
    Return all unique non-null values in a column.

    Args:
        column_name: Column to inspect.

    Returns:
        JSON string with column name and unique values.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})

    unique_vals = df[column_name].dropna().unique().tolist()
    return json.dumps({"column": column_name, "unique_values": unique_vals})


def get_row_count() -> str:
    """
    Return the total number of server rows in the master Excel.

    Returns:
        JSON string with the row count.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    return json.dumps({"total_rows": len(df)})


def filter_by_column_value(column_name: str, value: str) -> str:
    """
    Filter rows where *column_name* contains *value* (case-insensitive).

    Args:
        column_name: Column to filter on.
        value:       Value to search for.

    Returns:
        JSON string with count and matching rows (important columns only).
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})

    mask     = df[column_name].astype(str).str.contains(re.escape(value), case=False, na=False)
    filtered = df[mask]

    # Return only important columns that actually exist in the DataFrame
    cols = [c for c in IMPORTANT_COLUMNS if c in filtered.columns]
    return json.dumps({"count": len(filtered), "results": filtered[cols].to_dict(orient="records")})


def get_all_rows() -> str:
    """
    Return every row in the master Excel (important columns only, max 200 rows).

    Returns:
        JSON string with count and rows.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    cols     = [c for c in IMPORTANT_COLUMNS if c in df.columns]
    limited  = df[cols].head(200)

    return json.dumps({"count": len(df), "results": limited.to_dict(orient="records")})


def get_lyric_servers() -> str:
    """
    Return all servers whose Application Name contains 'lyric'.

    Returns:
        JSON string with count and server details (important columns).
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    mask     = df["Application Name"].str.contains("lyric", case=False, na=False)
    filtered = df[mask][IMPORTANT_COLUMNS].head(50)

    return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


def lyric_summary() -> str:
    """
    Aggregate summary for all lyric application servers.

    Returns:
        JSON with total count, reboot distribution, and unique patch windows.
    """
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    lyric = df[df["Application Name"].str.contains("lyric", case=False, na=False)]

    summary = {
        "total_servers":    len(lyric),
        "reboot_required":  lyric["Reboot Required"].value_counts().to_dict(),
        "patch_windows":    lyric["Patch Window"].dropna().unique().tolist(),
    }

    return json.dumps(summary)



def _resolve_folder_id(folder_name: str) -> str | None:
    """
    Resolve an Outlook subfolder name under Inbox to its Graph API folder ID.

    Args:
        folder_name: Display name of the subfolder (e.g. 'Enterprise Patching').

    Returns:
        The folder ID string, or None if not found.
    """
    url      = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/childFolders"
    response = requests.get(url, headers=get_headers(), timeout=15)
    response.raise_for_status()

    for folder in response.json().get("value", []):
        if folder["displayName"] == folder_name:
            return folder["id"]

    return None


def _save_attachment(att: dict, subject: str) -> str | None:
    """
    Decode and save an email attachment to the correct sub-folder.

    The destination sub-folder is determined by keywords in the email subject:
        'Maintenance Notification' → Maintenance/
        'Reschedule Maintenance'   → Rescheduled/
        'Implementation Status'    → ImplementationStatus/

    Args:
        att:     Attachment dict from the Graph API (must contain 'name' and 'contentBytes').
        subject: Email subject string used to route the file.

    Returns:
        The saved file path as a string, or None if the subject did not match
        any known category or the file extension was not an Excel type.
    """
    file_name = att.get("name", "")
    ext       = Path(file_name).suffix.lower()

    if ext not in EXCEL_EXTENSIONS:
        logger.debug("Skipping non-Excel attachment: %s", file_name)
        return None

    # Route by subject keyword
    if "Maintenance Notification" in subject:
        sub_folder = "Maintenance"
        save_name  = "maintenance_latest.xlsx"
    elif "Reschedule Maintenance" in subject:
        sub_folder = "Rescheduled"
        save_name  = "rescheduled_latest.xlsx"
    elif "Implementation Status" in subject:
        sub_folder = "ImplementationStatus"
        save_name  = "implementation_latest.xlsx"
    else:
        logger.debug("Subject '%s' did not match any routing rule — skipping.", subject)
        return None

    dest_folder = os.path.join(EXCELS_FOLDER, sub_folder)
    os.makedirs(dest_folder, exist_ok=True)
    save_path = os.path.join(dest_folder, save_name)

    try:
        file_data = base64.b64decode(att["contentBytes"])
        # Atomic write
        tmp_path  = save_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(file_data)
        os.replace(tmp_path, save_path)
        logger.info("Attachment saved: %s", save_path)
        return save_path
    except Exception as exc:
        logger.error("Failed to save attachment '%s': %s", file_name, exc)
        return None


def get_latest_mail(folder_name: str = "") -> str:
    """
    Fetch the most recent email from the monitored folder.

    If the email subject matches a known patching category and contains
    Excel attachments, those attachments are automatically saved and the
    master Excel is rebuilt.

    Args:
        folder_name: Override the default FOLDER_NAME from .env (optional).

    Returns:
        JSON string with message metadata and a list of saved attachment paths.
    """
    target = folder_name or FOLDER_NAME

    try:
        folder_id = _resolve_folder_id(target)
        if not folder_id:
            return json.dumps({"error": f"Folder '{target}' not found in Inbox."})

        msgs_url  = (
            f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder_id}/messages"
            f"?$top=1&$orderby=receivedDateTime desc"
        )
        msgs_resp = requests.get(msgs_url, headers=get_headers(), timeout=15)
        msgs_resp.raise_for_status()
        messages  = msgs_resp.json().get("value", [])

        if not messages:
            return json.dumps({"error": "No messages found in folder."})

        mail       = messages[0]
        message_id = mail["id"]
        subject    = mail.get("subject", "")
        sender     = mail["from"]["emailAddress"]["address"]
        body       = mail.get("bodyPreview", "")
        received   = mail.get("receivedDateTime", "")

        attachments_saved: list[str] = []

        # Download attachments only when subject matches a known category
        patching_keywords = [
            "Maintenance Notification",
            "Reschedule Maintenance",
            "Implementation Status",
        ]
        is_patching_mail = any(kw in subject for kw in patching_keywords)

        if is_patching_mail and mail.get("hasAttachments"):
            att_url  = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"
            att_resp = requests.get(att_url, headers=get_headers(), timeout=30)
            att_resp.raise_for_status()

            for att in att_resp.json().get("value", []):
                saved_path = _save_attachment(att, subject)
                if saved_path:
                    attachments_saved.append(saved_path)

        # Rebuild master only if we actually saved something
        if attachments_saved:
            build_master_excel()

        return json.dumps({
            "message_id":        message_id,
            "subject":           subject,
            "from":              sender,
            "received":          received,
            "body_preview":      body,
            "attachments_saved": attachments_saved,
        })

    except requests.RequestException as exc:
        logger.error("get_latest_mail failed: %s", exc)
        return json.dumps({"error": str(exc)})


def search_mails_by_subject(keyword: str) -> str:
    """
    Search emails in the monitored folder whose subject contains *keyword*.

    Args:
        keyword: Case-insensitive substring to match against email subjects.

    Returns:
        JSON string with matching email summaries (up to 10).
    """
    try:
        folder_id = _resolve_folder_id(FOLDER_NAME)
        if not folder_id:
            return json.dumps({"error": f"Folder '{FOLDER_NAME}' not found."})

        msgs_url = (
            f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder_id}/messages"
            f"?$filter=contains(subject,'{keyword}')"
            f"&$select=subject,from,receivedDateTime,hasAttachments,bodyPreview"
            f"&$top=10&$orderby=receivedDateTime desc"
        )
        resp = requests.get(msgs_url, headers=get_headers(), timeout=15)
        resp.raise_for_status()

        results = [
            {
                "subject":      m.get("subject"),
                "from":         m["from"]["emailAddress"]["address"],
                "received":     m.get("receivedDateTime"),
                "body_preview": m.get("bodyPreview", "")[:200],
            }
            for m in resp.json().get("value", [])
        ]

        return json.dumps({"keyword": keyword, "count": len(results), "emails": results})

    except requests.RequestException as exc:
        logger.error("search_mails_by_subject failed: %s", exc)
        return json.dumps({"error": str(exc)})




TOOL_REGISTRY: dict[str, callable] = {
    # Mail
    "get_latest_mail":         get_latest_mail,
    "search_mails_by_subject": search_mails_by_subject,
    # Excel — query
    "filter_by_application_name": filter_by_application_name,
    "get_column_names":           get_column_names,
    "get_summary_stats":          get_summary_stats,
    "get_unique_values":          get_unique_values,
    "get_row_count":              get_row_count,
    "filter_by_column_value":     filter_by_column_value,
    "get_all_rows":               get_all_rows,
    "get_lyric_servers":          get_lyric_servers,
    "lyric_summary":              lyric_summary,
}




TOOL_SCHEMAS: list[dict] = [
    # ---- Mail tools --------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name":        "get_latest_mail",
            "description": (
                "Fetch the single most recent email from the monitored inbox folder. "
                "Returns subject, sender, received time, body preview, and paths of any "
                "Excel attachments that were automatically saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_name": {
                        "type":        "string",
                        "description": "Optional: override the default monitored folder name.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "search_mails_by_subject",
            "description": "Search emails in the monitored folder by a subject keyword (up to 10 results).",
            "parameters": {
                "type":     "object",
                "properties": {
                    "keyword": {
                        "type":        "string",
                        "description": "Keyword to search for in email subjects.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    # ---- Excel query tools -------------------------------------------------
    {
        "type": "function",
        "function": {
            "name":        "filter_by_application_name",
            "description": "Filter server rows where Application Name contains a keyword.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "keyword": {
                        "type":        "string",
                        "description": "Partial or full application name to search for.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_column_names",
            "description": "Return all column names in the master patch Excel file.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_summary_stats",
            "description": "Return descriptive statistics for a numeric column in the Excel.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "column_name": {
                        "type":        "string",
                        "description": "Name of the column to describe.",
                    }
                },
                "required": ["column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_unique_values",
            "description": "Return all unique non-null values in a given column.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "column_name": {
                        "type":        "string",
                        "description": "Column to retrieve unique values from.",
                    }
                },
                "required": ["column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_row_count",
            "description": "Return the total number of server entries in the master Excel.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "filter_by_column_value",
            "description": "Filter rows where a specific column contains a given value (case-insensitive).",
            "parameters": {
                "type":       "object",
                "properties": {
                    "column_name": {
                        "type":        "string",
                        "description": "Column to filter on.",
                    },
                    "value": {
                        "type":        "string",
                        "description": "Value to search for within that column.",
                    },
                },
                "required": ["column_name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_all_rows",
            "description": (
                "Return all server rows from the master Excel (up to 200, important columns only). "
                "Use when the user wants a full list without any specific filter."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_lyric_servers",
            "description": "Return all servers belonging to the Lyric application (up to 50 rows).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "lyric_summary",
            "description": (
                "Return an aggregate summary for Lyric application servers: "
                "total count, reboot-required distribution, and unique patch windows."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]