# """
# email_tool.py
# -------------
# All executable tool functions and their Groq-compatible JSON schemas.

# Two registries are exported:
#     TOOL_REGISTRY : dict[str, callable]
#         Maps tool name → Python function.
#         Consumed by email_agent.py for dispatch.

#     TOOL_SCHEMAS : list[dict]
#         OpenAI-style function definitions passed to the Groq API so the
#         model knows which tools exist and what arguments they accept.

# Sections:
#     1. Configuration & shared state
#     2. Excel helpers  (load, build master, file utilities)
#     3. Excel query tools   (filter, stats, unique values, row count …)
#     4. Mail tools          (fetch latest, search by subject)
#     5. TOOL_REGISTRY
#     6. TOOL_SCHEMAS

# Key behaviour — Implementation Status mails
# --------------------------------------------
# Each Implementation Status mail covers a DIFFERENT subset of servers.
# To avoid overwriting earlier servers when a new mail arrives we:

#   * Save every Implementation Status attachment with a UNIQUE timestamped
#     filename  (implementation_<YYYYMMDD_HHMMSS>.xlsx)  instead of always
#     writing to 'implementation_latest.xlsx'.  All files accumulate in the
#     ImplementationStatus/ sub-folder.

#   * build_master_excel() reads ALL files in ImplementationStatus/ and
#     merges them together before deduplication, so every batch of servers
#     is always present in the master.

#   * Maintenance and Rescheduled still use a single 'latest' file because
#     those mails replace each other by design (most-recent wins).

#   * The master rebuild NEVER touches the 'Boot Time',
#     'Application Team Validation Status', or 'Error' columns that the
#     Validation Agent has already written.  Those columns are preserved by
#     a merge-with-existing-master step inside build_master_excel().
# """

# from __future__ import annotations

# import hashlib
# import base64
# import json
# import logging
# import os
# import re
# import threading
# from datetime import datetime, timedelta
# from pathlib import Path

# import pandas as pd
# import requests
# from dotenv import load_dotenv

# from auth import get_headers
# from validation_agent import run_agent as run_validation_agent

# load_dotenv()
# logger = logging.getLogger(__name__)


# EXCELS_FOLDER   : str       = os.environ.get("EXCELS_FOLDER", "Excels")
# FOLDER_NAME     : str       = os.environ["FOLDER_NAME"]          # Outlook subfolder
# EXCEL_EXTENSIONS: set[str]  = {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"}

# IMPORTANT_COLUMNS: list[str] = [
#     "Server Name",
#     "Application Name",
#     "Patch Window",
#     "Reboot Required",
#     "Implementation Status",
# ]

# # Columns written exclusively by the Validation Agent.
# # build_master_excel() NEVER overwrites these — it carries them forward
# # from the existing master so previous run data is never lost.
# _VALIDATION_COLUMNS: list[str] = [
#     "Boot Time",
#     "Error",
#     "Application Team Validation Status",
#     "Change Ticket"
# ]

# # Sub-folder priority: higher number wins on deduplication across folders.
# # Within ImplementationStatus itself ALL files are merged (no dedup needed
# # until the combined frame is deduped against Maintenance/Rescheduled).
# _SUBFOLDER_PRIORITY: dict[str, int] = {
#     "Maintenance":          1,
#     "Rescheduled":          2,
#     "ImplementationStatus": 3,
# }

# # Ensure directory structure exists
# for _sub in _SUBFOLDER_PRIORITY:
#     os.makedirs(os.path.join(EXCELS_FOLDER, _sub), exist_ok=True)

# # Threading lock — prevents concurrent writes to the master Excel
# _excel_lock: threading.Lock = threading.Lock()


# # ---------------------------------------------------------------------------
# # Content-based mail dedup
# # ---------------------------------------------------------------------------

# _processed_mail_hashes: set[str] = set()
# _mail_hash_lock: threading.Lock  = threading.Lock()

# # Module-level: maps saved filename → CHG ticket (for master rebuild)
# _pending_chg_tickets: dict[str, str] = {}
# _tickets_lock: threading.Lock = threading.Lock()

# def _make_mail_hash(subject: str, received: str, sender: str) -> str:
#     """
#     Stable content fingerprint for a mail, independent of Graph's message_id.
#     Graph can assign different message_ids to the same physical email across
#     duplicate notifications — this hash collapses them to one identity.
#     """
#     raw = f"{subject}|{received}|{sender}".lower().strip()
#     return hashlib.sha256(raw.encode()).hexdigest()


# # ---------------------------------------------------------------------------
# # Excel helpers
# # ---------------------------------------------------------------------------

# def _get_latest_file(folder: str) -> Path | None:
#     """Return the most-recently-modified Excel/CSV file in *folder*, or None."""
#     candidates = [
#         f for f in Path(folder).iterdir()
#         if f.is_file() and f.suffix.lower() in EXCEL_EXTENSIONS
#     ]
#     return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


# def _get_all_files(folder: str) -> list[Path]:
#     """Return ALL Excel/CSV files in *folder*, sorted oldest → newest."""
#     candidates = [
#         f for f in Path(folder).iterdir()
#         if f.is_file() and f.suffix.lower() in EXCEL_EXTENSIONS
#     ]
#     return sorted(candidates, key=lambda f: f.stat().st_mtime)


# def _read_file(path: Path) -> pd.DataFrame:
#     """Read an Excel or CSV file into a DataFrame."""
#     if path.suffix.lower() == ".csv":
#         return pd.read_csv(path)
#     return pd.read_excel(path)


# def build_master_excel(default_impl_status: str = "Pending") -> pd.DataFrame | None:
#     """
#     Merge source files from all sub-folders into a single master Excel.

#     Implementation Status folder
#     ----------------------------
#     ALL files in ImplementationStatus/ are read and stacked — this is how
#     servers from multiple emails coexist without overwriting each other.

#     Maintenance / Rescheduled folders
#     ----------------------------------
#     Only the latest file is read (most-recent-wins for these mail types).

#     Deduplication
#     -------------
#     Rows are deduplicated on 'Server Name'.
#     Higher-priority sub-folder wins (ImplementationStatus > Rescheduled > Maintenance).
#     Within ImplementationStatus, the LAST occurrence of a server (newest file)
#     wins — handles the rare case of the same server appearing in two mails.

#     Validation data preservation
#     ----------------------------
#     After rebuilding from source files the function loads the existing master
#     (if any) and carries forward Boot Time, Error, and
#     Application Team Validation Status for every server that already has
#     those values.  This ensures the Validation Agent's work is NEVER erased
#     by a subsequent Implementation Status mail arriving.

#     Returns:
#         The merged DataFrame, or None if no source files exist.
#     """
#     dfs: list[pd.DataFrame] = []

#     for folder_name, priority in _SUBFOLDER_PRIORITY.items():
#         folder_path = os.path.join(EXCELS_FOLDER, folder_name)

#         if folder_name == "ImplementationStatus":
#             # ----------------------------------------------------------------
#             # Read ALL implementation status files so every mail's servers
#             # are included.  Sort oldest→newest so that when the same server
#             # appears in two mails the newer file's row wins after drop_duplicates.
#             # ----------------------------------------------------------------
#             files = _get_all_files(folder_path)
#             if not files:
#                 logger.debug("No files found in %s — skipping.", folder_path)
#                 continue

#             for file_path in files:
#                 try:
#                     df = _read_file(file_path)
#                     df.columns = df.columns.str.strip()
#                     df["_source_folder"] = folder_name
#                     df["_source_file"]   = file_path.name
#                     df["_priority"]      = priority
#                     dfs.append(df)
#                     logger.debug("Loaded %d rows from %s", len(df), file_path)
#                 except Exception as exc:
#                     logger.error("Failed to read %s: %s", file_path, exc)

#         else:
#             # Maintenance / Rescheduled — latest file only
#             latest_file = _get_latest_file(folder_path)
#             if not latest_file:
#                 logger.debug("No file found in %s — skipping.", folder_path)
#                 continue
#             try:
#                 df = _read_file(latest_file)
#                 df.columns = df.columns.str.strip()
#                 df["_source_folder"] = folder_name
#                 df["_source_file"]   = latest_file.name
#                 df["_priority"]      = priority
#                 dfs.append(df)
#                 logger.debug("Loaded %d rows from %s", len(df), latest_file)
#             except Exception as exc:
#                 logger.error("Failed to read %s: %s", latest_file, exc)

#     if not dfs:
#         logger.warning("build_master_excel: no source files found — nothing to merge.")
#         return None

#     combined = pd.concat(dfs, ignore_index=True)

#     # ── NEW: stamp Change Ticket column from pending tickets dict ─────────────
#     # Build a lookup: absolute save_path → ticket
#     with _tickets_lock:
#         ticket_snapshot = dict(_pending_chg_tickets)
    
#     if ticket_snapshot:
#         # Build a reverse map: source_file basename → ticket
#         # The combined frame has a _source_file column with the basename
#         basename_to_ticket: dict[str, str] = {}
#         for full_path, ticket in ticket_snapshot.items():
#             basename_to_ticket[Path(full_path).name] = ticket

#         if "_source_file" in combined.columns:
#             combined["Change Ticket"] = combined.apply(
#                 lambda row: basename_to_ticket.get(row["_source_file"],
#                             row.get("Change Ticket") if "Change Ticket" in row.index else None),
#                 axis=1,
#             )

#     # Ensure required columns exist
#     for col in IMPORTANT_COLUMNS:
#         if col not in combined.columns:
#             combined[col] = None

#     # Deduplicate: sort so highest-priority (and newest within same priority)
#     # row ends up last, then keep='last'
#     combined.sort_values(["_priority"], inplace=True)
#     combined.drop_duplicates(subset=["Server Name"], keep="last", inplace=True)

#     combined["Implementation Status"] = combined["Implementation Status"].fillna(default_impl_status)

#     # Drop internal helper columns before saving
#     combined.drop(columns=["_priority"], inplace=True)

#     # ------------------------------------------------------------------
#     # Preserve Validation Agent data from the existing master
#     # ------------------------------------------------------------------
#     # Load the current master (if it exists) and carry forward any
#     # non-empty values in the validation columns.  This means that even
#     # when a brand-new Implementation Status mail triggers a rebuild,
#     # servers whose boot time / validation status was already recorded
#     # keep that data intact.
#     master_path = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

#     if os.path.exists(master_path):
#         try:
#             with _excel_lock:
#                 existing = pd.read_excel(master_path)
#             existing.columns = existing.columns.str.strip()

#             # Build a lookup: server_name (lower) → {col: value}
#             val_cols_present = [c for c in _VALIDATION_COLUMNS if c in existing.columns]

#             if val_cols_present:
#                 existing["_key"] = existing["Server Name"].astype(str).str.strip().str.lower()
#                 val_lookup = (
#                     existing[["_key"] + val_cols_present]
#                     .drop_duplicates(subset=["_key"], keep="last")
#                     .set_index("_key")
#                 )

#                 combined["_key"] = combined["Server Name"].astype(str).str.strip().str.lower()

#                 # Ensure validation columns exist in combined
#                 for col in val_cols_present:
#                     if col not in combined.columns:
#                         combined[col] = None

#                 # For each validation column, fill from existing master
#                 # only where the new combined frame has an empty cell
#                 for col in val_cols_present:
#                     def _carry_forward(row, col=col):
#                         current = row[col]
#                         # If already populated in combined, keep it
#                         if current is not None and not (
#                             isinstance(current, float) and pd.isna(current)
#                         ) and str(current).strip() != "":
#                             return current
#                         # Otherwise look up from existing master
#                         key = row["_key"]
#                         if key in val_lookup.index:
#                             existing_val = val_lookup.at[key, col]
#                             if existing_val is not None and not (
#                                 isinstance(existing_val, float) and pd.isna(existing_val)
#                             ) and str(existing_val).strip() != "":
#                                 return existing_val
#                         return current

#                     combined[col] = combined.apply(_carry_forward, axis=1)

#                 combined.drop(columns=["_key"], inplace=True)
#                 logger.info(
#                     "Carried forward validation data from existing master for columns: %s",
#                     val_cols_present,
#                 )
#         except Exception as exc:
#             logger.warning(
#                 "Could not carry forward validation data from existing master: %s", exc
#             )

#     # Write master atomically
#     with _excel_lock:
#         combined.to_excel(master_path, index=False)
#         logger.info(
#             "Master Excel rebuilt — %d unique servers → %s", len(combined), master_path
#         )
#         print(f"Master Excel updated: {master_path}")

#     return combined


# def load_excel() -> pd.DataFrame | None:
#     """
#     Load the master Excel.  If it does not exist yet, build it first.

#     Returns:
#         DataFrame or None if no source data is available at all.
#     """
#     master_path = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

#     if not os.path.exists(master_path):
#         logger.info("Master Excel not found — building now…")
#         return build_master_excel()

#     try:
#         with _excel_lock:
#             df = pd.read_excel(master_path)
#         logger.debug("Master Excel loaded — %d rows.", len(df))
#         return df
#     except Exception as exc:
#         logger.error("Failed to load master Excel: %s", exc)
#         return None


# def delete_stale_files(days: int = 14) -> int:
#     """
#     Delete files older than *days* from the Excels folder (non-recursive).

#     Returns:
#         Number of files deleted.
#     """
#     cutoff  = datetime.now() - timedelta(days=days)
#     deleted = 0

#     for file_path in Path(EXCELS_FOLDER).iterdir():
#         if file_path.is_file():
#             modified = datetime.fromtimestamp(file_path.stat().st_mtime)
#             if modified < cutoff:
#                 file_path.unlink()
#                 logger.info("Deleted stale file: %s", file_path.name)
#                 deleted += 1

#     return deleted


# # ---------------------------------------------------------------------------
# # Excel query tools
# # ---------------------------------------------------------------------------

# def filter_by_application_name(keyword: str) -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})

#     mask     = df["Application Name"].str.contains(re.escape(keyword), case=False, na=False)
#     filtered = df[mask][IMPORTANT_COLUMNS]
#     return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


# def get_column_names() -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     return json.dumps({"columns": list(df.columns)})


# def get_summary_stats(column_name: str) -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     if column_name not in df.columns:
#         return json.dumps({"error": f"Column '{column_name}' not found."})
#     return json.dumps(df[column_name].describe().to_dict())


# def get_unique_values(column_name: str) -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     if column_name not in df.columns:
#         return json.dumps({"error": f"Column '{column_name}' not found."})
#     return json.dumps({"column": column_name, "unique_values": df[column_name].dropna().unique().tolist()})


# def get_row_count() -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     return json.dumps({"total_rows": len(df)})


# def filter_by_column_value(column_name: str, value: str) -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     if column_name not in df.columns:
#         return json.dumps({"error": f"Column '{column_name}' not found."})
#     mask     = df[column_name].astype(str).str.contains(re.escape(value), case=False, na=False)
#     filtered = df[mask]
#     cols     = [c for c in IMPORTANT_COLUMNS if c in filtered.columns]
#     return json.dumps({"count": len(filtered), "results": filtered[cols].to_dict(orient="records")})


# def get_all_rows() -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     cols    = [c for c in IMPORTANT_COLUMNS if c in df.columns]
#     limited = df[cols].head(200)
#     return json.dumps({"count": len(df), "results": limited.to_dict(orient="records")})


# def get_lyric_servers() -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     mask     = df["Application Name"].str.contains("lyric", case=False, na=False)
#     filtered = df[mask][IMPORTANT_COLUMNS].head(50)
#     return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


# def lyric_summary() -> str:
#     df = load_excel()
#     if df is None:
#         return json.dumps({"error": "Master Excel could not be loaded."})
#     lyric   = df[df["Application Name"].str.contains("lyric", case=False, na=False)]
#     summary = {
#         "total_servers":   len(lyric),
#         "reboot_required": lyric["Reboot Required"].value_counts().to_dict(),
#         "patch_windows":   lyric["Patch Window"].dropna().unique().tolist(),
#     }
#     return json.dumps(summary)


# # ---------------------------------------------------------------------------
# # Mail helpers
# # ---------------------------------------------------------------------------

# def _resolve_folder_id(folder_name: str) -> str | None:
#     url      = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/childFolders"
#     response = requests.get(url, headers=get_headers(), timeout=15)
#     response.raise_for_status()
#     for folder in response.json().get("value", []):
#         if folder["displayName"] == folder_name:
#             return folder["id"]
#     return None


# # ---------------------------------------------------------------------------
# # CHG ticket extractor
# # ---------------------------------------------------------------------------

# def _extract_chg_ticket(subject: str) -> str | None:
#     """
#     Extract a change ticket number (e.g. CHG083232) from an email subject.
#     Returns the ticket string, or None if not found.
#     """
#     m = re.search(r'\b(CHG\d+)\b', subject, re.IGNORECASE)
#     return m.group(1).upper() if m else None

# def _save_attachment(att: dict, subject: str) -> str | None:
#     """
#     Decode and save an email attachment to the correct sub-folder.

#     Routing rules
#     -------------
#     'Maintenance Notification'  → Maintenance/maintenance_latest.xlsx
#         (single file, overwritten each time — newest mail wins)
#     'Reschedule Maintenance'    → Rescheduled/rescheduled_latest.xlsx
#         (single file, overwritten each time — newest mail wins)
#     'Implementation Status'     → ImplementationStatus/implementation_<timestamp>.xlsx
#         (NEW timestamped file per mail — ALL files are kept so that servers
#         from different mails accumulate in the master Excel instead of
#         overwriting each other)
#     """
#     file_name = att.get("name", "")
#     ext       = Path(file_name).suffix.lower()

#     if ext not in EXCEL_EXTENSIONS:
#         logger.debug("Skipping non-Excel attachment: %s", file_name)
#         return None

#     if "Maintenance Notification" in subject:
#         sub_folder = "Maintenance"
#         save_name  = "maintenance_latest.xlsx"

#     elif "Reschedule Maintenance" in subject:
#         sub_folder = "Rescheduled"
#         save_name  = "rescheduled_latest.xlsx"

#     elif "Implementation Status" in subject:
#         sub_folder = "ImplementationStatus"
#         # Unique filename per mail — preserves ALL implementation status files
#         timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
#         save_name  = f"implementation_{timestamp}.xlsx"

#     else:
#         logger.debug("Subject '%s' did not match any routing rule — skipping.", subject)
#         return None

#     dest_folder = os.path.join(EXCELS_FOLDER, sub_folder)
#     os.makedirs(dest_folder, exist_ok=True)
#     save_path = os.path.join(dest_folder, save_name)

#     try:
#         file_data = base64.b64decode(att["contentBytes"])
#         tmp_path  = save_path + ".tmp"
#         with open(tmp_path, "wb") as fh:
#             fh.write(file_data)
#         os.replace(tmp_path, save_path)
#         logger.info("Attachment saved: %s", save_path)

#         # ── NEW: store CHG ticket keyed to this file ──────────────────────
#         ticket = _extract_chg_ticket(subject)
#         if ticket:
#             with _tickets_lock:
#                 _pending_chg_tickets[save_path] = ticket
#             logger.info("CHG ticket recorded for %s: %s", save_name, ticket)

#         return save_path
#     except Exception as exc:
#         logger.error("Failed to save attachment '%s': %s", file_name, exc)
#         return None


# # ---------------------------------------------------------------------------
# # Validation agent runner
# # ---------------------------------------------------------------------------

# _validation_lock:   threading.Lock  = threading.Lock()
# _validation_pending: threading.Event = threading.Event()


# def _run_validation_safe(query: str) -> None:
#     _validation_pending.set()

#     acquired = _validation_lock.acquire(blocking=False)
#     if not acquired:
#         logger.info("[Validation Thread] Queued — will run after current finishes.")
#         return

#     try:
#         while _validation_pending.is_set():
#             _validation_pending.clear()
#             logger.info("[Validation Thread] Starting validation agent...")
#             try:
#                 run_validation_agent(query)
#             except Exception as exc:
#                 logger.error("[Validation Thread] Agent failed: %s", exc, exc_info=True)
#     finally:
#         _validation_lock.release()
#         logger.info("[Validation Thread] Validation agent finished.")


# # ---------------------------------------------------------------------------
# # Mail tools
# # ---------------------------------------------------------------------------

# def get_latest_mail(folder_name: str = "") -> str:
#     """
#     Fetch the most recent email from the monitored folder.

#     If the email subject matches a known patching category and contains
#     Excel attachments, those attachments are automatically saved and the
#     master Excel is rebuilt.
#     """
#     target = folder_name or FOLDER_NAME

#     try:
#         folder_id = _resolve_folder_id(target)
#         if not folder_id:
#             return json.dumps({"error": f"Folder '{target}' not found in Inbox."})

#         msgs_url  = (
#             f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder_id}/messages"
#             f"?$top=1&$orderby=receivedDateTime desc"
#         )
#         msgs_resp = requests.get(msgs_url, headers=get_headers(), timeout=15)
#         msgs_resp.raise_for_status()
#         messages  = msgs_resp.json().get("value", [])

#         if not messages:
#             return json.dumps({"error": "No messages found in folder."})

#         mail       = messages[0]
#         message_id = mail["id"]
#         subject    = mail.get("subject", "")
#         sender     = mail["from"]["emailAddress"]["address"]
#         body       = mail.get("bodyPreview", "")
#         received   = mail.get("receivedDateTime", "")

#         # Content-based dedup
#         mail_hash = _make_mail_hash(subject, received, sender)
#         with _mail_hash_lock:
#             if mail_hash in _processed_mail_hashes:
#                 logger.info(
#                     "Duplicate mail content detected (subject='%s', received='%s') — skipping.",
#                     subject, received,
#                 )
#                 return json.dumps({
#                     "message_id":        message_id,
#                     "subject":           subject,
#                     "from":              sender,
#                     "received":          received,
#                     "body_preview":      "",
#                     "attachments_saved": [],
#                     "skipped":           True,
#                     "reason":            "Duplicate mail content already processed.",
#                 })
#             _processed_mail_hashes.add(mail_hash)

#         attachments_saved: list[str] = []

#         patching_keywords = [
#             "Maintenance Notification",
#             "Reschedule Maintenance",
#             "Implementation Status",
#         ]
#         is_patching_mail = any(kw in subject for kw in patching_keywords)

#         if is_patching_mail and mail.get("hasAttachments"):
#             att_url  = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"
#             att_resp = requests.get(att_url, headers=get_headers(), timeout=30)
#             att_resp.raise_for_status()

#             for att in att_resp.json().get("value", []):
#                 saved_path = _save_attachment(att, subject)
#                 if saved_path:
#                     attachments_saved.append(saved_path)

#         if attachments_saved:
#             # Rebuild master — carries forward all previously written
#             # boot times and validation statuses automatically
#             build_master_excel()

#             if "Implementation Status" in subject:
#                 logger.info(
#                     "[Mail Tool] Implementation Status mail arrived — starting Validation Agent..."
#                 )
#                 threading.Thread(
#                     target=_run_validation_safe,
#                     args=(
#                         "Get all lyric servers where Implementation Status is Completed, "
#                         "connect to each via WinRM to fetch the boot time/errors, save it to Excel, "
#                         "then validate if the boot time (if present) is within the patch window and "
#                         "update the Application Team Validation Status for every server.",
#                     ),
#                     daemon=True,
#                 ).start()

#                 return json.dumps({
#                     "message_id":        message_id,
#                     "subject":           subject,
#                     "from":              sender,
#                     "received":          received,
#                     "body_preview":      body,
#                     "attachments_saved": attachments_saved,
#                     "delegated":         True,
#                     "message":           (
#                         "Implementation Status mail received. Excel attachment saved with a unique "
#                         "timestamped filename so previous servers are preserved. Master Excel rebuilt "
#                         "(existing Boot Time / Validation Status data carried forward). "
#                         "Validation Agent triggered for all completed Lyric servers. "
#                         "No further action required from the email agent."
#                     ),
#                 })
#             else:
#                 logger.info("[Mail Tool] '%s' mail processed — validation agent not triggered.", subject)

#         return json.dumps({
#             "message_id":        message_id,
#             "subject":           subject,
#             "from":              sender,
#             "received":          received,
#             "body_preview":      body,
#             "attachments_saved": attachments_saved,
#         })

#     except requests.RequestException as exc:
#         logger.error("get_latest_mail failed: %s", exc)
#         return json.dumps({"error": str(exc)})


# def search_mails_by_subject(keyword: str) -> str:
#     """Search emails in the monitored folder by a subject keyword (up to 10 results)."""
#     try:
#         folder_id = _resolve_folder_id(FOLDER_NAME)
#         if not folder_id:
#             return json.dumps({"error": f"Folder '{FOLDER_NAME}' not found."})

#         msgs_url = (
#             f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder_id}/messages"
#             f"?$filter=contains(subject,'{keyword}')"
#             f"&$select=subject,from,receivedDateTime,hasAttachments,bodyPreview"
#             f"&$top=10&$orderby=receivedDateTime desc"
#         )
#         resp = requests.get(msgs_url, headers=get_headers(), timeout=15)
#         resp.raise_for_status()

#         results = [
#             {
#                 "subject":      m.get("subject"),
#                 "from":         m["from"]["emailAddress"]["address"],
#                 "received":     m.get("receivedDateTime"),
#                 "body_preview": m.get("bodyPreview", "")[:200],
#             }
#             for m in resp.json().get("value", [])
#         ]

#         return json.dumps({"keyword": keyword, "count": len(results), "emails": results})

#     except requests.RequestException as exc:
#         logger.error("search_mails_by_subject failed: %s", exc)
#         return json.dumps({"error": str(exc)})


# # ---------------------------------------------------------------------------
# # TOOL_REGISTRY
# # ---------------------------------------------------------------------------

# TOOL_REGISTRY: dict[str, callable] = {
#     "get_latest_mail":            get_latest_mail,
#     "search_mails_by_subject":    search_mails_by_subject,
#     "filter_by_application_name": filter_by_application_name,
#     "get_column_names":           get_column_names,
#     "get_summary_stats":          get_summary_stats,
#     "get_unique_values":          get_unique_values,
#     "get_row_count":              get_row_count,
#     "filter_by_column_value":     filter_by_column_value,
#     "get_all_rows":               get_all_rows,
#     "get_lyric_servers":          get_lyric_servers,
#     "lyric_summary":              lyric_summary,
# }


# # ---------------------------------------------------------------------------
# # TOOL_SCHEMAS
# # ---------------------------------------------------------------------------

# TOOL_SCHEMAS: list[dict] = [
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_latest_mail",
#             "description": (
#                 "Fetch the single most recent email from the monitored inbox folder. "
#                 "Returns subject, sender, received time, body preview, and paths of any "
#                 "Excel attachments that were automatically saved."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "folder_name": {
#                         "type":        "string",
#                         "description": "Optional: override the default monitored folder name.",
#                     }
#                 },
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "search_mails_by_subject",
#             "description": "Search emails in the monitored folder by a subject keyword (up to 10 results).",
#             "parameters": {
#                 "type":       "object",
#                 "properties": {
#                     "keyword": {
#                         "type":        "string",
#                         "description": "Keyword to search for in email subjects.",
#                     }
#                 },
#                 "required": ["keyword"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "filter_by_application_name",
#             "description": "Filter server rows where Application Name contains a keyword.",
#             "parameters": {
#                 "type":       "object",
#                 "properties": {
#                     "keyword": {
#                         "type":        "string",
#                         "description": "Partial or full application name to search for.",
#                     }
#                 },
#                 "required": ["keyword"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_column_names",
#             "description": "Return all column names in the master patch Excel file.",
#             "parameters":  {"type": "object", "properties": {}},
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_summary_stats",
#             "description": "Return descriptive statistics for a numeric column in the Excel.",
#             "parameters": {
#                 "type":       "object",
#                 "properties": {
#                     "column_name": {
#                         "type":        "string",
#                         "description": "Name of the column to describe.",
#                     }
#                 },
#                 "required": ["column_name"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_unique_values",
#             "description": "Return all unique non-null values in a given column.",
#             "parameters": {
#                 "type":       "object",
#                 "properties": {
#                     "column_name": {
#                         "type":        "string",
#                         "description": "Column to retrieve unique values from.",
#                     }
#                 },
#                 "required": ["column_name"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_row_count",
#             "description": "Return the total number of server entries in the master Excel.",
#             "parameters":  {"type": "object", "properties": {}},
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "filter_by_column_value",
#             "description": "Filter rows where a specific column contains a given value (case-insensitive).",
#             "parameters": {
#                 "type":       "object",
#                 "properties": {
#                     "column_name": {
#                         "type":        "string",
#                         "description": "Column to filter on.",
#                     },
#                     "value": {
#                         "type":        "string",
#                         "description": "Value to search for within that column.",
#                     },
#                 },
#                 "required": ["column_name", "value"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_all_rows",
#             "description": (
#                 "Return all server rows from the master Excel (up to 200, important columns only). "
#                 "Use when the user wants a full list without any specific filter."
#             ),
#             "parameters": {"type": "object", "properties": {}},
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "get_lyric_servers",
#             "description": "Return all servers belonging to the Lyric application (up to 50 rows).",
#             "parameters":  {"type": "object", "properties": {}},
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name":        "lyric_summary",
#             "description": (
#                 "Return an aggregate summary for Lyric application servers: "
#                 "total count, reboot-required distribution, and unique patch windows."
#             ),
#             "parameters": {"type": "object", "properties": {}},
#         },
#     },
# ]





"""
email_tool.py
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

Key behaviour — Implementation Status mails
--------------------------------------------
Each Implementation Status mail covers a DIFFERENT subset of servers.
To avoid overwriting earlier servers when a new mail arrives we:

  * Save every Implementation Status attachment with a UNIQUE timestamped
    filename  (implementation_<YYYYMMDD_HHMMSS>.xlsx)  instead of always
    writing to 'implementation_latest.xlsx'.  All files accumulate in the
    ImplementationStatus/ sub-folder.

  * build_master_excel() reads ALL files in ImplementationStatus/ and
    merges them together before deduplication, so every batch of servers
    is always present in the master.

  * Maintenance and Rescheduled still use a single 'latest' file because
    those mails replace each other by design (most-recent wins).

  * The master rebuild NEVER touches the 'Boot Time',
    'Application Team Validation Status', or 'Error' columns that the
    Validation Agent has already written.  Those columns are preserved by
    a merge-with-existing-master step inside build_master_excel().

CHANGES MADE (v2):
  ✓ Added _extract_timestamp_from_impl_filename() to parse filename timestamps
  ✓ Modified build_master_excel() to track _file_timestamp during file read
  ✓ Updated deduplication sort to use both _priority AND _file_timestamp
  ✓ This ensures "Pending" → "Completed" updates work across multiple mails
"""

from __future__ import annotations

import hashlib
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
from validation_agent import run_agent as run_validation_agent

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

# Columns written exclusively by the Validation Agent.
# build_master_excel() NEVER overwrites these — it carries them forward
# from the existing master so previous run data is never lost.
_VALIDATION_COLUMNS: list[str] = [
    "Boot Time",
    "Error",
    "Application Team Validation Status",
    "Change Ticket"
]

# Sub-folder priority: higher number wins on deduplication across folders.
# Within ImplementationStatus itself ALL files are merged (no dedup needed
# until the combined frame is deduped against Maintenance/Rescheduled).
_SUBFOLDER_PRIORITY: dict[str, int] = {
    "Maintenance":          1,
    "Rescheduled":          2,
    "ImplementationStatus": 3,
}

# Ensure directory structure exists
for _sub in _SUBFOLDER_PRIORITY:
    os.makedirs(os.path.join(EXCELS_FOLDER, _sub), exist_ok=True)

# Threading lock — prevents concurrent writes to the master Excel
_excel_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Content-based mail dedup
# ---------------------------------------------------------------------------

_processed_mail_hashes: set[str] = set()
_mail_hash_lock: threading.Lock  = threading.Lock()

# Module-level: maps saved filename → CHG ticket (for master rebuild)
_pending_chg_tickets: dict[str, str] = {}
_tickets_lock: threading.Lock = threading.Lock()

def _make_mail_hash(subject: str, received: str, sender: str) -> str:
    """
    Stable content fingerprint for a mail, independent of Graph's message_id.
    Graph can assign different message_ids to the same physical email across
    duplicate notifications — this hash collapses them to one identity.
    """
    raw = f"{subject}|{received}|{sender}".lower().strip()
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ CHANGE 1: NEW FUNCTION - Extract timestamp from implementation filename   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _extract_timestamp_from_impl_filename(filename: str) -> datetime | None:
    """
    Extract and parse the timestamp from an Implementation Status filename.
    
    Expected format: implementation_YYYYMMDD_HHMMSS.xlsx
    Example: 'implementation_20250315_143022.xlsx'
    
    Returns:
        datetime object representing when the file was created, or None if 
        filename doesn't match the expected pattern.
    
    Why this matters:
    ───────────────
    When multiple Implementation Status mails arrive for the same servers,
    we need to know which file is NEWER to determine if a server's status
    should be updated from "Pending" to "Completed" etc.
    
    We use the FILENAME timestamp (created when email arrived) instead of
    filesystem modification time, because the latter can get mixed up if
    files are moved, backed up, or copied.
    
    Example flow:
      Mail 1 (2025-03-10 12:00) → implementation_20250310_120000.xlsx
              Server A: Implementation Status = "Pending"
      
      Mail 2 (2025-03-15 14:30) → implementation_20250315_143022.xlsx
              Server A: Implementation Status = "Completed"
      
      build_master_excel() reads both files, compares timestamps,
      and keeps the NEWER "Completed" status ✓
    """
    # Regex pattern: match 'implementation_' followed by 8 digits, underscore,
    # 6 digits, then any extension
    match = re.search(r'implementation_(\d{8})_(\d{6})', filename)
    
    if match:
        date_str, time_str = match.groups()  # e.g., ('20250315', '143022')
        try:
            # Combine date and time strings, then parse
            # Format: YYYYMMDD_HHMMSS
            return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except ValueError as e:
            logger.warning(f"Could not parse timestamp from {filename}: {e}")
            return None
    
    return None


def _get_latest_file(folder: str) -> Path | None:
    """Return the most-recently-modified Excel/CSV file in *folder*, or None."""
    candidates = [
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in EXCEL_EXTENSIONS
    ]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def _get_all_files(folder: str) -> list[Path]:
    """Return ALL Excel/CSV files in *folder*, sorted oldest → newest."""
    candidates = [
        f for f in Path(folder).iterdir()
        if f.is_file() and f.suffix.lower() in EXCEL_EXTENSIONS
    ]
    return sorted(candidates, key=lambda f: f.stat().st_mtime)


def _read_file(path: Path) -> pd.DataFrame:
    """Read an Excel or CSV file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def build_master_excel(default_impl_status: str = "Pending") -> pd.DataFrame | None:
    """
    Merge source files from all sub-folders into a single master Excel.

    Implementation Status folder
    ----------------------------
    ALL files in ImplementationStatus/ are read and stacked — this is how
    servers from multiple emails coexist without overwriting each other.

    Maintenance / Rescheduled folders
    ----------------------------------
    Only the latest file is read (most-recent-wins for these mail types).

    Deduplication
    ─────────────
    Rows are deduplicated on 'Server Name'.
    
    UPDATED LOGIC (v2):
    ✓ Primary sort: Higher-priority sub-folder wins 
      (ImplementationStatus > Rescheduled > Maintenance)
    ✓ Secondary sort (NEW): WITHIN same priority, NEWER file timestamp wins
      (ensures "Pending" → "Completed" transitions work correctly)
    ✓ After sorting, keep the LAST occurrence of each server name
      (last = highest priority + newest timestamp)
    
    Example scenario with SAME server in two Implementation Status files:
    ──────────────────────────────────────────────────────────────────
    Server A appears in:
      • Mail 1 (received 2025-03-10 12:00) → file: implementation_20250310_120000.xlsx
        Data: Implementation Status = "Pending", Reboot Required = "Yes"
      
      • Mail 2 (received 2025-03-15 14:30) → file: implementation_20250315_143022.xlsx
        Data: Implementation Status = "Completed", Reboot Required = "Yes"
    
    Before v2:
      ⚠ Dedup result was unpredictable (relied on filesystem timestamps)
    
    After v2:
      ✓ Both files have same priority (ImplementationStatus = 3)
      ✓ File timestamps are compared: 20250315_143022 > 20250310_120000
      ✓ After sort, Mail 2's row is last
      ✓ drop_duplicates(keep='last') keeps Mail 2's "Completed" status ✓

    Validation data preservation
    ───────────────────────────
    After rebuilding from source files the function loads the existing master
    (if any) and carries forward Boot Time, Error, and
    Application Team Validation Status for every server that already has
    those values.  This ensures the Validation Agent's work is NEVER erased
    by a subsequent Implementation Status mail arriving.

    Returns:
        The merged DataFrame, or None if no source files exist.
    """
    dfs: list[pd.DataFrame] = []

    for folder_name, priority in _SUBFOLDER_PRIORITY.items():
        folder_path = os.path.join(EXCELS_FOLDER, folder_name)

        if folder_name == "ImplementationStatus":
            # ────────────────────────────────────────────────────────────────
            # Read ALL implementation status files so every mail's servers
            # are included.  Sort oldest→newest so that when the same server
            # appears in two mails the newer file's row can win during dedup.
            # ────────────────────────────────────────────────────────────────
            files = _get_all_files(folder_path)
            if not files:
                logger.debug("No files found in %s — skipping.", folder_path)
                continue

            for file_path in files:
                try:
                    df = _read_file(file_path)
                    df.columns = df.columns.str.strip()
                    df["_source_folder"] = folder_name
                    df["_source_file"]   = file_path.name
                    df["_priority"]      = priority
                    
                    # ╔════════════════════════════════════════════════════════╗
                    # ║ CHANGE 2: Extract timestamp from filename              ║
                    # ║ This timestamp will be used to determine which file's   ║
                    # ║ data "wins" when the same server appears in multiple    ║
                    # ║ Implementation Status files.                            ║
                    # ╚════════════════════════════════════════════════════════╝
                    df["_file_timestamp"] = _extract_timestamp_from_impl_filename(file_path.name)
                    
                    dfs.append(df)
                    logger.debug("Loaded %d rows from %s (timestamp: %s)", 
                               len(df), file_path, df["_file_timestamp"].iloc[0] if len(df) > 0 else "N/A")
                except Exception as exc:
                    logger.error("Failed to read %s: %s", file_path, exc)

        else:
            # Maintenance / Rescheduled — latest file only
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
                
                # ╔════════════════════════════════════════════════════════╗
                # ║ CHANGE 3: Set _file_timestamp to None for              ║
                # ║ Maintenance/Rescheduled (they don't use timestamped    ║
                # ║ filenames, so None acts as a flag that they don't      ║
                # ║ participate in timestamp-based sorting)                 ║
                # ╚════════════════════════════════════════════════════════╝
                df["_file_timestamp"] = None
                
                dfs.append(df)
                logger.debug("Loaded %d rows from %s", len(df), latest_file)
            except Exception as exc:
                logger.error("Failed to read %s: %s", latest_file, exc)

    if not dfs:
        logger.warning("build_master_excel: no source files found — nothing to merge.")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # ── NEW: stamp Change Ticket column from pending tickets dict ─────────────
    # Build a lookup: absolute save_path → ticket
    with _tickets_lock:
        ticket_snapshot = dict(_pending_chg_tickets)
    
    if ticket_snapshot:
        # Build a reverse map: source_file basename → ticket
        # The combined frame has a _source_file column with the basename
        basename_to_ticket: dict[str, str] = {}
        for full_path, ticket in ticket_snapshot.items():
            basename_to_ticket[Path(full_path).name] = ticket

        if "_source_file" in combined.columns:
            combined["Change Ticket"] = combined.apply(
                lambda row: basename_to_ticket.get(row["_source_file"],
                            row.get("Change Ticket") if "Change Ticket" in row.index else None),
                axis=1,
            )

    # Ensure required columns exist
    for col in IMPORTANT_COLUMNS:
        if col not in combined.columns:
            combined[col] = None

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ CHANGE 4: UPDATED DEDUPLICATION SORT LOGIC                           ║
    # ║                                                                       ║
    # ║ OLD (v1):                                                            ║
    # ║   combined.sort_values(["_priority"], inplace=True)                  ║
    # ║   Problem: Only sorted by folder priority, relied on file order      ║
    # ║                                                                       ║
    # ║ NEW (v2):                                                            ║
    # ║   combined.sort_values(["_priority", "_file_timestamp"], inplace=True)
    # ║   Benefit: Within same priority, newer timestamps sort last          ║
    # ║                                                                       ║
    # ║ Sort order (after this):                                             ║
    # ║   1. All Maintenance rows (priority=1, _file_timestamp=None)         ║
    # ║   2. All Rescheduled rows (priority=2, _file_timestamp=None)         ║
    # ║   3. All ImplementationStatus rows, sorted by timestamp              ║
    # ║      - Oldest Implementation Status files first                       ║
    # ║      - NEWEST Implementation Status files LAST                        ║
    # ║                                                                       ║
    # ║ Result: drop_duplicates(keep='last') keeps the newest file's data    ║
    # ║ for each server. "Pending" → "Completed" transitions WORK! ✓         ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    combined.sort_values(
        by=["_priority", "_file_timestamp"],
        na_position="first",  # None timestamps (Maintenance/Rescheduled) sort to front
        inplace=True
    )
    
    combined.drop_duplicates(subset=["Server Name"], keep="last", inplace=True)

    combined["Implementation Status"] = combined["Implementation Status"].fillna(default_impl_status)

    # Drop internal helper columns before saving
    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║ CHANGE 5: Drop _file_timestamp column (it was only for sorting)       ║
    # ║ We don't want this internal tracking column in the final Excel file    ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    combined.drop(columns=["_priority", "_file_timestamp"], inplace=True)

    # ------------------------------------------------------------------
    # Preserve Validation Agent data from the existing master
    # ------------------------------------------------------------------
    # Load the current master (if it exists) and carry forward any
    # non-empty values in the validation columns.  This means that even
    # when a brand-new Implementation Status mail triggers a rebuild,
    # servers whose boot time / validation status was already recorded
    # keep that data intact.
    master_path = os.path.join(EXCELS_FOLDER, "master_patch_data.xlsx")

    if os.path.exists(master_path):
        try:
            with _excel_lock:
                existing = pd.read_excel(master_path)
            existing.columns = existing.columns.str.strip()

            # Build a lookup: server_name (lower) → {col: value}
            val_cols_present = [c for c in _VALIDATION_COLUMNS if c in existing.columns]

            if val_cols_present:
                existing["_key"] = existing["Server Name"].astype(str).str.strip().str.lower()
                val_lookup = (
                    existing[["_key"] + val_cols_present]
                    .drop_duplicates(subset=["_key"], keep="last")
                    .set_index("_key")
                )

                combined["_key"] = combined["Server Name"].astype(str).str.strip().str.lower()

                # Ensure validation columns exist in combined
                for col in val_cols_present:
                    if col not in combined.columns:
                        combined[col] = None

                # For each validation column, fill from existing master
                # only where the new combined frame has an empty cell
                for col in val_cols_present:
                    def _carry_forward(row, col=col):
                        current = row[col]
                        # If already populated in combined, keep it
                        if current is not None and not (
                            isinstance(current, float) and pd.isna(current)
                        ) and str(current).strip() != "":
                            return current
                        # Otherwise look up from existing master
                        key = row["_key"]
                        if key in val_lookup.index:
                            existing_val = val_lookup.at[key, col]
                            if existing_val is not None and not (
                                isinstance(existing_val, float) and pd.isna(existing_val)
                            ) and str(existing_val).strip() != "":
                                return existing_val
                        return current

                    combined[col] = combined.apply(_carry_forward, axis=1)

                combined.drop(columns=["_key"], inplace=True)
                logger.info(
                    "Carried forward validation data from existing master for columns: %s",
                    val_cols_present,
                )
        except Exception as exc:
            logger.warning(
                "Could not carry forward validation data from existing master: %s", exc
            )

    # Write master atomically
    with _excel_lock:
        combined.to_excel(master_path, index=False)
        logger.info(
            "Master Excel rebuilt — %d unique servers → %s", len(combined), master_path
        )
        print(f"Master Excel updated: {master_path}")

    return combined


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


def delete_stale_files(days: int = 14) -> int:
    """
    Delete files older than *days* from the Excels folder (non-recursive).

    Returns:
        Number of files deleted.
    """
    cutoff  = datetime.now() - timedelta(days=days)
    deleted = 0

    for file_path in Path(EXCELS_FOLDER).iterdir():
        if file_path.is_file():
            modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            if modified < cutoff:
                file_path.unlink()
                logger.info("Deleted stale file: %s", file_path.name)
                deleted += 1

    return deleted


# ---------------------------------------------------------------------------
# Excel query tools
# ---------------------------------------------------------------------------

def filter_by_application_name(keyword: str) -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})

    mask     = df["Application Name"].str.contains(re.escape(keyword), case=False, na=False)
    filtered = df[mask][IMPORTANT_COLUMNS]
    return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


def get_column_names() -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    return json.dumps({"columns": list(df.columns)})


def get_summary_stats(column_name: str) -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})
    return json.dumps(df[column_name].describe().to_dict())


def get_unique_values(column_name: str) -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})
    return json.dumps({"column": column_name, "unique_values": df[column_name].dropna().unique().tolist()})


def get_row_count() -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    return json.dumps({"total_rows": len(df)})


def filter_by_column_value(column_name: str, value: str) -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    if column_name not in df.columns:
        return json.dumps({"error": f"Column '{column_name}' not found."})
    mask     = df[column_name].astype(str).str.contains(re.escape(value), case=False, na=False)
    filtered = df[mask]
    cols     = [c for c in IMPORTANT_COLUMNS if c in filtered.columns]
    return json.dumps({"count": len(filtered), "results": filtered[cols].to_dict(orient="records")})


def get_all_rows() -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    cols    = [c for c in IMPORTANT_COLUMNS if c in df.columns]
    limited = df[cols].head(200)
    return json.dumps({"count": len(df), "results": limited.to_dict(orient="records")})


def get_lyric_servers() -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    mask     = df["Application Name"].str.contains("lyric", case=False, na=False)
    filtered = df[mask][IMPORTANT_COLUMNS].head(50)
    return json.dumps({"count": len(filtered), "results": filtered.to_dict(orient="records")})


def lyric_summary() -> str:
    df = load_excel()
    if df is None:
        return json.dumps({"error": "Master Excel could not be loaded."})
    lyric   = df[df["Application Name"].str.contains("lyric", case=False, na=False)]
    summary = {
        "total_servers":   len(lyric),
        "reboot_required": lyric["Reboot Required"].value_counts().to_dict(),
        "patch_windows":   lyric["Patch Window"].dropna().unique().tolist(),
    }
    return json.dumps(summary)


# ---------------------------------------------------------------------------
# Mail helpers
# ---------------------------------------------------------------------------

def _resolve_folder_id(folder_name: str) -> str | None:
    url      = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/childFolders"
    response = requests.get(url, headers=get_headers(), timeout=15)
    response.raise_for_status()
    for folder in response.json().get("value", []):
        if folder["displayName"] == folder_name:
            return folder["id"]
    return None


# ---------------------------------------------------------------------------
# CHG ticket extractor
# ---------------------------------------------------------------------------

def _extract_chg_ticket(subject: str) -> str | None:
    """
    Extract a change ticket number (e.g. CHG083232) from an email subject.
    Returns the ticket string, or None if not found.
    """
    m = re.search(r'\b(CHG\d+)\b', subject, re.IGNORECASE)
    return m.group(1).upper() if m else None

def _save_attachment(att: dict, subject: str) -> str | None:
    """
    Decode and save an email attachment to the correct sub-folder.

    Routing rules
    -------------
    'Maintenance Notification'  → Maintenance/maintenance_latest.xlsx
        (single file, overwritten each time — newest mail wins)
    'Reschedule Maintenance'    → Rescheduled/rescheduled_latest.xlsx
        (single file, overwritten each time — newest mail wins)
    'Implementation Status'     → ImplementationStatus/implementation_<timestamp>.xlsx
        (NEW timestamped file per mail — ALL files are kept so that servers
        from different mails accumulate in the master Excel instead of
        overwriting each other)
    """
    file_name = att.get("name", "")
    ext       = Path(file_name).suffix.lower()

    if ext not in EXCEL_EXTENSIONS:
        logger.debug("Skipping non-Excel attachment: %s", file_name)
        return None

    if "Maintenance Notification" in subject:
        sub_folder = "Maintenance"
        save_name  = "maintenance_latest.xlsx"

    elif "Reschedule Maintenance" in subject:
        sub_folder = "Rescheduled"
        save_name  = "rescheduled_latest.xlsx"

    elif "Implementation Status" in subject:
        sub_folder = "ImplementationStatus"
        # Unique filename per mail — preserves ALL implementation status files
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name  = f"implementation_{timestamp}.xlsx"

    else:
        logger.debug("Subject '%s' did not match any routing rule — skipping.", subject)
        return None

    dest_folder = os.path.join(EXCELS_FOLDER, sub_folder)
    os.makedirs(dest_folder, exist_ok=True)
    save_path = os.path.join(dest_folder, save_name)

    try:
        file_data = base64.b64decode(att["contentBytes"])
        tmp_path  = save_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(file_data)
        os.replace(tmp_path, save_path)
        logger.info("Attachment saved: %s", save_path)

        # ── NEW: store CHG ticket keyed to this file ──────────────────────
        ticket = _extract_chg_ticket(subject)
        if ticket:
            with _tickets_lock:
                _pending_chg_tickets[save_path] = ticket
            logger.info("CHG ticket recorded for %s: %s", save_name, ticket)

        return save_path
    except Exception as exc:
        logger.error("Failed to save attachment '%s': %s", file_name, exc)
        return None


# ---------------------------------------------------------------------------
# Validation agent runner
# ---------------------------------------------------------------------------

_validation_lock:   threading.Lock  = threading.Lock()
_validation_pending: threading.Event = threading.Event()


def _run_validation_safe(query: str) -> None:
    _validation_pending.set()

    acquired = _validation_lock.acquire(blocking=False)
    if not acquired:
        logger.info("[Validation Thread] Queued — will run after current finishes.")
        return

    try:
        while _validation_pending.is_set():
            _validation_pending.clear()
            logger.info("[Validation Thread] Starting validation agent...")
            try:
                run_validation_agent(query)
            except Exception as exc:
                logger.error("[Validation Thread] Agent failed: %s", exc, exc_info=True)
    finally:
        _validation_lock.release()
        logger.info("[Validation Thread] Validation agent finished.")


# ---------------------------------------------------------------------------
# Mail tools
# ---------------------------------------------------------------------------

def get_latest_mail(folder_name: str = "") -> str:
    """
    Fetch the most recent email from the monitored folder.

    If the email subject matches a known patching category and contains
    Excel attachments, those attachments are automatically saved and the
    master Excel is rebuilt.
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

        # Content-based dedup
        mail_hash = _make_mail_hash(subject, received, sender)
        with _mail_hash_lock:
            if mail_hash in _processed_mail_hashes:
                logger.info(
                    "Duplicate mail content detected (subject='%s', received='%s') — skipping.",
                    subject, received,
                )
                return json.dumps({
                    "message_id":        message_id,
                    "subject":           subject,
                    "from":              sender,
                    "received":          received,
                    "body_preview":      "",
                    "attachments_saved": [],
                    "skipped":           True,
                    "reason":            "Duplicate mail content already processed.",
                })
            _processed_mail_hashes.add(mail_hash)

        attachments_saved: list[str] = []

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

        if attachments_saved:
            # Rebuild master — carries forward all previously written
            # boot times and validation statuses automatically
            build_master_excel()

            if "Implementation Status" in subject:
                logger.info(
                    "[Mail Tool] Implementation Status mail arrived — starting Validation Agent..."
                )
                threading.Thread(
                    target=_run_validation_safe,
                    args=(
                        "Get all lyric servers where Implementation Status is Completed, "
                        "connect to each via WinRM to fetch the boot time/errors, save it to Excel, "
                        "then validate if the boot time (if present) is within the patch window and "
                        "update the Application Team Validation Status for every server.",
                    ),
                    daemon=True,
                ).start()

                return json.dumps({
                    "message_id":        message_id,
                    "subject":           subject,
                    "from":              sender,
                    "received":          received,
                    "body_preview":      body,
                    "attachments_saved": attachments_saved,
                    "delegated":         True,
                    "message":           (
                        "Implementation Status mail received. Excel attachment saved with a unique "
                        "timestamped filename so previous servers are preserved. Master Excel rebuilt "
                        "(existing Boot Time / Validation Status data carried forward). "
                        "Validation Agent triggered for all completed Lyric servers. "
                        "No further action required from the email agent."
                    ),
                })
            else:
                logger.info("[Mail Tool] '%s' mail processed — validation agent not triggered.", subject)

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
    """Search emails in the monitored folder by a subject keyword (up to 10 results)."""
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


# ---------------------------------------------------------------------------
# TOOL_REGISTRY
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "get_latest_mail":            get_latest_mail,
    "search_mails_by_subject":    search_mails_by_subject,
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


# ---------------------------------------------------------------------------
# TOOL_SCHEMAS
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
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
                "type":       "object",
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
            "parameters":  {"type": "object", "properties": {}},
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
            "parameters":  {"type": "object", "properties": {}},
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
            "parameters":  {"type": "object", "properties": {}},
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