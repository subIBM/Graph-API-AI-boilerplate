"""

Standalone CLI for the Email Patch Agent.
Completely independent of Flask / server.py.

TWO RUN MODES:
─────────────────────────────────────────────────────────────
  python main.py            → Interactive CLI only
                              You type queries manually.
                              Agent runs when you ask.

  python main.py --watch    → Interactive CLI + background poller
                              Polls for new emails every CHECK_INTERVAL seconds.
                              Agent runs automatically on new patching emails.
                              You can STILL type queries while it watches.
─────────────────────────────────────────────────────────────

CLI Commands (available in both modes):
    /run <key>      Run a predefined query by key
    /reload         Hot-reload all prompt files from disk
    /prompts        List all available predefined prompt keys
    /verify         Verify the Microsoft Graph API connection
    /rebuild        Force-rebuild the master Excel from source files
    /status         Show watch mode status (interval, last check, emails processed)
    /help           Show this help text
    exit | quit     Quit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap — load .env before any project imports
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level   = logging.WARNING,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from auth import verify_connection
from email_agent import PREDEFINED_PROMPTS, reload_prompts, run_agent, run_predefined
from email_tool import build_master_excel, get_latest_mail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECK_INTERVAL : int = int(os.environ.get("CHECK_INTERVAL", 60))   # seconds
PROCESSED_FILE : str = os.environ.get("PROCESSED_FILE", "processed_ids.txt")

# ---------------------------------------------------------------------------
# Watch mode state  (shared between poller thread and CLI thread)
# ---------------------------------------------------------------------------
_watch_stats: dict = {
    "running":          False,
    "last_check":       None,       # datetime of last poll
    "emails_processed": 0,          # count of patching emails handled
    "last_email":       None,       # subject of last processed email
}
_watch_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Processed-ID helpers  (prevents reprocessing same email across restarts)
# ---------------------------------------------------------------------------

def _load_processed_ids() -> set[str]:
    """Load persisted message IDs from disk."""
    path = Path(PROCESSED_FILE)
    if not path.exists():
        return set()
    try:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError:
        return set()


def _save_processed_id(message_id: str) -> None:
    """Append a single message ID to the persistent file."""
    try:
        with open(PROCESSED_FILE, "a", encoding="utf-8") as fh:
            fh.write(message_id + "\n")
    except OSError as exc:
        logger.error("Could not persist message ID: %s", exc)


# In-memory set — loaded once at startup, appended as new IDs arrive
_processed_ids      : set[str]       = _load_processed_ids()
_processed_ids_lock : threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Watch mode — poller logic
# ---------------------------------------------------------------------------

_PATCHING_KEYWORDS = [
    "Maintenance Notification",
    "Reschedule Maintenance",
    "Implementation Status",
]

_AUTO_QUERY = (
    "A new patching email just arrived. "
    "Fetch the latest email, check its subject, download any Excel attachments, "
    "then summarise the Lyric servers found in the updated data — "
    "include their patch windows and reboot requirements."
)


def _poll_once() -> None:
    """
    Single poll cycle:
      1. Fetch latest email directly via tool (lightweight — no agent overhead)
      2. Skip if already seen (dedup check)
      3. Skip if subject is not a patching email
      4. Otherwise — run the full agent and print the result
    """
    import json

    try:
        raw  = get_latest_mail()
        mail = json.loads(raw)

        if "error" in mail:
            logger.warning("Poll: get_latest_mail returned error — %s", mail["error"])
            return

        message_id = mail.get("message_id", "")
        subject    = mail.get("subject", "")
        received   = mail.get("received", "")

        # Update last-check timestamp regardless of outcome
        with _watch_lock:
            _watch_stats["last_check"] = datetime.now().strftime("%H:%M:%S")

        # Deduplication check
        with _processed_ids_lock:
            if message_id in _processed_ids:
                logger.debug("Poll: already processed %s — skipping.", message_id)
                return
            _processed_ids.add(message_id)
            _save_processed_id(message_id)

        # Subject filter — only act on patching emails
        is_patching = any(kw in subject for kw in _PATCHING_KEYWORDS)
        if not is_patching:
            logger.debug("Poll: new email but not patching related ('%s') — skipping.", subject)
            return

        # New patching email found — run agent
        print(f"\n{'─' * 60}")
        print(f"  [Watch] New patching email detected!")
        print(f"  Subject  : {subject}")
        print(f"  Received : {received}")
        print(f"{'─' * 60}\n")

        result = run_agent(_AUTO_QUERY, stream=False)

        print(f"\n[Auto-Response]\n{result}\n")
        print(f"{'─' * 60}")
        print("You: ", end="", flush=True)   # restore prompt after auto output

        with _watch_lock:
            _watch_stats["emails_processed"] += 1
            _watch_stats["last_email"] = subject

    except Exception as exc:
        logger.error("Poll cycle error: %s", exc)


def _watch_loop() -> None:
    """
    Background daemon thread.
    Calls _poll_once() every CHECK_INTERVAL seconds until stopped.
    Sleeps in 1-second increments to respond to stop signal quickly.
    """
    with _watch_lock:
        _watch_stats["running"] = True

    logger.info("Watch loop started — polling every %ds", CHECK_INTERVAL)

    while True:
        with _watch_lock:
            if not _watch_stats["running"]:
                break

        _poll_once()

        for _ in range(CHECK_INTERVAL):
            with _watch_lock:
                if not _watch_stats["running"]:
                    return
            time.sleep(1)


def _start_watch() -> threading.Thread:
    """Spawn and return the background poller thread."""
    t = threading.Thread(target=_watch_loop, name="WatchPoller", daemon=True)
    t.start()
    return t


def _stop_watch() -> None:
    """Signal the poller thread to exit on its next sleep tick."""
    with _watch_lock:
        _watch_stats["running"] = False


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(watch_mode: bool) -> None:
    mode_label = "CLI + Watch Mode" if watch_mode else "CLI Mode"
    print()
    print("=" * 60)
    print(f"  Enterprise Patch Intelligence Agent  —  {mode_label}")
    print("=" * 60)
    print(f"  Model    : {os.environ.get('GPT_MODEL', 'openai/gpt-oss-120b')}")
    print(f"  Folder   : {os.environ.get('FOLDER_NAME', 'Enterprise Patching')}")
    if watch_mode:
        print(f"  Interval : every {CHECK_INTERVAL}s")
        print(f"  Watching : YES — auto-processes new patching emails")
    print()
    print("  Predefined prompts (type /run <key>):")
    for key in PREDEFINED_PROMPTS:
        print(f"    /run {key}")
    print()
    print("  Type /help for all commands  |  exit to quit")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

HELP_TEXT = """
Available commands:
  /run <key>    Run a predefined prompt (see /prompts for keys)
  /reload       Hot-reload all prompt files without restarting
  /prompts      List predefined prompt keys
  /verify       Test Microsoft Graph API connection
  /rebuild      Force-rebuild master_patch_data.xlsx from source files
  /status       Show watch mode status (interval, last check, emails processed)
  /help         Show this help message
  exit / quit   Exit
"""


def _handle_command(user_input: str) -> bool:
    """
    Handle a slash command.

    Returns:
        True  — command recognised and handled
        False — not a slash command, treat as agent query
    """
    parts = user_input.strip().split(maxsplit=1)
    cmd   = parts[0].lower()

    if cmd == "/help":
        print(HELP_TEXT)
        return True

    if cmd == "/prompts":
        print("\nPredefined prompt keys:")
        for key, query in PREDEFINED_PROMPTS.items():
            print(f"  {key:<30}  →  {query[:60]}…")
        print()
        return True

    if cmd == "/verify":
        print("Verifying Microsoft Graph API connection…")
        ok = verify_connection()
        print("✓ Connection OK\n" if ok else "✗ Connection FAILED — check auth / credentials\n")
        return True

    if cmd == "/rebuild":
        print("Rebuilding master Excel from source files…")
        df = build_master_excel()
        if df is not None:
            print(f"✓ Master Excel rebuilt — {len(df)} unique servers.\n")
        else:
            print("✗ Rebuild failed — no source files found.\n")
        return True

    if cmd == "/reload":
        prompts = reload_prompts()
        print(f"✓ Prompts reloaded: {list(prompts.keys())}\n")
        return True

    if cmd == "/status":
        with _watch_lock:
            running   = _watch_stats["running"]
            last      = _watch_stats["last_check"] or "never"
            count     = _watch_stats["emails_processed"]
            last_mail = _watch_stats["last_email"] or "none"

        status = "RUNNING" if running else "STOPPED  (start with --watch flag)"
        print(f"\n  Watch mode            : {status}")
        print(f"  Poll interval         : every {CHECK_INTERVAL}s")
        print(f"  Last poll             : {last}")
        print(f"  Emails auto-processed : {count}")
        print(f"  Last patching email   : {last_mail}\n")
        return True

    if cmd == "/run":
        if len(parts) < 2:
            print("Usage: /run <key>   (see /prompts for available keys)\n")
            return True
        key = parts[1].strip()
        try:
            print(f"\nRunning predefined prompt: '{key}'\n")
            run_predefined(key, stream=True)
            print()
        except ValueError as exc:
            print(f"Error: {exc}\n")
        return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise Patch Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py              # interactive CLI only\n"
            "  python main.py --watch      # CLI + auto email polling\n"
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            f"Poll the mailbox every {CHECK_INTERVAL}s and auto-process "
            "new patching emails in the background."
        ),
    )
    args = parser.parse_args()

    _print_banner(watch_mode=args.watch)

    if args.watch:
        print(f"  [Watch] Polling every {CHECK_INTERVAL}s — first check in {CHECK_INTERVAL}s.")
        print(f"  [Watch] You can still type queries below at any time.\n")
        _start_watch()

    # Interactive CLI loop — runs on the main thread
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            if args.watch:
                print("\nStopping watch mode…")
                _stop_watch()
            print("Exiting — goodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            if args.watch:
                print("Stopping watch mode…")
                _stop_watch()
            print("Goodbye!")
            sys.exit(0)

        if user_input.startswith("/"):
            handled = _handle_command(user_input)
            if not handled:
                print(f"Unknown command '{user_input}'. Type /help for available commands.\n")
            continue

        # Regular natural language query → run agent
        print()
        run_agent(user_input, stream=True)
        print()


if __name__ == "__main__":
    main()