"""
─────────────
Alert Agent powered by Groq (openai/gpt-oss-120b) with event-driven scheduling.

Responsibilities:
    - Load all three prompt files once at startup from alert_prompts/
    - Expose run_agent(user_query) for manual and scheduled triggering
    - Drive the Groq function-calling loop (max 10 iterations)
    - Dispatch tool calls to TOOL_FUNCTIONS in alert_tool.py
    - Provide schedule_alert_for_window() for exact-time alert scheduling
    - Reschedule automatically when Implementation Status emails arrive
    - Provide reload_prompts() for hot-reloading without restart

Prompt files (loaded once at startup from alert_prompts/):
    system_prompt.txt    — agent identity, governance rules, email rules
    developer_prompt.txt — reasoning gates, tool call rules, composition logic
    response_prompt.txt  — email HTML structure, styling, subject format

Scheduling — EVENT-DRIVEN (not polling):
    Old: poll every 60s → check if in alert window → high CPU + disk I/O
    New: read patch window end once → schedule one job at exact trigger time
         → zero overhead between trigger and alert

    Trigger time = patch_window_end - ALERT_LEAD_MINUTES
    Job rescheduled whenever notify_implementation_status_updated() is called.

Usage:
    from alert_agent import run_agent, start_alert_scheduler
    from alert_agent import notify_implementation_status_updated
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tzlocal
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from dotenv import load_dotenv
from groq import Groq

from alert_tool import (
    MASTER_PATH,
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    _parse_patch_window_end,
)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY        : str = os.environ["GROQ_API_KEY"]
GPT_MODEL           : str = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")
ALERT_LEAD_MINUTES  : int = int(os.environ.get("ALERT_LEAD_MINUTES", "10"))

PROMPTS_DIR: Path = Path(__file__).parent / "alert_prompts"

# Groq client — single instance
_groq_client: Groq = Groq(api_key=GROQ_API_KEY)

# Scheduler state
_scheduler      : BackgroundScheduler | None = None
_scheduler_lock : threading.Lock             = threading.Lock()

# ---------------------------------------------------------------------------
# Prompt loading — once at startup
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    """
    Read a prompt file from the alert_prompts/ directory.

    Args:
        filename: File name (e.g. 'system_prompt.txt').

    Returns:
        File contents as a stripped string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected directory: {PROMPTS_DIR}"
        )
    content = path.read_text(encoding="utf-8").strip()
    logger.debug("Loaded prompt: %s (%d chars)", filename, len(content))
    return content


def load_prompts() -> dict[str, str]:
    """
    Load all three prompt files and return them as a dict.

    Returns:
        {
            "system":    contents of system_prompt.txt,
            "developer": contents of developer_prompt.txt,
            "response":  contents of response_prompt.txt,
        }
    """
    prompts = {
        "system":    _load_prompt("system_prompt.txt"),
        "developer": _load_prompt("developer_prompt.txt"),
        "response":  _load_prompt("response_prompt.txt"),
    }
    logger.info("Alert agent prompts loaded successfully.")
    return prompts


def reload_prompts() -> dict[str, str]:
    """
    Hot-reload all prompts from disk without restarting.
    Call this after editing any prompt file, or use the /reload CLI command.

    Returns:
        Fresh prompt dict.
    """
    logger.info("Hot-reloading alert prompts…")
    global _PROMPTS
    _PROMPTS = load_prompts()
    return _PROMPTS


# Load once at module import time
_PROMPTS: dict[str, str] = load_prompts()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_system_message() -> str:
    """
    Combine system, developer, and response prompts into a single
    system message for the Groq API.

    Returns:
        Combined prompt string with section separators.
    """
    return "\n\n---\n\n".join([
        _PROMPTS["system"],
        _PROMPTS["developer"],
        _PROMPTS["response"],
    ])


def _dispatch_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Look up and execute a tool by name.

    Args:
        tool_name: The function name the model requested.
        tool_args: Parsed argument dict from the model.

    Returns:
        JSON string result from the tool function.
    """
    func = TOOL_FUNCTIONS.get(tool_name)

    if func is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        logger.info("  [Tool] %s(%s)", tool_name, tool_args)
        result  = func(**tool_args)
        preview = result[:300] + ("…" if len(result) > 300 else "")
        logger.info("  [Result] %s", preview)
        return result
    except TypeError as exc:
        logger.error("Tool %s bad arguments: %s", tool_name, exc)
        return json.dumps({"error": f"Invalid arguments for {tool_name}: {exc}"})
    except Exception as exc:
        logger.error("Tool %s error: %s", tool_name, exc)
        return json.dumps({"error": f"Tool '{tool_name}' failed: {exc}"})


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        mins  = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"
    else:
        days  = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run_agent(user_query: str) -> str:
    """
    Run the alert agent loop for a single query.

    Drives the Groq function-calling loop until no tool calls remain,
    then returns the final text response.

    Called either:
      1. Manually — user types a query or 'check' in the CLI
      2. Automatically — APScheduler DateTrigger fires at alert time

    Args:
        user_query: Natural language query or instruction.

    Returns:
        The agent's final text response.
    """
    logger.info("[Alert Agent] Query: %s", user_query)

    messages: list[dict] = [
        {"role": "system", "content": _build_system_message()},
        {"role": "user",   "content": user_query},
    ]

    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        time.sleep(1.5)   # throttle Groq API calls

        response = _groq_client.chat.completions.create(
            model                 = GPT_MODEL,
            messages              = messages,
            tools                 = TOOL_SCHEMAS,
            tool_choice           = "auto",
            temperature           = 1,
            max_completion_tokens = 4096,
            top_p                 = 1,
            reasoning_effort      = "medium",
            stream                = False,
        )

        message    = response.choices[0].message
        tool_calls = message.tool_calls or []

        # No tool calls → model ready to write final answer
        if not tool_calls:
            answer = message.content or ""
            logger.info("[Alert Agent] Completed in %d iteration(s) (%d chars)",
                        iteration + 1, len(answer))
            return answer

        # Append assistant turn with tool calls to history
        messages.append({
            "role":       "assistant",
            "content":    message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute each tool and append results
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Strip empty keys that some models inject
            args   = {k: v for k, v in args.items() if k}
            result = _dispatch_tool_call(tc.function.name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    logger.error("[Alert Agent] Exceeded max iterations (%d).", MAX_ITERATIONS)
    return "Alert agent exceeded maximum iterations. Check logs for details."


# ---------------------------------------------------------------------------
# Event-driven scheduling
# ---------------------------------------------------------------------------

def _get_latest_lyric_window_end() -> datetime | None:
    """
    Read the master Excel and return the LATEST patch window end time
    across all Lyric servers.

    Returns:
        Timezone-aware datetime of the latest window end, or None if
        no parseable windows are found.
    """
    if not os.path.exists(MASTER_PATH):
        logger.debug("[Alert Scheduler] Master Excel not found.")
        return None

    try:
        df = pd.read_excel(MASTER_PATH)
        df.columns = df.columns.str.strip()

        lyric_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]

        if lyric_df.empty:
            logger.debug("[Alert Scheduler] No Lyric servers in Excel.")
            return None

        local_tz = tzlocal.get_localzone()
        now      = datetime.now(local_tz)
        ends     = []

        for _, row in lyric_df.iterrows():
            end_dt = _parse_patch_window_end(
                row.get("Patch Window"), reference_date=now
            )
            if end_dt:
                ends.append(end_dt)

        if not ends:
            logger.debug("[Alert Scheduler] No parseable patch windows found.")
            return None

        latest = max(ends)
        logger.debug("[Alert Scheduler] Latest window end: %s", latest.isoformat())
        return latest

    except Exception as exc:
        logger.error("[Alert Scheduler] Failed to read patch windows: %s", exc)
        return None


def schedule_alert_for_window(window_end: datetime | None = None) -> None:
    """
    Schedule the alert job to fire at (window_end - ALERT_LEAD_MINUTES).

    If window_end is None, reads the latest patch window from the master Excel.
    Cancels any previously scheduled alert job before creating the new one.
    Does nothing if the calculated trigger time is already in the past.

    Called:
      1. At startup via start_alert_scheduler()
      2. When a new Implementation Status email arrives via
         notify_implementation_status_updated()

    Args:
        window_end: Patch window end datetime. If None, reads from Excel.
    """
    if _scheduler is None:
        logger.error("[Alert Scheduler] Scheduler not initialized.")
        return

    if window_end is None:
        window_end = _get_latest_lyric_window_end()

    if window_end is None:
        logger.info("[Alert Scheduler] No patch window found — clearing any existing alert job.")
        with _scheduler_lock:
            try:
                _scheduler.remove_job("alert_window_end_trigger")
            except Exception:
                pass
        return

    local_tz     = tzlocal.get_localzone()
    now          = datetime.now(local_tz)
    trigger_time = window_end - timedelta(minutes=ALERT_LEAD_MINUTES)

    logger.info("[Alert Scheduler] Latest window end : %s",
                window_end.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("[Alert Scheduler] Alert trigger time: %s (%d min before end)",
                trigger_time.strftime("%Y-%m-%d %H:%M:%S"), ALERT_LEAD_MINUTES)

    if trigger_time <= now:
        logger.warning(
            "[Alert Scheduler] Trigger time is in the past (%s) — not scheduling.",
            trigger_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    query = (
        f"The Lyric application patch window ends at "
        f"{window_end.strftime('%Y-%m-%d %H:%M')}. "
        f"We are {ALERT_LEAD_MINUTES} minutes away from the end. "
        "Check all Lyric servers and send an alert email if any servers "
        "are unreachable, failed, or still pending validation."
    )

    with _scheduler_lock:
        # Cancel any existing alert job
        try:
            existing = _scheduler.get_job("alert_window_end_trigger")
            if existing:
                logger.info(
                    "[Alert Scheduler] Removing previous alert job (was: %s).",
                    existing.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
                    if existing.next_run_time else "unknown",
                )
            _scheduler.remove_job("alert_window_end_trigger")
        except Exception:
            pass

        _scheduler.add_job(
            _trigger_alert_agent,
            trigger = DateTrigger(run_date=trigger_time),
            args    = (query, window_end),
            id      = "alert_window_end_trigger",
            name    = f"Alert for window end {window_end.isoformat()}",
        )

    seconds_until = (trigger_time - now).total_seconds()
    logger.info("=" * 65)
    logger.info("[Alert Scheduler] ALERT SCHEDULED")
    logger.info("[Alert Scheduler] Fires at : %s",
                trigger_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("[Alert Scheduler] In       : %s", _format_duration(seconds_until))
    logger.info("=" * 65)


def _trigger_alert_agent(query: str, window_end: datetime) -> None:
    """
    APScheduler callback — called at the exact trigger time.

    Args:
        query:      The alert query passed to run_agent.
        window_end: The patch window end time (for logging).
    """
    logger.info(
        "[Alert Scheduler] ALERT TRIGGERED — window ends at %s",
        window_end.strftime("%Y-%m-%d %H:%M"),
    )
    try:
        result = run_agent(query)
        logger.info("[Alert Scheduler] Agent completed:\n%s", result)
    except Exception as exc:
        logger.error("[Alert Scheduler] Agent failed: %s", exc, exc_info=True)


def notify_implementation_status_updated() -> None:
    """
    Called by email_tool.py when a new Implementation Status email is processed.

    Reads the updated patch windows from the master Excel and reschedules
    the alert job for the new window end time. This is the event-driven
    replacement for polling.

    Integration in email_tool.py (get_latest_mail):
        from alert_agent import notify_implementation_status_updated

        if "Implementation Status" in subject and attachments_saved:
            build_master_excel()
            notify_implementation_status_updated()   ← call this
    """
    logger.info("=" * 65)
    logger.info("[Alert Agent] Implementation Status email processed — rescheduling alert.")

    try:
        latest_window_end = _get_latest_lyric_window_end()

        if latest_window_end is None:
            logger.warning("[Alert Agent] No patch windows found — alert scheduling cancelled.")
            logger.info("=" * 65)
            return

        logger.info("[Alert Agent] New window end: %s",
                    latest_window_end.strftime("%Y-%m-%d %H:%M"))
        schedule_alert_for_window(latest_window_end)

        logger.info("[Alert Agent] Alert rescheduled successfully.")
        logger.info("=" * 65)

    except Exception as exc:
        logger.error("[Alert Agent] Failed to reschedule alert: %s", exc, exc_info=True)
        logger.info("=" * 65)


def start_alert_scheduler(
    scheduler: BackgroundScheduler | None = None,
) -> BackgroundScheduler:
    """
    Initialize and start the event-driven alert scheduler.

    Creates a BackgroundScheduler if one is not provided, then reads the
    current patch windows from the master Excel and schedules the initial
    alert job. No polling — one job, exact trigger time.

    Args:
        scheduler: Optional existing BackgroundScheduler instance to reuse.

    Returns:
        The BackgroundScheduler instance.
    """
    global _scheduler

    if scheduler is None:
        scheduler = BackgroundScheduler(timezone="local")
        scheduler.start()

    _scheduler = scheduler

    logger.info(
        "[Alert Scheduler] Event-driven scheduler started "
        "(alert %d min before window end).",
        ALERT_LEAD_MINUTES,
    )

    schedule_alert_for_window()
    return scheduler


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    print("=" * 65)
    print("  Patch Alert Agent — Event-Driven Scheduling")
    print("=" * 65)
    print(f"  Master Excel    : {MASTER_PATH}")
    print(f"  Alert recipient : {os.getenv('ALERT_RECIPIENT_EMAIL', '(not set)')}")
    print(f"  Alert lead time : {ALERT_LEAD_MINUTES} min before window end")
    print(f"  Prompts dir     : {PROMPTS_DIR}")
    print()
    print("  Scheduling: EVENT-DRIVEN (not polling)")
    print("    Alert scheduled once at startup")
    print("    Rescheduled on each Implementation Status email")
    print()
    print("  Commands:")
    print("    check    — run agent now (manual trigger)")
    print("    sched    — start scheduler and keep running")
    print("    next     — show next scheduled alert time")
    print("    /reload  — hot-reload all prompt files from disk")
    print("    exit     — quit")
    print("=" * 65)
    print()

    _sched = None

    while True:
        try:
            cmd = input("Command: ").strip().lower()

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                if _sched:
                    _sched.shutdown()
                print("Exiting.")
                break

            elif cmd == "check":
                print()
                result = run_agent(
                    "Check all Lyric servers for connection errors, validation "
                    "failures, and servers still pending validation. Send an "
                    "alert email if any servers need attention."
                )
                print(f"\n{result}\n")

            elif cmd == "sched":
                _sched = start_alert_scheduler()
                print("Event-driven scheduler running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    _sched.shutdown()
                    print("\nScheduler stopped.")
                    break

            elif cmd == "next":
                if _scheduler is None:
                    print("Scheduler not initialized — run 'sched' first.\n")
                else:
                    job = _scheduler.get_job("alert_window_end_trigger")
                    if job and job.next_run_time:
                        print(f"\nNext alert: {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    else:
                        print("\nNo alert currently scheduled.\n")

            elif cmd == "/reload":
                prompts = reload_prompts()
                print(f"✓ Prompts reloaded: {list(prompts.keys())}\n")

            else:
                print()
                result = run_agent(cmd)
                print(f"\n{result}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            if _sched:
                _sched.shutdown()
            break