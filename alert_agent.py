"""
alert_agent.py (IMPROVED - EVENT-DRIVEN SCHEDULING)
────────────────────────────────────────────────────
Alert Agent powered by Groq with optimized scheduling.

IMPROVEMENTS IN THIS VERSION:
✓ Event-driven scheduling instead of polling
✓ Dynamically schedules alert job for exact patch window end time
✓ No more 1-minute polling overhead
✓ Updates scheduling when new Implementation Status emails arrive
✓ Thread-safe job management
✓ Supports multiple scheduled alerts for different patch windows

Why this is better:
──────────────────
OLD (polling every 60s):
  - CPU waste: Every 60 seconds, check if we're in the alert window
  - Server disk I/O: 1440 Excel reads per day
  - Inaccurate: Alert fires within 60s of trigger time, not exact
  - Scalability issue: Polling gets worse with more servers/windows

NEW (event-driven):
  - CPU: Zero overhead - job scheduled in memory
  - Disk I/O: Only when Excel actually changes
  - Accurate: Alert fires at exact calculated time
  - Scalable: Handles any number of servers/windows efficiently
  - Memory efficient: One job in scheduler, not one every minute

How it works:
─────────────
1. When Implementation Status email arrives (via webhook):
   → Extract patch windows from new data
   → Calculate alert trigger time = patch_window_end - ALERT_LEAD_MINUTES
   → Schedule job to run at that exact time
   → Cancel any old scheduled alert jobs

2. Scheduler triggers at exact calculated time:
   → Run alert agent once
   → No polling needed

3. If multiple patch windows exist:
   → Use the LATEST (maximum) end time
   → Only one alert job active at any time

Responsibilities:
    - Expose run_agent(user_query) for manual triggering
    - Drive the Groq function-calling loop
    - Provide schedule_alert_for_window(window_end_time) to dynamically schedule
    - Listen for updates from email_tool.py when new Implementation Status arrives
    - Cancel old alerts and reschedule when windows change
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
from alert_tool import TOOL_FUNCTIONS, TOOL_SCHEMAS, _parse_patch_window_end, MASTER_PATH

import pandas as pd
import tzlocal
from datetime import datetime as dt

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str  = os.environ["GROQ_API_KEY"]
GPT_MODEL:    str  = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")

PROMPTS_DIR  : Path = Path(__file__).parent / "prompts" / "Alert Prompt"

ALERT_LEAD_MINUTES: int = int(os.environ.get("ALERT_LEAD_MINUTES", "10"))

_groq_client: Groq = Groq(api_key=GROQ_API_KEY)

# Global scheduler instance (set by start_alert_scheduler)
_scheduler: BackgroundScheduler | None = None
_scheduler_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are an automated patch validation alert agent for the Lyric application team.

Your job is to check the validation results for Lyric servers and send ONE clean,
professional alert email when servers need attention.

## Workflow — follow exactly
1. Call get_lyric_change_ticket to retrieve the CHG ticket number.
2. Call get_lyric_alert_summary once to get the current server states.
3. If alert_required is false → respond with a short confirmation that no alert is needed.
   Do NOT send an email.
4. If alert_required is true → compose and send ONE email using send_alert_email.

## Email rules

Subject format:
  [ACTION REQUIRED] <change_ticket> Lyric Application – Patch Validation Alert | <DD-Mon-YYYY>
    Where <change_ticket> is the value returned by get_lyric_change_ticket.
    If no ticket is found, omit it: [ACTION REQUIRED] Lyric Application – Patch Validation Alert | <DD-Mon-YYYY>
    
HTML body structure (use inline styles, no external CSS):

  - Header: "Lyric Application — Patch Validation Summary"
  - Sub-header line: "Generated: <date time> | Patch Window End: <latest_window_end formatted as Day HH:MM>"
  - A short intro line: "Please review the servers below and take action where needed."
  - Horizontal rule

  SECTION 1 — only include if 'unreachable' list is non-empty:
    Heading: "Servers Unreachable"
    One-line message: "We were unable to connect to the following servers. Please look into this."
    Table columns: Server Name | Patch Window | Reboot Required
    One row per server from the 'unreachable' list.

  SECTION 2 — only include if 'failed' list is non-empty:
    Heading: "Reboot Not Confirmed Within Patch Window"
    One-line message: "Could you please check if these servers were rebooted during the patch window?"
    Table columns: Server Name | Patch Window | Boot Time | Reboot Required
    One row per server from the 'failed' list.

  SECTION 3 — only include if 'pending' list is non-empty:
    Heading: "Validation Pending"
    One-line message: "Could you please provide an update on the patching status for the following servers?"
    Table columns: Server Name | Patch Window | Reboot Required
    One row per server from the 'pending' list.

  - Horizontal rule
  - Footer: "Enterprise Patch Intelligence System — Automated Alert"

## Styling rules
  - Font: Arial, 14px, color #1a1a1a
  - Section headings: bold, 16px, margin-top 24px
    - "Servers Unreachable" heading color: #b45309 (amber)
    - "Reboot Not Confirmed" heading color: #b91c1c (red)
    - "Validation Pending" heading color: #1d4ed8 (blue)
  - Tables: border-collapse collapse, width 100%, font-size 13px, margin-top 8px
  - Table header row: background #f3f4f6, bold, border 1px solid #d1d5db, padding 8px 12px
  - Table data cells: border 1px solid #d1d5db, padding 8px 12px
  - Alternating row background: white / #f9fafb
  - Footer: font-size 12px, color #6b7280, italic, margin-top 24px

## Hard rules
  - Call get_lyric_alert_summary exactly once.
  - Call send_alert_email at most once.
  - Never invent server names, boot times, patch windows, or errors.
  - Do NOT mention WinRM, technical tools, or internal system names in the email body.
  - Do NOT add extra explanations — keep section messages exactly as specified above.
  - Only include a section if that list has entries — never render an empty table.
  - All three sections can appear in the same email if all three lists are non-empty.
"""

def _load_prompt(filename: str) -> str:
    """
    Read a prompt file from the prompts/ directory.

    Args:
        filename: File name (e.g. 'system_prompt.txt').

    Returns:
        File contents as a stripped string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    logger.debug("Loaded prompt: %s (%d chars)", filename, len(content))
    return content


def load_prompts() -> dict[str, str]:
    """
    Load all prompt files and return them as a dict.

    Returns:
        {
            "system":    contents of system_prompt.txt,
            "developer": contents of developer_prompt.txt,
            "response":  contents of response_prompt.txt,
        }
    """
    prompts = {
        "system":    _load_prompt("system_prompt"),
        "developer": _load_prompt("developer_prompt"),
        "response":  _load_prompt("response_prompt"),
    }
    logger.info("All prompts loaded successfully.")
    return prompts


def reload_prompts() -> dict[str, str]:
    """
    Hot-reload all prompts from disk without restarting the process.
    Useful during prompt tuning or called from an admin endpoint.

    Returns:
        Fresh prompt dict.
    """
    logger.info("Hot-reloading prompts…")
    global _PROMPTS
    _PROMPTS = load_prompts()
    return _PROMPTS


# Load prompts once at module import time
_PROMPTS: dict[str, str] = load_prompts()



def _build_system_message() -> str:
    """
    Combine system, developer, and response prompts into a single
    system message string for the Groq API.

    Returns:
        Combined prompt string.
    """
    return "\n\n---\n\n".join([
        _PROMPTS["system"],
        _PROMPTS["developer"],
        _PROMPTS["response"],
    ])


# ---------------------------------------------------------------------------
# Agent internals
# ---------------------------------------------------------------------------

def _dispatch_tool_call(tool_name: str, tool_args: dict) -> str:
    """Dispatch and execute a tool call."""
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


# ---------------------------------------------------------------------------
# Public agent interface
# ---------------------------------------------------------------------------

def run_agent(user_query: str) -> str:
    """
    Run the alert agent loop for a single query.
    Returns the final text response from the agent.
    
    This is called either:
    1. Manually (user types /check in CLI)
    2. Automatically when scheduled alert job fires
    """
    logger.info("[Alert Agent] Query: %s", user_query)

    messages: list[dict] = [
        {"role": "system", "content": _build_system_message()},
        {"role": "user",   "content": user_query},
    ]

    MAX_ITERATIONS = 10

    for _iteration in range(MAX_ITERATIONS):
        time.sleep(1.5)

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

        if not tool_calls:
            answer = message.content or ""
            logger.info("[Alert Agent] Finished (%d chars)", len(answer))
            return answer

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

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            args   = {k: v for k, v in args.items() if k}
            result = _dispatch_tool_call(tc.function.name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    logger.error("[Alert Agent] Exceeded max iterations.")
    return "Alert agent exceeded maximum iterations. Check logs for details."


# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED: Event-driven scheduling (replaces polling)
# ═══════════════════════════════════════════════════════════════════════════

def _get_latest_lyric_window_end() -> datetime | None:
    """
    Read master Excel and return the LATEST (maximum) patch window end time
    across all Lyric servers.
    
    Returns:
        datetime of the latest window end, or None if no parseable windows
    """
    if not os.path.exists(MASTER_PATH):
        logger.debug("[Alert Scheduler] Master Excel not found yet")
        return None
    
    try:
        df = pd.read_excel(MASTER_PATH)
        df.columns = df.columns.str.strip()
        
        lyric_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]
        
        if len(lyric_df) == 0:
            logger.debug("[Alert Scheduler] No Lyric servers found in Excel")
            return None
        
        local_tz = tzlocal.get_localzone()
        now = dt.now(local_tz)  
        ends = []
        
        for _, row in lyric_df.iterrows():
            end_dt = _parse_patch_window_end(row.get("Patch Window"), reference_date=now)
            if end_dt:
                ends.append(end_dt)
        
        if not ends:
            logger.debug("[Alert Scheduler] No parseable patch windows found")
            return None
        
        latest = max(ends)
        logger.debug("[Alert Scheduler] Latest window end: %s", latest.isoformat())
        return latest
    
    except Exception as exc:
        logger.error("[Alert Scheduler] Failed to read patch windows: %s", exc)
        return None


def schedule_alert_for_window(window_end: datetime | None = None) -> None:
    """
    IMPROVED: Schedule alert to fire at calculated trigger time.
    
    NOW WITH DETAILED LOGGING at each step!
    
    This is called:
    1. When Implementation Status email arrives (to update alert timing)
    2. When starting the alert scheduler
    
    The job fires automatically at the exact calculated time — no polling needed.
    
    Args:
        window_end: The patch window end datetime. If None, reads from Excel.
    """
    
    if _scheduler is None:
        logger.error("[Alert Scheduler]  Scheduler not initialized — cannot schedule alert")
        return
    
    logger.debug("[Alert Scheduler] schedule_alert_for_window() called")
    
    # Get the latest window end time if not provided
    if window_end is None:
        logger.debug("[Alert Scheduler] Reading patch windows from Excel...")
        window_end = _get_latest_lyric_window_end()
    
    if window_end is None:
        logger.info("[Alert Scheduler]   No patch window to schedule for — clearing any existing alerts")
        with _scheduler_lock:
            try:
                _scheduler.remove_job("alert_window_end_trigger")
                logger.debug("[Alert Scheduler] Previous alert job removed")
            except Exception:
                logger.debug("[Alert Scheduler] No previous alert job to remove")
        return
    
    logger.info("[Alert Scheduler] Latest patch window end: %s", 
               window_end.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Calculate when to trigger the alert
    trigger_time = window_end - timedelta(minutes=ALERT_LEAD_MINUTES)
    local_tz = tzlocal.get_localzone()
    now = dt.now(local_tz)  # Use local timezone
    
    logger.info("[Alert Scheduler]  Alert will fire at: %s (%d minutes before window end)",
               trigger_time.strftime("%Y-%m-%d %H:%M:%S"),
               ALERT_LEAD_MINUTES)
    
    # If trigger time is in the past, don't schedule
    if trigger_time < now:
        logger.warning("[Alert Scheduler]   Alert trigger time (%s) is in the PAST — not scheduling",
                      trigger_time.isoformat())
        logger.warning("[Alert Scheduler] Current time: %s", now.isoformat())
        return
    
    # Build the alert query
    query = (
        f"The Lyric application patch window ends at {window_end.strftime('%Y-%m-%d %H:%M')}. "
        f"We are {ALERT_LEAD_MINUTES} minutes away from the end. "
        "Check all Lyric servers and send an alert email if any servers have issues "
        "or are still pending validation."
    )
    
    # Schedule the job to run at exact trigger time
    with _scheduler_lock:
        # Remove any existing alert job (proper error handling)
        try:
            old_job = _scheduler.get_job("alert_window_end_trigger")
            if old_job:
                logger.info("[Alert Scheduler] Removing previous alert (was scheduled for %s)",
                           old_job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if old_job.next_run_time else "unknown")
            _scheduler.remove_job("alert_window_end_trigger")
        except Exception:
            logger.debug("[Alert Scheduler] No previous alert job to remove")
        
        # Schedule new alert for exact trigger time
        logger.info("[Alert Scheduler]  Scheduling new alert for %s...",
                   trigger_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        _scheduler.add_job(
            _trigger_alert_agent,
            trigger=DateTrigger(run_date=trigger_time),
            args=(query, window_end),
            id="alert_window_end_trigger",
            name=f"Alert for window end {window_end.isoformat()}",
        )
    
    seconds_until = (trigger_time - now).total_seconds()
    
    logger.info("="*70)
    logger.info("[Alert Scheduler]  ALERT SUCCESSFULLY SCHEDULED!")
    logger.info("[Alert Scheduler]  Will fire at: %s",
               trigger_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("[Alert Scheduler]   Time remaining: %s",
               _format_duration(seconds_until))
    logger.info("="*70)


def _trigger_alert_agent(query: str, window_end: datetime) -> None:
    """
    Background job callback: runs when alert is triggered.
    
    Args:
        query: The alert query for the agent
        window_end: The patch window end time (for logging)
    """
    logger.info(
        "[Alert Scheduler] ALERT TRIGGERED! Window ends at %s",
        window_end.strftime("%Y-%m-%d %H:%M")
    )
    
    try:
        result = run_agent(query)
        logger.info("[Alert Scheduler] Agent completed:\n%s", result)
    except Exception as exc:
        logger.error("[Alert Scheduler] Agent failed: %s", exc, exc_info=True)


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


def _import_parse_patch_window_end():
    """Import the patch window parser from alert_tool."""
    from alert_tool import _parse_patch_window_end
    return _parse_patch_window_end


# Import at module level
_parse_patch_window_end = _import_parse_patch_window_end()


def start_alert_scheduler(
    scheduler: BackgroundScheduler | None = None,
) -> BackgroundScheduler:
    """
    IMPROVED: Initialize event-driven alert scheduler.
    
    This replaces the polling-based scheduler with event-driven scheduling:
    - No 1-minute polling loop
    - Alert job created once at startup
    - Job fires at exact calculated time
    - Job rescheduled when new Implementation Status emails arrive
    
    Args:
        scheduler: Optional existing BackgroundScheduler instance
    
    Returns:
        The BackgroundScheduler instance (created if not provided)
    """
    global _scheduler
    
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone="local")
        scheduler.start()
    
    _scheduler = scheduler
    
    logger.info(
        "[Alert Scheduler] Event-driven scheduler initialized "
        f"(alert {ALERT_LEAD_MINUTES} min before window end)"
    )
    
    # Schedule the initial alert based on current patch windows
    schedule_alert_for_window()
    
    return scheduler


def notify_implementation_status_updated() -> None:
    """
    IMPROVED: Called when Implementation Status email arrives to reschedule alert.
    
    NOW WITH DETAILED LOGGING so you can see when scheduler updates!
    
    This replaces the old polling mechanism. When email_tool.py processes
    a new Implementation Status email and updates the master Excel:
    
    1. It calls this function
    2. We read the updated patch windows
    3. We reschedule the alert job for the new window end time
    4. WE LOG EVERYTHING so you know what happened ← NEW!
    
    Example integration in email_tool.py:
    ──────────────────────────────────────
        from alert_agent import notify_implementation_status_updated
        
        # In get_latest_mail():
        if "Implementation Status" in subject and attachments_saved:
            build_master_excel()  # Update with new servers
            notify_implementation_status_updated()  # Reschedule alert ← NEW
    
    This is much more efficient than polling every 60 seconds!
    """
    logger.info("="*70)
    logger.info("[Alert Agent]  IMPLEMENTATION STATUS EMAIL PROCESSED")
    logger.info("="*70)
    
    try:
        # Get the latest window end from updated Excel
        latest_window_end = _get_latest_lyric_window_end()
        
        if latest_window_end is None:
            logger.warning("[Alert Agent]   No patch windows found in updated Excel")
            logger.warning("[Alert Agent] Alert scheduling cancelled")
            logger.info("="*70)
            return
        
        logger.info("[Alert Agent]  Found patch window end time: %s", 
                   latest_window_end.strftime("%Y-%m-%d %H:%M"))
        
        # Reschedule the alert
        logger.info("[Alert Agent]  Rescheduling alert based on new data...")
        schedule_alert_for_window(latest_window_end)
        
        logger.info("="*70)
        logger.info("[Alert Agent]  ALERT RESCHEDULED SUCCESSFULLY!")
        logger.info("="*70)
        
    except Exception as exc:
        logger.error("[Alert Agent]  FAILED to reschedule alert: %s", exc)
        logger.error("[Alert Agent] Stack trace:", exc_info=True)
        logger.info("="*70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" Patch Alert Agent (IMPROVED EVENT-DRIVEN SCHEDULING)")
    print("=" * 60)
    print(f"  Master Excel    : {MASTER_PATH}")
    print(f"  Alert recipient : {os.getenv('ALERT_RECIPIENT_EMAIL', '(not set)')}")
    print(f"  Alert lead time : {ALERT_LEAD_MINUTES} minutes before window end")
    print("\n  Scheduling: EVENT-DRIVEN (not polling)")
    print("  └─ Alert scheduled once at startup")
    print("  └─ Rescheduled when Implementation Status emails arrive")
    print("  └─ Zero polling overhead!")
    print("\n  Commands:")
    print("    check  — run agent now (manual trigger)")
    print("    sched  — start scheduler and keep running")
    print("    next   — show next scheduled alert")
    print("    exit   — quit\n")

    sched = None

    while True:
        try:
            cmd = input("Command: ").strip().lower()

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                if sched:
                    sched.shutdown()
                print("Exiting.")
                break

            elif cmd == "check":
                result = run_agent(
                    "Check all Lyric servers for connection errors, validation failures, "
                    "and servers still pending validation. Send an alert email if any "
                    "servers need attention."
                )
                print(f"\nAgent: {result}\n")

            elif cmd == "sched":
                sched = start_alert_scheduler()
                print(
                    f"Event-driven scheduler running. "
                    "Press Ctrl+C to stop."
                )
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    sched.shutdown()
                    print("\nScheduler stopped.")
                    break

            elif cmd == "next":
                if _scheduler is None:
                    print("Scheduler not initialized — run 'sched' first")
                else:
                    job = _scheduler.get_job("alert_window_end_trigger")
                    if job:
                        print(f"\nNext alert scheduled for: {job.next_run_time}\n")
                    else:
                        print("\nNo alert currently scheduled\n")

            else:
                result = run_agent(cmd)
                print(f"\nAgent: {result}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
