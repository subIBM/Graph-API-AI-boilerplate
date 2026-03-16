"""
alert_agent.py
--------------
Alert Agent powered by Groq.

Responsibilities:
    - Expose run_agent(user_query) for manual triggering
    - Drive the Groq function-calling loop to:
        1. Call get_lyric_alert_summary to assess server states
        2. If alerts exist, compose and send a professional HTML email
    - Run a scheduler that:
        * Checks the master Excel every minute
        * Computes the latest patch window end across all Lyric servers
        * Fires the alert agent exactly ALERT_LEAD_MINUTES before that end time
        * Does not fire again for the same window end time
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from groq import Groq

from alert_tool import TOOL_FUNCTIONS, TOOL_SCHEMAS, _parse_patch_window_end, MASTER_PATH

import pandas as pd

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str  = os.environ["GROQ_API_KEY"]
GPT_MODEL:    str  = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")

ALERT_LEAD_MINUTES:     int = int(os.environ.get("ALERT_LEAD_MINUTES",     "10"))
SCHEDULER_POLL_SECONDS: int = int(os.environ.get("SCHEDULER_POLL_SECONDS", "60"))

_groq_client: Groq        = Groq(api_key=GROQ_API_KEY)
_last_alerted_window_end: str | None = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are an automated patch validation alert agent for the Lyric application team.

Your job is to check the validation results for Lyric servers and send ONE clean,
professional alert email when servers need attention.

## Workflow — follow exactly
1. Call get_lyric_alert_summary once to get the current server states.
2. If alert_required is false → respond with a short confirmation that no alert is needed.
   Do NOT send an email.
3. If alert_required is true → compose and send ONE email using send_alert_email.

## Email rules

Subject format:
  [ACTION REQUIRED] Lyric Application – Patch Validation Alert | <DD-Mon-YYYY>

HTML body structure (use inline styles, no external CSS):

  - Header: "Lyric Application — Patch Validation Summary"
  - Sub-header line: "Generated: <date time> | Patch Window End: <latest_window_end formatted as Day HH:MM>"
  - A short intro line: "Please review the servers below and take action where needed."
  - Horizontal rule

  SECTION 1 — only include if 'unreachable' list is non-empty:
    Heading: "⚠ Servers Unreachable"
    One-line message: "We were unable to connect to the following servers. Please look into this."
    Table columns: Server Name | Patch Window | Reboot Required
    One row per server from the 'unreachable' list.

  SECTION 2 — only include if 'failed' list is non-empty:
    Heading: "✗ Reboot Not Confirmed Within Patch Window"
    One-line message: "Could you please check if these servers were rebooted during the patch window?"
    Table columns: Server Name | Patch Window | Boot Time | Reboot Required
    One row per server from the 'failed' list.

  SECTION 3 — only include if 'pending' list is non-empty:
    Heading: "? Validation Pending"
    One-line message: "Could you please provide an update on the patching status for the following servers?"
    Table columns: Server Name | Patch Window | Reboot Required
    One row per server from the 'pending' list.

  - Horizontal rule
  - Footer: "Enterprise Patch Intelligence System — Automated Alert"

## Styling rules
  - Font: Arial, 14px, color #1a1a1a
  - Section headings: bold, 16px, margin-top 24px
    - "⚠ Servers Unreachable" heading color: #b45309 (amber)
    - "✗ Reboot Not Confirmed" heading color: #b91c1c (red)
    - "? Validation Pending" heading color: #1d4ed8 (blue)
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

# ---------------------------------------------------------------------------
# Agent internals
# ---------------------------------------------------------------------------

def _dispatch_tool_call(tool_name: str, tool_args: dict) -> str:
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
    """
    logger.info("[Alert Agent] Query: %s", user_query)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
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


# ---------------------------------------------------------------------------
# Scheduler logic
# ---------------------------------------------------------------------------

def _get_latest_lyric_window_end() -> datetime | None:
    if not os.path.exists(MASTER_PATH):
        return None
    try:
        df = pd.read_excel(MASTER_PATH)
        df.columns = df.columns.str.strip()
        lyric_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]
        now  = datetime.now()
        ends = []
        for _, row in lyric_df.iterrows():
            end_dt = _parse_patch_window_end(row.get("Patch Window"), reference_date=now)
            if end_dt:
                ends.append(end_dt)
        return max(ends) if ends else None
    except Exception as exc:
        logger.error("[Alert Scheduler] Failed to read patch windows: %s", exc)
        return None


def _scheduler_tick() -> None:
    global _last_alerted_window_end

    latest_end = _get_latest_lyric_window_end()
    if latest_end is None:
        logger.debug("[Alert Scheduler] No parseable patch windows — skipping.")
        return

    now        = datetime.now()
    alert_from = latest_end - timedelta(minutes=ALERT_LEAD_MINUTES)
    window_key = latest_end.isoformat()

    if _last_alerted_window_end == window_key:
        logger.debug("[Alert Scheduler] Already alerted for %s — skipping.", window_key)
        return

    if not (alert_from <= now <= latest_end):
        logger.debug("[Alert Scheduler] Outside alert window — not firing yet.")
        return

    logger.info(
        "[Alert Scheduler] %d min before patch window end (%s) — triggering alert agent.",
        ALERT_LEAD_MINUTES, latest_end.strftime("%Y-%m-%d %H:%M"),
    )
    _last_alerted_window_end = window_key

    query = (
        f"The Lyric application patch window ends at {latest_end.strftime('%Y-%m-%d %H:%M')}. "
        f"We are {ALERT_LEAD_MINUTES} minutes away from the end. "
        "Check all Lyric servers and send an alert email if any servers have issues "
        "or are still pending validation."
    )

    try:
        result = run_agent(query)
        logger.info("[Alert Scheduler] Agent completed:\n%s", result)
    except Exception as exc:
        logger.error("[Alert Scheduler] Agent failed: %s", exc, exc_info=True)


def start_alert_scheduler(
    scheduler: BackgroundScheduler | None = None,
) -> BackgroundScheduler:
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone="local")
        scheduler.start()
        logger.info(
            "[Alert Scheduler] Started (poll every %ds, fire %d min before window end).",
            SCHEDULER_POLL_SECONDS, ALERT_LEAD_MINUTES,
        )

    scheduler.add_job(
        _scheduler_tick,
        "interval",
        seconds          = SCHEDULER_POLL_SECONDS,
        id               = "alert_scheduler_tick",
        replace_existing = True,
    )
    logger.info("[Alert Scheduler] Job registered — polling every %ds.", SCHEDULER_POLL_SECONDS)
    return scheduler


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(" Patch Alert Agent")
    print("=" * 55)
    print(f"  Master Excel    : {MASTER_PATH}")
    print(f"  Alert recipient : {os.getenv('ALERT_RECIPIENT_EMAIL', '(not set)')}")
    print(f"  Alert lead time : {ALERT_LEAD_MINUTES} minutes before window end")
    print(f"  Scheduler poll  : every {SCHEDULER_POLL_SECONDS}s")
    print("\n  Commands:")
    print("    check  — run agent now (manual trigger)")
    print("    sched  — start scheduler and keep running")
    print("    exit   — quit\n")

    while True:
        try:
            cmd = input("Command: ").strip().lower()

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
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
                    f"Scheduler running — polling every {SCHEDULER_POLL_SECONDS}s. "
                    "Press Ctrl+C to stop."
                )
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    sched.shutdown()
                    print("\nScheduler stopped.")
                    break

            else:
                result = run_agent(cmd)
                print(f"\nAgent: {result}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
