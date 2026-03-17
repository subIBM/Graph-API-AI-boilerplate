"""
validation_agent.py
-------------------
Core agent loop powered by Groq (openai/gpt-oss-120b).

Responsibilities:
    - Expose run_agent(user_query) — the primary public interface
    - Drive the Groq function-calling loop until no more tool calls remain
    - Dispatch tool calls to TOOL_FUNCTIONS in validation_tool.py
    - Stream the final response token-by-token to stdout (optional)

Usage:
    from validation_agent import run_agent, run_predefined
    answer = run_agent("Fetch boot times and validate all lyric servers")
"""

from __future__ import annotations

import json
import logging
import os
import time

from dotenv import load_dotenv
from groq import Groq

from validation_tool import TOOL_FUNCTIONS, TOOL_SCHEMAS, MASTER_PATH, WINRM_USER

load_dotenv()
logger = logging.getLogger(__name__)
TOOL_CALL_DELAY_SECONDS: int = int(os.environ.get("TOOL_CALL_DELAY", "1"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GPT_MODEL:    str = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")

# Groq client — single instance
_groq_client: Groq = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a Patch Validation Agent. Your job is to:\n"
    "1. Read lyric application servers from the master Excel.\n"
    "2. Connect to each server via WinRM and fetch its last boot time.\n"
    "3. Write the boot time into the master Excel ('Boot Time' column).\n"
    "4. Validate whether the boot time falls within the server's Patch Window.\n"
    "5. Update 'Application Team Validation Status' as 'Successful' or 'Failed'.\n\n"
    "Always use the provided tools — never guess server data.\n"
    "Process servers STRICTLY one at a time in this exact sequence for each server:\n"
    "  get_server_boot_time → update_boot_time_in_excel → validate_boot_within_patch_window\n"
    "Complete all three steps for one server before moving to the next.\n"
    "Do NOT batch multiple servers in a single tool-call round.\n"
    "Be concise and professional."
)
# ---------------------------------------------------------------------------
# Predefined prompts
# ---------------------------------------------------------------------------

PREDEFINED_PROMPTS: dict[str, str] = {
    "full_validation": (
        "Get all lyric servers from Excel, connect to each one via WinRM to fetch "
        "the boot time, save it to Excel, then validate if it's within the patch window "
        "and update the Application Team Validation Status for every server."
    ),
    "boot_times_only": (
        "Get all lyric servers from Excel and fetch the boot time for each server "
        "via WinRM. Save each boot time to the master Excel."
    ),
    "validate_only": (
        "For all lyric servers in the master Excel that already have a Boot Time recorded, "
        "validate whether the boot time is within the patch window and update the "
        "Application Team Validation Status column."
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        preview = result[:200] + ("…" if len(result) > 200 else "")
        logger.info("  [Result] %s", preview)
        return result
    except TypeError as exc:
        logger.error("Tool call %s failed with bad arguments: %s", tool_name, exc)
        return json.dumps({"error": f"Invalid arguments for {tool_name}: {exc}"})
    except Exception as exc:
        logger.error("Tool call %s raised unexpected error: %s", tool_name, exc)
        return json.dumps({"error": f"Tool '{tool_name}' failed: {exc}"})


def _stream_final_response(stream) -> str:
    """
    Consume a streaming Groq response, printing tokens as they arrive.

    Args:
        stream: Groq streaming completion object.

    Returns:
        The complete assembled response text.
    """
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        token = delta.content or ""
        if token:
            print(token, end="", flush=True)
            full_text += token
    print()  # newline after streaming completes
    return full_text


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_query: str, stream: bool = True) -> str:
    """
    Run the Groq agent loop for a single user query.

    The loop continues until the model stops requesting tool calls,
    at which point the final text response is returned.

    Args:
        user_query: Natural language question or instruction from the user.
        stream:     If True, stream the final response tokens to stdout.

    Returns:
        The agent's final text response.
    """
    logger.info("User: %s", user_query)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    while True:
        time.sleep(1.5) 
        response = _groq_client.chat.completions.create(
            model                 = GPT_MODEL,
            messages              = messages,
            tools                 = TOOL_SCHEMAS,
            tool_choice           = "auto",
            temperature           = 1,
            max_completion_tokens = 8192,
            top_p                 = 1,
            reasoning_effort      = "medium",
            stream                = False,   # stream=False during tool-call phase
        )

        message    = response.choices[0].message
        tool_calls = message.tool_calls or []

        # No tool calls → model is ready to produce its final answer
        if not tool_calls:
            break

        # Append the model's assistant turn (with tool_calls) to history
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

            result = _dispatch_tool_call(tc.function.name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })
            time.sleep(TOOL_CALL_DELAY_SECONDS)

    # ---- Final response ----------------------------------------------------
    if stream:
        final_stream = _groq_client.chat.completions.create(
            model                 = GPT_MODEL,
            messages              = messages,
            tools                 = TOOL_SCHEMAS,
            tool_choice           = "auto",   
            temperature           = 1,
            max_completion_tokens = 8192,
            top_p                 = 1,
            reasoning_effort      = "medium",
            stream                = True,
        )
        return _stream_final_response(final_stream)

    # Non-streaming fallback
    final_response = _groq_client.chat.completions.create(
        model                 = GPT_MODEL,
        messages              = messages,
        tools                 = TOOL_SCHEMAS,
        tool_choice           = "auto",   
        temperature           = 1,
        max_completion_tokens = 8192,
        top_p                 = 1,
        reasoning_effort      = "medium",
        stream                = False,
    )
    answer = final_response.choices[0].message.content or ""
    logger.info("Agent response (%d chars)", len(answer))
    return answer


def run_predefined(prompt_key: str, stream: bool = True) -> str:
    """
    Run a predefined query by its registry key.

    Args:
        prompt_key: Key from PREDEFINED_PROMPTS (e.g. 'full_validation').
        stream:     Whether to stream the final response.

    Returns:
        The agent's final text response.

    Raises:
        ValueError: If the key does not exist in PREDEFINED_PROMPTS.
    """
    if prompt_key not in PREDEFINED_PROMPTS:
        available = list(PREDEFINED_PROMPTS.keys())
        raise ValueError(f"Unknown prompt key '{prompt_key}'. Available: {available}")

    return run_agent(PREDEFINED_PROMPTS[prompt_key], stream=stream)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(" Patch Validation Agent")
    print("=" * 55)
    print(f"  Master Excel : {MASTER_PATH}")
    print(f"  WinRM user   : {WINRM_USER}")
    print(f"  Groq model   : {GPT_MODEL}")
    print("\n  Predefined prompts (type /run <key>):")
    for key in PREDEFINED_PROMPTS:
        print(f"    /run {key}")
    print("  Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting!")
                break
            if user_input.startswith("/run "):
                key    = user_input[5:].strip()
                result = run_predefined(key)
                print(f"\nAgent: {result}")
            else:
                result = run_agent(user_input)
                print(f"\nAgent: {result}")
        except KeyboardInterrupt:
            print("\nExiting!")
            break
