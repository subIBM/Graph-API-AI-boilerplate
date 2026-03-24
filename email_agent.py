"""

Core agent loop powered by Groq (openai/gpt-oss-120b).

Responsibilities:
    - Load all three prompt files once at startup
    - Expose run_agent(user_query) — the primary public interface
    - Drive the Groq function-calling loop until no more tool calls remain
    - Dispatch tool calls to TOOL_REGISTRY in email_tool.py
    - Stream the final response token-by-token to stdout (optional)

Usage:
    from email_agent import run_agent
    answer = run_agent("Show me all Lyric servers with their patch windows")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from email_tool import TOOL_REGISTRY, TOOL_SCHEMAS


load_dotenv()
logger = logging.getLogger(__name__)


GROQ_API_KEY : str = os.environ["GROQ_API_KEY"]
GPT_MODEL    : str = os.environ.get("GPT_MODEL", "openai/gpt-oss-120b")

PROMPTS_DIR  : Path = Path(__file__).parent / "prompts" / "Email Prompt"

# Groq client — single instance
_groq_client: Groq = Groq(api_key=GROQ_API_KEY)


PREDEFINED_PROMPTS: dict[str, str] = {
    "lyric_servers_patch": (
        "Find all Lyric application servers and tell me their patch day window."
    ),
    "full_summary": (
        "Give me a complete summary of all servers: total count, how many need reboots, "
        "unique patch windows, and any servers with a downtime flag."
    ),
    "mail_and_patch_check": (
        "Get the latest email, extract any server names mentioned in the subject or body, "
        "then check the Excel data to see if those servers have patch windows defined."
    ),
}



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


def _dispatch_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Look up and execute a tool by name.

    Args:
        tool_name: The function name the model requested.
        tool_args: Parsed argument dict from the model.

    Returns:
        JSON string result from the tool function.
    """
    func = TOOL_REGISTRY.get(tool_name)

    if func is None:
        logger.warning("Unknown tool requested: %s", tool_name)
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})

    try:
        logger.info("  [Tool] %s(%s)", tool_name, tool_args)
        result = func(**tool_args)
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
        {"role": "system",  "content": _build_system_message()},
        {"role": "user",    "content": user_query},
    ]


    while True:
        response = _groq_client.chat.completions.create(
            model               = GPT_MODEL,
            messages            = messages,
            tools               = TOOL_SCHEMAS,
            tool_choice         = "auto",
            temperature         = 1,
            max_completion_tokens = 8192,
            top_p               = 1,
            reasoning_effort    = "medium",
            stream              = False,    # stream=False during tool-call phase
        )

        message     = response.choices[0].message
        tool_calls  = message.tool_calls or []

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


    if stream:
        final_stream = _groq_client.chat.completions.create(
            model                 = GPT_MODEL,
            messages              = messages,
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
        prompt_key: Key from PREDEFINED_PROMPTS (e.g. 'lyric_servers_patch').
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