"""
Summarizer callback for LiteLLM.

When the combined message history approaches the truncation threshold,
this callback replaces the oldest non-system turns with a compact LLM-generated
summary, keeping total token count within the context window.

Flow:
  1. async_pre_call_hook fires before every request.
  2. Token count is estimated (char-based heuristic, ~4 chars/token).
  3. If estimated tokens > SUMMARIZE_THRESHOLD, call the local model to
     summarize the middle portion of the conversation.
  4. The summary replaces the middle turns with a single assistant message
     prefixed with "[SUMMARY]".
"""

from __future__ import annotations

import os
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Start summarizing when estimated input tokens exceed this fraction of
# truncate_input_tokens.  Tune via SUMMARIZER_THRESHOLD_RATIO env var.
_THRESHOLD_RATIO = float(os.getenv("SUMMARIZER_THRESHOLD_RATIO", "0.75"))
_TRUNCATE_LIMIT = int(os.getenv("LITELLM_TRUNCATE_INPUT_TOKENS", "30000"))
SUMMARIZE_THRESHOLD = int(_TRUNCATE_LIMIT * _THRESHOLD_RATIO)

# How many of the most recent turns to keep verbatim (not summarized).
KEEP_RECENT_TURNS = int(os.getenv("SUMMARIZER_KEEP_RECENT", "6"))

_SUMMARY_SYSTEM_PROMPT = (
    "You are a concise assistant. Summarize the following conversation excerpt "
    "in 3-5 bullet points, preserving all key decisions, file paths, code snippets, "
    "and tool results mentioned. Be terse."
)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total characters / 4."""
    return sum(len(str(m.get("content", ""))) for m in messages) // 4


def _partition(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split messages into:
      head  — system messages at the start
      middle — turns eligible for summarization
      tail  — the most recent KEEP_RECENT_TURNS turns
    """
    head = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= KEEP_RECENT_TURNS:
        return head, [], non_system

    tail = non_system[-KEEP_RECENT_TURNS:]
    middle = non_system[: -KEEP_RECENT_TURNS]
    return head, middle, tail


async def _summarize(middle: list[dict], api_base: str, model: str) -> str:
    """Call the local model to produce a summary of the middle turns."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {str(m.get('content', ''))[:1000]}" for m in middle
    )
    try:
        resp = await litellm.acompletion(
            model=f"anthropic/{model}",
            api_base=api_base,
            api_key="local",
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        # Fallback: simple truncation of middle turns if summarization fails.
        return f"[SUMMARY unavailable: {exc}]\n" + transcript[:500]


class SummarizerCallback(CustomLogger):
    """
    LiteLLM pre-call hook that compresses conversation history via summarization
    when the estimated token count exceeds SUMMARIZE_THRESHOLD.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        messages = data.get("messages", [])
        if _estimate_tokens(messages) <= SUMMARIZE_THRESHOLD:
            return data

        head, middle, tail = _partition(messages)
        if not middle:
            return data

        # Resolve api_base and model from the outgoing request params.
        litellm_params = data.get("litellm_params", {})
        api_base = (
            litellm_params.get("api_base")
            or os.getenv("LLAMA_API_BASE", "http://llama-server:8080")
        )
        model = str(data.get("model", "local-model")).removeprefix("anthropic/")

        summary_text = await _summarize(middle, api_base, model)
        summary_message = {
            "role": "assistant",
            "content": f"[SUMMARY of earlier conversation]\n{summary_text}",
        }

        data["messages"] = head + [summary_message] + tail
        return data
