"""
Middle-truncation callback for LiteLLM.

llama.cpp does not implement proxy-side "truncate middle" behavior. This hook
enforces it before forwarding requests upstream:
  - keep leading system messages
  - keep the most recent turns
  - drop oldest non-system middle turns until under token budget
"""

from __future__ import annotations

import os
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

TRUNCATE_LIMIT = int(os.getenv("LITELLM_TRUNCATE_INPUT_TOKENS", "30000"))
KEEP_RECENT_TURNS = int(os.getenv("TRUNCATE_MIDDLE_KEEP_RECENT", "8"))
MIN_HEAD_NON_SYSTEM = int(os.getenv("TRUNCATE_MIDDLE_KEEP_HEAD", "2"))


def _content_len(content: Any) -> int:
    """Estimate text length for string/list/dict message content."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    total += len(str(image_url.get("url", ""))) // 4
                elif isinstance(image_url, str):
                    total += len(image_url) // 4
            else:
                total += len(str(item))
        return total
    if isinstance(content, dict):
        return len(str(content.get("text", ""))) + len(str(content)) // 4
    return len(str(content))


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    # Rough, intentionally conservative estimate for mixed text/image payloads.
    chars = 0
    for msg in messages:
        chars += 16  # per-message overhead
        chars += len(str(msg.get("role", "")))
        chars += _content_len(msg.get("content"))
    return chars // 4


def _split_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    head: list[dict[str, Any]] = []
    rest_start = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            head.append(msg)
            rest_start = i + 1
            continue
        break
    rest = messages[rest_start:]
    return head, rest


class TruncateMiddleCallback(CustomLogger):
    """LiteLLM pre-call hook that drops middle turns when context is too large."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return data

        return truncate_data_messages(data)


def truncate_data_messages(data: dict[str, Any]) -> dict[str, Any]:
    """Apply the same truncation logic to a raw request data dict."""
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return data

    typed_messages = [m for m in messages if isinstance(m, dict)]
    if _estimate_tokens(typed_messages) <= TRUNCATE_LIMIT:
        return data

    head, rest = _split_messages(typed_messages)
    if not rest:
        data["messages"] = head[:1] if head else typed_messages[:1]
        return data

    keep_tail = min(KEEP_RECENT_TURNS, len(rest))
    tail = rest[-keep_tail:]
    middle = rest[:-keep_tail]

    keep_head_non_system = min(MIN_HEAD_NON_SYSTEM, len(middle))
    early = middle[:keep_head_non_system]
    drop_pool = middle[keep_head_non_system:]

    candidate = head + early + tail
    if _estimate_tokens(candidate) <= TRUNCATE_LIMIT:
        data["messages"] = candidate
        return data

    while drop_pool and _estimate_tokens(candidate) > TRUNCATE_LIMIT:
        drop_pool.pop(0)
        candidate = head + early + drop_pool + tail

    while len(tail) > 1 and _estimate_tokens(candidate) > TRUNCATE_LIMIT:
        tail.pop(0)
        candidate = head + early + drop_pool + tail

    data["messages"] = candidate
    return data
