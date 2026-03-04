"""
Tool-call memory callback for LiteLLM.

Stores tool call results per session in memory and injects a compact summary
of previous tool results into the system prompt on the next request.

This lets the model "remember" what tools returned earlier in the session
without repeating the full tool output in every turn.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Maximum number of tool-call results to retain per session.
MAX_TOOL_RESULTS_PER_SESSION = 20

_lock = threading.Lock()
# session_id → deque of {"tool": str, "result": str}
_store: dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TOOL_RESULTS_PER_SESSION))


def _session_id(kwargs: dict) -> str:
    """Derive a stable session key from LiteLLM call kwargs."""
    metadata = kwargs.get("litellm_params", {}).get("metadata") or {}
    return (
        metadata.get("session_id")
        or metadata.get("user_id")
        or kwargs.get("user")
        or "default"
    )


def _extract_tool_results(response) -> list[dict]:
    """Pull tool-call name+result pairs out of a completed response."""
    results = []
    try:
        for choice in response.choices:
            msg = choice.message
            if not msg:
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                name = getattr(tc.function, "name", "unknown_tool") if tc.function else "unknown_tool"
                raw = getattr(tc.function, "arguments", "") if tc.function else ""
                try:
                    parsed = json.loads(raw) if raw else {}
                except Exception:
                    parsed = {"raw": raw}
                results.append({"tool": name, "args": parsed})
    except Exception:
        pass
    return results


def _build_memory_note(session_results: deque) -> str:
    """Format stored tool results as a compact memory block."""
    if not session_results:
        return ""
    lines = ["[Tool-call memory from earlier in this session:]"]
    for entry in session_results:
        tool = entry.get("tool", "?")
        result = entry.get("result", entry.get("args", ""))
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False)[:300]
        lines.append(f"  • {tool}: {str(result)[:300]}")
    return "\n".join(lines)


class ToolMemoryLogger(CustomLogger):
    """
    LiteLLM CustomLogger that:
    - pre-call:  injects prior tool results into the system message
    - post-call: stores new tool results for the session
    """

    # ── Pre-call hook: inject tool memory ───────────────────────────────────
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        if call_type not in ("completion", "acompletion"):
            return data

        sid = _session_id(data)
        with _lock:
            past = _store.get(sid)

        if not past:
            return data

        memory_note = _build_memory_note(past)
        messages = data.get("messages", [])
        if not messages:
            return data

        # Append to existing system message or prepend a new one.
        if messages[0].get("role") == "system":
            existing = messages[0].get("content", "")
            messages[0]["content"] = existing + "\n\n" + memory_note
        else:
            messages.insert(0, {"role": "system", "content": memory_note})

        data["messages"] = messages
        return data

    # ── Post-call hook: save tool results ───────────────────────────────────
    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time, end_time):
        results = _extract_tool_results(response_obj)
        if not results:
            return
        sid = _session_id(kwargs)
        with _lock:
            for r in results:
                _store[sid].append(r)

    # Sync variant for non-async callers
    def log_success_event(self, kwargs: dict, response_obj: Any, start_time, end_time):
        results = _extract_tool_results(response_obj)
        if not results:
            return
        sid = _session_id(kwargs)
        with _lock:
            for r in results:
                _store[sid].append(r)
