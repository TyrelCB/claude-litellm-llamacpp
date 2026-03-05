#!/usr/bin/env python3
"""
Patch LiteLLM Anthropic proxy handlers to enforce truncate-middle.

This modifies litellm.proxy.proxy_server in the active Python environment.
It injects:
  from callbacks.truncate_middle import truncate_data_messages
  data = truncate_data_messages(data)
immediately after Anthropic pre_call_hook blocks.
"""

from __future__ import annotations

from pathlib import Path
import sys


OLD_BLOCK = (
    "        data = await proxy_logging_obj.pre_call_hook(  # type: ignore\n"
    "            user_api_key_dict=user_api_key_dict, data=data, call_type=\"text_completion\"\n"
    "        )\n"
)

INSERT = (
    "\n"
    "        from callbacks.truncate_middle import truncate_data_messages\n"
    "        data = truncate_data_messages(data)\n"
)


def main() -> int:
    try:
        import litellm.proxy.proxy_server as proxy_server
    except Exception as exc:
        print(f"error: could not import litellm.proxy.proxy_server: {exc}", file=sys.stderr)
        return 1

    target = Path(proxy_server.__file__)
    text = target.read_text(encoding="utf-8")

    if "data = truncate_data_messages(data)" in text:
        print(f"already patched: {target}")
        return 0

    occurrences = text.count(OLD_BLOCK)
    if occurrences == 0:
        print("error: did not find target pre_call_hook blocks", file=sys.stderr)
        return 2

    patched = text.replace(OLD_BLOCK, OLD_BLOCK + INSERT)
    target.write_text(patched, encoding="utf-8")
    print(f"patched {occurrences} block(s) in: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
