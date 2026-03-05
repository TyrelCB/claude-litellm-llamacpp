"""
Normalize multimodal image blocks before forwarding to llama.cpp.

Claude/Anthropic-style image blocks can arrive in shapes that llama.cpp's
OpenAI endpoint rejects ("Invalid url value"). This callback rewrites those
blocks to a strict OpenAI-compatible image_url payload:

  {"type":"image_url","image_url":{"url":"data:<mime>;base64,<...>"}}
"""

from __future__ import annotations

import base64
import ast
import mimetypes
import os
import re
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _to_data_url(source: dict[str, Any]) -> str | None:
    """Convert an Anthropic-style image source dict into a data URL."""
    source_type = source.get("type")
    if source_type != "base64":
        return None

    media_type = source.get("media_type") or "image/png"
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None

    # Validate base64 to avoid forwarding malformed values.
    try:
        base64.b64decode(data, validate=True)
    except Exception:
        return None

    return f"data:{media_type};base64,{data}"


def _normalize_url_string(url: str) -> str:
    """Normalize URL-ish strings into valid image URL values for llama.cpp."""
    if url.startswith(("http://", "https://", "data:")):
        return url

    if url.startswith("base64,"):
        return "data:image/png;" + url

    # Local file path passed through by some clients/adapters.
    if os.path.isabs(url) and os.path.isfile(url):
        guessed_mime = mimetypes.guess_type(url)[0] or "image/png"
        with open(url, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{guessed_mime};base64,{encoded}"

    # Raw base64 blob with no data: prefix.
    if len(url) > 64 and _BASE64_RE.match(url):
        try:
            base64.b64decode(url, validate=True)
            return f"data:image/png;base64,{url}"
        except Exception:
            return url

    return url


def _fix_broken_data_url(url: str) -> str:
    """
    Fix buggy Anthropic adapter URLs like:
    data:image;base64,{'type': 'base64', 'media_type': 'image/png', 'data': '...'}
    """
    prefix = "data:image;base64,"
    if not url.startswith(prefix):
        return url

    payload = url[len(prefix) :]
    try:
        parsed = ast.literal_eval(payload)
    except Exception:
        return url

    if not isinstance(parsed, dict):
        return url

    media_type = parsed.get("media_type") or "image/png"
    data = parsed.get("data")
    if not isinstance(data, str) or not data:
        return url

    return f"data:{media_type};base64,{data}"


def _patch_anthropic_adapter_translation() -> None:
    """Patch LiteLLM Anthropic adapter output to repair malformed image URLs."""
    try:
        from litellm.llms.anthropic.experimental_pass_through.transformation import (
            AnthropicExperimentalPassThroughConfig,
        )
    except Exception:
        return

    if getattr(AnthropicExperimentalPassThroughConfig, "_image_url_patch_applied", False):
        return

    original = AnthropicExperimentalPassThroughConfig.translate_anthropic_messages_to_openai

    def patched(self, messages):  # type: ignore[no-untyped-def]
        out = original(self, messages)
        if not isinstance(out, list):
            return out

        for message in out:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                image_url = block.get("image_url")
                if isinstance(image_url, str):
                    block["image_url"] = {"url": _fix_broken_data_url(image_url)}
                elif isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str):
                        image_url["url"] = _fix_broken_data_url(url)
        return out

    AnthropicExperimentalPassThroughConfig.translate_anthropic_messages_to_openai = patched
    AnthropicExperimentalPassThroughConfig._image_url_patch_applied = True


def _normalize_block(block: Any) -> Any:
    """Return a normalized content block when possible, otherwise unchanged."""
    if not isinstance(block, dict):
        return block

    block_type = block.get("type")

    # Anthropic-style image block.
    if block_type == "image":
        source = block.get("source")
        if isinstance(source, dict):
            data_url = _to_data_url(source)
            if data_url:
                return {"type": "image_url", "image_url": {"url": data_url}}
        return block

    # OpenAI-style image block with non-string or nested URL value.
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            return {
                "type": "image_url",
                "image_url": {"url": _normalize_url_string(image_url)},
            }

        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                return {
                    "type": "image_url",
                    "image_url": {"url": _normalize_url_string(url)},
                }

            # Sometimes "url" is itself a source-style dict.
            if isinstance(url, dict):
                data_url = _to_data_url(url)
                if data_url:
                    return {"type": "image_url", "image_url": {"url": data_url}}

            # Sometimes image_url dict itself looks like source.
            data_url = _to_data_url(image_url)
            if data_url:
                return {"type": "image_url", "image_url": {"url": data_url}}

        return block

    # Some adapters emit "input_image" blocks.
    if block_type == "input_image":
        image_url = block.get("image_url") or block.get("url")
        if isinstance(image_url, str):
            return {
                "type": "image_url",
                "image_url": {"url": _normalize_url_string(image_url)},
            }
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                return {
                    "type": "image_url",
                    "image_url": {"url": _normalize_url_string(url)},
                }
            data_url = _to_data_url(image_url)
            if data_url:
                return {"type": "image_url", "image_url": {"url": data_url}}
        source = block.get("source")
        if isinstance(source, dict):
            data_url = _to_data_url(source)
            if data_url:
                return {"type": "image_url", "image_url": {"url": data_url}}
        return block

    return block


class ImageNormalizerCallback(CustomLogger):
    """LiteLLM pre-call hook that normalizes message image blocks."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        _patch_anthropic_adapter_translation()

        if call_type not in ("completion", "acompletion"):
            return data

        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        changed = False
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                normalized_messages.append(message)
                continue

            content = message.get("content")
            if not isinstance(content, list):
                normalized_messages.append(message)
                continue

            new_content = []
            for block in content:
                normalized_block = _normalize_block(block)
                if normalized_block is not block:
                    changed = True
                new_content.append(normalized_block)

            if changed:
                updated = dict(message)
                updated["content"] = new_content
                normalized_messages.append(updated)
            else:
                normalized_messages.append(message)

        if changed:
            data["messages"] = normalized_messages

        return data
