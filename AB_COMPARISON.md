# Anthropic Endpoint A/B Comparison

Date: 2026-03-05

Goal: keep context engineering in proxy while using Anthropic endpoints on both sides.

## Setup

- Backend: `llama.cpp` at `http://172.17.0.1:8082`
- Client API: Anthropic `/v1/messages`
- Test payload: 10 very large user turns (~289k prompt tokens untruncated)

## Branches

1. `exp/direct-anthropic-patch` (`/tmp/wt-direct`)
- LiteLLM version: `1.57.2`
- Added deterministic runtime patch in `litellm.proxy.proxy_server` to run
  `truncate_data_messages(data)` after Anthropic pre-call hook.
- Added repeatable script: `scripts/patch-litellm-anthropic-truncate.py`
- `scripts/start-stack.sh` applies patch automatically by default (`LITELLM_AUTO_PATCH_ANTHROPIC=1`).

2. `exp/litellm-upgrade-path` (`/tmp/wt-upgrade`)
- LiteLLM version: `1.82.0`
- No site-packages patch, only config/callback path.

## Results

- Direct patch branch: PASS
  - Request completed with usage: `input_tokens=86688, output_tokens=8`
  - No context overflow from llama.cpp.

- Upgrade branch: FAIL
  - Request failed with context overflow:
  - `request (288975 tokens) exceeds the available context size (262144 tokens)`

## Verdict

`exp/direct-anthropic-patch` is currently the more functional solution in this environment.

The upgrade-only path did not apply middle truncation on Anthropic `/v1/messages` requests.
