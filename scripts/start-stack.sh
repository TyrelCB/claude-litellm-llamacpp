#!/usr/bin/env bash
# Start llama-server (native, GPU) + LiteLLM proxy.
# Run this once after boot, then use start-claude.sh to launch Claude Code.
#
# Usage:
#   ./scripts/start-stack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# Load .env
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

LLAMA_BIN="${LLAMA_BIN:-/home/tyrel/llama.cpp/build/bin/llama-server}"
LLAMA_PORT="${LLAMA_PORT:-8082}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-model.gguf}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-32768}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-4}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-512}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}"
LLAMA_CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-q8_0}"
LLAMA_CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-q8_0}"
LLAMA_CACHE_REUSE="${LLAMA_CACHE_REUSE:-256}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-99}"
LITELLM_PORT="${LITELLM_PORT:-4000}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-litellm-local}"
LLAMA_CHAT_MODEL="${LLAMA_CHAT_MODEL:-local-model}"

MODEL_PATH="$ROOT/models/$LLAMA_MODEL_PATH"

# ── llama-server ─────────────────────────────────────────────────────────────
if curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
  echo "✓ llama-server already running on :${LLAMA_PORT}"
else
  echo "→ Starting llama-server on :${LLAMA_PORT}…"
  nohup "$LLAMA_BIN" \
    -m "$MODEL_PATH" \
    -c "$LLAMA_CTX_SIZE" \
    -np "$LLAMA_PARALLEL" \
    -b "$LLAMA_BATCH_SIZE" \
    --ubatch-size "$LLAMA_UBATCH_SIZE" \
    --cache-type-k "$LLAMA_CACHE_TYPE_K" \
    --cache-type-v "$LLAMA_CACHE_TYPE_V" \
    --cache-reuse "$LLAMA_CACHE_REUSE" \
    -ngl "$LLAMA_N_GPU_LAYERS" \
    --host 0.0.0.0 --port "$LLAMA_PORT" \
    > /tmp/llama-server.log 2>&1 &
  echo "  PID $! — logs: /tmp/llama-server.log"

  echo -n "  Waiting for llama-server"
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
      echo " ✓"
      break
    fi
    echo -n "."
    sleep 2
  done
fi

# ── LiteLLM proxy ────────────────────────────────────────────────────────────
if curl -sf "http://localhost:${LITELLM_PORT}/health" \
     -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" >/dev/null 2>&1; then
  echo "✓ LiteLLM already running on :${LITELLM_PORT}"
else
  echo "→ Starting LiteLLM on :${LITELLM_PORT}…"
  PYTHONPATH="$ROOT/litellm" \
  LLAMA_CHAT_MODEL="$LLAMA_CHAT_MODEL" \
  LITELLM_MASTER_KEY="$LITELLM_MASTER_KEY" \
  SUMMARIZER_THRESHOLD_RATIO="${SUMMARIZER_THRESHOLD_RATIO:-0.75}" \
  SUMMARIZER_KEEP_RECENT="${SUMMARIZER_KEEP_RECENT:-6}" \
  nohup ~/.local/bin/litellm \
    --config "$ROOT/litellm/config.yaml" \
    --port "$LITELLM_PORT" \
    > /tmp/litellm.log 2>&1 &
  echo "  PID $! — logs: /tmp/litellm.log"

  echo -n "  Waiting for LiteLLM"
  for i in $(seq 1 30); do
    if curl -sf "http://localhost:${LITELLM_PORT}/health" \
         -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" >/dev/null 2>&1; then
      echo " ✓"
      break
    fi
    echo -n "."
    sleep 2
  done
fi

echo ""
echo "Stack is ready. Launch Claude Code with:"
echo "  ./scripts/start-claude.sh --model qwen"
