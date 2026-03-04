#!/usr/bin/env bash
# Launch Claude Code pointed at the local LiteLLM proxy.
#
# Usage:
#   ./scripts/start-claude.sh [claude args...]
#
# Examples:
#   ./scripts/start-claude.sh
#   ./scripts/start-claude.sh --model claude-opus-4-5
#   ./scripts/start-claude.sh --model claude-sonnet-4-5 --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# Load .env if present
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

LITELLM_PORT="${LITELLM_PORT:-4000}"
LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-litellm-local}"

export ANTHROPIC_BASE_URL="http://localhost:${LITELLM_PORT}"
export ANTHROPIC_API_KEY="$LITELLM_MASTER_KEY"

# Verify the proxy is reachable
if ! curl -sf "${ANTHROPIC_BASE_URL}/health" >/dev/null 2>&1; then
  echo "⚠  LiteLLM proxy not reachable at ${ANTHROPIC_BASE_URL}" >&2
  echo "   Start the stack first:" >&2
  echo "   docker compose --profile <cpu|cuda|metal> up -d" >&2
  exit 1
fi

echo "→ ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
echo "→ Starting Claude Code…"
exec claude "$@"
