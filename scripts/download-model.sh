#!/usr/bin/env bash
# Download a GGUF model from Hugging Face into ./models/
#
# Usage:
#   ./scripts/download-model.sh <hf-repo> <filename>
#
# Examples:
#   ./scripts/download-model.sh Qwen/Qwen2.5-Coder-7B-Instruct-GGUF Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf
#   ./scripts/download-model.sh bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf
#   ./scripts/download-model.sh bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf

set -euo pipefail

REPO="${1:-}"
FILE="${2:-}"

if [[ -z "$REPO" || -z "$FILE" ]]; then
  echo "Usage: $0 <hf-repo> <filename>" >&2
  exit 1
fi

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DEST_DIR"

DEST="$DEST_DIR/$FILE"

if [[ -f "$DEST" ]]; then
  echo "✓ Model already downloaded: $DEST"
  exit 0
fi

echo "Downloading $FILE from $REPO …"

# Prefer huggingface-cli if available
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST_DIR" --local-dir-use-symlinks False
elif command -v hf &>/dev/null; then
  hf download "$REPO" "$FILE" --local-dir "$DEST_DIR"
else
  # Fall back to plain curl with HF redirect support
  HF_URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
  echo "Using curl: $HF_URL"
  curl -L --progress-bar -o "$DEST" \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
    "$HF_URL"
fi

echo ""
echo "✓ Saved to $DEST"
echo ""
echo "Next steps:"
echo "  1. Update LLAMA_MODEL_PATH=$FILE in your .env"
echo "  2. Run: docker compose --profile <cpu|cuda|metal> up"
