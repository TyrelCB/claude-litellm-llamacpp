# Claude Code + LiteLLM + llama.cpp

Run [Claude Code](https://claude.ai/code) against **local open models** using
[llama.cpp](https://github.com/ggml-org/llama.cpp) for inference and
[LiteLLM](https://github.com/BerriAI/litellm) as a proxy/gateway.

```
Claude Code
    │  ANTHROPIC_BASE_URL=http://localhost:4000
    ▼
LiteLLM proxy  (:4000)
  ├─ Anthropic API ↔ OpenAI API translation
  ├─ Truncate-middle  (keeps system prompt + recent context)
  ├─ Summarization    (compresses old turns before sending)
  ├─ In-memory cache  (deduplicates identical requests)
  └─ Tool-call memory (re-injects past tool results)
    │  http://llama-server:8080
    ▼
llama.cpp server (:8080)
  ├─ KV cache  (quantized q8_0, saves VRAM)
  ├─ Prefix caching (--cache-reuse)
  └─ Batching  (--parallel, --batch-size)
```

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Compose](https://docs.docker.com/compose/install/) v2
- [Claude Code CLI](https://docs.anthropic.com/claude-code) (`npm i -g @anthropic-ai/claude-code` or `pip install claude-code`)
- A GGUF model file (see [Downloading a model](#downloading-a-model))

---

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set LLAMA_MODEL_PATH to your model filename
```

### 2. Download a model

```bash
# Qwen2.5-Coder 7B — excellent for coding tasks, fits 8 GB VRAM
./scripts/download-model.sh Qwen/Qwen2.5-Coder-7B-Instruct-GGUF Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf

# Then set in .env:
# LLAMA_MODEL_PATH=Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf
```

Other good choices:

| Model | Size | Strengths |
|-------|------|-----------|
| `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~5 GB Q5 | Code, instruction, tool use |
| `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | ~9 GB Q5 | Code quality, context |
| `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | ~5 GB Q5 | General, fast |
| `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | ~5 GB Q5 | Fast, solid tool-call |

### 3. Start the stack

**CPU:**
```bash
docker compose --profile cpu up -d
```

**NVIDIA GPU (CUDA):**
```bash
# Requires nvidia-container-toolkit
docker compose --profile cuda up -d
```

**Apple Silicon (Metal):**
> Docker on macOS cannot access the GPU. For Metal acceleration, run
> llama-server natively (see [Metal native](#metal-native)) and start
> only LiteLLM via Docker:
```bash
docker compose --profile metal up -d litellm
```

### 4. Find the model name (one-time)

After llama-server starts, check the reported model name:

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
# Look for "id" field, e.g. "Qwen2.5-Coder-7B-Instruct-Q5_K_M"
```

Set it in `.env`:
```
LLAMA_CHAT_MODEL=Qwen2.5-Coder-7B-Instruct-Q5_K_M
```

Then restart LiteLLM: `docker compose --profile cpu restart litellm`

### 5. Launch Claude Code

```bash
./scripts/start-claude.sh --model claude-opus-4-5
# or
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-litellm-local
claude --model claude-opus-4-5
```

---

## Features

### KV Cache (llama.cpp)

The KV cache stores intermediate transformer states so repeated prefixes
(system prompt, shared code context) are not recomputed.

| Flag | Default | Description |
|------|---------|-------------|
| `LLAMA_CTX_SIZE` | 32768 | Total token capacity (split across slots) |
| `LLAMA_CACHE_TYPE_K/V` | `q8_0` | Quantized KV — ~50 % VRAM saving |
| `LLAMA_CACHE_REUSE` | 256 | Min tokens to match for prefix reuse |

### Batching (llama.cpp)

| Flag | Default | Description |
|------|---------|-------------|
| `LLAMA_PARALLEL` | 4 | Concurrent decode slots |
| `LLAMA_BATCH_SIZE` | 512 | Tokens per prompt-processing batch |
| `LLAMA_UBATCH_SIZE` | 512 | Micro-batch size within a batch |

Increase `LLAMA_PARALLEL` for higher throughput; decrease for lower latency
per request. `LLAMA_CTX_SIZE` should be at least `PARALLEL × max_tokens`.

### Truncate-Middle (LiteLLM)

When the conversation exceeds `LITELLM_TRUNCATE_INPUT_TOKENS` (default 30 000),
LiteLLM drops turns from the **middle** of the history — preserving the system
prompt at the top and the most recent turns at the bottom.

Configure in `.env`:
```
LITELLM_TRUNCATE_INPUT_TOKENS=30000
```

### Summarization (custom callback)

Before hitting the truncation limit, `litellm/callbacks/summarizer.py` fires
and asks the local model to summarize the oldest turns into bullet points.
The summary replaces those turns as a single `[SUMMARY]` message.

Configure in `.env`:
```
SUMMARIZER_THRESHOLD_RATIO=0.75   # fire at 75 % of truncation limit
SUMMARIZER_KEEP_RECENT=6          # keep last 6 turns verbatim
```

### In-Memory Cache (LiteLLM)

Identical prompts (same messages hash) return the cached response instantly,
saving inference time and token budget.

Enabled in `litellm/config.yaml`:
```yaml
cache: true
cache_params:
  type: local
```

### Tool-Call Memory (custom callback)

`litellm/callbacks/tool_memory.py` tracks every tool call result per session.
On the next request, past tool results are injected into the system prompt as
a compact bullet list — the model can reference earlier tool outputs without
them being explicitly in the context window.

---

## Project Structure

```
.
├── docker-compose.yml          # Services: llama-server + litellm
├── .env.example                # All configurable variables
├── models/                     # GGUF model files (gitignored)
├── litellm/
│   ├── config.yaml             # LiteLLM proxy configuration
│   └── callbacks/
│       ├── __init__.py
│       ├── tool_memory.py      # Tool-call memory callback
│       └── summarizer.py       # Summarization callback
└── scripts/
    ├── download-model.sh       # HuggingFace GGUF downloader
    └── start-claude.sh         # Sets env vars, launches claude
```

---

## Metal Native

Docker on macOS does not expose the GPU. To use Metal acceleration:

```bash
# Build llama.cpp with Metal support
brew install cmake
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j

# Run the server (adjust flags as needed)
./build/bin/llama-server \
  -m ../models/your-model.gguf \
  -c 32768 -np 4 -b 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-reuse 256 \
  -ngl 99 \
  --host 0.0.0.0 --port 8080
```

Then start only LiteLLM:
```bash
docker compose --profile metal up -d litellm
```

---

## Troubleshooting

**LiteLLM can't reach llama-server**
Check the container is healthy: `docker compose ps`
Inspect logs: `docker compose logs llama-server`

**"model not found" errors**
Run `curl http://localhost:8080/v1/models` and update `LLAMA_CHAT_MODEL` in `.env`.

**Out of memory / context overflow**
Reduce `LLAMA_CTX_SIZE` or `LLAMA_PARALLEL`, or use a smaller model/quant.

**Slow first token**
Normal for large context sizes on CPU. Enable GPU offload with `LLAMA_N_GPU_LAYERS=99`.

**Callbacks not loading**
Ensure `./litellm/callbacks` is bind-mounted (check docker-compose.yml) and
that `callbacks/` is importable as a Python package (needs `__init__.py`).
