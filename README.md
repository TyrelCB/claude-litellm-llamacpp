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
    │  http://localhost:8082
    ▼
llama.cpp server (:8082)
  ├─ KV cache  (quantized q8_0, saves VRAM)
  ├─ Prefix caching (--cache-reuse)
  └─ Batching  (--parallel, --batch-size)
```

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Compose](https://docs.docker.com/compose/install/) v2 (for LiteLLM)
- [Claude Code CLI](https://docs.anthropic.com/claude-code) (`npm i -g @anthropic-ai/claude-code` or `pip install claude-code`)
- [LiteLLM](https://github.com/BerriAI/litellm) Python package: `pip install 'litellm[proxy]'`
- A GGUF model file (see [Downloading a model](#downloading-a-model))
- A built [llama.cpp](https://github.com/ggml-org/llama.cpp) binary (see [Building llama.cpp](#building-llamacpp))

---

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set LLAMA_MODEL_PATH, LLAMA_BIN, and LLAMA_PORT at minimum
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
| `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF` | ~21 GB Q4 | Large MoE, strong reasoning |
| `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | ~9 GB Q5 | Code quality, context |
| `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | ~5 GB Q5 | General, fast |
| `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | ~5 GB Q5 | Fast, solid tool-call |

If you want image input, use a vision-capable model and set its projector file
in `.env`:

```bash
# Text+image models require BOTH files:
LLAMA_MODEL_PATH=<vision-model>.gguf
LLAMA_MMPROJ_PATH=<matching-mmproj>.gguf
```

The same `LLAMA_MMPROJ_PATH` variable is used by both `start-stack.sh` and
`docker compose` profiles.

### 3. Start the stack

```bash
./scripts/start-stack.sh
```

This starts llama-server (natively, with GPU) and LiteLLM (as a local Python process).
Logs go to `/tmp/llama-server.log` and `/tmp/litellm.log`.

### 4. Launch Claude Code

```bash
./scripts/start-claude.sh
# or with an explicit model name (any name works — all route to the local model):
./scripts/start-claude.sh --model qwen
```

The `--model` flag accepts any string; the LiteLLM catch-all routes everything
to the local llama-server. Defaults to `--model local` if omitted.

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
├── docker-compose.yml          # Optional: Docker profiles for llama-server (cpu/cuda/metal/native)
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
    ├── start-stack.sh          # Starts llama-server + LiteLLM
    └── start-claude.sh         # Sets env vars, launches claude
```

---

## Building llama.cpp

llama-server runs natively (outside Docker) for GPU access.

**Linux — CUDA:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
# Binary: build/bin/llama-server
```

**macOS — Metal:**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j
```

**CPU only:**
```bash
cmake -B build && cmake --build build --config Release -j
```

Set `LLAMA_BIN` in `.env` to the path of the built binary.

---

## Troubleshooting

**Proxy not reachable / health check fails**
Ensure the stack is started: `./scripts/start-stack.sh`
Check logs: `tail -f /tmp/litellm.log` and `tail -f /tmp/llama-server.log`

**"model not found" errors**
llama.cpp serves whichever model is loaded regardless of the model name in the
request, so this usually means llama-server isn't running. Check:
`curl http://localhost:${LLAMA_PORT}/v1/models`

**"image input is not supported ... provide the mmproj"**
Your model is running without a projector or is text-only.
Set `LLAMA_MMPROJ_PATH` to the matching mmproj GGUF, then restart:
`./scripts/start-stack.sh`

**Out of memory / context overflow**
Reduce `LLAMA_CTX_SIZE` or `LLAMA_PARALLEL`, or use a smaller model/quant.

**Slow first token**
Normal for large context sizes on CPU. Enable GPU offload with `LLAMA_N_GPU_LAYERS=99`.

**Callbacks not loading**
`start-stack.sh` sets `PYTHONPATH` to `litellm/` automatically. If running
LiteLLM manually, ensure `PYTHONPATH=<repo>/litellm` is exported so the
`callbacks` package is importable.

**LiteLLM version compatibility**
`master_key` auth in LiteLLM ≥ v1.58 requires a database. The config has
`master_key` commented out so the proxy runs unauthenticated (fine for local
use). To re-enable auth, set up a database and uncomment `master_key` in
`litellm/config.yaml`.
