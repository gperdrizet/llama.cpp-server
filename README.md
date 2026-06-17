# llama.cpp inference server

[![llama.cpp](https://img.shields.io/badge/llama.cpp-inference-6B7280?logo=meta&logoColor=white)](https://github.com/ggml-org/llama.cpp)
[![CUDA](https://img.shields.io/badge/CUDA-P100%2016GB-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository documents and centralizes the configuration of a `llama.cpp` inference server running as a systemd service on a dedicated model server. The server exposes an OpenAI-compatible API and supports multiple concurrent projects.

> **API gateway**: [gperdrizet/model-gateway](https://github.com/gperdrizet/model-gateway), providing authentication, token metering, billing, and an admin panel


## Table of contents

- [API usage](#api-usage)
- [Hardware](#hardware)
- [Performance](#performance)
- [Available models](#available-models)
- [Deployment](#deployment)
- [Parallelism](#parallelism)
- [Systemd service](#systemd-service)
- [Service management and logs](#service-management-and-logs)
- [Testing](#testing)
- [Dashboard](#dashboard)


## API usage

The server exposes an OpenAI-compatible API.

```bash
# Chat completion — direct (internal / local network)
curl http://localhost:8502/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api-key>" \
  -d '{
    "model": "gpt-oss-20b-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Chat completion — external (through gateway)
curl https://model.perdrizet.org/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <gateway-issued-key>" \
  -d '{
    "model": "gpt-oss-20b-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Health check
curl http://localhost:8502/health
```

When configuring clients (LangChain, LlamaIndex, OpenWebUI, etc.), set:
- **Base URL**: `http://<model-server-ip>:8502/v1` (internal) or `https://model.perdrizet.org/v1` (external via gateway)
- **API Key**: value from the unit file (internal) or a gateway-issued key (external)


## Hardware

| Device | Model                  | VRAM   | Role                            |
|--------|------------------------|--------|---------------------------------|
| `0`    | Tesla P100-PCIE-16GB   | 16 GiB | Active inference GPU (selected via `CUDA_VISIBLE_DEVICES=0`) |
| `1`    | GeForce GTX 1070       | 8 GiB  | Available; not currently used by the service |

The service pins to GPU 0 (the P100) via `CUDA_VISIBLE_DEVICES=0`. The P100's 16 GiB VRAM comfortably fits the active model (12 GiB) fully on-device. The system has 24 CPU threads and 251 GiB RAM.

The binary was built from source with CMake in **Release** mode with CUDA support (`GGML_CUDA=ON`), CUDA flash attention (`GGML_CUDA_FA=ON`), and CUDA graphs (`GGML_CUDA_GRAPHS=ON`).


## Performance

Results produced by `tests/load_test.py` against the active model (`gpt-oss-20b-mxfp4.gguf`) on the P100, with the server configured to various `--parallel` slot counts. Analysis notebooks and saved figures live in `notebooks/`.


### Latency vs concurrency

![Latency vs concurrency by slot count](notebooks/figures/latency_vs_concurrency.png)

Each line is one slot configuration. At low concurrency all slot counts perform similarly. As concurrency rises, servers with more slots sustain lower latency because requests are served in parallel rather than queued behind one another.


### Latency at concurrency = 8 vs slot count

![Latency at concurrency 8 vs slot count](notebooks/figures/latency_vs_slots_c8.png)

At a fixed concurrency of 8 simultaneous requests, increasing the slot count reduces both mean latency and p95 latency significantly. Beyond 4 slots the gains diminish as the GPU becomes the bottleneck rather than the queuing.


### Context length per slot

![Context length per slot](notebooks/figures/context_per_slot.png)

The server's total context window (`-c 65536`, 64k tokens) is divided equally across all slots. More slots means less context available per individual request. For most short chat turns and one-shot completions 8–16k tokens is ample; workloads with long system prompts or multi-turn histories may require fewer slots to preserve context.


### Latency vs input context length

![Latency vs input context length](notebooks/figures/latency_vs_context_length.png)

Measured by `tests/context_length_test.py` at fixed concurrency. Both mean and p95 latency increase with prompt length as the model must process more tokens during the prefill phase before generating any output.


## Available models

All models live in `/opt/models/`. The service must be restarted to switch models (update `-m` in `utils/llamacpp.service` and redeploy).

| Filename | Size | Type | Status | Notes |
|---|---|---|---|---|
| `gpt-oss-20b-mxfp4.gguf` | 12 GiB | Chat | **Active** | Microsoft MXFP4 quantization. Fits entirely on P100. |
| `mxbai-embed-large-v1-f16.gguf` | 639 MiB | Embedding | On disk | mixedbread-ai embedding model, FP16. Not currently served. |
| `Qwen2.5-32B-Instruct-Q3_K_M.gguf` | 15 GiB | Chat | On disk | **Does not fit on P100.** Weights leave insufficient room for KV cache at `-c 65536`. Tested; OOM at context allocation. Requires `-c ≤ 8192`. |
| `Qwen2.5-32B-Instruct-Q4_K_M.gguf` | 19 GiB | Chat | On disk | Exceeds P100 VRAM; requires CPU offload or both GPUs. |
| `Mistral-Small-3.1-24B-Instruct-Q4_K_M.gguf` | ~14 GiB | Chat | Not downloaded | Strong reasoning; 128k native context window (cap to 8k–16k for VRAM headroom). |
| `Phi-4-Q8_0.gguf` | ~15 GiB | Chat | Not downloaded | Microsoft Phi-4 (14B); strong on reasoning and code. Q5_K_M (~10 GiB) also an option. |
| `gemma-3-27b-it-Q3_K_M.gguf` | ~11 GiB | Chat | Not downloaded | Google Gemma 3 27B; highest parameter count that fits comfortably. Q4_K_M (~14 GiB) also fits. |
| `Qwen2.5-14B-Instruct-Q8_0.gguf` | ~14 GiB | Chat | Not downloaded | Same family as the 32B but properly fits at full Q8 precision. |
| `DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf` | ~14 GiB | Chat | Not downloaded | Reasoning-focused distill of DeepSeek-R1; good for structured/multi-step tasks. |


## Deployment

The unit file template lives in `utils/llamacpp.service` — that is the source of truth. Deploy it with:

```bash
# Copy and fill in the env file
cp .env.template .env
# edit .env: set LLAMA_API_KEY and LLAMA_SLOTS

# Deploy the unit file (runs daemon-reload)
bash utils/deploy_service.sh

# Deploy and immediately restart the service
bash utils/deploy_service.sh --restart
```

`deploy_service.sh` substitutes `YOUR_API_KEY_HERE` and `YOUR_SLOTS_HERE` from `.env`, copies the result to `/etc/systemd/system/llamacpp.service`, and runs `systemctl daemon-reload`.

> **Note:** `.env` contains the real API key — do not commit it. It is listed in `.gitignore`.

Model files are not included in this repository. Download them separately with `huggingface-cli` or `wget` into `/opt/models/`.

### File layout

```
/opt/llama.cpp/                      # llama.cpp source + build tree (read-only to service)
└── build/
    └── bin/
        └── llama-server             # the inference server binary

/opt/models/                         # model storage (read-write to service)
├── gpt-oss-20b-mxfp4.gguf           # (12 GiB) ← currently active
├── mxbai-embed-large-v1-f16.gguf    # (639 MiB)
├── Qwen2.5-32B-Instruct-Q3_K_M.gguf # (15 GiB)
└── Qwen2.5-32B-Instruct-Q4_K_M.gguf # (19 GiB)

/etc/systemd/system/
├── llamacpp.service                 # deployed unit file (generated by deploy_service.sh)
└── llamacpp.service.d/
    └── override.conf                # drop-in: sets CUDA_VISIBLE_DEVICES
```


## Parallelism

llama.cpp splits its KV cache into **slots** using the `--parallel` flag. Each slot handles one concurrent request; when all slots are busy, additional requests queue.

The number of slots is configured via `LLAMA_SLOTS` in `.env` and substituted into the unit file by `deploy_service.sh`.

| `LLAMA_SLOTS` | Slots | Tokens per slot (with `-c 65536`) | Behavior |
|---|---|---|---|
| `1` | 1 | 65 536 | Full context per request; no concurrency, requests queue |
| `4` | 4 | 16 384 | 4 simultaneous requests; 16k context each |
| `8` | 8 | 8 192 | Higher throughput; short context limit per request |

Start with `LLAMA_SLOTS=1` and use the load test to benchmark before increasing. Most short chat turns and one-shot completions fit comfortably within 16k tokens, making `LLAMA_SLOTS=4` a reasonable first step on the P100.


## Systemd service

The unit file template is `utils/llamacpp.service` — that is the source of truth. The deployed copy is at `/etc/systemd/system/llamacpp.service`. See [Deployment](#deployment) for how to build and apply it.

**CUDA probe:** Before starting, the service polls `nvidia-smi -L` for up to 30 seconds to confirm the GPU is available. This guards against `nvidia-persistenced` race conditions on boot — if the GPU isn't ready, the service fails immediately rather than silently falling back to CPU inference.

**Security hardening:**

| Directive | Effect |
|---|---|
| `NoNewPrivileges=true` | Prevents privilege escalation via setuid/setgid |
| `PrivateTmp=true` | Isolated `/tmp` namespace |
| `ProtectSystem=strict` | Filesystem mounted read-only except listed paths |
| `ProtectHome=true` | `/home`, `/root`, `/run/user` invisible to the process |
| `ReadWritePaths=/opt/models` | Allows cache writes to the model directory |
| `ReadOnlyPaths=/opt/llama.cpp` | Marks the install tree read-only |

The service runs as the unprivileged `llama` user/group.

**Restart policy:**

| Setting | Value | Meaning |
|---|---|---|
| `Restart` | `on-failure` | Restart if the process exits non-zero or is killed by a signal |
| `RestartSec` | `10` | Wait 10 seconds before restarting |
| `StartLimitInterval` | `300` | Rolling window for the burst limit |
| `StartLimitBurst` | `5` | Stop retrying after 5 failures within 5 minutes |


## Service management and logs

```bash
# Status
systemctl status llamacpp.service

# Start / stop / restart
sudo systemctl start llamacpp.service
sudo systemctl stop llamacpp.service
sudo systemctl restart llamacpp.service

# Apply unit file changes
sudo systemctl daemon-reload && sudo systemctl restart llamacpp.service

# Enable / disable autostart on boot
sudo systemctl enable llamacpp.service
sudo systemctl disable llamacpp.service
```

**Logs** — all output goes to the systemd journal tagged with `llama-server`:

```bash
# Follow live logs
journalctl -u llamacpp.service -f

# Show logs since last boot
journalctl -u llamacpp.service -b

# Show last 100 lines (full, not ellipsized)
journalctl -u llamacpp.service -n 100 --no-pager -l

# Filter by time range
journalctl -u llamacpp.service --since "2026-04-24 00:00" --until "2026-04-24 12:00"
```

## Testing

### Load test

`tests/load_test.py` measures response latency as a function of the number of concurrent callers. For each concurrency level it fires a batch of requests simultaneously, waits for all to complete, repeats that batch a configurable number of times, then prints statistics.

**Setup:**

```bash
pip install -r requirements.txt
cp .env.template .env   # edit and set LLAMA_API_KEY
```

**Usage:**

```bash
# Run with defaults (levels 1 2 4 8, 3 repetitions each, non-streaming)
python tests/load_test.py

# Custom levels and repetitions
python tests/load_test.py --levels 1 2 4 8 16 --requests 5

# Enable streaming (measures time-to-first-token in addition to total latency)
python tests/load_test.py --stream

# Target a specific host (bypass nginx rate limits when testing locally)
python tests/load_test.py --url http://localhost:8502
```

> **Note:** `.env` sets `LLAMA_BASE_URL=https://model.perdrizet.org`, which routes through nginx (12 req/min limit). Use `--url http://localhost:8502` to bypass it when running load tests on the server itself.

**CLI options:**

| Option | Default | Description |
|---|---|---|
| `--url` | `$LLAMA_BASE_URL` or `http://localhost:8502` | Server base URL |
| `--api-key` | `$LLAMA_API_KEY` | Bearer token |
| `--slots N` | `$LLAMA_SLOTS` or `1` | Parallel slots the server is configured with (recorded in CSV) |
| `--levels N [N ...]` | `1 2 4 8` | Concurrency levels to test |
| `--requests N` | `3` | Repetitions per level (for averaging) |
| `--prompt TEXT` | one-sentence transformer question | Prompt sent to the model |
| `--max-tokens N` | `128` | Max completion tokens per request |
| `--output FILE` | `tests/results/YYYYmmdd_HHMM.csv` | Path for raw results CSV |
| `--stream` | off | Use streaming responses (enables TTFT measurement) |

Results are written to `tests/results/` as CSV with one row per request (timestamp, slots, concurrency, latency_s, ttft_s, tokens, error).


### Context length test

`tests/context_length_test.py` measures how response latency scales with input prompt length. For each target token count it constructs a prompt of approximately that size, verifies the actual count via `/tokenize`, fires concurrent requests, and repeats for several replicates.

**Usage:**

```bash
# Run with defaults (targets 128 256 512 1024 2048 4096 8192, concurrency 4, 5 replicates)
python tests/context_length_test.py

# Custom targets and replicates
python tests/context_length_test.py --targets 256 1024 4096 --replicates 10

# Enable streaming
python tests/context_length_test.py --stream
```

**CLI options:**

| Option | Default | Description |
|---|---|---|
| `--url` | `$LLAMA_BASE_URL` or `http://localhost:8502` | Server base URL |
| `--api-key` | `$LLAMA_API_KEY` | Bearer token |
| `--slots N` | `$LLAMA_SLOTS` or `1` | Parallel slots the server is configured with (recorded in CSV) |
| `--targets N [N ...]` | `128 256 512 1024 2048 4096 8192` | Target prompt token counts |
| `--concurrency N` | `4` | Simultaneous requests per replicate |
| `--replicates N` | `5` | Repetitions per target length (for averaging) |
| `--max-tokens N` | `64` | Max completion tokens per request |
| `--output FILE` | `tests/results/context_test_YYYY-MM-DD_HH-MM.csv` | Path for raw results CSV |
| `--stream` | off | Use streaming responses (enables TTFT measurement) |

Results are written to `tests/results/` as CSV (timestamp, slots, target_tokens, prompt_tokens, concurrency, latency_s, ttft_s, output_tokens, error).


## Dashboard

`dashboard/app.py` is a [Streamlit](https://streamlit.io) application that visualizes load test CSV results.

![Dashboard screenshot](dashboard/screen_shot.png)

```bash
streamlit run dashboard/app.py
```

Open the URL printed by Streamlit (default: `http://localhost:8501`). The sidebar lets you select any CSV from `tests/results/`. Charts show mean ± SEM and p95 latency vs. concurrency level.
