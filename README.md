# llama.cpp inference server

[![llama.cpp](https://img.shields.io/badge/llama.cpp-inference-6B7280?logo=meta&logoColor=white)](https://github.com/ggml-org/llama.cpp)
[![CUDA](https://img.shields.io/badge/CUDA-P100%2016GB-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI compatible](https://img.shields.io/badge/API-OpenAI%20compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository documents and centralizes the configuration of a `llama.cpp` inference server running as a systemd service on a dedicated model server. The server exposes a local OpenAI-compatible API and supports multiple concurrent projects.

> **Public API gateway**: [promptlyapi.com](https://promptlyapi.com/register), providing authentication, token metering, billing, and an admin panel for indie devs and hobbyists on a budget - 100k free tokens for new registrations.


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


## Deployment

### Prerequisites

The service runs as the `llama` system user. Create it once before deploying:

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin llama
```

### Deploy

The unit file template lives in `utils/llamacpp.service` - that is the source of truth. Deploy it with:

```bash
# Copy and fill in the env file
cp .env.template .env

# Deploy and immediately restart the service
bash utils/deploy_service.sh --restart
```

`deploy_service.sh` substitutes `SUB_API_KEY_HERE`, etc from `.env`, copies the result to `/etc/systemd/system/llamacpp.service`, and runs `systemctl daemon-reload`.

> **Note:** `.env` contains the real API key - do not commit it. It is listed in `.gitignore`.

Model files are not included in this repository. Download them separately with `huggingface-cli` or `wget`.

## Systemd service

The unit file template is `utils/llamacpp.service` - that is the source of truth. The deployed copy is at `/etc/systemd/system/llamacpp.service`. See [Deployment](#deployment) for how to build and apply it.


### Service management

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

### Logs

All log output goes to the systemd journal tagged with `llama-server`:

```bash
# Follow live logs
journalctl -u llamacpp.service -f

# Show logs since last system boot
journalctl -u llamacpp.service -b

# Show last 100 lines (full, not ellipsized)
journalctl -u llamacpp.service -n 100 --no-pager -l

# Filter by time range
journalctl -u llamacpp.service --since "2026-04-24 00:00" --until "2026-04-24 12:00"
```

### Restart policy

By default, the service will restart on failure with the following settings.

| Setting | Value | Meaning |
|---|---|---|
| `Restart` | `on-failure` | Restart if the process exits non-zero or is killed by a signal |
| `RestartSec` | `10` | Wait 10 seconds before restarting |
| `StartLimitInterval` | `300` | Rolling window for the burst limit |
| `StartLimitBurst` | `5` | Stop retrying after 5 failures within 5 minutes |

**CUDA probe:** Before starting, the service polls `nvidia-smi -L` for up to 30 seconds to confirm the GPU is available. This guards against `nvidia-persistenced` race conditions on boot. If the GPU isn't ready, the service fails immediately rather than silently falling back to CPU inference.

### Security hardening

The service runs as the unprivileged `llama` user/group and several flags are set in the unit file to protect the host system.

| Directive | Effect |
|---|---|
| `NoNewPrivileges=true` | Prevents privilege escalation via setuid/setgid |
| `PrivateTmp=true` | Isolated `/tmp` namespace |
| `ProtectSystem=strict` | Filesystem mounted read-only except listed paths |
| `ProtectHome=true` | `/home`, `/root`, `/run/user` invisible to the process |
| `ReadOnlyPaths=/opt/llama.cpp /opt/models` | Both the install tree and model directory are read-only (model files are memory-mapped for reading only) |


## Performance

Results produced by `tests/load_test.py` against the active model (`gpt-oss-20b-mxfp4.gguf`) on a NVIDIA P100 16GB GPU, with the server configured to various `--parallel` slot counts. Analysis notebooks and saved figures live in `notebooks/`.

### Parallelism

llama.cpp splits its KV cache into **slots** using the `--parallel` flag. Each slot handles one concurrent request; when all slots are busy, additional requests queue.

The number of slots is configured via `LLAMA_SLOTS` in `.env` and substituted into the unit file by `deploy_service.sh`.

| `LLAMA_SLOTS` | Slots | Tokens per slot (with `-c 65536`) | Behavior |
|---|---|---|---|
| `1` | 1 | 65 536 | Full context per request; no concurrency, requests queue |
| `4` | 4 | 16 384 | 4 simultaneous requests; 16k context each |
| `8` | 8 | 8 192 | Higher throughput; short context limit per request |

Start with `LLAMA_SLOTS=1` and use the load test to benchmark before increasing. Most short chat turns and one-shot completions fit comfortably within 16k tokens, making `LLAMA_SLOTS=4` a reasonable first step on the P100.


### Latency vs concurrency

![Latency vs concurrency by slot count](notebooks/figures/latency_vs_concurrency.png)

Each line is one slot configuration. At low concurrency all slot counts perform similarly. As concurrency rises, servers with more slots sustain lower latency because requests are served in parallel rather than queued behind one another.


### Latency at concurrency = 8 vs slot count

![Latency at concurrency 8 vs slot count](notebooks/figures/latency_vs_slots_c8.png)

At a fixed concurrency of 8 simultaneous requests, increasing the slot count reduces both mean latency and p95 latency significantly. Beyond 4 slots the gains diminish as the GPU becomes the bottleneck rather than the queuing.


### Context length per slot

![Context length per slot](notebooks/figures/context_per_slot.png)

The server's total context window (`-c 131072`, 100k tokens for `gpt-oss-20b) is divided equally across all slots. More slots means less context available per individual request. For most short chat turns and one-shot completions 8–16k tokens is ample; workloads with long system prompts or multi-turn histories may require fewer slots to preserve context.


### Latency vs input context length

![Latency vs input context length](notebooks/figures/latency_vs_context_length.png)

Measured by `tests/context_length_test.py` at fixed concurrency. Both mean and p95 latency increase with prompt length as the model must process more tokens during the prefill phase before generating any output.


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

# Target a specific host
python tests/load_test.py --url http://localhost:8502
```

**CLI options:**

| Option | Default | Description |
|---|---|---|
| `--url` | `$BASE_URL` | Server base URL |
| `--api-key` | `$API_KEY` | Bearer token |
| `--slots N` | `$SLOTS` or `1` | Parallel slots the server is configured with (recorded in CSV) |
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
| `--url` | `$BASE_URL` | Server base URL |
| `--api-key` | `$API_KEY` | Bearer token |
| `--slots N` | `$SLOTS` or `1` | Parallel slots the server is configured with (recorded in CSV) |
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
