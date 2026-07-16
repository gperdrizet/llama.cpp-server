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
- [Deployment](#deployment)
- [Systemd service](#systemd-service)
- [Testing](#testing)
  - [Max context size](#max-context-size)
  - [Results](#results)
  - [Load test](#load-test)
  - [Analysis notebook](#analysis-notebook)


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

The unit file template lives in `utils/llamacpp.service`. Deploy it with:

```bash
# Copy and fill in the env file
cp .env.template .env

# Deploy and immediately restart the service
bash utils/deploy_service.sh --restart
```

`deploy_service.sh` substitutes `SUB_API_KEY_HERE`, etc from `.env`, copies the result to `/etc/systemd/system/llamacpp.service`, and runs `systemctl daemon-reload`.

> **Note:** `.env` contains the real API key - do not commit it. It is listed in `.gitignore`.

Model files are not included in this repository. Download them separately with `huggingface-cli` or `wget`.


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


## Testing

### Max context size

The maximum context size that will fit within the avalible GPU memory is determined with `tests/run_fit_context_size.py`.

The runner has two phases:
1. **Coarse scan** over a standard context list (`4096` to `262144`).
2. **Refinement scan** after first OOM:
   - fit a linear model (`peak_vram_total_mib` vs `ctx_size`) from successful runs
   - predict the memory-limit context
    - probe around that estimate for a tighter boundary.

It does the max context determination three times, once for each KV-cache quantization level (`q4_0`, `q8_0`, `f16`) and aggregates all results into the same CSV/log/summary/plot artifacts.

- `context_fit.csv`: one row per attempted context (`ok`, `failed_oom`, or `failed`)
- `context_fit.log`: full command, stdout, stderr, and VRAM summary per run
- `context_fit_summary.json`: compact run summary (boundary estimates and regression details)
- `context_fit_plot.png`: matplotlib plot combining all KV-cache runs on one chart with color-separated series


```bash
# Example: run context-fit on two P100 GPUs
.venv/bin/python tests/run_fit_context_size.py \
  --model /opt/models/Qwen3.6-27B-Q4_K_M.gguf \
  --gpus 1,2 \
  --tensor-split 1,1 \
  --split-mode layer
```

**Useful options**:

| Option | Purpose |
|---|---|
| `--model` | Model path (absolute or filename under `/opt/models`) |
| `--gpus` | Physical GPU indexes for `CUDA_VISIBLE_DEVICES` |
| `--tensor-split` | Tensor split ratio for multi-GPU runs |
| `--memory-cap-mib` | Explicit regression target memory cap |
| `--memory-headroom-mib` | VRAM safety margin when auto-computing memory cap |
| `--refine-step` | Granularity for refinement probe contexts |
| `--kv-cache-types` | Comma-separated KV cache types to run (default: `q4_0,q8_0,f16`) |
| `--results-dir` / `--run-name` | Output location and run label |
| `--service-name` / `--no-manage-service` | Service lifecycle control around benchmark runs |

### Results

The table below uses the completed context-fit runs for the Qwen Q3 and Q4 quants on GPUs `1,2` with `split-mode layer` and `tensor-split 1/1`. The throughput columns are taken from the max-context benchmark rows (`ctx=262144`) in the run logs.

| Model | Model quant | KV quant | GPU config | Max context | Peak VRAM @ max context (GiB) | PP rate @ max ctx | TG rate @ max ctx |
|---|---|---|---|---:|---:|---:|---:|
| Qwen3.6-27B | Q3_K_S | f16 | `1,2` / layer / 1/1 | 256k | 29.7 | 38.9 | 5.35 |
| Qwen3.6-27B | Q3_K_S | q8  | `1,2` / layer / 1/1 | 256k | 25.7 | 39.0 | 3.83 |
| Qwen3.6-27B | Q3_K_S | q4  | `1,2` / layer / 1/1 | 256k | 21.6 | 38.8 | 3.96 |
| Qwen3.6-27B | Q4_K_M | f16 | `1,2` / layer / 1/1 | 256k | 30.8 | 36.5 | 0.88 |
| Qwen3.6-27B | Q4_K_M | q8  | `1,2` / layer / 1/1 | 256k | 29.5 | 38.8 | 4.17 |
| Qwen3.6-27B | Q4_K_M | q4  | `1,2` / layer / 1/1 | 256k | 25.4 | 38.7 | 4.32 |


### Load test

`tests/load_test.py` supports both one-off runs and YAML-defined benchmark suites.

Single run mode measures end-to-end response latency against the running `llamacpp.service` as a function of concurrent callers. Unlike the standalone benchmark runner, which bypasses the server binary, this exercises the full HTTP stack and is useful for tuning `--parallel` slot count.

```bash
# Run with defaults (concurrency levels 1 2 4 8 16 32, 3 repetitions each)
.venv/bin/python tests/load_test.py

# Custom concurrency levels and repetitions
.venv/bin/python tests/load_test.py --levels 1 2 4 8 --requests 5

# Enable streaming (also measures time-to-first-token)
.venv/bin/python tests/load_test.py --stream
```

#### Suite mode (YAML, recommended)

Use `--suite-config` to run a sequence of load-test experiments defined in YAML.

```bash
# Run the default suite
.venv/bin/python tests/load_test.py --suite-config tests/benchmarks/load-test.yaml

# Preview actions without redeploying or sending requests
.venv/bin/python tests/load_test.py --suite-config tests/benchmarks/load-test.yaml --dry-run
```

In suite mode, each case can set model/deployment settings (`model`, `slots`, `ctx_size`, `gpu_layers`, `cuda_device`, `tensor_split`, `prompt_cache_size`) and test settings (`levels`, `requests`, `max_tokens`, `stream`, `url`).

For each case the runner:
1. Updates `.env` with case-specific server settings.
2. Calls `utils/deploy_service.sh --restart`.
3. Runs the load test.
4. Writes results to `tests/results/YYYY-MM-DD_<case-label>_slotsN/load_test.csv`.

The `.env` file is restored to its original contents when the suite finishes.

> **Note:** If `.env` points to a public URL behind nginx rate limits, set per-case `url: http://localhost:8502` for on-server benchmarking.

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--suite-config FILE` | _(none)_ | Run YAML-defined suite with automated redeploy between cases |
| `--url` | `$BASE_URL` or `$LLAMA_BASE_URL` or `http://localhost:8502` | Server base URL |
| `--api-key` | `$API_KEY` or `$LLAMA_API_KEY` | Bearer token |
| `--levels N [N ...]` | `1 2 4 8 16 32` | Concurrency levels to test |
| `--requests N` | `3` | Repetitions per level |
| `--slots N` | `$SLOTS` or `$LLAMA_SLOTS` or `1` | Slot count recorded in CSV |
| `--stream` | off | Streaming mode (enables TTFT measurement) |
| `--model-label` | _(empty)_ | Model identifier recorded in CSV |
| `--ctx-size N` | _(none)_ | Context size recorded in CSV |
| `--output FILE` | `tests/results/load_test_YYYY-mm-dd_HH-MM.csv` | Output path (single-run mode) |

Use `notebooks/load_test_results.ipynb` to analyze suite outputs across configurations.
