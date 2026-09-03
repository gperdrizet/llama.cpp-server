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

Model files are not included in this repository. Download them separately with `huggingface-cli` or `wget` into the repository's `models/` path.

In this setup, `models/` is a symlink to storage under `/mnt/fast_scratch`. Keep `MODEL_DIR` set to `<repo>/models` in `.env` so service configuration stays repo-relative.

When deploying from benchmark-derived context sizes, keep server KV-cache quantization aligned with benchmark settings. Set `KV_CACHE_TYPE` in `.env` (for example `q4_0`) so `llama-server` uses the same K/V cache type. Mismatched cache type (for example default `f16`) can cause startup OOM even when context-fit benchmarks succeeded.

Because the service runs as user `llama` and `ProtectHome=read-only` is enabled, the full path to model files must be traversable/readable by `llama` (including parent directories). If needed, grant access with ACLs, for example:

```bash
setfacl -m u:llama:x /home/<user>
setfacl -m u:llama:rx <repo>
setfacl -m u:llama:rx /mnt/fast_scratch/llama-models
```


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
| `ProtectHome=read-only` | `/home`, `/root`, `/run/user` are visible read-only to the process |
| `ReadOnlyPaths=/opt/llama.cpp <repo>/models` | Both the install tree and model directory are read-only (model files are memory-mapped for reading only) |


## Testing

### Max context size

The maximum context size that will fit within the available GPU memory is determined with `tests/context_fit.py`.

The benchmark is driven by `tests/config/context_fit/context_fit.yaml`, which supplies the default run settings, score breakpoints, and the model list file. The model list itself lives in `tests/config/context_fit/models.csv`; comment out any model you want to skip before the next run. Each model row may also include a per-model `max_context` value, which the runner uses to derive that model's coarse context scan.

The runner has three phases:
1. **Coarse scan** over four context sizes derived per model as `max//8, max//4, max//2, max`.
2. **Bisection refinement** around the first failing context, at `--refine-step` granularity.
3. **Verification** re-runs the candidate max context `--verify-runs` times; any single failure marks that context unstable.

It runs the max-context search once per KV-cache quantization level listed in `--kv-cache-types` (default `q4_0`, `q8_0`, `f16`) and aggregates all results into the same CSV/log/summary/plot artifacts.

- `results.csv`: one row per attempted context (`ok`, `failed_oom`, or `failed`)
- `run.log`: full command, stdout, stderr, and VRAM summary per run
- `summary.json`: compact run summary with context boundary estimates and VRAM observations
- `plot.png`: matplotlib plot combining all KV-cache runs on one chart with color-separated series


```bash
# Example: run the full context-fit suite on two P100 GPUs
.venv/bin/python tests/context_fit.py \
  --config tests/config/context_fit/context_fit.yaml \
  --model-list tests/config/context_fit/models.csv
```

**Useful options**:

| Option | Purpose |
|---|---|
| `--config` | YAML file with run defaults and score breakpoints |
| `--model` | Single model path or filename under `models/` |
| `--model-list` | CSV file with one model path and optional per-model max context per line |
| `--max-context` | Max context for single-model runs; coarse sweep derives from this value |
| `--gpus` | Physical GPU indexes for `CUDA_VISIBLE_DEVICES` |
| `--bench-bin` | Path to `llama-bench` |
| `--results-dir` | Output directory |
| `--run-name` | Output run label |
| `--tensor-split` | Tensor split ratio for multi-GPU runs |
| `--split-mode` | Tensor split mode (`layer` in the current setup) |
| `--allow-host` | Allow `llama-bench` to spill past GPU VRAM into system RAM; default is off for strict GPU-fit discovery |
| `--fit-target` / `--fit-ctx` | Only used when `--allow-host` is set; these are not used in the default GPU-only fit pass |
| `--n-prompt` / `--n-gen` / `--repetitions` | Throughput benchmark settings |
| `--flash-attn` | Flash attention mode: `on` (default, matches `llamacpp.service`), `off`, or `auto` |
| `--refine-step` | Granularity for bisection refinement probes |
| `--verify-runs` | Confirmation runs at the final context |
| `--max-run-seconds` | Hard timeout for each `llama-bench` invocation |
| `--kv-cache-types` | Comma-separated KV cache types to run (default: `q4_0,q8_0,f16`) |
| `--skip-completed` | Resume helper: skip models whose `summary.json` already exists |
| `--stop-after-model` | Run only N models, then exit cleanly |
| `--service-name` / `--no-manage-service` | Service lifecycle control around benchmark runs |

### Results

The `dual_gpu` sweep (2026-09-01) measured the maximum stable context for five models on two Tesla P100-PCIE-16GB GPUs (32 GiB total), driven by `tests/config/context_fit/context_fit.dual_gpu.yaml`. Artifacts are in `tests/results/context_fit/dual_gpu/`.

Run settings: GPUs `1,2`, `split-mode layer`, `tensor-split 1/1`, KV cache `f16` and `q8_0`, strict GPU-only fit (`allow-host false`, `fit-target 0`), `n_prompt 128`, `n_gen 32`, `repetitions 1`, `refine-step 4096`, `verify-runs 1`.

**Max context** is the largest context that passed the verification run, bounded above by the per-model ceiling in `models_dual_gpu.csv`. Peak VRAM is the summed peak across both GPUs at that context. All boundaries below verified stable, and every model reached the `interactive` deployment tier.

| Model | KV cache | Max context | Peak VRAM (GiB) |
|---|---|---:|---:|
| Qwen3.8-27B-UD-Q4_K_M | f16 | 208k | 29.8 |
| Qwen3.8-27B-UD-Q4_K_M | q8_0 | 256k | 28.5 |
| gemma-4-31B-it-Q4_K_M | f16 | 140k | 31.1 |
| gemma-4-31B-it-Q4_K_M | q8_0 | 212k | 31.5 |
| GLM-4.7-Flash-Q4_K_M | f16 | 198k | 29.8 |
| GLM-4.7-Flash-Q4_K_M | q8_0 | 198k | 25.2 |
| Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M | f16 | 96k | 30.6 |
| Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M | q8_0 | 128k | 27.1 |
| gpt-oss-20b-Q4_K_M | f16 | 128k | 15.7 |
| gpt-oss-20b-Q4_K_M | q8_0 | 128k | 14.5 |

Notes:

- Quantizing the KV cache to `q8_0` extended the reachable context for three of the five models (Qwen `+48k`, gemma `+72k`, Mistral `+32k`) at lower peak VRAM.
- GLM-4.7-Flash and gpt-oss-20b reached the same context under both KV types because they were capped by their configured ceilings in `models_dual_gpu.csv` (198k and 128k), not by VRAM.
- gpt-oss-20b is a 20B model and leaves large headroom (14-16 GiB peak of 32 GiB), so its ceiling is the model's own trained context limit rather than available memory.


### Speculative decoding (unsupported for the qwen35 / M-RoPE models)

Speculative decoding does not work for the `Qwen3.8-27B` (`qwen35` architecture) models on the current llama.cpp build (`b4024af6c`, build 9687), by any path:

- A standalone draft (`Qwen3.5-2B`, vocab-compatible) loads and drafts tokens, but the target's verification batch fails with `decode: failed to initialize batch` / `for M-RoPE, it is required that the position satisfies: X < Y`, yielding ~0% acceptance and no speedup.
- The purpose-built MTP head (`mtp-Qwen3.8-27B-Q4_0.gguf`) segfaults when loaded as a `-md` draft; `llama-speculative-simple` aborts.

Root cause: `qwen35` uses **M-RoPE** (`rope.dimension_sections = [11, 11, 10, 0]`), and this build's speculative batch construction cannot assign the non-contiguous positions M-RoPE requires. This is a llama.cpp limitation, not a configuration problem - the drafts are vocab-compatible and the flags are correct. Re-test after a llama.cpp update that adds M-RoPE support to the speculative path (and MTP support for this architecture). Until then, the bankable generation win is `-sm row` (~+14% tg) plus concurrency/batching.


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
# Run a suite
.venv/bin/python tests/load_test.py --suite-config tests/config/load_test/load-test-GTP-OSS-20B.yaml

# Preview actions without redeploying or sending requests
.venv/bin/python tests/load_test.py --suite-config tests/config/load_test/load-test-GTP-OSS-20B.yaml --dry-run
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
