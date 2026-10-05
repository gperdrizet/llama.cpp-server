# llama.cpp-server

Multi-profile `llama.cpp` inference deployment for **pyrite**: two 16 GB Tesla P100s, 24 CPU cores, 251 GB RAM, 2.7 TB NVMe fronted bcache scratch. One OpenAI-compatible API on a single port (8502), fronted by nginx and backed by whichever profile you select:

- **agents**: a dynamic personal agent router that loads one of several models on demand into shared GPU memory
- **pool**: a dual-replica multiuser pool, one `llama-server` per GPU, always hot

Two CPU-only sidecars are independent of the profile: a resident embedding server (8082) and an opt-in 125 B MoE for long-horizon coding (8085).

> **Public API gateway**: [promptlyapi.com](https://promptlyapi.com/register): authentication, token metering, billing, and an admin panel for indie devs and hobbyists on a budget. 1 M free tokens for new registrations.

## What is running

| Service | Port | Model | Where | Purpose |
|---|---|---|---|---|
| nginx gateway | **8502** (0.0.0.0) | — | `/etc/nginx/llama-profiles/` | Single public port; routes to the active profile |
| `agent-router` | 8503 (loopback) | any preset model | systemd unit | On-demand multi-model router (profile: `agents`) |
| `llama-pool@replica0/1` | 8083 / 8084 | `gpt-oss-20b` | systemd template | Student pool, 4 slots/replica (profile: `pool`) |
| `llama-embed` | 8082 (loopback) | `bge-m3-Q8_0` | systemd unit | CPU-only embedding server, always available |
| `llama-flash-next` | 8085 (loopback) | `Qwen3.8-Flash-Next-Q8_0` (125 B MoE, 178 GB) | systemd unit | CPU-only long-horizon coding; start manually |

All inference runs as the unprivileged `llama` user with `ProtectSystem=strict` / `ProtectHome=read-only` / `ReadOnlyPaths` hardening.

```
                    ┌───────────────┐
 clients ── 8502 ──►│ nginx (pyrite)│──► active-profile.conf (symlink)
                    └───────────────┘             │
               ┌───────────────────┬──────────────┴───┐
               ▼                   ▼                  ▼
        agents profile        pool profile       none profile
        127.0.0.1:8503        8083 + 8084    127.0.0.1:1 (discard)
              │
       ┌──────┴──────────┬──────────────────┐
       ▼                 ▼                  ▼
  Qwen3.8-27B       gpt-oss-20b      Qwen2.5-Coder-7B
  (tensor-split,    (pinned CUDA0)   (pinned CUDA1)
   both GPUs)

  sidecars (independent of profile):
  - 8082 bge-m3 embeddings (CPU)
  - 8085 Qwen3.8-Flash-Next (CPU, opt-in)
```

## Repository layout

| Path | What |
|---|---|
| `utils/` | Everything deploy.sh installs: unit files, router preset (`agent-router.ini`), pool config, secrets template, profile switcher |
| `models/` | **Symlink** → `/mnt/fast_scratch/llama-models` (fast bcache). Never in git. See [Fast storage](#fast-storage-for-model-files) |
| `tests/` | Benchmark runners (context-fit, load test, generation benchmark) + configs; see `tests/README.md` |
| `notebooks/` | Result analysis (`load_test_results.ipynb`) |
| `legacy/` | Old single-server deployment (`llamacpp.service` + `deploy_service.sh`); kept for reference, not deployed |
| `/opt/llama.cpp/build/bin/` | Prebuilt binaries (`llama-server`, `llama-cli`, …) — not part of this repo |
| `/etc/llama/` | Runtime config (owned by root): `agent-router.ini`, `llama-pool-*`, `llama-secrets.env` (0600) |
| `/usr/local/bin/switch-llama-profile` | Profile switcher installed by deploy.sh |

## Quick start

Prerequisites (once, per machine): the `llama` system user (`sudo useradd --system --no-create-home --shell /usr/sbin/nologin llama`), a built llama.cpp tree at `/opt/llama.cpp` (current: `0.5.0-dev`, build 11206, CUDA enabled), nginx, and the fast storage mount (`/mnt/fast_scratch`).

```bash
# 1. Clone this repo (anywhere; the units reference the models symlink by absolute path)
git clone git@github.com:gperdrizet/llama.cpp-server.git

# 2. Point models/ at fast storage (see below if it isn't already a symlink)
ln -s /mnt/fast_scratch/llama-models $PWD/models

# 3. Deploy: installs units, /etc/llama config, and the switcher.
#    First run scaffolds /etc/llama/llama-secrets.env from the template and stops.
bash utils/deploy.sh

# 4. Fill in the keys (see Secrets) and re-run deploy to finish:
sudo nano /etc/llama/llama-secrets.env
bash utils/deploy.sh

# 5. Pick the profile you want running:
sudo bash utils/switch-llama-profile agents    # personal router + embed
# sudo bash utils/switch-llama-profile pool    # student pool
# sudo bash utils/switch-llama-profile none    # GPUs free

# 6. Verify
curl -s http://127.0.0.1:8502/health
```

`deploy.sh` never starts, stops, or restarts services — only the switcher does that. It also never overwrites an existing secrets file.

## Profiles and switching

**Only one profile at a time**. Both share the same GPUs. `none` is the right choice when you want the P100s for something else; `agents` additionally starts the embed sidecar.

```
switch-llama-profile [agents|pool|none]`
```

Re-points the nginx symlink (`/etc/nginx/llama-profiles/active-profile.conf`), restarts nginx, clears VRAM, and starts the selected backend.

## Adding a new model

### Router model (GPU, on-demand)

1. Download the `.gguf` into `models/`.
2. Add a section to `utils/agent-router.ini` (deploys to `/etc/llama/agent-router.ini`):

   ```ini
   [MyNewModel]
   alias                = MyNewModel          # the name clients request
   model                = /home/siderealyear/llama.cpp/models/MyNew-Model-Q4_K_M.gguf
   ctx-size             = 131072
   device               = CUDA1               # pin to one GPU…
   # …or split across both:
   # split-mode         = tensor
   ```

   `[*]` holds defaults for all sections: `n-gpu-layers = -1` (all layers on GPU) and `sleep-idle-seconds = 300` (auto-unload after 5 min of silence). Reasoning models can add `spec-type = draft-mtp`, `spec-draft-n-max`, `reasoning-budget`, and `chat-template-kwargs` (see the `Qwen3.8-27B` section).

3. Redeploy and restart. **The router reads the preset only at startup**; an edited-but-unrestarted router silently keeps the old model list:

   ```bash
   bash utils/deploy.sh
   sudo bash utils/switch-llama-profile agents
   ```

4. Verify with a real request (auth required):

   ```bash
   curl -s http://127.0.0.1:8503/v1/chat/completions \
     -H "Authorization: Bearer <LLAMA_ROUTER_KEY>" -H "Content-Type: application/json" \
     -d '{"model":"MyNewModel","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
   ```

Constraints: `--models-max 1` means **one model resident at a time** — a new request evicts the previous one and pays its load time (a few seconds for ~16 GB Q4). Pick ctx-size to fit 16–32 GB VRAM; `sleep-idle-seconds` reclaims VRAM automatically.

### Pool model

`llama-pool@.service` is a systemd template: each replica instance expands `$MODEL_PATH $MODEL_ALIAS $SERVER_LIMITS $OPTIMIZATIONS $SECURITY $ARGS_<instance>` from the three `/etc/llama/llama-pool-*` files. Currently two replicas, one per GPU (`--tensor-split 1,0` / `0,1`), 4 slots each, `Restart=always`.

Edit `utils/llama-pool-global` (shared: `MODEL_PATH`, `MODEL_ALIAS`, `SERVER_LIMITS`, `OPTIMIZATIONS`) and/or `utils/llama-pool-instances` (per-replica `ARGS_replicaN`), re-deploy, then `switch-llama-profile pool`.

### Standalone service (dedicated port, e.g. CPU-only)

Copy `utils/llama-flash-next.service` as a template: change the `Description`, `--model`, `--alias`, and `--port` (pick a free one), then add one line to `utils/deploy.sh`'s unit-install block and re-deploy. Use `Environment=CUDA_VISIBLE_DEVICES=` and `--n-gpu-layers 0` for CPU-only so the GPUs stay free.

## Secrets

`/etc/llama/llama-secrets.env` is **root-only (0600)** and is a **systemd EnvironmentFile, not a shell script** — do not `source` it. It holds:

```
LLAMA_ROUTER_KEY=<key>            # referenced as ${LLAMA_ROUTER_KEY} by agent-router.service
SECURITY=--api-key <key> --host 127.0.0.1   # expanded as $SECURITY by llama-pool@.service
```

The `SECURITY` line is the gotcha: valid for systemd, but bash parses `SECURITY=--api-key` as an assignment prefix and tries to *execute* the key. If you need the key in a script, extract it: `grep -E '^LLAMA_ROUTER_KEY=' /etc/llama/llama-secrets.env | cut -d= -f2-`. After editing the file, restart affected services — systemd re-reads EnvironmentFiles only at unit start.

## Operations

```bash
systemctl status agent-router llama-embed llama-flash-next "llama-pool@*"
journalctl -u agent-router -f                 # or -u llama-pool@replica0, etc.
journalctl -u agent-router --since "10:00"    # time-range
nvidia-smi                                     # VRAM per GPU
curl -s http://127.0.0.1:8502/health          # gateway
```

Useful log lines (router, `SyslogIdentifier=llama-server`):
- `slot print_timing: … prompt processing, n_tokens = N, … t = X s / Y tokens per second` — prefill rate
- `slot print_timing: … n_gen = N, tg = X t/s, tg_3s = Y t/s` — generation rate (sustained / 3-second window)
- `load_model: loading model '…'` and `slot release: … stop processing` — load/evict lifecycle

## Fast storage for model files

Model weights are memory-mapped, so first-touch page-in speed is limited by disk. Keep them on the bcache, not the system root:

```bash
sudo mkdir -p /mnt/fast_scratch/llama-models
ln -s /mnt/fast_scratch/llama-models <repo>/models     # repo-local, gitignored
```

Then make the path traversable by the `llama` service user (the units run with `ProtectHome=read-only`, so plain permissions on `/home/...` are not enough):

```bash
sudo setfacl -m u:llama:x /home/<user>
sudo setfacl -m u:llama:rx <repo>
sudo setfacl -m u:llama:rx /mnt/fast_scratch/llama-models
```

Notes:
- Download models into `models/` with `huggingface-cli` or `wget`; the repo stores none of them (`.gitignore` excludes `models`).
- All unit files reference models through the repo path (`/home/<user>/llama.cpp/models/...`), so moving the *target* of the symlink requires no unit edits — just restart the affected services.
- Check headroom before big downloads: `df -h /mnt/fast_scratch` (this box's bcache runs near full).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `401 Invalid API Key` | Client key ≠ key in the running router's environment. Compare with `sudo tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value agent-router)/environ \| grep ^LLAMA_ROUTER_KEY=`; mismatch after a secrets edit means the router needs a restart. Trailing whitespace/CRLF in the env file also causes this. |
| `500 model name=X failed to load` for the *wrong* model name | The running router has a stale preset (ini edited, router not restarted) — section headers matter: keys without a `[Section]` line merge into the previous section. Restart via the switcher. |
| `model name=X is not found` | Alias not in the loaded preset — check `/etc/llama/agent-router.ini` spelling vs the request's `model` field, then restart. |
| Model load failure on a new quant | Build may not support the format (e.g. some ternary/experimental quants). Check the `llama-server` journal around the request for the loader error. |
| VRAM OOM at startup | ctx-size × KV cache + weights must fit. Lower `ctx-size`, or keep `n-gpu-layers = -1` so nothing leaks to CPU. |
| Generation much slower than expected | CPU-only sidecars (8085) are ~0.3–0.4 tok/s warm on this box — by design. For GPU models, confirm the model isn't mid-reload (see load lines above). |
| bcache nearly full | `df -h /mnt/fast_scratch`; prune old quants before downloading new ones. |

## Benchmarks

Runners live in `tests/` (see `tests/README.md`): `context_fit.py` (largest GPU-resident context per model/KV type), `load_test.py` (end-to-end latency vs concurrency, incl. YAML suite mode), and `generation_benchmark.py`. Outputs land under `tests/results/...`; analysis in `notebooks/load_test_results.ipynb`.

> Benchmark numbers in this repo are being re-run against the current build; treat existing plots in `assets/` as provisional until then.

Known limitation (re-test after upstream updates): speculative decoding with MTP draft model does not work for the `qwen35`/M-RoPE family (Qwen3.8-27B) on the current build. The MTP draft model loads but verification fails.
