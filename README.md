# llama.cpp-server

Distributed `llama.cpp` inference deployment, managed by **Ansible + Docker** with
**LiteLLM** as the unified router. Serves a mixed fleet of local and remote models
through one OpenAI-compatible API.

## Topology

Two machines on a Tailscale LAN:

| Node | Tailscale IP | Hardware | Role |
|---|---|---|---|
| **pyrite** | 100.64.0.2 | 2× Tesla P100 16 GB, 24 cores, 251 GB RAM, bcache NVMe scratch | Local models + LiteLLM router + nginx front |
| **ethics-gradient** | 100.64.0.5 | 1× CUDA GPU | Remote models (coder + embeddings) |

```
                        ┌──────────────────────────────────────┐
                        │ VPS gateway  (promptlyapi.com)       │
                        │ public OpenAI-compatible API (HTTPS) │
                        └──────────────────────────────────────┘
                                         │
                                         │
                                         │
                         ┌───────────────┘
                         ▼
    ┌───────────────────────────────┐        ┌───────────────────────────────┐
    │ pyrite                        │        │ ethics-gradient               │
    │                               │        │                               │
    │ nginx :8502                   │        │                               │
    │    │                          │        │                               │
    │    ▼                          │        │                               │
    │ litellm-router :8506          │──LAN──►│ llama-cuda :8080              │
    │    │                          │        │    • Qwen2.5-Coder-7B         │
    │    ▼                          │        │    • BGE-M3  (embeddings)     │
    │ llama-cuda :8080              │        │                               │
    │    • Qwen3.8-27B              │        │                               │
    │    • Qwen3.8-Flash-Next       │        │                               │
    └───────────────────────────────┘        └───────────────────────────────┘
```

- **nginx (`:8502`)** — public entry point. Proxies everything to LiteLLM and exposes a
  fast `/healthz` (LiteLLM `/health/readiness`); the deep `/health` endpoint probes all
  models and is intentionally *not* the public probe.
- **LiteLLM (`:8506`)** — the only router. Owns the model list (see `inventory/host_vars/pyrite.yml`,
  `managed_models`), latency-based routing, and model selection. No per-profile switching exists.
- **llama.cpp (`llama-cuda` container)** — one per node, `ghcr.io/ggml-org/llama.cpp:server-cuda`,
  started with `--models-preset`. Holds `models_max` models resident; a new request evicts and
  reloads.

### Models

| Model | Node | Preset | Notes |
|---|---|---|---|
| `Qwen3.8-27B` | pyrite | `[Qwen3.8-27B]` in `llama_router_presets/pyrite.ini` | Dual-GPU tensor split, MTP speculative decode, 128K ctx, low reasoning effort |
| `Qwen3.8-Flash-Next` | pyrite | `[Qwen3.8-Flash-Next]` | 125B MoE, 40 layers on GPU + `cpu-moe` offload with mlock, slow cold load (~400 s) |
| `Qwen2.5-Coder-7B` | ethics-gradient | `[Qwen2.5-Coder-7B]` in `llama_router_presets/ethics_gradient.ini` | Instruct model; used for fast utility/classifier workloads |
| `BGE-M3` | ethics-gradient | `[BGE-M3]` | Embeddings (`/v1/embeddings`) |

Both presets use q8_0 KV caches. Model files live in `/mnt/fast_scratch/llama-models` on each
node (pyrite: bcache, check free space before big downloads), mounted into the container at `/models`.

## Repository layout

| Path | What |
|---|---|
| `playbooks/deploy_llm_stack.yml` | Deploys the `llama-cuda` container (and LiteLLM where enabled) per node: presets, chat template, network, GPU passthrough, volumes |
| `playbooks/deploy_nginx_gateway.yml` | One-time cutover playbook: backs up and replaces the nginx public gateway, removes old profile machinery, validates + graceful reload |
| `nginx/llama-gateway` | The public gateway site (`:8502 → :8506`, `/healthz`) |
| `ansible.cfg` | Points at `inventory/hosts.ini` |
| `inventory/hosts.ini` | Node list (pyrite local, ethics-gradient over SSH/Tailscale) |
| `inventory/host_vars/*.yml` | Per-node config: preset source, `models_max`, LAN exposure, image, LiteLLM `managed_models` |
| `llama_router_presets/*.ini` | llama.cpp `--models-preset` files per node |
| `chat_templates/qwen_chat.jinja` | Patched Qwen chat template (accepts `developer` role) |
| `templates/litellm_config.json.j2` | LiteLLM `config.json` template |
| `tests/` | Benchmark runners + configs (see `tests/README.md`; update pending) |
| `legacy/` | Old systemd/nginx deployment, **reference only** — do not deploy from it |
| `models/` | **Symlink** → `/mnt/fast_scratch/llama-models` (gitignored) |

## Deployment

Prereqs on each node: Docker, NVIDIA container runtime (GPU passthrough), Ansible with
`community.docker` (pyrite only, for the `docker_*` modules). `inventory/hosts.ini` targets
pyrite locally and ethics-gradient over SSH on the Tailscale IP.

Deploy a node (idempotent; recreates the `llama-cuda` container only when its preset or
chat template changed, or when `ansible_docker_recreate: true` is passed):

```bash
ansible-playbook playbooks/deploy_llm_stack.yml --limit pyrite
ansible-playbook playbooks/deploy_llm_stack.yml --limit ethics-gradient
```

The playbook:
1. Copies the node's preset from `llama_router_presets/` → `/opt/llama/config/presets.ini`
   (and the chat template → `/config/template.jinja` when `has_custom_template`).
2. Renders `templates/litellm_config.json.j2` → `/opt/litellm/config.json` (pyrite only).
3. Creates the internal `llm-network` bridge.
4. Starts `llama-cuda` with all GPUs, `shm 2 GB`, host IPC (P2P lanes), `restart_policy: unless-stopped`.
   Port 8080 is published to the LAN only when `expose_llama_port_to_lan` (ethics-gradient).
5. Starts `litellm-router` (pyrite only) on `:8506` with no master key (`LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY`).

### Adding / changing a model

1. Add a section to the node's preset in `llama_router_presets/` (model path, ctx, device split,
   speculative decode, …).
2. If it must be routable, add it to `managed_models` in that node's `host_vars` (the entry is
   what LiteLLM's `model_list` is built from — exact name casing matters, both for the LiteLLM
   alias and for the GGUF filename it references in the preset).
3. Re-run the playbook for that node — a changed preset redeploys the container; a changed
   `host_vars` re-renders the LiteLLM config, which recreates the router.

The llama.cpp router reads its preset **only at container start** — editing the file on
`/opt/llama/config` without recreating the container silently keeps the old model list.

## Operations

```bash
docker ps -a                                   # containers on this node
docker logs -f llama-cuda                      # llama.cpp server logs
docker logs -f litellm-router                  # router logs (pyrite)

curl -s http://127.0.0.1:8506/v1/models        # router: full model list
curl -s http://127.0.0.1:8506/health/liveliness   # fast liveness  (~ms)
curl -s http://127.0.0.1:8506/health/readiness    # fast readiness (~ms)
# /health is the DEEP check (probes every model, 10-30s+); don't use it for health-checking

curl -s http://127.0.0.1:8502/healthz          # public gateway health (5 ms)
curl -s http://127.0.0.1:8502/v1/models        # public gateway -> router
curl -s https://promptlyapi.com/v1/models -H "Authorization: Bearer $KEY"   # full public chain

nvidia-smi                                     # VRAM on this node
```

### Quick API calls

```bash
# chat
curl -s http://127.0.0.1:8506/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'

# embeddings
curl -s http://127.0.0.1:8506/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BGE-M3","input":"hello world"}'
```

## Gotchas

- **Model file names are case-sensitive.** Preset `model =` paths must match the GGUF on disk
  byte-for-byte; a single-case mismatch fails the load with a bare `failed to load model` at
  request time (the router lists the model regardless).
- **pyrite's bcache runs near full.** Check `df -h /mnt/fast_scratch` before large downloads.
- **`models_max = 1` on pyrite** means model *swap*, not stack: requesting Flash-Next evicts
  the 27B and its ~400 s cold load is the price. Warm loads are ~40 s.
- **LiteLLM `/health` is deep and slow.** Point health checkers at `/healthz` (gateway) or
  `/health/liveliness`/`/health/readiness` (router) — not raw `/health`.
- **The public gateway auth lives in promptly** (gatekeeper → promptlyapi.com), not in this
  repo or in nginx. The router runs keyless on the private side; never expose `:8506` to the
  LAN.
- **Qwen3.5+ chat templates**: the stock template rejects the `developer` role (breaks Claude
  Code-style clients). The patched `chat_templates/qwen_chat.jinja` is deployed to
  `/config/template.jinja` and referenced by the presets.

## Claude Code backend

This stack is the backend for local Claude Code sessions (via `~/.claude/settings.json`):

| Slot | Env var | Model | Why |
|---|---|---|---|
| main | `ANTHROPIC_DEFAULT_SONNET_MODEL` | Qwen3.8-27B | conversation |
| fast | `ANTHROPIC_DEFAULT_OPUS_MODEL` | Qwen3.8-Flash-Next | extended work |
| utility | `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Qwen2.5-Coder-7B | auto-mode tool classification, compaction — fast, non-reasoning |

`ANTHROPIC_BASE_URL` points at the public gateway (`https://promptlyapi.com/v1`) with a
promptly key, so sessions exercise the full production chain. Note: model slots are resolved
at **session start** — changing them requires a new session.
