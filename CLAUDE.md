# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this repo is

**Not the llama.cpp source.** This is `llama.cpp-server`: the deployment/ops repo for a
Dockerized, Ansible-managed `llama.cpp` inference stack across two Tailscale nodes —
*pyrite* (2× Tesla P100 16 GB, 24 cores; local models + the LiteLLM router, which is the
public front) and *ethics-gradient* (remote models). It contains the Ansible playbooks,
per-node inventory, llama.cpp router presets, the LiteLLM config template, and benchmark
tooling. The `llama.cpp` server image
(`ghcr.io/ggml-org/llama.cpp:server-cuda`) is pulled, not built here.

One public OpenAI-compatible port (8502), the **LiteLLM router** itself (container
port 8506, mapped to host 8502), which routes to the models:

- pyrite local: `Qwen3.8-27B` (dual-GPU, MTP), `Qwen3.8-Flash-Next` (MoE, CPU-offloaded)
- ethics-gradient remote (over LAN): `Qwen2.5-Coder-7B`, `BGE-M3` (embeddings)
- the public chain is `promptlyapi.com → gatekeeper → pyrite:8502 (litellm-router)`

## Key commands

```bash
ansible-playbook playbooks/deploy_llm_stack.yml --limit pyrite             # deploy/redeploy a node's llama.cpp (+ litellm on pyrite)
docker ps -a && docker logs -f llama-cuda                                  # container state / llama.cpp logs (per node)
curl -s http://127.0.0.1:8502/v1/models                                    # router model list
curl -s http://127.0.0.1:8502/health/readiness                             # public health (fast)
```

Benchmarks (see `tests/README.md` — docs pending update): runners are Python driven by
configs in `tests/config/<benchmark>/`. Results go to `tests/results/<benchmark>/...`.
Prefer YAML/CSV configs over ad-hoc shell for repeatable runs.

## Non-obvious gotchas

- **The llama.cpp router reads its preset only at container (re)start.** Editing
  `/opt/llama/config/presets.ini` (or the repo `llama_router_presets/*.ini`) without
  recreating the container silently keeps the old model list. The playbook re-creates the
  container when the preset/chat-template changes; to force it, pass
  `-e ansible_docker_recreate=true`.
- **LiteLLM reads its config at start too.** A changed `managed_models` in
  `inventory/host_vars/<node>.yml` re-renders `/opt/litellm/config.json` and recreates the
  router on the next `deploy_llm_stack.yml` run.
- **Model file names are case-sensitive.** The preset `model =` path must match the GGUF on
  disk byte-for-byte; a case mismatch fails at *request* time with a bare "failed to load
  model" even though the router lists the model. Same for the LiteLLM `model_name` alias.
- **`models_max = 1` on pyrite** → model *swap*, not stack. Requesting Flash-Next evicts
  the 27B; its cold load is ~400 s (warm ~40 s).
- **`models/` is a gitignored symlink** → `/mnt/fast_scratch/llama-models` (bcache) on each
  node. Pyrite's bcache runs near full — check `df -h /mnt/fast_scratch` before big downloads.
- **LiteLLM `/health` is the deep check** (probes every model, 10–30 s+). Health-check
  against `:8502/health/liveliness` or `:8502/health/readiness` instead.
- **Auth lives in promptly, not here.** The LiteLLM router fronts `:8502` keyless on the
  private side; keying + metering happen at promptly/gatekeeper. Keep `:8502` on the
  Tailscale LAN only — its only client is the VPS gateway.
- **Qwen3.5+ chat templates**: the stock template rejects the `developer` role (breaks
  Claude Code et al.). `chat_templates/qwen_chat.jinja` is deployed to `/config/template.jinja`
  and referenced by the presets. `--chat-template chatml` "works" but silently disables thinking.

## Conventions & rules

- Never commit secrets: the promptly API key (lives in `~/.claude/settings.json`), `.env`,
  and anything containing real keys. `.claude/` and `models/` are gitignored.
- Benchmark result artifacts (`tests/results/...`) are run outputs; ask before committing a
  batch of them.
- `legacy/` (the old systemd/nginx deployment, including `legacy/utils/` and
  `legacy/utils/switch-llama-profile`) is **reference only** — do not modify or deploy from it.
- Config deployed to nodes lands under `/opt/llama/config/` and `/opt/litellm/` (root-owned);
  editing files in this repo does nothing until you re-run the relevant playbook.
- This session's own Claude Code runs *through* this deployment (LiteLLM router on
  `:8502`, Qwen models), so re-running a deploy that recreates the router can interrupt
  this session. Warn before `deploy_llm_stack.yml` on pyrite.
