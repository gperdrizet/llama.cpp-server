# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this repo is

**Not the llama.cpp source.** This is `llama.cpp-server`: the deployment/ops repo for a
multi-profile llama.cpp inference stack on *pyrite* (2× Tesla P100 16 GB, 24 cores).
It contains systemd units, router/pool config, the profile switcher, and benchmark
tooling. The prebuilt llama.cpp binaries live in `/opt/llama.cpp/build/bin/` (built
from the upstream v0.7.0 clone at `/opt/llama.cpp`) and are **not** part of this repo.

One public OpenAI-compatible port (8502) behind nginx, routed to a profile:

- `agents` → on-demand multi-model router (8503, loopback) + CPU embedding sidecar (8082)
- `pool` → dual-replica `llama-server` pool (8083/8084, one per GPU)
- `none` → GPUs freed
- `llama-flash-next` (8085) is a CPU-only sidecar, started manually

## Key commands

```bash
bash utils/deploy-llama                # install units + /etc/llama config; never starts/stops services
sudo bash utils/switch-llama-profile [agents|pool|none]   # the ONLY thing that starts/stops services
curl -s http://127.0.0.1:8502/health   # gateway health
journalctl -u agent-router -f          # or -u llama-pool@replica0/1, llama-embed
```

Benchmarks (see `tests/README.md`): runners are Python (`tests/router_benchmark.py`,
`load_test.py`, `context_fit.py`, `generation_benchmark.py`) driven by YAML configs in
`tests/config/<benchmark>/`. Results go to `tests/results/<benchmark>/<run>/` as
`results.csv` / `run.log` / `summary.json` / `plot.png`. Prefer YAML configs over ad-hoc
shell scripts for repeatable runs.

## Non-obvious gotchas

- **The router reads its preset (`agent-router.ini`) only at startup.** Editing the ini
  without restarting the router silently keeps the old model list — always
  `deploy-llama` then re-run the switcher.
- **`--models-max 1`**: the router holds one model resident; a new request evicts and
  reloads the previous one.
- **`/etc/llama/llama-secrets.env` is a systemd EnvironmentFile, not a shell script** —
  never `source` it (the `SECURITY=--api-key <key>` line breaks bash). Extract values
  with `grep ... | cut -d= -f2-`. It is root-only (0600); after edits, restart affected
  services. The repo-local template is `utils/llama-secrets.env.example`.
- **`models/` is a gitignored symlink** → `/mnt/fast_scratch/llama-models` (bcache).
  Unit files reference models through the repo path, so retargeting the symlink needs no
  unit edits, just service restarts. The bcache runs near full — check `df -h /mnt/fast_scratch`
  before big downloads.
- **Unit hardening**: services run as unprivileged `llama` user with
  `ProtectSystem=strict` / `ProtectHome=read-only`; repo files need ACLs
  (`setfacl u:llama:rx ...`) to be readable by services.
- **Runtime config lives in `/etc/llama/`** (root-owned), deployed from `utils/`.
  Editing `utils/*` does nothing until you redeploy.
- **Qwen3.5+ chat templates**: stock template rejects `developer` role (breaks Claude
  Code et al.). Use the patched `utils/qwen_chat.jinja` via `--chat-template-file`
  (see README troubleshooting). `--chat-template chatml` also "works" but silently
  disables thinking.

## Conventions & rules

- Never commit secrets: `llama-secrets.env`, `.env`, and anything containing real keys.
  `.claude/` and `models/` are gitignored.
- Benchmark result CSVs (`tests/results/...`) are run artifacts; ask before committing
  a batch of them.
- `utils/switch-llama-profile` in the repo is the source of truth for the installed
  `/usr/local/bin/switch-llama-profile`; keep them in sync when editing.
- `legacy/` is reference only — do not modify or deploy from it.
- This session's own Claude Code runs *through* this deployment (local gateway, Qwen
  models), so restarting services mid-conversation can interrupt this session. Warn
  before running the profile switcher.
