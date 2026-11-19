# Benchmark tooling

## Layout

```
tests/
├── bench/                  # Python package: all tooling + the shared engine
│   ├── paths.py            # canonical repo/tests/results paths
│   ├── llama_server_lib.py # shared Docker launch + measurement engine
│   ├── llama_server_optuna_bench.py
│   ├── llama_context_tuner.py
│   ├── inference_api_bench.py
│   └── make_bench_prompt.py
├── self_tests/             # host-side self-tests (no GPU/Docker/network)
├── config/<tool>/<model>.json
├── prompts/
├── results/<tool>/         # run outputs
└── llama-server-bench-results.ipynb   # legacy analysis of old results
```

Run a tool as a script from the repository root:
`.venv/bin/python tests/bench/<tool>.py --config tests/config/<tool>/<model>.json`.
They are also importable as `from bench import <tool>` (with `tests/` on
`sys.path`) — both forms resolve paths identically via `bench/paths.py`.

Four tools:

| Tool | What it answers | How it drives inference |
| --- | --- | --- |
| `llama_server_optuna_bench.py` | Best llama-server flags for a model (joint Optuna search) | Local Dockerized `llama-server` |
| `llama_context_tuner.py` | Largest safe `--ctx-size` for a model | Local Dockerized `llama-server` |
| `inference_api_bench.py` | PP / generation rates of a **live** OpenAI-compatible endpoint | HTTPS (no Docker, no GPU) |
| `make_bench_prompt.py` | Builds the long realistic test prompts | — (offline helper) |

The two local tools share `llama_server_lib`, which contains the Docker
launch driver, the PP decay-curve log parser, the standardized-workload
objective, and the VRAM / production guards. They need no local llama.cpp
install — the container image is pulled. Results go to
`tests/results/<tool>/...`.

> **⚠️ The two local tools stop and restart the production server for you.**
> Each launch loads the full model onto the GPU(s) and processes a long prompt
> (~32k tokens; a minute or two of PP each, plus the model load). On pyrite,
> loading the 27B alongside the live `llama-cuda` (or the
> `Qwen3.8-Flash-Next` swap) will OOM the P100s — and can drop the LiteLLM
> router this session routes through. So at startup these tools **stop the
> production container(s)** (`production_containers`, default `llama-cuda`) to
> free their VRAM, and **restart them** in a `finally` block — on a clean
> finish **or** a crash / Ctrl-C. **This takes the node's live models (and, if
> you are running Claude Code on pyrite, that session's own model) offline for
> the duration of the run.** The router itself keeps running but returns 502
> for the pyrite-local models until the restart. Pass `--no-stop-production`
> to opt out (production must then already be down, or the
> `min_free_vram_gib` gate refuses the launches). `inference_api_bench.py`
> never touches Docker or the local GPUs.

## Shared engine: `llama_server_lib`

Not a CLI — the reusable machinery imported by the Optuna search and the
context tuner (and exercised by `self_tests/test_llama_server_lib.py`):

- `run_bench_launch()` — one launch: preflight the model/prompt files and free
  VRAM, `docker run` the image (model → `/models`, prompt → `/prompts`, a temp
  work dir → `/work`, all GPUs), then in-container start `llama-server` (log
  to a file so a long model-load log can't deadlock), wait for the model to
  appear in `/v1/models` (`/health` answers 200 mid-load on current builds),
  POST the prompt with **`stream: true` and `timings_per_token: true`**, consume
  the SSE stream, and shut the server down. Every chunk carries a `timings`
  snapshot, so the result holds the per-token decode curve, `finish_reason`,
  and exact HTTP token counts.
- `parse_server_log()` — turns the server's progress lines (a line every
  ~4096 tokens with cumulative count / seconds / rate) into the PP
  rate-vs-occupancy **decay curve** (`points` of `n_tokens_cum` + `tps_seg`).
- `standardized_objective()` and friends — the versioned T\* objective (below).
- `model_specs_from_config()`, `expand_cells()`, VRAM gate, stale-container
  cleanup, and the production stop/restore guard.

Bare CLI switches (`--cpu-moe`, `--repack`, `--kv-offload`, `--no-host`,
`--spec-draft-cpu-moe`, …) take `"on"`/`"off"` in configs; the driver emits the
flag alone for `on`, and its negated form where one exists (`--no-repack`,
`--no-kv-offload`) for `off`. All other flags keep the normal `"key value"`
form.

## Standardized-workload objective (T\*)

Used by the Optuna search. Raw wall time is a poor objective: a model that
emits EOS early finishes fast without doing the requested work, and the PP
rate decays with context while decode slows as the KV cache grows, so runs at
different output / prompt lengths are not comparable. Every launch is scored
on the **same reference workload**:

```
T* = P(L0) + G(L0, N0)
```

`P(L0)` is the PP time interpolated off the measured decay curve at `L0`
reference prompt tokens (`objective.prompt_tokens`, default 32768; the final
segment's marginal rate extends it when the observed prompt is shorter).
`G(L0, N0)` is the generation time at reference context `L0` for `N0` output
tokens (`objective.output_tokens`, default `ntg`). The driver streams with
`timings_per_token`, so `G` is normally a least-squares fit of per-step decode
time vs context over a tail window (`objective.gen_fit_tail_steps`, default 64)
— mode `stream-fit`; with no usable per-token curve it scales the observed mean
decode step by the context-midpoint ratio (`linear-midpoint`). A short output
is **scaled UP** to `N0`, never rewarded, and an extrapolation too short to
trust sets `objective_low_confidence` + `objective_low_confidence_reasons`
(e.g. fewer than `objective.min_gen_tokens`, too few fit steps, an early
`finish_reason=stop`). Flagged trials stay in the study. `objective_pp_s`,
`objective_gen_s`, the fit coefficients and the observed token count are all
kept so any score can be re-derived offline; the raw `wall_request_s`,
`finish_reason` and per-token `gen_curve` (`[[predicted_n, predicted_ms], …]`)
are retained too.

## `llama_server_optuna_bench.py` — Optuna search over the whole parameter space

The current tuning tool. It hands the **whole joint space** to Optuna's TPE
sampler, which learns interactions between axes (e.g. cache dtype only matters
at certain `n-gpu-layers` / `split-mode` combos) and focuses its budget on the
promising regions.

* **Objective (minimize):** the mean over every prompt in the config of T\*
  (above). The raw host-side `wall_request_s` is recorded per launch as
  `wall_objective_s` so the old metric stays comparable offline. ⚠️ The
  objective is **versioned** (`objective_version`, currently `standardized-v1`);
  resuming a study whose stored version differs fails fast **before**
  production is stopped, so give a changed objective a new `study_name` /
  `storage` (the shipped configs use a `-std` suffix) or set
  `objective.allow_objective_migration: true`.
* **Search space (all from the config file):** `quants` (categorical),
  `space` (flag → list of choices, each a TPE categorical: `--flash-attn`,
  `--split-mode` (`layer,row,tensor`), `--cache-type-k`, `--cache-type-v`,
  `--batch-size`, `--ubatch-size`, `--threads`, `--threads-batch`), and a
  **flexible spec axis** — `spec.modes` is a list of modes, the sampler picks
  a mode, then that mode's sub-params (categorical/int/float). The default
  config has `off`, `mtp`, `ngram-simple`, `ngram-map-k`, `ngram-mod` and
  `ngram-map-k4v`. **Important:** ngram modes use
  `--spec-ngram-<mode>-size-m` (default 48), *not* `--spec-draft-n-max`, as
  their max draft length — sweeping `--spec-draft-n-max` on an ngram arm is a
  no-op. Draft-model modes (`draft-mtp`, dflash-style, …) do use
  `--spec-draft-n-max`.
* **`--threads` and `--threads-batch` are independent axes.** Tune them
  separately; the P100 box has 24 cores.
* **Data:** one JSON **file per launch** (trial × prompt) in
  `<output_dir>/runs/` with the full PP decay curve + params + timings +
  the per-token `gen_curve` + the standardized-objective block + the raw
  `wall_objective_s` + server-log tail — for per-run plotting; the Optuna
  study itself in a **SQLite** DB (`storage`), so a run is **resumable**
  (re-run with the same config to add trials; `n_trials` counts trials added
  THIS invocation). `summary.json` records the best trial on finish and the
  study's `objective_version`/`objective_config`. Failed trials are **pruned**
  (not fake values) with the error in the trial's `user_attrs` AND its per-run
  files. Never discard a short-output trial — it is scaled, flagged, kept.
* **Pruned before launch:** `ubatch > batch`; a quantized K/V cache with
  `--flash-attn off`; `--split-mode tensor` with `--flash-attn off` (tensor
  split requires flash attention upstream); `--split-mode tensor` at partial
  GPU offload (NCCL crash on the two-P100 box); and a spec mode whose `n_min`
  exceeds its `n_max`. After launch, a `--split-mode row` trial is pruned if
  the server log shows the split was a no-op (no GPU-offload support, no
  split-buffer support, or fewer than two CUDA devices). `row` is deliberately
  **not** pre-pruned for `--flash-attn off` — upstream only requires flash
  attention for `tensor`, not `row`.
* **Deliberately out of scope for the 27B config** (despite being in the
  upstream server README): prompt-cache flags (`--cache-reuse`, `--cache-ram`,
  `--cache-idle-slots`, `--prompt-cache`) — the three prompt files share no
  prefix and each trial is a single request per fresh server; `kv-offload` /
  `repack` / `kv-unified` / `swa-full` / context-shift — this is a hybrid
  `qwen35` model, fully GPU-offloaded, and 32k ≪ 131k context (`kv-offload`
  and `repack` ARE swept for the CPU-offloaded Flash-Next config below);
  `--fit` (on by default, and a pinned `--ctx-size` disables its context
  reduction); CPU affinity / `--prio` / `--poll` / `--numa` (GPU-bound); and
  draft sampling knobs (`--spec-draft-p-split`, `-p-min`,
  `-backend-sampling`) because the request uses `temperature: 0`.

### Config

`tests/config/llama-server-optuna-bench/Qwen3.8-27B.json` and
`Qwen3.8-Flash-Next.json`. Common knobs: `model_base` + `quants` (comma list or
JSON list of quant tags → `<model_base>-<quant>.gguf`; for sharded GGUFs use
`model_files`: a `quant -> filename` map, giving the `00001-of-NNNNN` shard),
`image`, `load_timeout`, `request_timeout`,
`production_containers` (default `["llama-cuda"]`; `[]` = leave production
running), `stop_production` (config-side switch; `--no-stop-production`
overrides to `false`), and `min_free_vram_gib` (default 24 — below this a
launch is recorded as `error` instead of spinning a container that would OOM;
lower it on smaller hosts).

`prompt_file` is a file or comma-separated list of versions; every trial runs
with **every** prompt and the T\* objective averages across them. The shipped
configs use the three ~32k-token prompts from `make_bench_prompt.py` and
`ntg: 4096`.

### Flash-Next (MoE, CPU-offloaded) config

`Qwen3.8-Flash-Next.json` uses the same three prompts and `ntg: 4096`, but the
three quants are sharded GGUFs (~71–104 GiB) that cannot fit in 2×16 GB of VRAM:

* `--cpu-moe on` is **pinned** (all 512 experts, 10 active per token, stay on
  CPU — the production arrangement); `--n-gpu-layers` sweeps 12/24/36/48/-1
  over the model's **48 layers** (48 and -1 are treated as full offload for the
  tensor-split prune via `full_offload_ngl`). `--load-mode mlock` also matches
  production, keeping the resident expert weights out of swap.
* MoE-specific axes: `--moe-cache-mib` (GPU cache of hot CPU experts),
  `--moe-cache-cpu-share` (CPU vs GPU handling of cache misses), `--repack`
  (weight repacking for quantized CPU experts) and `--kv-offload`.
* Spec arms: spec-off, the two MTP drafts (`mtp` and `mtp-shared`), and the
  four ngram families. Timeouts are raised to 1200 s load / 5400 s request for
  cold shard loading and 4096-token generation.
* Reuses the shared engine's launch driver, log parser, VRAM gate, stale
  cleanup, and production stop/restart guard (same ⚠️: on pyrite this takes the
  live models offline for the WHOLE optimization — hours at a realistic
  `n_trials`).

```bash
# validate config/search space without stopping production or launching:
.venv/bin/python tests/bench/llama_server_optuna_bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --dry-run
# run (resumable — re-run to add trials):
.venv/bin/python tests/bench/llama_server_optuna_bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --n-trials 20
```

## `llama_context_tuner.py` — find the max safe `--ctx-size`

Answers: *given every other server flag for a model (from its config), what is
the largest `--ctx-size` that still loads and serves a request?*

The config supplies the complete `server_parameters` (threads, device, GPU
layers, split mode, cache dtypes, spec decoding, …), the GGUF, and the model's
**native context limit** (`native_ctx`). Only `--ctx-size` is varied:

1. **Exponential probe** upward from `min_ctx` (doubling by default), capped at
   `native_ctx`. A launch that loads and completes the probe request is a
   success; an OOM / allocation failure is a failure. Any **other** error (bad
   flag, missing file, VRAM gate) aborts — it is not evidence about context
   size.
2. **Bisection** between the last good and first bad context down to
   `ctx_granularity` tokens. If `native_ctx` itself succeeds, the model is
   reported as reaching its native limit (no OOM found).
3. **Characterization** at the winning context: re-launch with the config's
   long `prompt_file` (truncated to fit) and record the PP rate-vs-occupancy
   decay curve from the server log.

Output: `tests/results/llama-context-tuner/<model>_<timestamp>.jsonl` (one row
per probe/bisection/confirm launch) plus a `.summary.json` carrying
`max_safe_ctx`, `first_bad_ctx`, `reached_native_max`, and `pp_curve` (the
`[[n_tokens_cum, tps_seg], …]` curve at the winning context). Failed probes
retain their error and server-log tail.

Config knobs (beyond the shared ones above): `native_ctx` (required),
`min_ctx` (default 4096), `ctx_multiplier` (default 2), `ctx_granularity`
(default 1024), `prompt_file` (the long prompt for the final curve),
`probe_prompt` (tiny prompt for the probes), `ntg`, and `chars_per_token`
(3.07 — sizing estimate for truncating the long prompt). The context tuner
sets `min_free_vram_gib: 1` in its configs: this search deliberately walks the
VRAM boundary, so the tuning-oriented VRAM floor must not refuse launches.

```bash
# validate inputs, print the probe ladder, no launches:
.venv/bin/python tests/bench/llama_context_tuner.py \
    --config tests/config/llama-context-tuner/Qwen3.8-27B.json --dry-run
# run the search:
.venv/bin/python tests/bench/llama_context_tuner.py \
    --config tests/config/llama-context-tuner/Qwen3.8-27B.json
# quick overrides: --min-ctx, --native-ctx, --no-stop-production
```

## `inference_api_bench.py` — benchmark a live inference API

Tests every model exposed by a live OpenAI-compatible endpoint, intended for
`https://promptlyapi.com/v1` (see `/home/siderealyear/promptly`). **No Docker,
no GPU, no local model files** — it discovers the catalog with
`GET /v1/models`, skips embedding models (the listing has no capability flag,
so exclusion is a configurable regex, default `(?i)(embed|bge|rerank)`), and
sends a short streaming chat completion to each remaining model. On pyrite,
`BGE-M3` is the embedding alias that gets skipped.

Timing model: each request streams with `stream_options.include_usage` and
`timings_per_token`. When the backend returns a `timings` object the **exact
server-side figures** are used (`prompt_n/prompt_ms`,
`predicted_n/predicted_ms`); otherwise it falls back to wall-clock
instrumentation of the SSE stream — time-to-first-token gives the PP rate, the
first-to-last-token span gives generation. Raw fields (usage, `finish_reason`,
final `timings`, per-token arrival curve, wall time) are always kept so any
rate can be recomputed offline. A tiny warmup request per model excludes the
cold model load and is recorded in the summary. Failed models are recorded,
never silently dropped.

Config-driven: `tests/config/inference-api-bench/promptly-prod.json` (and
`promptly-staging.json`). Knobs: `base_url`, `api_key_env` (default
`PROMPTLY_API_KEY` — the key is passed as `Authorization: Bearer …`),
`prompt_file` (file or list), `ntg` (default 512), `runs`, `temperature`,
`warmup` / `warmup_max_tokens`, `timings_per_token`, `include_models` /
`exclude_models`, `include_regex` / `exclude_regex`, and `request_timeout`.
The API key comes from `--api-key` or the env var named by `api_key_env` (set
`PROMPTLY_API_KEY` in your shell; never commit it).

Output: `tests/results/inference-api-bench/<name>_<timestamp>.jsonl` (one row
per measured request) plus a `.summary.json` with per-model means
(`mean_pp_tps`, `mean_gen_tps`) and the skip reasons.

```bash
# validate config/prompts, no network:
.venv/bin/python tests/bench/inference_api_bench.py \
    --config tests/config/inference-api-bench/promptly-prod.json --dry-run
# list the discovered (filtered) catalog without benchmarking:
PROMPTLY_API_KEY=... .venv/bin/python tests/bench/inference_api_bench.py \
    --config tests/config/inference-api-bench/promptly-prod.json --list-models
# benchmark all text models:
PROMPTLY_API_KEY=... .venv/bin/python tests/bench/inference_api_bench.py \
    --config tests/config/inference-api-bench/promptly-prod.json
# override per-run: --url, --api-key, --models M1,M2, --exclude REGEX,
#                  --runs N, --ntg N
```

## `make_bench_prompt.py` — build the test prompt

Builds a long, realistic prompt from real Claude Code transcripts. It harvests
the full realistic content of an agentic session — assistant/user **text**,
**tool-call arguments** (`tool_use.input`), and **tool-result payloads** (file
reads / command outputs, which are the bulk of the tokens) — so the token
distribution (prose + code + tool I/O) is natural rather than synthetic filler.
Only meta/sidechain lines and image blocks are skipped.

The on-disk size is a *chars-per-token estimate*; the server run reports the
true `n_tokens`, which is the ground truth for how full the context got. The
default `--ratio 3.07` is the value this Qwen model measured on the current
prompt content (471113 chars → 153533 tokens). The last chunk is trimmed so
the prompt lands on the char target instead of overshooting `--ctx-size` with
one big tool result. Point a config at any file (including a hand-written one)
to try variations.

```bash
.venv/bin/python tests/bench/make_bench_prompt.py \
    --out tests/prompts/qwen3_8_27b_v1.txt --target 32000 --offset 0
.venv/bin/python tests/bench/make_bench_prompt.py \
    --out tests/prompts/qwen3_8_27b_v2.txt --target 32000 --offset 487000
.venv/bin/python tests/bench/make_bench_prompt.py \
    --out tests/prompts/qwen3_8_27b_v3.txt --target 32000 --offset 974000
```

## Self-tests

No GPU, no real docker, no network. Run before trusting a real sweep:

```bash
.venv/bin/python tests/self_tests/test_llama_server_lib.py             # shared engine: log parser, launch, guards
.venv/bin/python tests/self_tests/test_llama_server_optuna_bench.py    # config→space, T* math, version guard, tiny study
.venv/bin/python tests/self_tests/test_llama_context_tuner.py          # probe/bisect, OOM classification, non-OOM abort
.venv/bin/python tests/self_tests/test_inference_api_bench.py          # model filtering, body, timings math, wall fallback
```

`test_llama_server_optuna_bench.py` includes a real (tiny) `study.optimize`
against a temp SQLite DB with `run_bench_launch` mocked — per-run files,
pruning, objective migration guard and resume included. All tools also support
`--dry-run` for config validation without touching GPUs or the network.
