# Benchmark tooling

Three tools measure prompt-processing (PP) performance, all driving the
`llama.cpp` docker images over `docker-py` (no local llama.cpp install needed).
`llama-server-bench.py` writes **long-format JSONL** (one row per
measurement, tagged with the condition); `llama-server-optuna-bench.py`
writes **one JSON file per launch** plus an Optuna SQLite database — see each
section.

> **The script stops and restarts the production server for you.** Each launch
> loads the full model onto the GPU(s) and processes a long prompt (~32k tokens;
> a minute or two of PP each, plus the model load). On pyrite, loading the 27B
> alongside the live `llama-cuda` (or the `Qwen3.8-Flash-Next` swap) will OOM
> the P100s — and can drop the LiteLLM router this session routes through. So
> at startup the script **stops the production container(s)**
> (`production_containers`, default `llama-cuda`) to free their VRAM, and
> **restarts them** in a `finally` block — on a clean finish **or** a crash /
> Ctrl-C. ⚠️ **This takes the node's live models (and, if you are running
> Claude Code on pyrite, that session's own model) offline for the duration of
> the sweep.** The router itself keeps running but returns 502 for the
> pyrite-local models until the restart. Pass `--no-stop-production` to opt
> out (production must then already be down, or the `min_free_vram_gib` gate
> below refuses the launches).

## `llama-server-bench.py` — PP rate vs context occupancy (decay curve)

The question this answers: *how does the PP token rate change as the number of
processed tokens grows, out to the full context size?*

A single `llama-server` run of a long prompt gives the whole curve in one run:
the server logs a progress line every ~4096 tokens with the **cumulative**
token count, seconds, and rate. The script:

1. `docker run`s `ghcr.io/ggml-org/llama.cpp:full-cuda` (model → `/models`,
   prompt → `/prompts`, a temp work dir → `/work`), all GPUs.
2. In-container, a driver starts `llama-server` (log to a file, so a long
   model-load log can't deadlock), waits for the model to appear in
   `/v1/models` (`/health` answers 200 mid-load on current builds), POSTs the
   prompt to `/v1/completions`, reads the response, shuts the server down.
3. Host side, the server log is parsed into per-occupancy rows.

Because it drives the *real* server (not `llama-bench`/`llama-batched-bench`),
it can also sweep settings those tools can't — most importantly
**speculative decoding (MTP)**, which `llama-batched-bench` in that image does
not support at all.

### Usage

```bash
# 1. Build the test prompts (~32k tokens of realistic transcript text;
#    keep them comfortably UNDER --ctx-size: prompt + ntg generation tokens
#    must fit the context or the request is rejected). Distinct --offset
#    values carve distinct versions out of the same transcript stream.
#    (They used to be ~131k — shortened to 32k to cut each run's duration.)
.venv/bin/python tests/make-bench-prompt.py \
    --out tests/prompts/qwen3_8_27b_v1.txt --target 32000 --offset 0
.venv/bin/python tests/make-bench-prompt.py \
    --out tests/prompts/qwen3_8_27b_v2.txt --target 32000 --offset 487000
.venv/bin/python tests/make-bench-prompt.py \
    --out tests/prompts/qwen3_8_27b_v3.txt --target 32000 --offset 974000

# 2. Run the four-phase sweep (see the config)
.venv/bin/python tests/llama-server-bench.py \
    --config tests/config/llama-server-bench/Qwen3.8-27B.json \
    --runs 3

# or run a single phase
.venv/bin/python tests/llama-server-bench.py --config ... --phase 1 --runs 3
.venv/bin/python tests/llama-server-bench.py --config ... --phase 2 --runs 3
#   --phase 2 requires overrides.phase1 in the config pinning the phase-1
#   winner (the winning quant)
.venv/bin/python tests/llama-server-bench.py --config ... --phase 3 --runs 3
#   --phase 3 requires overrides.phase1 AND overrides.phase2 (the COMPLETE
#   winning server cell, fixed + swept flags)
.venv/bin/python tests/llama-server-bench.py --config ... --phase 4 --runs 3
#   --phase 4 requires all three override blocks plus overrides.phase3
#   (the winning pp cell)
```

`--prompt <file>` overrides the config prompt (path or comma list) without
editing; `--limit N` caps the total number of launches (phases chain in
order) for a quick smoke test; `--runs N` repeats each phase's launches;
`--no-stop-production` skips the automatic stop/restart of the production
container(s) — use when production is already down or you don't want the
bench touching it (the VRAM gate will then refuse launches while the
production model is resident).

### Four-phase run

Like `llama-bench-tuner.py`, the run is a chain of phases, not one big cross
(each phase runs ONLY its winning condition from the earlier phases, so the
sweep stays small):

- **Phase 1** (`phase: "quant"` in the rows): the quant sweep — every entry
  in `quants` at the **fixed** `server_parameters` (ctx-size, threads,
  device, cache dtypes), with the server's **default split-mode / flash-attn
  / batch / ubatch** (none of those flags are passed), speculative decoding
  **off**. Each launch traces the decay curve. Winner = the quant with the
  highest average **`pp_auc`**.
- **Phase 2** (`phase: "server"`): the server-settings sweep —
  `server_sweep_parameters` cells (flash-attn × split-mode, …) on top of the
  fixed params, at the **phase-1 winning quant**, spec-off. Winner = the
  cell with the highest average `pp_auc`; its **complete** flag set (fixed +
  swept) is inherited by the later phases.
- **Phase 3** (`phase: "batch"`): the PP-parameters sweep — `pp_parameters`
  cells (batch/ubatch and any other prompt-preprocessing flags) at the
  **phase-1 + phase-2 winners**, still spec-off and `ntg: 1` (pure PP).
  Winner = the pp cell with the highest average `pp_auc`.
- **Phase 4** (`phase: "spec"`): the speculative-decoding scan — every
  `spec_variants` entry (first = the spec-off control) at the **phase-1 +
  phase-2 + phase-3 winners**, generating `spec_ntg` tokens. Winner = the
  variant with the highest average `pp_auc` (spec decoding shouldn't move
  prompt processing much, so this is mostly a guard — the control winning is
  the expected outcome); the per-variant average `gen_tps` and a naive
  accept-rate estimate (variant/control `gen_tps` ratio − 1, an **upper
  bound**, assuming the verify cost is free) are printed alongside for
  reference.

**Winner metric, all four phases:** `pp_auc` — the "area under the curve" of
each launch's PP decay curve, i.e. the **sum of the per-segment prompt
preprocessing rates** (the `tps_seg` values, in t/s). A condition that is fast
*everywhere* along the context scores higher than one that is fast only in
the early, cheap part of the prompt. `pp_auc` is recorded per launch on every
point row; the winner is the condition with the highest **average** `pp_auc`.

**Prompt coverage:** every condition in every phase is run with **every**
prompt version (all three), and winners are chosen on the average across
those launches — no condition wins on a single prompt.

With the default config (2 quants; 4 server cells; 4 pp cells; 5 spec
variants; 3 prompt versions): phase 1 = 6 launches/run, phase 2 = 12,
phase 3 = 12, phase 4 = 15 → 45/run, × 3 runs = 135 launches.

### Config (`tests/config/llama-server-bench/<model>.json`)

- `model_base` + `quants` — the model axis. `quants` is one or more (comma
  list or JSON list) quant tags; the model file is constructed as
  `<model_base>-<quant>.gguf` under `models/` (e.g. base `Qwen3.8-27B` +
  quant `UD-Q4_K_M` → `Qwen3.8-27B-UD-Q4_K_M.gguf`). One launch per quant,
  each row tagged `quant`. (A plain `model_file` also works: single model,
  no sweep.)
- `server_parameters` — the **fixed** server settings, used in **every**
  phase (ctx-size, threads, device, cache dtypes, …). A `llama-server` flag →
  single value (a comma list here is still crossed — but prefer to keep these
  single-valued; the swept server axes live in `server_sweep_parameters`).
  `--device` is atomic (not split on commas). Batch/ubatch are NOT here (if
  listed they are stripped) — phase 1 runs at the server's default
  batch/ubatch.
- `server_sweep_parameters` — **phase 2's** axes: the swept server settings
  (flash-attn, split-mode, …). A flag → single value or comma list; every
  list-valued flag is crossed — **one container launch per combination**
  (e.g. `--flash-attn: "on,off"` × `--split-mode: "layer,tensor"` = 4 cells).
  Each phase-2 launch runs the fixed `server_parameters` **plus** this swept
  cell, at the phase-1 winning quant.
- `pp_parameters` — **phase 3's** axes: the prompt-preprocessing flags
  (`--batch-size`, `--ubatch-size`, …), crossed the same way. Batch × ubatch
  combos with `ubatch > batch` are dropped (no-op). Each phase-3 launch runs
  the phase-1 + phase-2 winner's server cell **plus** this pp cell.
- `spec_variants` — **phase 4's** axis: a list of `{label, flags}`; the first
  is the spec-off control. Flags are appended to the server command. Phases
  1–3 always run spec-off, so these never appear in their rows.
- `prompt_file` — a file (swappable) or comma-separated list of files (each a
  "version"). **Every condition in every phase runs with every version**, and
  winners average across them. The prompts are long, realistic inputs (the
  current set is ~32k tokens each — deliberately shorter than `--ctx-size` to
  keep runs fast; the decay curve is still long enough to see the trend).
- `ntg` — completion tokens to generate after the prompt (phases 1–3). `1` =
  pure PP (the 1-token decode is negligible); raise it to also capture
  generation.
- `spec_ntg` — tokens generated in **phase 4 only** (default = `ntg`).
  Set it to a real number (e.g. 128–256) so phase 4's `gen_tps` reference is
  over actual decode tokens while phases 1–3 stay pure PP.
- `image`, `load_timeout`, `request_timeout` — container/timeout knobs.
- `min_free_vram_gib` (default 24) — each launch preflights total free VRAM
  across the GPUs; below this the row is recorded as `error` immediately
  ("only X GiB total VRAM free …") instead of spinning a container that would
  OOM at model load. Lower it if the host has less total VRAM.
- `production_containers` (default `["llama-cuda"]`) — the **exact** names of
  the production container(s) to stop at startup and restart when the run
  ends (clean finish or crash). They are matched by name, not by docker's
  substring filter. Set to `[]` (or use `--no-stop-production`) to leave
  production running — in which case the `min_free_vram_gib` gate will refuse
  launches while it is resident.
- `stop_production` (default `true`) — the config-side switch for the same
  behavior; the `--no-stop-production` flag overrides it to `false`.
- `overrides` — used only with `--phase 2` / `3` / `4`, to pin the winners
  the earlier phases would have supplied (they are not running this
  invocation). Each pinned block is copied **verbatim** from the run's
  `.summary.json` `phases.<N>` entry:
  - `--phase 2` needs `overrides.phase1` — the winning quant, e.g.
    `{"phase1": {"quant": "UD-Q4_K_M"}}` (from `phases.1`; the fixed
    `server_parameters` come straight from the config).
  - `--phase 3` needs **both** `overrides.phase1` (as above) and
    `overrides.phase2` — the **complete** winning server cell (fixed AND
    swept flags, so flash-attn / split-mode / cache / threads are inherited
    exactly), e.g. `{"phase2": {"--ctx-size": "131072",
    "--threads": "12", "--threads-batch": "12", "--device": "CUDA0,CUDA1",
    "--n-gpu-layers": "-1", "--cache-type-k": "q8_0", "--cache-type-v":
    "q8_0", "--flash-attn": "on", "--split-mode": "tensor"}}`
    (from `phases.2.cell`).
  - `--phase 4` needs all three of those plus `overrides.phase3` — the
    winning pp cell, e.g. `{"phase3": {"--batch-size": "4096",
    "--ubatch-size": "2048"}}` (from `phases.3.cell`).

Guards around a run:
- **Production stop/restart.** At startup the script stops the production
  container(s) (`production_containers`, default `llama-cuda`) so the bench
  has the full VRAM, and restarts them in a `finally` block — so they come
  back on a clean finish **and** on any crash / Ctrl-C / `parser.error`. Only
  containers that were actually running are started back up. A `--no-stop-
  production` flag (or `stop_production: false`) skips this entirely.
- At startup the script also removes any `llmsrvbench-*` containers and
  `/tmp/llmsrvbench_*` workdirs left behind by a previously killed run.
- The in-container driver waits for the **model** to finish loading (it polls
  `/v1/models` for the alias, because `/health` answers 200 mid-load on
  current builds), so a request is never sent to a server that can't answer
  it.

### Output

`tests/results/llama-server-bench/<model>_<timestamp>.jsonl`, long format.
Each successful launch writes one row per progress point (no separate summary
row):

```jsonc
{"phase":"batch","run":1,"spec":"spec-off","launch":13,"model":"Qwen3.8-27B","quant":"UD-Q4_K_M",
 "prompt":"qwen3_8_27b_v1.txt","n_tokens_prompt":32410,"pp_tps":310.22,"pp_auc":9804.1,
 "gen_tps":null,"batch_size":"4096","ubatch_size":"2048",
 "point":7,
 "n_tokens_cum":28672,"n_tokens_seg":4096,"t_cum_s":92.44,"t_seg_s":2.91,
 "tps_cum":310.17,"tps_seg":1407.56}
```

- Every row is an integer-`point` occupancy-curve row; the last point row of a
  launch is the highest-occupancy point.
- `tps_seg` = marginal PP rate over that ~2048-4096-token segment (**the decay
  curve**); `tps_cum` = running average.
- The per-launch summary fields — `pp_tps`/`gen_tps` (the server's final
  `prompt eval time` / `eval time` rates), `pp_auc` (sum of the `tps_seg`s —
  the winner metric), `n_tokens_prompt`, `wall_request_s`, the HTTP `*_http`
  token counts — are stamped on **every** point row for that launch, so the
  last point row carries the whole-prompt summary. To pivot per launch, take
  the max-`point` row per `(phase, quant, prompt, run, spec)` plus the swept
  flags of that phase (`flash_attn`/`split_mode`/… for `phase: "server"`,
  `batch_size`/`ubatch_size` for `phase: "batch"`).
- Phase 1 rows have **no** `batch_size`/`ubatch_size`/`flash_attn`/
  `split_mode` tags (the server defaults were used); phase 2/3 rows carry the
  full combined flag set (fixed + swept server +, in phase 3, the pp flags).
- Failed conditions still leave a row: `status: "error"` (with the stderr;
  driver errors also carry `server_log_tail` — the last 2000 chars of the
  server log, which is where load-timeout / early-crash causes show up) or
  `status: "no_output"`. Filter `status` when analyzing.

A `.summary.json` next to the file records the config and the launch list.

### Analysis sketch (pandas)

```python
import pandas as pd, glob
f = sorted(glob.glob("tests/results/llama-server-bench/*.jsonl"))[-1]
df = pd.read_json(f, lines=True)
ok = df[df["status"] == "ok"]                 # every ok row is an occupancy point

# PP decay curves live in phases 1-3 (the spec scan is phase 4)
pp = ok[ok["phase"].isin(["quant", "server", "batch"])]

# condition label built from the tags present on each row (phase-1 rows have
# no flash-attn/split-mode/batch/ubatch tags; phase-3 rows inherit the
# phase-2 server cell):
#   cond = ["phase", "quant", "flash_attn", "split_mode", "batch_size", "ubatch_size"]
# keep the columns that exist, drop the rest of the label to NaN-safe values

# per-launch pivot: the max-point row per condition carries the whole-prompt summary
# (see the notebook tests/llama-server-bench-results.ipynb for a working
# scatter of tps_seg vs n_tokens_cum, one series per condition)

# winner check: average pp_auc per condition (one value per prompt/run launch)
# ... groupby the condition columns above, agg pp_auc=("pp_auc", "mean")
```

## `llama-server-optuna-bench.py` — Optuna search over the whole parameter space

The phased chain above can't see *interactions* between axes (e.g. cache dtype
only matters at certain `n-gpu-layers` / `split-mode` combos), and the phase
order is a guess. This tool hands the **whole joint space** to Optuna's TPE
sampler, which learns the interactions from the trials it has already spent and
focuses its budget on the promising regions.

* **Objective (minimize):** the mean over every prompt in the config of the
  total host-side wall time of one `/v1/completions` request — the full
  (~32k) prompt's PP **plus** generation of a fixed number of output tokens
  (`ntg`, default 128). "Time to read the prompt and generate a fixed amount
  of output."
* **Search space (all from the config file):** `quants` (categorical),
  `space` (flag → list of choices, each a TPE categorical: `--n-gpu-layers`,
  `--flash-attn`, `--split-mode`, `--cache-type-k`, `--cache-type-v`,
  `--batch-size`, `--ubatch-size`), and a **flexible spec axis** — `spec.modes`
  is a list of modes, the sampler picks a mode, then that mode's sub-params
  (categorical/int/float). MTP is just one mode; a dflash-style
  `--model-draft` + `--spec-type` setup is another. `ubatch > batch` combos are
  pruned before any launch is spent.
* **Data:** one JSON **file per launch** (trial × prompt) in
  `<output_dir>/runs/` with the full PP decay curve + params + timings +
  server-log tail — for per-run plotting; the Optuna study itself in a
  **SQLite** DB (`storage`), so a run is **resumable** (re-run with the same
  config to add trials; `n_trials` counts trials added THIS invocation).
  `summary.json` records the best trial on finish. Failed trials are **pruned**
  (not fake values) with the error in the trial's `user_attrs` AND its per-run
  files.
* Reuses the sibling's launch driver, log parser, VRAM gate, stale cleanup,
  and the production stop/restart guard (same ⚠️: on pyrite this takes the live
  models offline for the WHOLE optimization — hours at a realistic `n_trials`).

```bash
.venv/bin/python tests/llama-server-optuna-bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --dry-run
.venv/bin/python tests/llama-server-optuna-bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --n-trials 20
```

## `make-bench-prompt.py` — build the test prompt

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
the prompt lands
on the char target instead of overshooting `--ctx-size` with one big tool
result. Point the config at any file (including a hand-written one) to try
variations.

## `llama-bench-tuner.py` — batched-bench three-phase tuner (legacy)

The earlier `llama-batched-bench`-based tuner (batch/ubatch scan → spec scan →
occupancy sweep). Kept for reference and for `speed_pp`/`speed_tg` point
measurements; `llama-server-bench.py` supersedes it for the decay-curve
question. See its module docstring.

## `test_llama_server_bench.py` — host-side self-tests

No GPU, no real docker: unit-tests the log parser and sweep expansion, and
drives `run_launch` with `run_bench_launch` mocked. Run before trusting a
real sweep:

```bash
.venv/bin/python tests/test_llama_server_bench.py
```

`test_llama_server_optuna_bench.py` does the same for the Optuna tool:
the config→space logic, the objective math, and a real (tiny)
`study.optimize` against a temp SQLite DB with `run_bench_launch` mocked —
per-run files, pruning, and resume included:

```bash
.venv/bin/python tests/test_llama_server_optuna_bench.py
```
