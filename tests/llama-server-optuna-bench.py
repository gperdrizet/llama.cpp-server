#!/usr/bin/env python3
'''Optuna-driven parameter search for the llama-server bench.

Older tooling swept parameters in a fixed phase order, which couldn't see
interactions between axes (e.g. cache dtype only matters at certain
n-gpu-layers / split-mode combos). This tool hands the WHOLE joint parameter
space to Optuna's TPE sampler, which learns the interactions from the trials
it has already spent and spends its budget on the promising regions.

Objective (minimize): the mean, across every prompt in the config, of the
**standardized-workload time** T* of one `/v1/completions` request — the time
to process a FIXED reference prompt length L0 (`objective.prompt_tokens`,
default 32768) and generate a FIXED reference output length N0
(`objective.output_tokens`, default `ntg`) at that context:

    T* = P(L0) + G(L0, N0)

P(L0) is interpolated off the measured PP decay curve; G(L0, N0) is the
generation time extrapolated to N0 tokens at reference context L0. The driver
streams with `timings_per_token`, so G is normally a least-squares fit of
per-step decode time vs context (mode `stream-fit`); if no per-token curve is
available it falls back to scaling the observed mean decode step by the
context-midpoint ratio (mode `linear-midpoint`). This is why a trial that emits
EOS early is NOT rewarded: its short output is scaled UP to N0, not measured as
a short (fast) request. Extrapolations that are too short to trust are flagged
(`objective_low_confidence` + reasons) and kept in the study, never pruned.

The raw host-side `wall_request_s`, the HTTP `finish_reason` /
`completion_tokens`, and the full per-token generation curve are all retained in
the per-run files so any objective can be recomputed offline.

⚠️ The objective is versioned (`OBJECTIVE_VERSION`). A study created with an
older objective cannot be mixed with the new values, so resuming a study whose
`objective_version` differs fails fast; give it a new `study_name`/`storage`
(the configs use a `-std` suffix) or set `objective.allow_objective_migration`.

The launch machinery is NOT duplicated: this script imports
`llama_server_lib.py` (same directory) and reuses its container driver, log
parser, VRAM gate, stale-container cleanup, standardized objective, and the
production stop/restart guard.

Data retention
--------------
* One JSON file per LAUNCH (trial x prompt) under `<output_dir>/runs/`,
  holding the trial number, every parameter, the full PP decay curve
  (parsed from the server log), the request timings, and the server-log
  tail. These are what you plot per-run later.
* The Optuna study itself in a SQLite database (`storage`), so a run is
  RESUMABLE: re-run with the same config and the sampler continues from
  where it left off (completed trials are not re-run).
* `n_trials` is the number of trials THIS invocation runs on top of what
  is already in the database.
* On finish, `<output_dir>/summary.json` records the best trial so far.

Search space (all from the config file)
---------------------------------------
* `quants`          — the model quant axis (`<model_base>-<quant>.gguf`)
* `space`           — a dict of server flag -> list of choices; EVERY key is
                      a TPE categorical. Keys used by the default config:
                      --flash-attn, --split-mode, --cache-type-k,
                      --cache-type-v, --batch-size, --ubatch-size,
                      --threads, --threads-batch.
* `fixed`           — server flags that are NEVER swept (ctx-size, device,
                      n-gpu-layers, main-gpu, ...) — always in the command.
* `spec.modes`      — the speculative-decoding axis as a list of MODES.
                      The sampler picks a mode, then that mode's optional
                      sub-parameters (categorical / int / float). Any mode
                      with any flags is valid, so e.g. a dflash-style
                      `--model-draft` + `--spec-type` setup is just another
                      entry alongside the MTP one:

                      { "name": "off" },
                      { "name": "mtp",
                        "base_flags": { "--model-draft": "...", "--spec-type": "draft-mtp" },
                        "params": { "--spec-draft-n-max": { "type": "int", "low": 1, "high": 5 } } },
                      { "name": "dflash",
                        "base_flags": { "--model-draft": "/models/xxx.gguf", "--spec-type": "draft-eagle" },
                        "params": { ... } }

Combos pruned BEFORE any launch is spent (the server would reject or crash
them):
  * `ubatch > batch` (meaningless);
  * a quantized (non-f16/bf16) `--cache-type-k`/`--cache-type-v` together
    with `--flash-attn off` — llama.cpp refuses to create that context
    ("quantized V cache requires flash_attn to be enabled");
  * `--split-mode tensor` together with `--flash-attn off` — tensor split
    mode itself requires flash attention ("SPLIT_MODE_TENSOR requires
    flash_attn to be enabled");
  (the last two apply only when `--flash-attn` is explicitly swept to off;
  if the flag is not in the space the server's default is in force and
  neither is pruned); and
  * `--split-mode tensor` at PARTIAL `--n-gpu-layers` offload — a RUNTIME
    (not startup) constraint: the tensor-split allreduce has repeatedly
    crashed the server in NCCL on this two-P100 box at partial offload.
    Full offload (the `-1` sentinel, or any value in the config's
    `full_offload_ngl`, e.g. `["999"]`) is NOT pruned — that is production's
    working tensor regime and stays testable; and
  * a spec mode's `n_min` > `n_max` (draft token counts, or ngram-mod's
    min/max) — rejected pre-launch.

AFTER a launch, `--split-mode row` trials are checked for a runtime no-op
(split mode ignored, no split-buffer support, or fewer than two CUDA devices
in the log) and pruned — unlike `tensor`, `row` has no flash-attn constraint
upstream, so it must NOT be pre-pruned on `--flash-attn off`.

Trial failure policy: if EVERY prompt of a trial fails (bad args, OOM,
load timeout, a runtime CUDA/NCCL crash, ...) the trial is PRUNED (not a
fake high value) and the error is kept in the trial's user_attrs AND in
that trial's per-launch files (which also carry the server log's ` E `
error lines — the real reason, since the driver's own message is a generic
"load timeout / exited early"). A trial where some prompts succeeded
reports the mean over the succeeding ones.

NOTE on resources
-----------------
As with the sibling tool, this STOPS the production container(s)
(`production_containers`, default `["llama-cuda"]`) for the duration of
the optimization and RESTARTS them in a `finally` block — on a clean
finish OR any crash / Ctrl-C. ⚠️ On pyrite that takes the node's live
models offline for the WHOLE optimization (hours at a realistic
n_trials), including a pyrite-hosted Claude Code session's own model: the
LiteLLM router stays up but 502s the pyrite-local models until the
restart. `--no-stop-production` opts out (the VRAM gate then refuses
launches while the production model is resident).

Usage:
    .venv/bin/python tests/llama-server-optuna-bench.py \\
        --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json
    .venv/bin/python tests/llama-server-optuna-bench.py --config ... --dry-run
    .venv/bin/python tests/llama-server-optuna-bench.py --config ... --n-trials 20
'''

import argparse
import importlib.util
import itertools
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import optuna
import optuna.storages

# Raise-for-pruning class. `optuna.TrialPruned` is the top-level alias in
# Optuna 5.x (== optuna.exceptions.TrialPruned); older versions expose
# `optuna.TrialPrunedError`. Use whichever exists.
if hasattr(optuna, "TrialPruned"):
    TrialPrunedError = optuna.TrialPruned
else:  # pragma: no cover - older Optuna
    TrialPrunedError = optuna.TrialPrunedError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Bumped whenever the objective's definition changes, so a study built with an
# older objective is never silently mixed with new trial values (the raw
# `wall_request_s` objective and the standardized T* are NOT comparable).
OBJECTIVE_VERSION = "standardized-v1"

# Import the shared engine (importlib so it works regardless of the caller's
# sys.path). Everything heavy (container driver, log parser, VRAM gate,
# objective, production stop/restore) is reused from there — keep in lockstep.
_spec = importlib.util.spec_from_file_location("llama_server_lib",
                                               HERE / "llama_server_lib.py")
LLM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LLM)


# --------------------------------------------------------------------------
# config → search space
# --------------------------------------------------------------------------

def prompt_list(config: dict) -> list:
    '''Config `prompt_file`: a single path or a comma-separated list.'''
    spec = config.get("prompt_file", "prompt.txt")
    if isinstance(spec, list):
        return [str(p) for p in spec]
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def quant_map(config: dict) -> dict:
    '''quant tag -> model_file, from the shared model-spec logic.'''
    return {s["quant"]: s["model_file"] for s in LLM.model_specs_from_config(config)}


def space_choices(config: dict) -> dict:
    '''The swept flags: {flag: [choice, ...]} — each key a TPE categorical.'''
    space = config.get("space", {})
    out = {}
    for flag, choices in space.items():
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.split(",") if c.strip()]
        choices = list(choices)
        if not choices:
            raise ValueError(f"space[{flag!r}] has no choices")
        out[flag] = choices
    return out


def spec_modes(config: dict) -> list:
    '''The spec axis: a list of modes, each {name, base_flags?, params?}.

    A missing/empty `spec` block means "always off" (a single mode).
    '''
    block = config.get("spec") or {}
    modes = block.get("modes")
    if not modes:
        return [{"name": "off"}]
    for m in modes:
        if not m.get("name"):
            raise ValueError(f"spec mode without a name: {m!r}")
    return modes


def is_full_offload(value, config: dict) -> bool:
    '''True if a sampled `--n-gpu-layers` value offloads the WHOLE model.

    `-1` is llama.cpp's "offload everything" sentinel, always full. Any other
    value counts as full only if it is listed in the config's
    `full_offload_ngl` (e.g. `["999"]` — a number above the layer count).
    This is what separates the working full-offload regime (production's
    tensor+MTP) from the partial-offload regime that crashes in NCCL.
    '''
    v = str(value).strip()
    if v == "-1":
        return True
    full = config.get("full_offload_ngl") or []
    if isinstance(full, str):
        full = [c.strip() for c in full.split(",") if c.strip()]
    return v in {str(x).strip() for x in full}


def sample_params(trial, config: dict) -> tuple:
    '''Sample one point of the search space. Returns (params, spec_flags).

    `params` = {flag: value} for every swept flag (ints/strings as-is) plus
    "quant". `spec_flags` = the full spec flag dict for the chosen mode
    (empty for "off"). Raises TrialPrunedError, before any launch is spent,
    on the known-bad combos: ubatch>batch, a quantized cache with flash-attn
    off, tensor split-mode with flash-attn off (both server-rejected at
    startup), and tensor split-mode at partial GPU offload (a known NCCL
    runtime crash — full offload stays testable).
    '''
    params = {"quant": trial.suggest_categorical("quant", list(quant_map(config)))}

    for flag, choices in space_choices(config).items():
        params[flag] = trial.suggest_categorical(flag, choices)

    batch = params.get("--batch-size")
    ubatch = params.get("--ubatch-size")
    if batch is not None and ubatch is not None and int(ubatch) > int(batch):
        trial.set_user_attr("prune_reason", f"ubatch {ubatch} > batch {batch}")
        raise TrialPrunedError()

    # llama.cpp hard constraints, pruned pre-launch so we never spend a full
    # load on a combo the server rejects deterministically at context
    # creation. Both only apply when `--flash-attn` was EXPLICITLY sampled to
    # off; if the flag isn't in the space the server uses its default
    # (auto → on when supported), so neither constraint is in force and we
    # must not prune.
    fa = params.get("--flash-attn")
    fa_off = fa is not None and str(fa).strip().lower() != "on"
    if fa_off:
        # (1) A quantized (non-f16/bf16) K or V cache can only be created with
        #     flash attention enabled ("quantized V cache requires flash_attn
        #     to be enabled").
        for cache_flag in ("--cache-type-k", "--cache-type-v"):
            ct = params.get(cache_flag)
            if ct is not None and str(ct) not in ("", "f16", "bf16", "auto"):
                trial.set_user_attr(
                    "prune_reason",
                    f"quantized {cache_flag} ({ct}) requires --flash-attn on "
                    f"(got {fa})")
                raise TrialPrunedError()
        # (2) SPLIT_MODE_TENSOR itself requires flash attention, regardless of
        #     cache dtype ("SPLIT_MODE_TENSOR requires flash_attn to be
        #     enabled").
        sm = params.get("--split-mode")
        if sm is not None and str(sm).strip().lower() == "tensor":
            trial.set_user_attr(
                "prune_reason",
                f"--split-mode tensor requires --flash-attn on (got {fa})")
            raise TrialPrunedError()

    # Runtime (not startup) constraint: `--split-mode tensor` at PARTIAL GPU
    # offload has repeatedly crashed the server in NCCL
    # (ggml_backend_cuda_comm_allreduce_nccl) on this two-P100 box — the
    # tensor-split allreduce dies mid-load. Full offload is NOT pruned: that
    # is production's working tensor regime (ngl -1 / 999) and stays
    # testable. The full-offload values are config-driven (`full_offload_ngl`,
    # plus the built-in `-1` sentinel); everything else at partial offload is
    # pruned pre-launch rather than spending a ~1-minute crashed launch.
    sm = params.get("--split-mode")
    ngl = params.get("--n-gpu-layers")
    if (sm is not None and str(sm).strip().lower() == "tensor"
            and ngl is not None
            and not is_full_offload(ngl, config)):
        trial.set_user_attr(
            "prune_reason",
            f"--split-mode tensor at partial offload (n-gpu-layers {ngl}) "
            f"crashes in NCCL on this box; full offload is testable")
        raise TrialPrunedError()

    modes = spec_modes(config)
    mode_name = trial.suggest_categorical("spec.mode", [m["name"] for m in modes])
    mode = next(m for m in modes if m["name"] == mode_name)

    spec_flags = dict(mode.get("base_flags", {}))
    for pname, pdef in (mode.get("params") or {}).items():
        # a bare list is shorthand for {"type": "choice", "choices": [...]}
        if isinstance(pdef, list):
            pdef = {"type": "choice", "choices": pdef}
        ptype = (pdef or {}).get("type", "choice")
        pname_key = f"spec:{pname}"
        if ptype == "int":
            spec_flags[pname] = trial.suggest_int(
                pname_key, int(pdef["low"]), int(pdef["high"]),
                step=int(pdef.get("step", 1)))
        elif ptype == "float":
            spec_flags[pname] = trial.suggest_float(
                pname_key, float(pdef["low"]), float(pdef["high"]),
                step=pdef.get("step"))
        else:
            spec_flags[pname] = trial.suggest_categorical(pname_key, list(pdef["choices"]))

    # A spec mode's minimum draft length can never exceed its maximum.
    for min_flag, max_flag in (("--spec-draft-n-min", "--spec-draft-n-max"),
                               ("--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max")):
        lo, hi = spec_flags.get(min_flag), spec_flags.get(max_flag)
        if lo is not None and hi is not None and int(lo) > int(hi):
            trial.set_user_attr(
                "prune_reason", f"{min_flag} {lo} > {max_flag} {hi}")
            raise TrialPrunedError()

    return params, spec_flags


def build_cell(params: dict, config: dict) -> dict:
    '''fixed + sampled flags -> the server flag dict for one launch.'''
    cell = {k: str(v) for k, v in (config.get("fixed") or {}).items()}
    for flag, value in params.items():
        if flag == "quant":
            continue
        cell[flag] = str(value)
    return cell


def first_error_lines(text: str, limit: int = 3):
    '''The server log's ` E ` lines — the real reason a launch failed,
    which the generic driver error ("load timeout / exited early") hides.

    llama.cpp log lines are `<clock> <sev> <tag> message`, e.g.
    `0.04.290.140 E llama_init_from_model: ...`; the clock never contains
    " E ", so the first occurrence is the severity marker.
    '''
    out = []
    for ln in (text or "").splitlines():
        i = ln.find(" E ")
        if i == -1:
            continue
        msg = ln[i + 3:].strip()
        if msg:
            out.append(msg)
    return out[:limit]


def row_is_noop(cell: dict, log: str) -> bool:
    '''True if a `--split-mode row` launch did not actually split across GPUs.

    Row split is only a no-op at RUNTIME (unlike tensor + flash-attn off,
    which is rejected deterministically at startup): upstream warns when the
    binary lacks GPU-offload support, throws when the device has no split
    buffer type, and needs >= 2 devices to split. Check the captured server
    log rather than pre-pruning a valid config combination.
    '''
    if str(cell.get("--split-mode", "")).strip().lower() != "row":
        return False
    log = log or ""
    if ("Setting the split mode has no effect" in log
            or "does not support split buffers" in log):
        return True
    return len(set(re.findall(r"using device (CUDA\d+)", log))) < 2


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------

def launch_objective(result: dict, parsed: dict):
    '''RAW host-side wall time for one launch, or None if unmeasurable.

    This is NOT the study objective any more (that is
    `LLM.standardized_objective`): it is recorded per launch as
    `wall_objective_s` so the old metric stays comparable offline. Prefers the
    host-side wall clock around the HTTP request; falls back to the server's own
    summary lines.
    '''
    resp = result.get("response") or {}
    wall = resp.get("wall_request_s")
    if wall is not None:
        return float(wall)
    pe = parsed.get("prompt_eval") or {}
    ge = parsed.get("gen_eval") or {}
    if pe.get("t_ms") is not None:
        return pe["t_ms"] / 1000.0 + (ge.get("t_ms") or 0.0) / 1000.0
    return None


def write_run_file(output_dir: Path, record: dict) -> Path:
    '''One JSON file per launch, written atomically (tmp + rename).'''
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record["prompt"]).stem
    path = runs_dir / f"trial-{record['trial']:04d}_r{record['run']:02d}_{stem}.json"
    fd, tmp = tempfile.mkstemp(dir=runs_dir, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(record, f, indent=1)
        Path(tmp).rename(path)
    except BaseException:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise
    return path


def check_study_objective(config: dict, db_path: Path, study_name: str) -> None:
    '''Fail fast if an existing study was built with a different objective.

    The raw wall-time objective and the standardized T* are not comparable, so
    resuming a study across the change would corrupt it. Called BEFORE the
    production containers are stopped, so a mismatch costs nothing. A config
    can opt out explicitly with `objective.allow_objective_migration: true`.
    '''

    if (config.get("objective") or {}).get("allow_objective_migration"):
        return
    if not Path(db_path).is_file():
        return
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    if study_name not in optuna.study.get_all_study_names(storage):
        return
    study = optuna.load_study(study_name=study_name, storage=storage)
    found = study.user_attrs.get("objective_version")
    if found == OBJECTIVE_VERSION:
        return
    raise RuntimeError(
        f"study {study_name!r} in {db_path} has objective_version={found!r}, "
        f"but this tool writes {OBJECTIVE_VERSION!r}. Trial values are not "
        f"comparable across objectives. Point the config at a new "
        f"study_name/storage (e.g. '{study_name}-std'), or set "
        f"objective.allow_objective_migration=true to accept the mismatch "
        f"explicitly.")


def run_study(config: dict, n_trials: int, seed=None) -> optuna.study.Study:
    '''Create/resume the study in the SQLite DB and run n_trials more trials.'''

    study_name = config.get("study_name") or \
        f"{config['model_name']}-optuna"
    db_path = Path(config.get("storage",
                              f"{ROOT / 'tests/results/llama-server-optuna-bench' / study_name}.db")
                   ).expanduser().resolve()
    output_dir = Path(config.get("output_dir",
                                 str(db_path.parent / study_name))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    # Optuna 5.x load_study does NOT verify the study exists — it returns a
    # shell over the (possibly empty) DB. Check existence explicitly so a
    # fresh run really creates the study with the seeded sampler.
    if study_name in optuna.study.get_all_study_names(storage):
        study = optuna.load_study(study_name=study_name, storage=storage,
                                  sampler=optuna.samplers.TPESampler(seed=seed))
        print(f"resumed existing study {study_name!r}: "
              f"{len(study.trials)} trial(s) in {db_path}")
    else:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        print(f"created new study {study_name!r} -> {db_path}")

    # stamp (or refresh, when a migration was explicitly allowed) the objective
    # identity so a later run can tell whether its values are comparable
    if study.user_attrs.get("objective_version") != OBJECTIVE_VERSION:
        study.set_user_attr("objective_version", OBJECTIVE_VERSION)
        study.set_user_attr(
            "objective_config",
            LLM.objective_config(config, int(config.get("ntg", 128))))

    prompts = prompt_list(config)
    qmap = quant_map(config)
    ntg = int(config.get("ntg", 128))
    launch_counter = itertools.count()

    def objective(trial: optuna.trial.Trial) -> float:
        params, spec_flags = sample_params(trial, config)
        model_file = qmap[params["quant"]]
        cell = build_cell(params, config)
        spec_mode = trial.params.get("spec.mode", "off")

        per_prompt = []
        values = []
        low_conf_prompts = []
        for i, pfile in enumerate(prompts, 1):
            launch_id = next(launch_counter)
            print(f"  [trial {trial.number}] launch {launch_id}: "
                  f"prompt={Path(pfile).name} quant={params['quant']} "
                  f"spec={spec_mode}", flush=True)
            result = LLM.run_bench_launch(config, cell, spec_flags, pfile,
                                          launch_id, model_file, ntg=ntg)
            status = result.get("status")

            record = {
                "tool": "llama-server-optuna-bench",
                "study": study_name,
                "trial": trial.number,
                "run": i,
                "prompt": Path(pfile).name,
                "params": params,
                "spec_mode": spec_mode,
                "spec_flags": spec_flags,
                "model_file": model_file,
                "cell": cell,
                "ntg": ntg,
                "status": "ok" if status == "ok" else status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if status == "ok":
                log = result.get("server_log") or ""
                if row_is_noop(cell, log):
                    record["status"] = "pruned-noop"
                    record["objective_s"] = None
                    record["server_log_tail"] = log[-8000:]
                    print(f"    !! --split-mode row was a no-op; pruning trial",
                          flush=True)
                else:
                    parsed = LLM.parse_server_log(log)
                    resp = result.get("response") or {}
                    record["points"] = parsed["points"]
                    record["prompt_eval"] = parsed["prompt_eval"]
                    record["gen_eval"] = parsed["gen_eval"]
                    record["response"] = resp
                    record["gen_curve"] = resp.get("gen_curve") or []
                    # raw host-side wall clock, kept for comparison/diagnostics
                    record["wall_objective_s"] = launch_objective(result, parsed)
                    std = LLM.standardized_objective(parsed, resp, config, ntg)
                    record["objective"] = std
                    value = std["objective_s"]
                    record["objective_s"] = value
                    record["objective_low_confidence"] = std["objective_low_confidence"]
                    if value is not None:
                        values.append(value)
                        if std["objective_low_confidence"]:
                            low_conf_prompts.append(record["prompt"])
                            print(f"    ~ low-confidence extrapolation "
                                  f"({'; '.join(std['objective_low_confidence_reasons'])})",
                                  flush=True)
                    if not parsed["points"]:
                        print(f"    !! no PP progress lines parsed "
                              f"(objective may be incomplete)", flush=True)
            else:
                log = result.get("server_log") or ""
                record["error"] = result.get("error")
                record["objective_s"] = None
                record["server_log_tail"] = log[-8000:]
                err_lines = first_error_lines(log)
                record["error_lines"] = err_lines
                shown = "; ".join(err_lines) or \
                    (str(result.get("error"))[:200].splitlines()[0]
                     if result.get("error") else "unknown")
                print(f"    !! launch failed: {str(result.get('error'))[:120]}\n"
                      f"       server said: {shown}", flush=True)

            path = write_run_file(output_dir, record)
            per_prompt.append({"prompt": record["prompt"],
                               "status": record["status"],
                               "objective_s": record.get("objective_s"),
                               "objective_mode": (record.get("objective") or {}).get("objective_mode"),
                               "objective_low_confidence": record.get("objective_low_confidence"),
                               "wall_objective_s": record.get("wall_objective_s"),
                               "error": record.get("error"),
                               "file": str(path)})

        trial.set_user_attr("params", params)
        trial.set_user_attr("spec_mode", spec_mode)
        trial.set_user_attr("per_prompt", per_prompt)
        trial.set_user_attr("low_confidence_prompts", low_conf_prompts)
        trial.set_user_attr("n_low_confidence", len(low_conf_prompts))

        if any(r["status"] == "pruned-noop" for r in per_prompt):
            trial.set_user_attr(
                "prune_reason",
                "--split-mode row was a no-op (single device / split buffers "
                "unsupported)")
            raise TrialPrunedError()

        if not values:
            trial.set_user_attr("prune_reason", "all prompts failed")
            raise TrialPrunedError()

        return sum(values) / len(values)

    study.optimize(objective, n_trials=n_trials)
    print(f"study {study_name!r}: {len(study.trials)} trial(s) total after "
          f"this invocation", flush=True)
    return study


def write_summary(study: optuna.study.Study, db_path: Path,
                  output_dir: Path) -> dict:
    summary = {
        "tool": "llama-server-optuna-bench",
        "study": study.study_name,
        "db": str(db_path),
        "output_dir": str(output_dir),
        "n_trials": len(study.trials),
        "n_complete": sum(1 for t in study.trials
                          if t.state.name == "COMPLETE"),
        "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
        "n_failed": sum(1 for t in study.trials if t.state.name == "FAIL"),
        "n_low_confidence": sum(1 for t in study.trials
                                if t.user_attrs.get("n_low_confidence")),
        "objective_version": study.user_attrs.get("objective_version"),
        "objective_config": study.user_attrs.get("objective_config"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        best = study.best_trial
        summary["best"] = {
            "trial": best.number,
            "value_s": best.value,
            "params": best.params,
            "spec_mode": best.user_attrs.get("spec_mode"),
            "per_prompt": best.user_attrs.get("per_prompt"),
            "low_confidence_prompts": best.user_attrs.get("low_confidence_prompts"),
        }
    except ValueError:
        summary["best"] = None
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=None,
                        help="number of trials THIS invocation runs (default: "
                             "config n_trials); the study is resumed from the DB")
    parser.add_argument("--seed", type=int, default=None,
                        help="TPE sampler seed (default: config seed; only used "
                             "when the study is newly created)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="override config study_name")
    parser.add_argument("--db", type=str, default=None,
                        help="override config storage (SQLite path)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="override config output_dir (per-run files + summary)")
    parser.add_argument("--no-stop-production", action="store_true",
                        help="do NOT stop/restart the production container(s) "
                             "(llama-cuda); the VRAM gate will refuse launches "
                             "while the production model is resident")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and print the search space; "
                             "no containers, no production stop")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("models_path", str(ROOT / "models"))
    config["stop_production"] = not args.no_stop_production
    if args.study_name:
        config["study_name"] = args.study_name
    if args.db:
        config["storage"] = args.db
    if args.output_dir:
        config["output_dir"] = args.output_dir
    n_trials = args.n_trials if args.n_trials is not None \
        else int(config.get("n_trials", 20))
    seed = args.seed if args.seed is not None else config.get("seed")

    study_name = config.get("study_name") or f"{config['model_name']}-optuna"
    db_path = Path(config.get("storage",
                              str(ROOT / "tests/results/llama-server-optuna-bench"
                                  / f"{study_name}.db"))).expanduser().resolve()
    output_dir = Path(config.get("output_dir", str(db_path.parent / study_name))
                      ).expanduser().resolve()

    # ---- validate everything that a failed trial would discover the hard
    # way (multi-minute loads), BEFORE touching production ----
    prompts = [Path(p).expanduser().resolve() for p in prompt_list(config)]
    missing = [str(p) for p in prompts if not p.is_file()]
    if missing:
        parser.error(f"prompt file(s) not found: {', '.join(missing)}")
    qmap = quant_map(config)
    models_path = Path(config["models_path"])
    missing = [str(models_path / mf) for mf in qmap.values()
               if not (models_path / mf).is_file()]
    if missing:
        parser.error(f"model file(s) not found: {', '.join(missing)}")
    for m in spec_modes(config):
        for v in (m.get("base_flags") or {}).values():
            if str(v).startswith("/models/"):
                p = models_path / str(v)[len("/models/"):]
                if not p.is_file():
                    parser.error(f"spec draft model not found: {p}")
    space = space_choices(config)
    obj_cfg = LLM.objective_config(config, int(config.get("ntg", 128)))

    if args.dry_run:
        print(f"study        : {study_name}")
        print(f"db           : {db_path}")
        print(f"output dir   : {output_dir}")
        print(f"prompts      : {[p.name for p in prompts]}")
        print(f"quants       : {list(qmap)}")
        print(f"ntg          : {config.get('ntg', 128)}")
        print(f"objective    : {OBJECTIVE_VERSION}  T* = P({obj_cfg['prompt_tokens']}) "
              f"+ G(..., {obj_cfg['output_tokens']})  "
              f"(stream-fit tail={obj_cfg['gen_fit_tail_steps']}, "
              f"low-confidence below {obj_cfg['min_gen_tokens']} gen tokens)")
        print(f"n_trials     : {n_trials} (this invocation)")
        total = 1
        for flag, choices in space.items():
            print(f"  space {flag:18s} {choices}")
            total *= len(choices)
        print(f"spec modes   : {[m['name'] for m in spec_modes(config)]}")
        print(f"space size   : {total} (before spec sub-params; pruned at sample "
              f"time: ubatch>batch, quantized-cache+flash-attn-off, "
              f"tensor-split+flash-attn-off, "
              f"tensor-split+partial-offload[full_offload_ngl="
              f"{config.get('full_offload_ngl', [])}], spec-n-min>n-max; "
              f"row no-ops pruned post-launch)")
        print(f"production   : stop/restart {config.get('production_containers', ['llama-cuda'])}"
              + ("" if config.get("stop_production", True) else " (DISABLED via --no-stop-production)"))
        return 0

    print(f"llama-server-optuna-bench: {config['model_name']} "
          f"({config['models_path']})")
    print(f"  db={db_path}  runs={output_dir / 'runs'}  n_trials={n_trials}")
    print(f"  objective = mean over {[p.name for p in prompts]} of T* "
          f"({OBJECTIVE_VERSION}): {obj_cfg['prompt_tokens']}-token PP + "
          f"{obj_cfg['output_tokens']}-token generation at that context; "
          f"raw wall_request_s + finish_reason + per-token gen curve retained "
          f"in the run files")
    if config.get("stop_production", True):
        names = LLM.production_container_names(config)
        print(f"  ⚠️ stopping production container(s) {names} for the WHOLE "
              f"optimization — live pyrite-local models (and any "
              f"pyrite-hosted Claude Code session's model) will 502 through "
              f"the router until the restart")

    # refuse to mix objectives in one study — BEFORE production is stopped
    check_study_objective(config, db_path, study_name)

    LLM.cleanup_stale_bench()

    stopped_production = []
    try:
        stopped_production = LLM.stop_production_containers(config)
        study = run_study(config, n_trials, seed=seed)
        summary = write_summary(study, db_path, output_dir)
        print(f"  summary -> {output_dir / 'summary.json'}")
        if summary.get("best"):
            b = summary["best"]
            print(f"  best: trial {b['trial']}  {b['value_s']:.1f} s  "
                  f"params={ {k: v for k, v in b['params'].items()} } "
                  f"spec={b['spec_mode']}")
        else:
            print("  (no COMPLETE trials yet — every trial so far pruned/failed)")
    finally:
        # Runs on clean exit AND on any crash (exceptions, Ctrl-C), so the
        # production container is always brought back however the run ends.
        LLM.restore_production_containers(stopped_production)

    return 0


if __name__ == "__main__":
    sys.exit(main())
