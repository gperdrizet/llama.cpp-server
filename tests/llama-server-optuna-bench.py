#!/usr/bin/env python3
'''Optuna-driven parameter search for the llama-server bench.

The sibling `llama-server-bench.py` sweeps parameters in a fixed 4-phase
order, which can't see interactions between axes (e.g. cache dtype only
matters at certain n-gpu-layers / split-mode combos). This tool hands the
WHOLE joint parameter space to Optuna's TPE sampler, which learns the
interactions from the trials it has already spent and spends its budget on
the promising regions.

Objective (minimize): the mean, across every prompt in the config, of the
total host-side wall time of one `/v1/completions` request — prompt
preprocessing of the full (~32k) prompt PLUS generation of a fixed number
of output tokens (`ntg`, default 128). That is "time to read the prompt and
then generate a fixed, reasonable amount of output" for a single request.

The launch machinery is NOT duplicated: this script imports
`llama-server-bench.py` (same directory, hyphenated filename → importlib)
and reuses its container driver, log parser, VRAM gate, stale-container
cleanup, and the production stop/restart guard.

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
                      --n-gpu-layers, --flash-attn, --split-mode,
                      --cache-type-k, --cache-type-v, --batch-size,
                      --ubatch-size.
* `fixed`           — server flags that are NEVER swept (ctx-size, threads,
                      device, ...) — always in the command.
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

Trial failure policy: if EVERY prompt of a trial fails (bad args, OOM,
load timeout, ...) the trial is PRUNED (not a fake high value) and the
error is kept in the trial's user_attrs AND in that trial's per-launch
files. A trial where some prompts succeeded reports the mean over the
succeeding ones. The `ubatch > batch` combo is pruned before any launch
(the server would reject it / it is meaningless).

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

# Import the sibling bench (hyphenated filename → not importable by name).
# Everything heavy (container driver, log parser, VRAM gate, production
# stop/restore) is reused from there — keep the two in lockstep.
_spec = importlib.util.spec_from_file_location("llm_srv_bench",
                                               HERE / "llama-server-bench.py")
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


def sample_params(trial, config: dict) -> tuple:
    '''Sample one point of the search space. Returns (params, spec_flags).

    `params` = {flag: value} for every swept flag (ints/strings as-is) plus
    "quant". `spec_flags` = the full spec flag dict for the chosen mode
    (empty for "off"). Raises TrialPrunedError on the ubatch>batch combo —
    before any launch is spent.
    '''
    params = {"quant": trial.suggest_categorical("quant", list(quant_map(config)))}

    for flag, choices in space_choices(config).items():
        params[flag] = trial.suggest_categorical(flag, choices)

    batch = params.get("--batch-size")
    ubatch = params.get("--ubatch-size")
    if batch is not None and ubatch is not None and int(ubatch) > int(batch):
        trial.set_user_attr("prune_reason", f"ubatch {ubatch} > batch {batch}")
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

    return params, spec_flags


def build_cell(params: dict, config: dict) -> dict:
    '''fixed + sampled flags -> the server flag dict for one launch.'''
    cell = {k: str(v) for k, v in (config.get("fixed") or {}).items()}
    for flag, value in params.items():
        if flag == "quant":
            continue
        cell[flag] = str(value)
    return cell


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------

def launch_objective(result: dict, parsed: dict):
    '''Total request time for one launch, or None if it can't be measured.

    Prefers the host-side wall clock around the HTTP request (prompt PP +
    generation, exactly "time to read the prompt and generate the output");
    falls back to the server's own summary lines.
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
                parsed = LLM.parse_server_log(result.get("server_log") or "")
                record["points"] = parsed["points"]
                record["prompt_eval"] = parsed["prompt_eval"]
                record["gen_eval"] = parsed["gen_eval"]
                record["response"] = result.get("response")
                value = launch_objective(result, parsed)
                record["objective_s"] = value
                if value is not None:
                    values.append(value)
                if not parsed["points"]:
                    print(f"    !! no PP progress lines parsed (wall clock only)",
                          flush=True)
            else:
                record["error"] = result.get("error")
                record["objective_s"] = None
                record["server_log_tail"] = (result.get("server_log") or "")[-8000:]
                print(f"    !! launch failed: "
                      f"{str(result.get('error'))[:200].splitlines()[0] if result.get('error') else 'unknown'}",
                      flush=True)

            path = write_run_file(output_dir, record)
            per_prompt.append({"prompt": record["prompt"],
                               "status": record["status"],
                               "objective_s": record.get("objective_s"),
                               "error": record.get("error"),
                               "file": str(path)})

        trial.set_user_attr("params", params)
        trial.set_user_attr("spec_mode", spec_mode)
        trial.set_user_attr("per_prompt", per_prompt)

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

    if args.dry_run:
        print(f"study        : {study_name}")
        print(f"db           : {db_path}")
        print(f"output dir   : {output_dir}")
        print(f"prompts      : {[p.name for p in prompts]}")
        print(f"quants       : {list(qmap)}")
        print(f"ntg          : {config.get('ntg', 128)}")
        print(f"n_trials     : {n_trials} (this invocation)")
        total = 1
        for flag, choices in space.items():
            print(f"  space {flag:18s} {choices}")
            total *= len(choices)
        print(f"spec modes   : {[m['name'] for m in spec_modes(config)]}")
        print(f"space size   : {total} (before spec sub-params; ubatch>batch "
              f"pruned)")
        print(f"production   : stop/restart {config.get('production_containers', ['llama-cuda'])}"
              + ("" if config.get("stop_production", True) else " (DISABLED via --no-stop-production)"))
        return 0

    print(f"llama-server-optuna-bench: {config['model_name']} "
          f"({config['models_path']})")
    print(f"  db={db_path}  runs={output_dir / 'runs'}  n_trials={n_trials}")
    print(f"  objective = mean over {[p.name for p in prompts]} of "
          f"wall_request_s (full prompt PP + {config.get('ntg', 128)} gen tokens)")
    if config.get("stop_production", True):
        names = LLM.production_container_names(config)
        print(f"  ⚠️ stopping production container(s) {names} for the WHOLE "
              f"optimization — live pyrite-local models (and any "
              f"pyrite-hosted Claude Code session's model) will 502 through "
              f"the router until the restart")

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
