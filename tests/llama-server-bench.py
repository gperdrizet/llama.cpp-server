'''Drive the real llama-server to measure prompt-preprocessing rate vs context occupancy.

Why a real server (and not llama-bench / llama-batched-bench)
---------------------------------------------------------------
The question is: *how does the prompt-processing (PP) token rate change as the
number of processed tokens grows, out to the full context size?* A single
llama-server run of a long prompt answers it directly -- the server logs a
progress line every ~4096 tokens carrying the CUMULATIVE token count,
CUMULATIVE seconds, and the CUMULATIVE rate, which together trace the whole
PP-rate-vs-occupancy decay curve in one run:

    slot   0: id 0 | task 1 | prompt processing, n_tokens =    4096, progress = 0.03, t =   11.94 s / 343.05 tokens per second
    ...
    slot   0: id 0 | task 1 | prompt processing, n_tokens =   85466, progress = 0.65, t =  367.36 s / 232.88 tokens per second

llama-bench / llama-batched-bench instead report ONE `speed_pp` per fixed `-npp`
(a single point at one occupancy) and cannot trace the curve in one run, and
they do not run at all with speculative decoding (MTP) -- which is one of the
settings we want on the curve. Driving the real server also lets us exercise
any llama-server flag (flash-attn, cache dtype, threads, MTP, split mode, ...),
not just the benchmark tools' subset.

How one launch works
---------------------
For every (sweep cell, spec variant, prompt version, run) the script:
  1. `docker run`s the benchmark image (ghcr.io/ggml-org/llama.cpp:full-cuda),
     mounting the model (-> /models), the prompt (-> /prompts) read-only, and a
     temp work dir (-> /work) read/write, with all GPUs.
  2. Inside, a small driver starts `llama-server` (log -> a file, so a long
     model-load log cannot deadlock a pipe), waits until the model is actually
     loaded (polls /v1/models for the alias, because /health answers 200 while
     the model is still being read in), then POSTs the prompt to
     `/v1/completions` with `max_tokens = ntg` (1 => PP only), reads the
     response, and shuts the server down.
  3. The server log is parsed (host side, pure functions) into the occupancy
     curve: the cumulative progress points are converted to per-segment
     (tokens, seconds, rate) rows, and the two summary lines (`prompt eval
     time`, `eval time`) give the final PP and generation rates.

Output
------
One JSONL per sweep, LONG format: one row per occupancy point, tagged with the
condition (batch/ubatch, spec, prompt, run); the run's final PP/TG rates and
other summary fields are stamped on EVERY point row, so the last point row of a
launch is the whole-prompt summary (there is no separate summary row).
Directly loadable by pandas: filter `status=="ok"` and plot `tps_cum`/`tps_seg`
(y) vs `n_tokens_cum` (x), grouped by condition. Every condition is recorded,
including failures (status `error`/`no_output`), so a sweep leaves a complete,
analyzable trail. A `.summary.json` records the config + launch list.

Four-phase sweep
----------------
Like the batched-bench tuner, the run is a CHAIN of phases, not one big cross
(each phase runs ONLY its winning condition from the previous phases, so the
sweep stays small -- the quant is never re-crossed with every server setting,
server settings are never crossed with batch/ubatch, and spec decoding is
NOT an axis of the PP sweeps):

  Phase 1 (tag `quant`) -- the QUANT sweep: every quant in `quants` at the
      FIXED server_parameters (context, threads, device, cache dtype), with
      the server's DEFAULT split-mode / flash-attn / batch / ubatch (none of
      those flags are passed) and speculative decoding OFF. Each launch
      traces the PP decay curve. Winner = the quant with the highest AVERAGE
      `pp_auc` -- the sum of the per-segment PP rates ("area under the
      curve" of the decay curve).
  Phase 2 (tag `server`) -- the SERVER-parameters sweep: the
      server_sweep_parameters cells (flash-attn, split-mode, ...) on top of
      the fixed server_parameters, at the phase-1 winning quant, still
      spec-off. Winner = the cell with the highest average `pp_auc`; its
      COMPLETE flag set (fixed + swept) is inherited by the later phases.
  Phase 3 (tag `batch`) -- the PP-parameters sweep: pp_parameters cells
      (batch/ubatch and any other prompt-preprocessing flags) at the phase-1
      + phase-2 winners, still spec-off, `ntg: 1` (pure PP). Winner = the pp
      cell with the highest average `pp_auc`.
  Phase 4 (tag `spec`) -- the speculative-decoding scan: every
      `spec_variants` entry (first = the spec-OFF control) at the phase-1 +
      phase-2 + phase-3 winners. Winner = the variant with the highest
      average `pp_auc`. (Spec decoding shouldn't move PP much, so this is
      mostly a guard; the per-variant average `gen_tps` and a naive
      accept-rate estimate -- variant/control `gen_tps` ratio - 1, an UPPER
      bound -- are printed alongside for reference.) Phase 4 generates
      `spec_ntg` tokens (default = `ntg`) so the gen_tps reference is over a
      real number of decode tokens -- with `ntg: 1` it would be a
      single-token estimate.

Every condition in every phase is run with EVERY prompt version, and each
winner's metric is AVERAGED across those launches (all prompts count
equally; `--runs` repeats the whole prompt set for extra robustness).

`server_parameters` (fixed in every phase), `server_sweep_parameters`
(phase-2 axes) and `pp_parameters` (phase-3 axes) each map a llama-server
flag to either a single value (fixed) or a comma-separated list (swept).
Every list-valued flag in a phase's parameter block is crossed: one container
launch per combination, like the batched-bench tuner. `--device` (and
anything in NON_SWEEP_FLAGS) is atomic and never split on commas. In phase 3,
batch x ubatch combos with ubatch > batch are dropped (no-op). Phases 1-2
pass NO batch/ubatch flags at all (the server defaults). The model files come
from `model_base` + `quants` (file = `<base>-<quant>.gguf`), one launch per
quant (the quant is a phase-1 axis). The prompt is a file (swappable);
`prompt_file` may be a single path or a comma-separated list of paths (each a
"version").

`--phase N` runs a single phase: `--phase 1` stops after the ranking;
`--phase 2` REQUIRES the config's `overrides.phase1` block (the winning
quant); `--phase 3` REQUIRES `overrides.phase1` AND `overrides.phase2` (the
COMPLETE winning server cell, fixed + swept flags); `--phase 4` REQUIRES all
three of those AND `overrides.phase3` (the winning pp cell) -- since the
earlier phases are not running to supply their winners.

NOTE on resources: each launch loads the full model on the GPU(s) and
processes a ~full-context prompt, so one launch is minutes to tens of minutes.
A sweep is many launches, and loading the 27B alongside the live `llama-cuda`
would OOM the P100s (and can drop the LiteLLM router the operator's own tools
route through). So at startup the script STOPS the production container(s)
(config `production_containers`, default `["llama-cuda"]`) to free their VRAM,
and RESTARTS them in a `finally` block -- on a clean finish OR any crash /
Ctrl-C / parser.error. Only containers that were running are started back up
(already-down ones stay down), and a `--no-stop-production` flag opts out
(production then must already be down, or the VRAM gate below refuses launches).
As a guard, each launch preflights total free VRAM (`min_free_vram_gib`,
default 24 GiB) and fails the row immediately instead of spending minutes on a
container that would OOM at model load, and any stale `llmsrvbench-*`
containers/workdirs left behind by a previously killed run are removed at
startup.
'''

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path
from datetime import datetime

import docker
import docker.errors

IMAGE_DEFAULT = "ghcr.io/ggml-org/llama.cpp:full-cuda"
SERVER_BIN = "/app/llama-server"

# Flags whose comma-separated values are atomic, never to be swept.
NON_SWEEP_FLAGS = {"--device"}

# Server log lines carry the curve. Matched per-line, most-specific first so
# "prompt eval time" is not mistaken for the generation "eval time".
RE_PROGRESS = re.compile(
    r"prompt processing, n_tokens =\s*(\d+), progress =\s*[\d.]+, "
    r"t =\s*([\d.]+) s / ([\d.]+) tokens per second")
RE_PROMPT_EVAL = re.compile(
    r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens "
    r"\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")
RE_GEN_EVAL = re.compile(
    r"eval time =\s*([\d.]+) ms /\s*(\d+) tokens "
    r"\(\s*([\d.]+) ms per token,\s*([\d.]+) tokens per second\)")


def parse_server_log(log_text: str) -> dict:
    '''Parse a llama-server log into the PP occupancy curve + final rates.

    Returns:
      points      -- list, in log order, each:
                       {n_tokens_cum, n_tokens_seg, t_cum_s, t_seg_s,
                        tps_cum, tps_seg}
      prompt_eval -- {n_tokens, t_ms, ms_per_tok, tps} or None
      gen_eval    -- {n_tokens, t_ms, ms_per_tok, tps} or None

    The cumulative progress points are converted to per-segment rows: segment
    tokens/seconds are the differences between consecutive progress lines, and
    the per-segment (marginal) rate is tokens/seconds for that segment -- that
    is the decay curve to plot. tps_cum is the running average (should match
    the summary line at the end).
    '''

    raw_points = []
    prompt_eval = None
    gen_eval = None

    for line in log_text.splitlines():
        if "prompt processing, n_tokens" in line:
            m = RE_PROGRESS.search(line)
            if m:
                raw_points.append(
                    (int(m.group(1)), float(m.group(2)), float(m.group(3))))
        elif "prompt eval time" in line:
            m = RE_PROMPT_EVAL.search(line)
            if m:
                prompt_eval = {"n_tokens": int(m.group(2)), "t_ms": float(m.group(1)),
                               "ms_per_tok": float(m.group(3)), "tps": float(m.group(4))}
        elif "eval time" in line:  # generation (the "        eval time" line)
            m = RE_GEN_EVAL.search(line)
            if m:
                gen_eval = {"n_tokens": int(m.group(2)), "t_ms": float(m.group(1)),
                            "ms_per_tok": float(m.group(3)), "tps": float(m.group(4))}

    # de-dup exact repeats, then sort by cumulative token count (defensive; the
    # log is already in order).
    raw_points = sorted(set(raw_points), key=lambda p: p[0])

    points = []
    prev_n, prev_t = 0, 0.0
    for n, t, _cum_rate in raw_points:
        seg_n = n - prev_n
        seg_t = t - prev_t
        if seg_n <= 0 or seg_t <= 0:
            prev_n, prev_t = n, t
            continue
        points.append({
            "n_tokens_cum": n,
            "n_tokens_seg": seg_n,
            "t_cum_s": round(t, 3),
            "t_seg_s": round(seg_t, 3),
            "tps_cum": round(n / t, 3),      # running-average rate
            "tps_seg": round(seg_n / seg_t, 3),  # marginal rate (the curve)
        })
        prev_n, prev_t = n, t

    return {"points": points, "prompt_eval": prompt_eval, "gen_eval": gen_eval}


def split_lists(parameters: dict) -> dict:
    return {key: [v.strip() for v in str(value).split(",")]
            for key, value in parameters.items()}


def expand_cells(parameters: dict) -> list:
    '''One parameter dict per sweep cell (cross of all list-valued, non-atomic
    flags). Single-valued and atomic flags pass through unchanged.'''

    split = split_lists(parameters)
    outer_keys = [
        k for k in parameters
        if k not in NON_SWEEP_FLAGS and len(split[k]) > 1
    ]

    if not outer_keys:
        return [dict(parameters)]

    out = []
    for values in product(*(split[k] for k in outer_keys)):
        cell = dict(parameters)
        for key, value in zip(outer_keys, values):
            cell[key] = value
        out.append(cell)
    return out


def valid_batch_ubatch_cells(base: dict) -> list:
    '''Drop batch x ubatch cells where ubatch > batch (a no-op).'''
    batch_key, ubatch_key = "--batch-size", "--ubatch-size"
    cells = []
    for cell in expand_cells(base):
        if batch_key in cell and ubatch_key in cell \
                and int(cell[ubatch_key]) > int(cell[batch_key]):
            continue
        cells.append(cell)
    return cells


def _num(v):
    '''True for a real (non-bool) number -- the guard used by the winner
    picks so a failed launch (None) never shifts a condition's average.'''
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def average_pp_auc(rows: list) -> dict:
    '''Average the per-launch `pp_auc` per (quant, full cell) condition.

    `pp_auc` = sum of the per-segment marginal PP rates (`tps_seg`) across the
    launch's decay curve -- "area under the curve" of rate-vs-occupancy: a
    condition that is fast everywhere scores higher than one that is fast
    only early. The cell carries EVERY swept flag for the phase, so
    conditions differing in any flag stay separate. Rows are the per-launch
    summaries the phase loop collected (tagged with the cell); failed
    launches carry no `pp_auc` and are skipped -- an uneven row count cannot
    shift another condition. The average spans every prompt version the
    condition was run with.'''

    vals = {}
    for r in rows:
        v = r.get("pp_auc")
        if not _num(v):
            continue
        key = (r.get("quant"),
               tuple(sorted((r.get("cell") or {}).items())))
        vals.setdefault(key, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in vals.items() if vs}


def pick_pp_auc_winner(rows: list):
    '''Winner of a PP phase = the (quant, cell) condition with the highest
    AVERAGE `pp_auc` (max of the condition means -- hard max, no noise-floor
    tolerance; same convention as the tuner's winner picks). Used by
    phases 1-3 (quant, server, batch sweeps). Returns {quant, cell, pp_auc}
    or None when no data landed; the caller resolves the model file and the
    inherited settings. The returned cell is the winner's COMPLETE flag set
    for that phase, so the next phase inherits every winning setting.'''

    avg = average_pp_auc(rows)
    if not avg:
        return None

    (quant, cell_items), best = max(avg.items(), key=lambda kv: kv[1])

    return {
        "quant": quant,
        "cell": dict(cell_items),
        "pp_auc": best,
    }


def average_gen_tps(variant_rows: dict[str, list]) -> dict[str, float]:
    '''Average per-launch `gen_tps` per spec-variant label (nan when a
    variant has no successful launch). Reference metric for the spec scan
    -- its WINNER is still picked by `pp_auc` (see
    pick_spec_variant_winner).'''

    out = {}
    for label, rows in variant_rows.items():
        vals = [r["gen_tps"] for r in rows if _num(r.get("gen_tps"))]
        out[label] = sum(vals) / len(vals) if vals else float("nan")
    return out


def pick_spec_variant_winner(variant_rows: dict[str, list],
                             spec_variants: list) -> dict:
    '''Phase-4 (spec scan) winner = the spec variant with the highest
    average `pp_auc` (spec decoding shouldn't move prompt processing much,
    so this is mostly a guard -- the spec-OFF control winning is the
    expected outcome). A failed launch (no pp_auc) cannot shift another
    variant's average.

    Also returned for reference: each variant's average `gen_tps` and the
    naive accept-rate estimate per variant, gen_tps(variant) /
    gen_tps(control) - 1 -- an UPPER bound on accepted draft tokens per
    target step, assuming the verify cost is free (same caveat as the
    tuner's accept estimate).'''

    aucs = {
        label: (sum([r["pp_auc"] for r in rows if _num(r.get("pp_auc"))])
                / len([r for r in rows if _num(r.get("pp_auc"))])
                if any(_num(r.get("pp_auc")) for r in rows) else float("nan"))
        for label, rows in variant_rows.items()
    }
    speeds = average_gen_tps(variant_rows)

    def sort_key(i):
        s = aucs.get(spec_variants[i]["label"], float("nan"))
        return s if s == s else float("-inf")   # nan -> worst

    if not spec_variants:
        return None

    best_i = max(range(len(spec_variants)), key=sort_key)
    best = aucs.get(spec_variants[best_i]["label"], float("nan"))
    if best != best:
        return None

    control = speeds.get("spec-off")
    accept = {}
    for v in spec_variants:
        s = speeds.get(v["label"], float("nan"))
        if v["label"] == "spec-off" or control in (None, 0) or s != s:
            continue
        accept[v["label"]] = round(s / control - 1, 3)

    return {
        "variant": spec_variants[best_i],
        "pp_auc": best,
        "ranking": [(v["label"], aucs.get(v["label"], float("nan")))
                    for v in spec_variants],
        "gen_tps": {v["label"]: speeds.get(v["label"], float("nan"))
                    for v in spec_variants},
        "accept_estimate": accept,
    }


def build_spec_variants(config: dict) -> list:
    variants = config.get("spec_variants")
    if not variants:
        return [{"label": "spec-off", "flags": {}}]
    return variants


def model_specs_from_config(config: dict) -> list:
    '''Resolve the model axis to a list of {quant, model_file} to sweep.

    From `model_base` + `quants` (one or more, comma list or JSON list):
    the model file is constructed as `<model_base>-<quant>.gguf`. Otherwise a
    single entry from a plain `model_file` (quant = the file stem).
    '''

    base = config.get("model_base")
    quants = config.get("quants")

    if isinstance(quants, str):
        quants = [q.strip() for q in quants.split(",") if q.strip()]

    if base and quants:
        return [{"quant": q, "model_file": f"{base}-{q}.gguf"} for q in quants]

    mf = config.get("model_file")
    if mf:
        return [{"quant": Path(mf).stem, "model_file": mf}]

    raise ValueError("config needs model_base + quants, or model_file")


SERVER_ALIAS = "bench"


def _build_server_command(config: dict, cell: dict, spec_flags: dict,
                          model_file: str) -> list:
    '''Assemble the llama-server argv for one launch (model + flags).

    Registers a fixed alias (`bench`) so the in-container driver can address
    the loaded model unambiguously -- llama-server 404s a completion whose
    `model` field doesn't match a loaded model/alias.
    '''

    cmd = [
        SERVER_BIN,
        "--model", f"/models/{model_file}",
        "--port", "8080",
        "--no-webui",
        "--parallel", "1",        # one slot; controlled KV memory
        "--alias", SERVER_ALIAS,
    ]

    # The "prompt processing" progress line (the decay curve) is gated behind
    # log-verbosity; use trace (5) so it is always present -- the log goes to
    # a file, so the extra verbosity is free. Production runs at verbosity 3
    # and still shows the line; lower it via the config to cut log volume.
    if "--log-verbosity" not in cell and "--log-verbosity" not in spec_flags:
        cmd.extend(["--log-verbosity", "5"])

    for key, value in cell.items():
        cmd.extend([key, str(value)])

    for key, value in spec_flags.items():
        cmd.extend([key, str(value)])

    return cmd


# In-container driver: launches llama-server, drives one completion, returns
# the server log + response. Stdlib only. Writes the full payload to
# /work/result.json (shared host dir); the server log goes to /work/server.log
# (a file, never the pipe) so a long model-load log cannot deadlock.
_DRIVER = r'''
import json, os, signal, subprocess, sys, time, urllib.request, urllib.error

def http_json(url, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def read_log():
    try:
        with open("/work/server.log", "rb") as f:
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""

def write_result(obj):
    with open("/work/result.json", "w") as f:
        f.write(json.dumps(obj))

def shutdown(proc):
    for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if proc.poll() is not None:
            break
        proc.send_signal(sig)
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < wait:
            time.sleep(0.2)
    try:
        proc.wait(timeout=5)
    except Exception:
        pass

def fail(msg):
    write_result({"status": "error", "error": msg, "server_log": read_log()})
    print("__RESULT__ error: " + msg, file=sys.stderr)
    sys.exit(0)

def main():
    cfg = json.load(open("/work/launch.json"))
    port = cfg["port"]
    try:
        proc = subprocess.Popen(
            cfg["server_cmd"],
            stdout=open("/work/server.log", "wb"),
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        fail(f"server launch: {e}")
        return

    # wait for the model to actually finish loading. /health answers 200 while
    # the model is still being read in, so polling it would let the request
    # out mid-load and the server would answer 400. Poll /v1/models until the
    # alias is listed instead.
    deadline = time.time() + cfg["load_timeout"]
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            data = http_json(f"http://127.0.0.1:{port}/v1/models", timeout=3)
            ids = [m.get("id") or m.get("name") for m in data.get("data", [])]
            if cfg["model_name"] in ids:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        shutdown(proc)
        fail("model did not finish loading within load_timeout "
             "(or the server exited early)")
        return

    # send the prompt. HTTPError carries the response body (400 says e.g.
    # the exact prompt token count when it overflows --ctx-size), so
    # capture it: a bare str(e) would drop it.
    try:
        with open(cfg["prompt_file"], "r", encoding="utf-8") as f:
            prompt = f.read()
        payload = {
            "model": cfg["model_name"],
            "prompt": prompt,
            "max_tokens": cfg["ntg"],
            "temperature": 0,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=data,
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=cfg["request_timeout"]) as r:
            resp = json.loads(r.read().decode("utf-8"))
        wall = time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:2000]
        except Exception:
            body = ""
        shutdown(proc)
        fail(f"completion request failed: HTTP {e.code} {e.reason} "
             f"-- server body: {body}")
        return
    except Exception as e:
        shutdown(proc)
        fail(f"completion request failed: {e!r}")
        return

    shutdown(proc)
    usage = (resp.get("usage") or {})
    write_result({
        "status": "ok",
        "response": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "wall_request_s": round(wall, 3),
        },
        "server_log": read_log(),
    })
    print("__RESULT__ ok")

main()
'''


def free_vram_gib() -> float:
    '''Total free VRAM across visible GPUs, via nvidia-smi (0.0 if n/a).

    The model is split across all the GPUs, so total free VRAM is the number
    that has to cover it.
    '''

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 0.0

    try:
        return sum(int(x) for x in out.stdout.split()) / 1024.0
    except ValueError:
        return 0.0


def cleanup_stale_bench() -> None:
    '''Remove containers/workdirs left behind by a previously killed run.

    A Ctrl-C between container.start() and container.remove() orphans the
    container, so a fresh run must not start on top of them. Best effort: if
    docker is unreachable, report and continue.
    '''

    try:
        client = docker.from_env()
        stale = client.containers.list(all=True, filters={"name": "llmsrvbench-"})
        for c in stale:
            c.remove(force=True)
    except docker.errors.DockerException as e:
        print(f"  !! could not check for stale bench containers: {e}")
        return

    if stale:
        print(f"  removed {len(stale)} stale bench container(s) from a previous run")

    for p in Path("/tmp").glob("llmsrvbench_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def production_container_names(config: dict) -> list:
    '''Names of the production containers this run should stop/restart.

    Read from the config's `production_containers` (a list, or a comma list);
    defaults to `["llama-cuda"]` — the production inference server. These are
    EXACT container names, not the substring `docker` name-filter uses
    (that would over-match, e.g. a `litellm-router` whose name merely
    contains the pattern), so we match `c.name == name`.
    '''

    names = config.get("production_containers", ["llama-cuda"])
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    return [str(n) for n in names]


def stop_production_containers(config: dict) -> list:
    '''Stop the production containers before the bench, to free their VRAM.

    Returns the (docker-py) container objects that were RUNNING and got
    stopped — the caller must hand these back to
    `restore_production_containers()` when the run ends. Containers that were
    already stopped are NOT started back up at the end (we didn't stop them).
    Best effort per container: a docker hiccup on one container must not
    abort the run, but a container we could not stop is reported loudly.
    '''

    if not config.get("stop_production", True):
        return []

    names = production_container_names(config)
    try:
        client = docker.from_env()
        all_containers = client.containers.list(all=True)
    except docker.errors.DockerException as e:
        print(f"  !! could not reach docker to stop production containers: {e}")
        print("  !! the VRAM gate may now refuse launches — stop them manually")
        return []

    by_name = {c.name: c for c in all_containers}
    stopped = []
    for name in names:
        c = by_name.get(name)
        if c is None:
            print(f"  (production container {name!r} not found; nothing to stop)")
            continue
        if c.status != "running":
            print(f"  (production container {name!r} already {c.status}; "
                  f"leaving it alone)")
            continue
        try:
            # Manual stop is honored by the container's restart_policy
            # (unless-stopped): docker will NOT auto-restart a container the
            # user explicitly stopped, so it stays down for the whole run.
            c.stop()
            stopped.append(c)
            print(f"  stopped production container {name!r} (was running)")
        except docker.errors.DockerException as e:
            print(f"  !! could not stop production container {name!r}: {e}")

    if stopped:
        # The driver's nvidia-smi reads free VRAM; give the GPU a moment to
        # release the stopped container's VRAM so the preflight gate passes on
        # the first launch instead of failing fast.
        time.sleep(5)
    return stopped


def restore_production_containers(stopped: list) -> None:
    '''Start back up the production containers `stop_production_containers`
    stopped. Called from a `finally` block, so it runs on a clean finish AND
    on any crash / Ctrl-C / parser.error. Only containers we actually stopped
    are started (ones that were already down stay down). Loud on failure: if
    production is not restored, the operator must do it by hand.'''

    for c in stopped:
        # Snapshot objects stay valid: docker-py's start posts by container
        # id, so the object captured at stop time is still addressable.
        try:
            if c.status == "running":
                # Something (a manual `docker start`, an ansible run) already
                # brought it back; don't double-start.
                continue
            c.start()
            print(f"  restarted production container {c.name!r}")
        except docker.errors.DockerException as e:
            print(f"  !! FAILED to restart production container {c.name!r}: {e}")
            print(f"  !! start it manually: docker start {c.name}")


def run_bench_launch(config: dict, cell: dict, spec_flags: dict,
                     prompt_file: str, launch_id: int, model_file: str,
                     ntg: int = None) -> dict:
    '''Run one llama-server bench launch (a full container).

    `ntg` overrides the config's `ntg` for this launch (phase 4 may use
    `spec_ntg` so the spec scan compares generation rates over a real
    number of tokens). Returns a dict: {status, server_log?, response?,
    error?}.
    '''

    # Pre-check the inputs on the host so a bad path fails fast with a clear
    # message instead of a multi-minute container that dies at model load.
    model_host = Path(config["models_path"]) / model_file
    if not model_host.is_file():
        return {"status": "error", "error": f"model file not found: {model_host}"}

    prompt_host = Path(prompt_file).expanduser().resolve()
    if not prompt_host.is_file():
        return {"status": "error", "error": f"prompt file not found: {prompt_host}"}

    # Preflight free VRAM. If another model (e.g. the production llama-cuda)
    # is resident, the server would OOM part-way through loading the full
    # quant; fail the row now instead of after minutes.
    min_vram = float(config.get("min_free_vram_gib", 24))
    free = free_vram_gib()
    if free < min_vram:
        return {"status": "error",
                "error": (f"only {free:.1f} GiB total VRAM free < min_free_vram_gib "
                          f"{min_vram:g} (is the production model still resident?)")}

    models_host = Path(config["models_path"]).resolve()
    prompt_in_container = f"/prompts/{prompt_host.name}"

    client = docker.from_env()
    workdir = Path(tempfile.mkdtemp(prefix="llmsrvbench_"))

    server_cmd = _build_server_command(config, cell, spec_flags, model_file)

    launch = {
        "server_cmd": server_cmd,
        "model_name": SERVER_ALIAS,   # matches the --alias registered above
        "prompt_file": prompt_in_container,
        "ntg": int(ntg if ntg is not None else config.get("ntg", 1)),
        "port": 8080,
        "load_timeout": int(config.get("load_timeout", 600)),
        "request_timeout": int(config.get("request_timeout", 3600)),
    }
    (workdir / "launch.json").write_text(json.dumps(launch))
    (workdir / "driver.py").write_text(_DRIVER)

    container = client.containers.create(
        image=config.get("image", IMAGE_DEFAULT),
        name=f"llmsrvbench-{launch_id}-{os.urandom(3).hex()}",
        entrypoint="sh",
        command=["-c", "python3 /work/driver.py"],
        volumes={
            str(models_host): {"bind": "/models", "mode": "ro"},
            str(prompt_host.parent): {"bind": "/prompts", "mode": "ro"},
            str(workdir): {"bind": "/work", "mode": "rw"},
        },
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
    )

    exit_status = 0
    logs = b""
    try:
        container.start()
        exit_status = container.wait()["StatusCode"]
        logs = container.logs(stdout=True, stderr=True)
    finally:
        try:
            container.remove()
        except docker.errors.NotFound:
            pass

    # Read the driver's payload from the shared host work dir (it survives
    # container removal because /work is a host bind mount). Salvage
    # server.log too, for the no-result case.
    result = None
    server_log = ""
    result_path = workdir / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            result = None
    server_log_path = workdir / "server.log"
    if server_log_path.exists():
        try:
            server_log = server_log_path.read_text(encoding="utf-8",
                                                   errors="replace")
        except OSError:
            pass

    shutil.rmtree(workdir, ignore_errors=True)

    if result is None:
        text = logs.decode("utf-8", "replace") if isinstance(logs, (bytes, bytearray)) \
            else str(logs)
        result = {"status": "error",
                  "error": f"no result (container exit {exit_status})\n{text[-1000:]}",
                  "server_log": server_log or text}

    return result


def run_launch(phase: str, config: dict, cell: dict, spec_label: dict,
               prompt_file: str, run_id: int, launch_counter: int,
               result_file, model_file: str, quant: str, ntg: int = None):
    '''Run one (cell, spec, prompt, run) launch and record it. Returns
    (launch_counter, launch_summary): the updated counter plus the per-launch
    summary dict (pp_tps, gen_tps, ...) for successful launches, or None on
    error/no_output. Every condition writes at least one row.'''

    spec_flags = spec_label.get("flags", {})
    desc = (f"batch={cell.get('--batch-size')}, ubatch={cell.get('--ubatch-size')}, "
            f"spec={spec_label['label']}, quant={quant}, "
            f"prompt={Path(prompt_file).name}")

    tags = {
        "phase": phase, "run": run_id, "spec": spec_label["label"],
        "launch": launch_counter, "model": config["model_name"],
        "quant": quant, "prompt": Path(prompt_file).name,
    }
    # every cell flag is tagged, so swept axes (flash-attn, split-mode,
    # cache types, threads, ...) separate conditions in analysis
    for key, value in cell.items():
        tags[key[2:].replace("-", "_")] = str(value)

    print(f"  [{phase}] run {run_id}: {desc}")
    launch_counter += 1

    try:
        result = run_bench_launch(config, cell, spec_flags, prompt_file,
                                  launch_counter, model_file, ntg)

    except docker.errors.ContainerError as e:
        raw = e.stderr or b""
        err = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray))
               else str(raw)).strip()
        print(f"    !! container exited {e.exit_status}; recorded error")
        result_file.write(json.dumps(
            {"status": "error", "exit_status": e.exit_status,
             "error": err[-2000:], **tags}, separators=(",", ":")) + "\n")
        return launch_counter, None

    except Exception as e:  # docker client / network / fs errors
        print(f"    !! launch error: {e}")
        result_file.write(json.dumps(
            {"status": "error", "error": str(e)[-2000:], **tags},
            separators=(",", ":")) + "\n")
        return launch_counter, None

    if result.get("status") != "ok":
        print(f"    !! launch failed: {str(result.get('error'))[:200]}")
        result_file.write(json.dumps(
            {"status": "error", "error": str(result.get("error", "unknown"))[-2000:],
             # a load-timeout / early crash leaves its cause in the server
             # log; the last lines (crash trace, OOM, load progress) are what
             # the host needs to diagnose it
             "server_log_tail": (result.get("server_log") or "")[-2000:],
             **tags}, separators=(",", ":")) + "\n")
        return launch_counter, None

    parsed = parse_server_log(result.get("server_log", ""))
    if not parsed["points"]:
        print(f"    !! no PP progress lines parsed; recorded no_output")
        result_file.write(json.dumps(
            {"status": "no_output",
             "server_log_tail": (result.get("server_log", ""))[-2000:],
             **tags}, separators=(",", ":")) + "\n")
        return launch_counter, None

    resp = result.get("response", {})
    common = dict(tags)
    common["status"] = "ok"
    common["n_points"] = len(parsed["points"])
    common["n_tokens_prompt"] = parsed["prompt_eval"]["n_tokens"] if parsed["prompt_eval"] else None
    common["pp_tps"] = parsed["prompt_eval"]["tps"] if parsed["prompt_eval"] else None
    common["pp_ms_per_tok"] = parsed["prompt_eval"]["ms_per_tok"] if parsed["prompt_eval"] else None
    # winner metric: "area under the curve" of the PP decay curve = the sum
    # of the per-segment marginal rates (tps_seg)
    common["pp_auc"] = round(sum(p["tps_seg"] for p in parsed["points"]), 3)
    common["gen_tps"] = parsed["gen_eval"]["tps"] if parsed["gen_eval"] else None
    common["gen_ms_per_tok"] = parsed["gen_eval"]["ms_per_tok"] if parsed["gen_eval"] else None
    common["prompt_tokens_http"] = resp.get("prompt_tokens")
    common["completion_tokens_http"] = resp.get("completion_tokens")
    common["wall_request_s"] = resp.get("wall_request_s")

    # The per-launch summary fields (pp_tps, gen_tps, wall_request_s, ...) live
    # in `common` and are stamped on EVERY point row, so the last point row
    # carries the whole-prompt summary — no separate summary row is needed.
    for i, p in enumerate(parsed["points"], start=1):
        row = dict(common)
        row["point"] = i
        row.update(p)
        result_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    return launch_counter, common


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Measure llama-server PP rate vs context occupancy")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3,
                        help="repeat each phase's launches this many times "
                             "(default 3)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="override config prompt_file (path or comma list)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the total number of launches (quick smoke "
                             "test; phases chain in order)")
    parser.add_argument("--no-stop-production", action="store_true",
                        help="do NOT stop/restart the production container(s) "
                             "(llama-cuda); use when production is already "
                             "down or you do not want the bench touching it. "
                             "The VRAM gate will then refuse launches while the "
                             "production model is resident.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="run only this phase; --phase 2 needs "
                             "overrides.phase1 (the winning quant), "
                             "--phase 3 needs overrides.phase1 AND "
                             "overrides.phase2 (the COMPLETE winning server "
                             "cell, fixed + swept flags), --phase 4 needs "
                             "all of those AND overrides.phase3 (the "
                             "winning pp cell), since the earlier phases do "
                             "not run this invocation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    root_path = Path(__file__).resolve().parents[1]
    config.setdefault("models_path", str(root_path / "models"))
    config["stop_production"] = not args.no_stop_production

    overrides = config.get("overrides", {})

    # resolve prompt versions (string or comma list; CLI overrides)
    prompt_spec = args.prompt if args.prompt else config.get("prompt_file", "prompt.txt")
    if isinstance(prompt_spec, list):
        prompts = prompt_spec
    else:
        prompts = [v.strip() for v in str(prompt_spec).split(",") if v.strip()]
    prompts = [str((root_path / p) if not Path(p).is_absolute() else p)
               for p in prompts]

    spec_variants = build_spec_variants(config)

    # Phase 1 runs the FIXED server_parameters alone (split-mode / flash-
    # attn / batch / ubatch left at the server's defaults -- none of those
    # flags passed); phase 2 crosses server_sweep_parameters (flash-attn,
    # split-mode, ...) on top of them; phase 3 crosses pp_parameters
    # (batch/ubatch + any other PP flags) at the winners, ubatch > batch
    # dropped.
    fixed_server = {k: str(v)
                    for k, v in (config.get("server_parameters") or {}).items()
                    if k not in ("--batch-size", "--ubatch-size")}
    sweep_server = dict(config.get("server_sweep_parameters") or {})
    server_sweep_cells = expand_cells(sweep_server)
    full_server_cells = [dict(fixed_server, **c) for c in server_sweep_cells]
    pp_cells = valid_batch_ubatch_cells(dict(config.get("pp_parameters") or {}))
    model_specs = model_specs_from_config(config)

    output_path = root_path / "tests/results/llama-server-bench"
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file_path = output_path / f"{config['model_name']}_{timestamp}.jsonl"

    # Per-phase launch counts (per run), and the total. Every condition is
    # run with EVERY prompt version (the runs loop); winners are the average
    # across those launches.
    run_p1 = args.phase is None or args.phase == 1
    run_p2 = args.phase is None or args.phase == 2
    run_p3 = args.phase is None or args.phase == 3
    run_p4 = args.phase is None or args.phase == 4
    n_p1 = len(model_specs) * len(prompts) if run_p1 else 0
    n_p2 = len(server_sweep_cells) * len(prompts) if run_p2 else 0
    n_p3 = len(pp_cells) * len(prompts) if run_p3 else 0
    n_p4 = len(spec_variants) * len(prompts) if run_p4 else 0
    n_launches = (n_p1 + n_p2 + n_p3 + n_p4) * args.runs
    if args.limit:
        n_launches = min(n_launches, args.limit)

    print("llama-server PP-vs-occupancy bench (four phases)")
    print(f"  models  : {', '.join(s['quant'] for s in model_specs)}"
          f"  ({config['models_path']})")
    print(f"  image   : {config.get('image', IMAGE_DEFAULT)}")
    print(f"  prompts : {prompts}  (ntg={config.get('ntg', 1)}, "
          f"spec_ntg={config.get('spec_ntg', config.get('ntg', 1))})")
    print(f"  fixed server params : {fixed_server}  "
          f"(split-mode/flash-attn/batch/ubatch = server defaults in p1-2)")
    print(f"  Phase 1 : {n_p1} launches/run (quants @ fixed server params, "
          f"spec-off)")
    print(f"  Phase 2 : {n_p2} launches/run (server_sweep_parameters @ "
          f"phase-1 winner, spec-off)")
    print(f"  Phase 3 : {n_p3} launches/run (pp_parameters @ phase-1+2 "
          f"winners, spec-off)")
    print(f"  Phase 4 : {n_p4} launches/run (spec_variants @ phase-1+2+3 "
          f"winners)")
    print(f"  runs: {args.runs} = {n_launches} launches total")
    print(f"  results : {result_file_path}")

    summary = {
        "model": config["model_name"],
        "model_specs": [{"quant": s["quant"], "model_file": s["model_file"]}
                        for s in model_specs],
        "image": config.get("image", IMAGE_DEFAULT), "runs": args.runs,
        "ntg": config.get("ntg", 1),
        "spec_ntg": int(config.get("spec_ntg", config.get("ntg", 1))),
        "prompts": prompts,
        "result_file": str(result_file_path),
        "fixed_server_params": fixed_server,
        "server_sweep_cells": full_server_cells,
        "pp_cells": pp_cells,
        "spec_variants": spec_variants,
        "phases": {}, "launches": [],
    }

    def do_launch(phase, config, cell, spec, prompt_file, run_id,
                  launch_counter, f, ms, summary, ntg=None):
        '''Run one launch, record its summary entry; return
        (new launch_counter, launch summary or None).'''
        launch_counter, launch_sum = run_launch(
            phase, config, cell, spec, prompt_file,
            run_id, launch_counter, f,
            ms["model_file"], ms["quant"], ntg)
        summary["launches"].append({
            "phase": phase, "launch": launch_counter, "spec": spec["label"],
            "quant": ms["quant"],
            "prompt": Path(prompt_file).name, "run": run_id,
            "cell": {k: v for k, v in cell.items()
                     if k not in NON_SWEEP_FLAGS},
        })
        return launch_counter, launch_sum

    # the spec-off control (phases 1 and 2) — defined before the loops so a
    # --phase 2/3 run (which skips phase 1) still has it
    control = {"label": "spec-off", "flags": {}}

    cleanup_stale_bench()

    # `stopped_production` holds the production container(s) we stop for the
    # bench (see the `try:` below) so the `finally` can restart exactly those.
    # It is initialised here (empty) so the `finally` always has a bound name.
    stopped_production = []

    def print_pp_ranking(avg: dict):
        '''Print a PP-phase ranking (avg pp_auc per (quant, cell)), best
        first. nan scores sort last.'''
        print("\nRanking (avg pp_auc -- sum of per-segment PP rates):")
        for (q, citems), s in sorted(
                avg.items(),
                key=lambda kv: (-kv[1] if kv[1] == kv[1]
                                else float("inf"))):
            c = dict(citems)
            parts = [f"{k[2:]}={v}" for k, v in sorted(c.items())]
            print(f"  quant={q}  {'  '.join(parts)}  "
                  f"pp_auc={s:12.1f} t/s-sum")

    def record_pp_winner(phase_key: str, winner: dict, source: str,
                         ms: dict = None):
        '''Store a PP-phase winner in the summary (with its ranking).'''
        entry = {
            "quant": winner["quant"],
            "cell": dict(winner["cell"]),
            "pp_auc": round(winner["pp_auc"], 2),
            "source": source,
        }
        if ms is not None:
            entry["model_file"] = ms["model_file"]
        summary["phases"][phase_key] = entry

    # Winner resolution for later phases.
    #
    # When a phase runs in this invocation, its winner is picked from the
    # rows it just collected (chained). When it is skipped (--phase N), the
    # winners of every EARLIER phase must be pinned in the config's
    # `overrides` block -- the phases do not run to supply them:
    #   overrides.phase1: {"quant": ...}
    #   overrides.phase2: the COMPLETE winning server cell (fixed + swept
    #                     --* flags)
    #   overrides.phase3: the winning pp cell (--batch-size, --ubatch-size,
    #                     ...)
    def pinned_quant(needed_by: str) -> dict:
        pinned = overrides.get("phase1") or {}
        quant = pinned.get("quant")
        if not quant:
            parser.error(f"--phase {needed_by} needs overrides.phase1 "
                         f"({{\"quant\": ...}}) in the config, since phase "
                         f"1 does not run this invocation")
        ms = next((s for s in model_specs if s["quant"] == quant), None)
        if ms is None:
            parser.error(f"overrides.phase1.quant {quant!r} is not in "
                         f"model_base + quants")
        return ms

    def pinned_server_cell(needed_by: str) -> dict:
        pinned2 = overrides.get("phase2") or {}
        cell = {k: str(v) for k, v in pinned2.items() if k.startswith("--")}
        if not cell:
            parser.error(f"--phase {needed_by} needs overrides.phase2 (the "
                         f"COMPLETE winning server cell -- every --* flag, "
                         f"fixed AND swept) in the config, since phase 2 "
                         f"does not run this invocation")
        return cell

    def pinned_pp_cell(needed_by: str) -> dict:
        pinned3 = overrides.get("phase3") or {}
        cell = {k: str(v) for k, v in pinned3.items() if k.startswith("--")}
        if not cell:
            parser.error(f"--phase {needed_by} needs overrides.phase3 (the "
                         f"winning pp cell: --batch-size, --ubatch-size, "
                         f"...) in the config, since phase 3 does not run "
                         f"this invocation")
        return cell

    launch_counter = 0
    stop = False
    p1_rows = []          # per-launch summaries, phase 1 (winner pick)
    p2_rows = []          # per-launch summaries, phase 2 (winner pick)
    p3_rows = []          # per-launch summaries, phase 3 (winner pick)
    p4_variant_rows = {}  # spec label -> per-launch summaries (winner pick)

    try:
        # Free the production model's VRAM for the bench: stop the production
        # container(s) NOW (before the first launch's VRAM gate). This is the
        # first protected statement, so even a failure right here still fires
        # the `finally` below. `stopped_production` holds exactly what we
        # stopped, so already-down containers are left down.
        stopped_production = stop_production_containers(config)

        with open(result_file_path, "a") as f:

            # PHASE 1: quants at the FIXED server params (default split-mode /
            # flash-attn / batch / ubatch), spec-off
            if run_p1:
                print("\n=== Phase 1: quant sweep (fixed server params, spec-off, "
                      "default batch/ubatch) ===")
                for run_id in range(1, args.runs + 1):
                    for ms in model_specs:
                        for prompt_file in prompts:      # EVERY prompt
                            if stop:
                                break
                            if args.limit and launch_counter >= args.limit:
                                print(f"\n(limit {args.limit} launches "
                                      f"reached; stopping)")
                                stop = True
                                break
                            launch_counter, launch_sum = do_launch(
                                "quant", config, dict(fixed_server), control,
                                prompt_file, run_id, launch_counter, f, ms,
                                summary)
                            if launch_sum:
                                entry = dict(launch_sum)
                                entry["cell"] = dict(fixed_server)
                                p1_rows.append(entry)
                            if args.limit and launch_counter >= args.limit:
                                stop = True
                                break

                avg = average_pp_auc(p1_rows)
                print_pp_ranking(avg)
                winner = pick_pp_auc_winner(p1_rows)
                if winner is not None:
                    record_pp_winner("1", winner, "auto",
                                     next(s for s in model_specs
                                          if s["quant"] == winner["quant"]))
                    print(f"Phase 1 winner: quant={winner['quant']} (avg "
                          f"pp_auc {winner['pp_auc']:.1f})")
                    if args.phase == 1:
                        print("(Stopped: --phase 1. Pin the winner in "
                              "overrides.phase1 to run phases 2-4 later.)")
                elif run_p2 or run_p3 or run_p4:
                    parser.error("phase 1 produced no pp_auc data; cannot "
                                 "chain to later phases")
                else:
                    print("!! phase 1 produced no pp_auc data")

            # PHASE 2: server_sweep_parameters cells at the phase-1 winning quant
            if run_p2:
                print("\n=== Phase 2: server-parameter sweep "
                      "(flash-attn, split-mode, ...) ===")

                if args.phase == 2:
                    # phase 1 is not running: its winner must be pinned
                    ms = pinned_quant(2)
                    record_pp_winner("1", {"quant": ms["quant"],
                                           "cell": dict(fixed_server),
                                           "pp_auc": None}, "override", ms)
                    fixed_src = "override (overrides.phase1)"
                else:
                    # picked right after the phase-1 loop; phases["1"] already
                    # recorded
                    ms = next(s for s in model_specs
                              if s["quant"] == winner["quant"])
                    fixed_src = "auto (phase 1)"

                print(f"(phase 2 runs @ {fixed_src}: quant={ms['quant']}, "
                      f"fixed={fixed_server})")

                for run_id in range(1, args.runs + 1):
                    for cell in full_server_cells:
                        for prompt_file in prompts:      # EVERY prompt
                            if stop:
                                break
                            if args.limit and launch_counter >= args.limit:
                                print(f"\n(limit {args.limit} launches reached; "
                                      f"stopping)")
                                stop = True
                                break
                            launch_counter, launch_sum = do_launch(
                                "server", config, cell, control, prompt_file,
                                run_id, launch_counter, f, ms, summary)
                            if launch_sum:
                                entry = dict(launch_sum)
                                entry["cell"] = dict(cell)   # full: fixed + sweep
                                p2_rows.append(entry)
                            if args.limit and launch_counter >= args.limit:
                                stop = True
                                break

                avg = average_pp_auc(p2_rows)
                print_pp_ranking(avg)
                winner2 = pick_pp_auc_winner(p2_rows)
                if winner2 is not None:
                    record_pp_winner("2", winner2, "auto")
                    print(f"Phase 2 winner: {winner2['cell']} (avg "
                          f"pp_auc {winner2['pp_auc']:.1f})")
                    if args.phase == 2:
                        print("(Stopped: --phase 2. Pin the winner in "
                              "overrides.phase2 (the COMPLETE cell: fixed + "
                              "swept flags) to run phases 3/4 later.)")
                elif run_p3 or run_p4:
                    parser.error("phase 2 produced no pp_auc data; cannot "
                                 "chain to later phases")
                else:
                    print("!! phase 2 produced no pp_auc data")

            # PHASE 3: pp_parameters (batch/ubatch) at the phase-1 + phase-2
            # winners
            if run_p3:
                print("\n=== Phase 3: batch/ubatch sweep ===")

                if args.phase == 3:
                    # phases 1 and 2 are not running: both winners must be pinned
                    ms = pinned_quant(3)
                    server_cell = pinned_server_cell(3)
                    record_pp_winner("1", {"quant": ms["quant"],
                                           "cell": dict(fixed_server),
                                           "pp_auc": None}, "override", ms)
                    record_pp_winner("2", {"quant": ms["quant"],
                                           "cell": server_cell,
                                           "pp_auc": None}, "override", ms)
                    fixed_src = "override (overrides.phase1+2)"
                else:
                    # chained: both winners picked this invocation
                    ms = next(s for s in model_specs
                              if s["quant"] == winner["quant"])
                    server_cell = dict(winner2["cell"])
                    fixed_src = "auto (phase 1 + 2)"

                print(f"(phase 3 runs @ {fixed_src}: quant={ms['quant']}, "
                      f"server={server_cell})")

                for run_id in range(1, args.runs + 1):
                    for cell in pp_cells:
                        for prompt_file in prompts:      # EVERY prompt
                            if stop:
                                break
                            if args.limit and launch_counter >= args.limit:
                                print(f"\n(limit {args.limit} launches reached; "
                                      f"stopping)")
                                stop = True
                                break
                            # the server command = the winning server cell +
                            # this pp cell (batch/ubatch, ...); the row is
                            # tagged with the full combined flag set
                            launch_counter, launch_sum = do_launch(
                                "batch", config, dict(server_cell, **cell),
                                control, prompt_file, run_id, launch_counter,
                                f, ms, summary)
                            if launch_sum:
                                entry = dict(launch_sum)
                                entry["cell"] = dict(cell)   # phase-3's own axis
                                p3_rows.append(entry)
                            if args.limit and launch_counter >= args.limit:
                                stop = True
                                break

                avg = average_pp_auc(p3_rows)
                print_pp_ranking(avg)
                winner3 = pick_pp_auc_winner(p3_rows)
                if winner3 is not None:
                    record_pp_winner("3", winner3, "auto")
                    print(f"Phase 3 winner: {winner3['cell']} (avg "
                          f"pp_auc {winner3['pp_auc']:.1f})")
                    if args.phase == 3:
                        print("(Stopped: --phase 3. Pin the winner in "
                              "overrides.phase3 to run phase 4 later.)")
                elif run_p4:
                    parser.error("phase 3 produced no pp_auc data; cannot "
                                 "chain to phase 4")
                else:
                    print("!! phase 3 produced no pp_auc data")

            # PHASE 4: speculative decoding scan at the winners of phases 1-3
            if run_p4:
                print("\n=== Phase 4: speculative decoding scan ===")

                if args.phase == 4:
                    # phases 1-3 are not running: all winners must be pinned
                    ms = pinned_quant(4)
                    server_cell = pinned_server_cell(4)
                    pp_cell = pinned_pp_cell(4)
                    record_pp_winner("1", {"quant": ms["quant"],
                                           "cell": dict(fixed_server),
                                           "pp_auc": None}, "override", ms)
                    record_pp_winner("2", {"quant": ms["quant"],
                                           "cell": server_cell,
                                           "pp_auc": None}, "override", ms)
                    record_pp_winner("3", {"quant": ms["quant"],
                                           "cell": pp_cell,
                                           "pp_auc": None}, "override", ms)
                    fixed_src = "override (overrides.phase1+2+3)"
                else:
                    # chained: all three winners picked this invocation
                    ms = next(s for s in model_specs
                              if s["quant"] == winner["quant"])
                    server_cell = dict(winner2["cell"])
                    pp_cell = dict(winner3["cell"])
                    fixed_src = "auto (phase 1 + 2 + 3)"

                fixed = {**server_cell, **pp_cell}
                spec_ntg = int(config.get("spec_ntg", config.get("ntg", 1)))
                print(f"(phase 4 runs @ {fixed_src}: quant={ms['quant']}, "
                      f"server={server_cell}, pp={pp_cell}, ntg={spec_ntg})")

                for run_id in range(1, args.runs + 1):
                    for spec in spec_variants:
                        for prompt_file in prompts:      # EVERY prompt
                            if stop:
                                break
                            if args.limit and launch_counter >= args.limit:
                                print(f"\n(limit {args.limit} launches reached; "
                                      f"stopping)")
                                stop = True
                                break
                            launch_counter, launch_sum = do_launch(
                                "spec", config,
                                dict(fixed, **spec.get("flags", {})),
                                spec, prompt_file, run_id, launch_counter, f,
                                ms, summary, ntg=spec_ntg)
                            if launch_sum:
                                p4_variant_rows.setdefault(
                                    spec["label"], []).append(launch_sum)
                            if args.limit and launch_counter >= args.limit:
                                stop = True
                                break

                winner4 = pick_spec_variant_winner(p4_variant_rows, spec_variants)
                if winner4 is None:
                    parser.error("phase 4 produced no pp_auc data")
                summary["phases"]["4"] = {
                    "variant": winner4["variant"]["label"],
                    "fixed": {"quant": ms["quant"],
                              "server": server_cell, "pp": pp_cell},
                    "pp_auc": (round(winner4["pp_auc"], 2)
                               if winner4["pp_auc"] == winner4["pp_auc"]
                               else None),
                    "ranking": [[lab, (round(s, 2) if s == s else None)]
                                for lab, s in winner4["ranking"]],
                    "gen_tps": {lab: (round(s, 2) if s == s else None)
                                for lab, s in winner4["gen_tps"].items()},
                    "accept_estimate": winner4["accept_estimate"],
                }
                print(f"Phase 4 winner: {winner4['variant']['label']} "
                      f"(avg pp_auc {winner4['pp_auc']:.1f})")
                print("  gen_tps (reference): "
                      f"{ {lab: (round(s, 1) if s == s else None) for lab, s in winner4['gen_tps'].items()} }")
                if winner4["accept_estimate"]:
                    print(f"  naive accept estimate (upper bound): "
                          f"{winner4['accept_estimate']}")

        summary_file = result_file_path.with_suffix(".summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nWrote results to {result_file_path}")
        print(f"Wrote summary to {summary_file}")
    finally:
        # Runs on clean exit AND on any crash (exceptions, Ctrl-C, and
        # parser.error/SystemExit alike), so the production container is
        # always brought back once the bench finishes, however it ends.
        restore_production_containers(stopped_production)
