'''Shared llama-server launch/measurement infrastructure.

This module is the reusable engine behind the benchmark tools in this
directory. It is imported by:

  * ``llama_server_optuna_bench`` -- the joint-space Optuna search (the
    current tuning tool), which uses ``run_bench_launch`` for every trial and
    ``standardized_objective`` to score it;
  * ``llama_context_tuner`` -- finds the largest safe ``--ctx-size`` for a
    model, reusing the same container launch + PP decay-curve parser;
  * the host-side self-tests in ``tests/self_tests``.

It deliberately contains NO command-line sweep of its own: the old four-phase
``llama-server-bench.py`` sweep was superseded by the Optuna search and has
been removed. What remains is the machinery:

  * ``extend_cli_flags`` / ``_build_server_command`` / ``_DRIVER`` -- build and
    run one Dockerized ``llama-server``, driving a streaming completion with
    ``timings_per_token`` (per-token decode curve, finish_reason, exact usage);
  * ``parse_server_log`` -- turn the server's progress lines into the PP
    rate-vs-occupancy decay curve;
  * ``standardized_objective`` and friends -- the versioned T* objective;
  * ``model_specs_from_config``, ``free_vram_gib``, ``cleanup_stale_bench``,
    ``stop_production_containers`` / ``restore_production_containers``,
    ``run_bench_launch`` -- config expansion, resource guards, and the launch
    primitive.

Anything written to the GPU needs the production containers stopped first;
see the tool docstrings for the ⚠️ note about the live models on pyrite.
'''

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from itertools import product
from pathlib import Path

import docker
import docker.errors

IMAGE_DEFAULT = "ghcr.io/ggml-org/llama.cpp:full-cuda"
SERVER_BIN = "/app/llama-server"

# Flags whose comma-separated values are atomic, never to be swept.
NON_SWEEP_FLAGS = {"--device"}

# Bare CLI switches: llama.cpp takes NO value for these. The config still says
# "on"/"off" so TPE can sweep them; emit the true form or the negated form
# (where one exists) instead of appending a stray token. A `None` false form
# means "off" is simply the default (omit the flag).
BOOL_FLAGS = {
    "--cpu-moe":            ("--cpu-moe", None),
    "--spec-draft-cpu-moe": ("--spec-draft-cpu-moe", None),
    "--repack":             ("--repack", "--no-repack"),
    "--kv-offload":         ("--kv-offload", "--no-kv-offload"),
    "--no-host":            ("--no-host", None),
}


def extend_cli_flags(cmd: list, flags: dict) -> None:
    '''Append a config flag dict to an argv list, handling bare bool switches.'''
    for key, value in flags.items():
        if key in BOOL_FLAGS:
            on, off = BOOL_FLAGS[key]
            v = str(value).strip().lower()
            if v in ("on", "true", "1", "yes"):
                cmd.append(on)
            elif v in ("off", "false", "0", "no", ""):
                if off:
                    cmd.append(off)
            else:
                raise ValueError(f"{key}: expected on/off, got {value!r}")
        else:
            cmd.extend([key, str(value)])

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


# ---------------------------------------------------------------------------
# standardized-workload objective
# ---------------------------------------------------------------------------
# Raw wall time is a bad objective for two reasons the operator flagged:
#   1. a model that emits EOS at 800 tokens finishes early and "wins" without
#      doing the requested 4096 tokens of work;
#   2. the PP rate decays as the context fills AND decode slows as the KV cache
#      grows, so runs at different output lengths / prompt lengths are not
#      comparable.
# We instead score every launch on the SAME reference workload:
#
#     T* = P(L0) + G(L0, N0)
#
# P(L0) is read off the measured PP decay curve (interpolated at L0 prompt
# tokens). G(L0, N0) is the generation time extrapolated to N0 tokens at
# reference context L0. With `timings_per_token` the driver records a
# per-token decode curve, so G is a direct least-squares fit of per-step decode
# time vs context (the "full-fat" path); without it, G scales the observed mean
# decode step by the context-midpoint ratio (the fallback). Nothing is
# discarded: short outputs are scaled UP to N0, and low-confidence
# extrapolations are flagged, never pruned.

OBJECTIVE_DEFAULTS = {
    "prompt_tokens": 32768,     # L0: reference prompt length
    "output_tokens": None,      # N0: reference generation length; None -> ntg
    "gen_fit_tail_steps": 64,   # OLS window (last N decode steps)
    "gen_fit_min_steps": 16,    # below this the stream fit is low-confidence
    "min_gen_tokens": 512,      # observed output below this is low-confidence
    "max_pp_extrapolation_ratio": 1.25,  # L0 / measured prompt length flag
}


def objective_config(config: dict, ntg: int) -> dict:
    '''Objective settings merged over OBJECTIVE_DEFAULTS (output_tokens->ntg).'''

    obj = dict(OBJECTIVE_DEFAULTS)
    obj.update(config.get("objective") or {})
    if obj.get("output_tokens") in (None, "", 0):
        obj["output_tokens"] = int(ntg)
    for key in ("prompt_tokens", "output_tokens", "gen_fit_tail_steps",
                "gen_fit_min_steps", "min_gen_tokens"):
        obj[key] = int(obj[key])
    obj["max_pp_extrapolation_ratio"] = float(obj["max_pp_extrapolation_ratio"])
    if obj["prompt_tokens"] < 1 or obj["output_tokens"] < 0:
        raise ValueError("objective.prompt_tokens must be >= 1 and "
                         "output_tokens >= 0 (0 = pure PP)")
    if obj["gen_fit_tail_steps"] < 2 or obj["gen_fit_min_steps"] < 2:
        raise ValueError("objective gen_fit_tail_steps / gen_fit_min_steps "
                         "must be >= 2")
    return obj


def pp_time_at(points: list, target_n: int):
    '''Seconds to process `target_n` prompt tokens from the measured curve.

    Returns (seconds_or_None, extrapolated_bool). Inside the observed range
    this is piecewise-linear interpolation through (0,0) and the progress
    points; beyond the last point it extends the final segment's marginal rate.
    '''

    pts = [(p["n_tokens_cum"], p["t_cum_s"]) for p in (points or [])
           if p.get("n_tokens_cum") is not None and p.get("t_cum_s") is not None]
    if not pts:
        return None, True
    by_n = {0: 0.0}
    for n, t in pts:
        by_n[n] = t
    ordered = sorted(by_n.items())

    if target_n <= ordered[-1][0]:
        for (n0, t0), (n1, t1) in zip(ordered, ordered[1:]):
            if n0 <= target_n <= n1:
                if n1 == n0:
                    return t0, False
                frac = (target_n - n0) / (n1 - n0)
                return t0 + frac * (t1 - t0), False
        return ordered[-1][1], False

    (n0, t0), (n1, t1) = ordered[-2], ordered[-1]
    if t1 <= t0 or n1 <= n0:
        return None, True
    rate = (n1 - n0) / (t1 - t0)          # tokens per second, final segment
    return t1 + (target_n - n1) / rate, True


def normalize_gen_curve(curve) -> list:
    '''Dedup + sort the driver's [[predicted_n, predicted_ms], ...] snapshots.

    Returns a list of (n_tokens, t_ms) with n ascending; a repeated n keeps the
    last snapshot.
    '''

    by_n = {}
    for pt in curve or []:
        try:
            n = int(pt[0])
            t = float(pt[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if n >= 1 and t >= 0:
            by_n[n] = t
    return sorted(by_n.items())


def _ols(xs: list, ys: list):
    '''Ordinary least squares -> (intercept, slope), or None if degenerate.'''
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def standardized_objective(parsed: dict, response: dict, config: dict,
                           ntg: int) -> dict:
    '''Score one launch as T* for the reference workload + audit diagnostics.

    `parsed` is `parse_server_log()` output, `response` the driver's `response`
    dict (HTTP counts, finish_reason, timings, gen_curve). Returns a flat dict
    that is safe to stamp on rows: `objective_s` (T*, seconds, lower better),
    `objective_mode` ("stream-fit" | "linear-midpoint"), a low-confidence flag
    with reasons, and the intermediate P/G numbers so a reviewer can re-derive
    the score offline.
    '''

    obj = objective_config(config, ntg)
    L0 = obj["prompt_tokens"]
    N0 = obj["output_tokens"]
    resp = response or {}
    parsed = parsed or {}
    points = parsed.get("points") or []
    ge = parsed.get("gen_eval") or {}
    timings = resp.get("timings") or {}
    curve = normalize_gen_curve(resp.get("gen_curve"))

    diag = {
        "objective_mode": None,
        "objective_s": None,
        "objective_low_confidence": False,
        "objective_low_confidence_reasons": [],
        "objective_ref_prompt_tokens": L0,
        "objective_ref_gen_tokens": N0,
        "objective_pp_s": None,
        "objective_gen_s": None,
        "objective_pp_extrapolated": False,
        "objective_gen_observed_tokens": None,
        "objective_gen_fit_steps": 0,
        "objective_gen_fit_a_ms": None,
        "objective_gen_fit_b_ms_per_ctx": None,
        "objective_gen_scale": None,
        "finish_reason": resp.get("finish_reason"),
    }
    reasons = diag["objective_low_confidence_reasons"]

    # --- P(L0): prompt processing on the measured decay curve ---------------
    pp_s, pp_extrapolated = pp_time_at(points, L0)
    diag["objective_pp_extrapolated"] = bool(pp_extrapolated)
    if pp_s is None:
        reasons.append("no PP curve parsed")
    else:
        diag["objective_pp_s"] = round(pp_s, 4)
        if pp_extrapolated and points:
            observed_n = points[-1]["n_tokens_cum"]
            if observed_n > 0 \
                    and L0 / observed_n > obj["max_pp_extrapolation_ratio"]:
                reasons.append(
                    f"P({L0}) extrapolates well beyond the measured "
                    f"{observed_n} prompt tokens")

    # observed prompt length: HTTP count first, then the server summary line
    L_obs = (resp.get("prompt_tokens")
             or (parsed.get("prompt_eval") or {}).get("n_tokens") or L0)
    n_seen = max(
        curve[-1][0] if curve else 0,
        int(timings.get("predicted_n") or 0),
        int(ge.get("n_tokens") or 0),
    )
    diag["objective_gen_observed_tokens"] = n_seen

    # --- G(L0, N0): generation, fitted if a per-token curve exists ----------
    gen_s = None
    if N0 <= 1:                          # pure-PP reference (0 or 1 token:
        gen_s = 0.0                      # the first token is free)
        diag["objective_mode"] = "pp-only"
    elif len(curve) >= 2:
        # per-token decode cost (ms) at context position L_obs + n
        deltas = []
        for (n_prev, t_prev), (n, t) in zip(curve, curve[1:]):
            dn = n - n_prev
            if dn < 1:
                continue
            dt = (t - t_prev) / dn
            if dt < 0:
                continue
            deltas.append((n, dt))
        if len(deltas) >= 2:
            tail = deltas[-obj["gen_fit_tail_steps"]:]
            fit = _ols([L_obs + n for n, _ in tail], [dt for _, dt in tail])
            if fit is None:
                reasons.append("generation fit needs >= 2 distinct context points")
            else:
                a, b = fit
                if b < 0:                       # decode cannot get faster
                    b, a = 0.0, sum(dt for _, dt in tail) / len(tail)
                if a + b * (L_obs + 1) <= 0:    # keep per-step cost positive
                    a = max(1e-6, sum(dt for _, dt in tail) / len(tail)
                            - b * (L_obs + 1))
                # step 1 is the "free" token (logits come from prompt eval)
                gen_ms = sum(max(0.0, a + b * (L0 + k))
                             for k in range(2, N0 + 1))
                gen_s = gen_ms / 1000.0
                diag["objective_mode"] = "stream-fit"
                diag["objective_gen_fit_steps"] = len(tail)
                diag["objective_gen_fit_a_ms"] = round(a, 6)
                diag["objective_gen_fit_b_ms_per_ctx"] = round(b, 9)
                if len(tail) < obj["gen_fit_min_steps"]:
                    reasons.append(
                        f"only {len(tail)} generation-fit steps "
                        f"(< {obj['gen_fit_min_steps']})")

    if gen_s is None:
        # fallback: scale the observed mean decode step by the context-midpoint
        # ratio (decode cost ~ KV length), from L_obs+n_obs/2 to L0+N0/2
        n_obs = n_seen
        t_ms = ge.get("t_ms") or timings.get("predicted_ms") or 0.0
        if not t_ms and len(curve) >= 2:
            t_ms = curve[-1][1] - curve[0][1]
            n_obs = curve[-1][0] - curve[0][0]
        if n_obs > 1 and t_ms > 0:
            dt = t_ms / (n_obs - 1)
            denom = L_obs + n_obs / 2.0
            scale = (L0 + N0 / 2.0) / denom if denom > 0 else 1.0
            gen_s = dt * max(N0 - 1, 0) * scale / 1000.0
            diag["objective_mode"] = "linear-midpoint"
            diag["objective_gen_scale"] = round(scale, 6)
            reasons.append("no usable per-token generation curve; used the "
                           "context-midpoint extrapolation")
        else:
            reasons.append("no generation timing available")

    if gen_s is not None:
        diag["objective_gen_s"] = round(gen_s, 4)

    # --- confidence flags ---------------------------------------------------
    if N0 > 1 and n_seen and n_seen < obj["min_gen_tokens"]:
        reasons.append(f"only {n_seen} generated tokens "
                       f"(< {obj['min_gen_tokens']}); extrapolating to {N0}")
    if N0 > 1 and str(diag["finish_reason"]).lower() == "stop":
        reasons.append("generation ended early (finish_reason=stop)")

    if pp_s is not None and gen_s is not None:
        diag["objective_s"] = round(pp_s + gen_s, 4)
    diag["objective_low_confidence"] = bool(reasons)
    return diag


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
    the model file is constructed as `<model_base>-<quant>.gguf`. An optional
    `model_files` dict maps quant -> explicit filename, needed for sharded
    GGUFs (pass the 00001-of-NNNNN shard; llama.cpp loads the siblings).
    Otherwise a single entry from a plain `model_file` (quant = the file stem).
    '''

    base = config.get("model_base")
    quants = config.get("quants")

    if isinstance(quants, str):
        quants = [q.strip() for q in quants.split(",") if q.strip()]

    if base and quants:
        files = config.get("model_files") or {}
        missing = [q for q in quants if q not in files] if files else []
        if missing:
            raise ValueError(f"model_files missing quant(s): {', '.join(missing)}")
        return [{"quant": q,
                 "model_file": str(files[q]) if files else f"{base}-{q}.gguf"}
                for q in quants]

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
        "--warmup",               # explicit pin; upstream default is already on
        "--parallel", "1",        # one slot; controlled KV memory
        "--alias", SERVER_ALIAS,
    ]

    # The "prompt processing" progress line (the decay curve) is gated behind
    # log-verbosity; use trace (5) so it is always present -- the log goes to
    # a file, so the extra verbosity is free. Production runs at verbosity 3
    # and still shows the line; lower it via the config to cut log volume.
    if "--log-verbosity" not in cell and "--log-verbosity" not in spec_flags:
        cmd.extend(["--log-verbosity", "5"])

    extend_cli_flags(cmd, cell)
    extend_cli_flags(cmd, spec_flags)

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
    #
    # STREAMING with timings_per_token: every SSE chunk then carries a
    # `timings` snapshot (predicted_n / predicted_ms), so the host can fit the
    # decode-time-vs-context curve instead of trusting one average.
    # stream_options.include_usage puts the final usage in the closing chunk.
    try:
        with open(cfg["prompt_file"], "r", encoding="utf-8") as f:
            prompt = f.read()
        payload = {
            "model": cfg["model_name"],
            "prompt": prompt,
            "max_tokens": cfg["ntg"],
            "temperature": 0,
            "stream": True,
            "timings_per_token": True,
            "stream_options": {"include_usage": True},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=data,
            headers={"Content-Type": "application/json"})
        chunks = []
        raw_all = []
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=cfg["request_timeout"]) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace")
                raw_all.append(line)
                s = line.strip()
                if not s.startswith("data:"):
                    continue
                body = s[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunks.append(json.loads(body))
                except Exception:
                    pass
        wall = time.time() - t0
        if not chunks:
            # a build that ignores `stream` returns one plain JSON object
            try:
                chunks = [json.loads("".join(raw_all))]
            except Exception:
                chunks = []
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

    # fold the SSE chunks into the raw response fields we keep
    curve = {}
    finish_reason = None
    usage = {}
    final_timings = {}
    for c in chunks:
        tm = c.get("timings") or {}
        if tm:
            final_timings = tm
            pn, pm = tm.get("predicted_n"), tm.get("predicted_ms")
            if pn is not None and pm is not None:
                try:
                    curve[int(pn)] = round(float(pm), 3)
                except (TypeError, ValueError):
                    pass
        choices = c.get("choices") or []
        if choices:
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
        if c.get("usage"):
            usage = c["usage"]
    if not usage:
        # no include_usage support -> derive the counts from the final timings
        usage = {
            "prompt_tokens": final_timings.get("prompt_n"),
            "completion_tokens": final_timings.get("predicted_n"),
        }

    write_result({
        "status": "ok",
        "response": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish_reason,
            "wall_request_s": round(wall, 3),
            "timings": final_timings or None,
            "gen_curve": [[n, curve[n]] for n in sorted(curve)],
            "stream_chunks": len(chunks),
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

    resp = result.get("response") or {}
    timings = resp.get("timings") or {}
    gen_curve = resp.get("gen_curve") or []
    effective_ntg = int(ntg if ntg is not None else config.get("ntg", 1))

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
    common["finish_reason"] = resp.get("finish_reason")
    common["wall_request_s"] = resp.get("wall_request_s")
    common["timings_predicted_n"] = timings.get("predicted_n")
    common["timings_predicted_ms"] = timings.get("predicted_ms")
    common["gen_curve_points"] = len(gen_curve)
    # standardized-workload objective T* + its audit fields (see
    # standardized_objective); `wall_request_s` is kept alongside as raw data
    common.update(standardized_objective(parsed, resp, config, effective_ntg))

    # The per-launch summary fields (pp_tps, gen_tps, wall_request_s, T*, ...)
    # live in `common` and are stamped on EVERY point row, so the last point row
    # carries the whole-prompt summary — no separate summary row is needed. The
    # full per-token generation curve (up to `ntg` entries) is written ONLY on
    # that last row, to keep the other rows small.
    last = len(parsed["points"])
    for i, p in enumerate(parsed["points"], start=1):
        row = dict(common)
        row["point"] = i
        row.update(p)
        if i == last and gen_curve:
            row["gen_curve"] = gen_curve
        result_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    return launch_counter, common

