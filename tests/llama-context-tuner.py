#!/usr/bin/env python3
'''Find the largest safe ``--ctx-size`` for a model, given all other server flags.

The model-specific config supplies the complete ``llama-server`` argument set
(``server_parameters``: threads, device, GPU layers, split mode, cache dtypes,
speculative decoding, ...), the GGUF file, and the model's NATIVE context
limit (``native_ctx``). This tool only varies ``--ctx-size``:

  1. Exponential probe upward from ``min_ctx`` (doubling by default), capped at
     ``native_ctx``. A launch that loads and completes the probe request is a
     success; an OOM / failed allocation is a failure. Any OTHER error (bad
     flag, missing file, VRAM gate) aborts the run -- it is not evidence about
     context size.
  2. Bisect between the last good and first bad context down to
     ``ctx_granularity`` tokens. If ``native_ctx`` itself succeeds, the model
     is reported as reaching its native limit (no OOM found).
  3. Re-launch at the winning context with the config's LONG ``prompt_file``
     (truncated to fit) and record the prompt-preprocessing rate-vs-occupancy
     decay curve from the server log.

Everything runs through ``llama_server_lib.run_bench_launch``: the same
Dockerized ``llama-server``, the same VRAM gate / stale-container cleanup and
the same production stop/restart guard as the Optuna tuner.

⚠️  On pyrite this stops the production containers (default ``llama-cuda``) for
the duration of the search and restarts them afterwards (including on a crash
or Ctrl-C). If you are running Claude Code through the local LiteLLM router,
that session's models go offline while this runs.

Output: ``tests/results/llama-context-tuner/<model>_<timestamp>.jsonl`` (one
row per probe) and a ``.summary.json`` next to it carrying ``max_safe_ctx``,
``reached_native_max`` and the PP decay curve at the winning context.
'''

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("llama_server_lib",
                                               HERE / "llama_server_lib.py")
LLM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LLM)

# A context too large for the KV cache fails at load / allocation with one of
# these. Anything else is a real error and must not be mistaken for "ctx too
# big" (that would silently search the wrong boundary).
OOM_RE = re.compile(
    r"out of memory|bad_alloc|failed to allocate|could not allocate|"
    r"insufficient (?:cuda |gpu )?memory|not enough (?:cuda |gpu )?memory|"
    r"cudaerror|cuda error|cudaMalloc|hip error|"
    r"ggml_cuda[^\n]*(?:error|fail)|"
    r"(?:cache|kv)[^\n]*(?:too large|exceeds)|"
    r"exceeds[^\n]*(?:memory|context|n_ctx)|"
    r"n_ctx[^\n]*(?:too large|exceeds)",
    re.IGNORECASE,
)

DEFAULT_PROBE_PROMPT = "Hello world. " * 32


def classify(result: dict) -> tuple[str, str]:
    '''Map a launch result to ("ok" | "oom" | "error", reason).'''
    if result.get("status") == "ok":
        return "ok", ""
    text = "\n".join(str(result.get(k) or "")
                     for k in ("error", "server_log"))
    if OOM_RE.search(text):
        return "oom", "OOM / allocation failure at this context size"
    return "error", str(result.get("error") or "unknown launch error")


def align(value: int, granularity: int) -> int:
    '''Round a token count down to a multiple of ``granularity`` (>= g).'''
    return max(granularity, (int(value) // granularity) * granularity)


def probe_sequence(min_ctx: int, native_ctx: int, multiplier: int,
                   granularity: int) -> list:
    '''The exponential ladder (for --dry-run / logging only).'''
    out = []
    ctx = align(min_ctx, granularity)
    while True:
        out.append(min(ctx, native_ctx))
        if ctx >= native_ctx:
            break
        ctx = min(ctx * multiplier, native_ctx)
    return out


def curve_from_log(result: dict) -> list:
    '''[[n_tokens_cum, tps_seg], ...] for the launch, if the log parsed.'''
    parsed = LLM.parse_server_log(result.get("server_log") or "")
    if not parsed:
        return []
    return [[p["n_tokens_cum"], p["tps_seg"]] for p in parsed["points"]]


def launch_ctx(config: dict, ctx: int, prompt_file: str, launch_id: int):
    '''One launch at a fixed context, with all other flags from the config.'''
    cell = dict(config.get("server_parameters") or {})
    cell["--ctx-size"] = str(ctx)
    return LLM.run_bench_launch(
        config, cell, {}, prompt_file, launch_id,
        config["model_file"], ntg=int(config.get("ntg", 1)))


def truncated_prompt(src: Path, workdir: Path, max_ctx: int,
                     ntg: int, chars_per_token: float) -> Path:
    '''Copy the long prompt into ``workdir``, cut to fit ``max_ctx`` tokens.

    The server reports the true token count; this is only a sizing estimate
    (the same chars/token ratio ``make-bench-prompt.py`` uses) with a margin so
    the request is not rejected for being a few tokens too long.
    '''
    budget = max(256, int((max_ctx - ntg - 512) * chars_per_token))
    text = src.read_text(encoding="utf-8")
    out = workdir / "prompt.txt"
    out.write_text(text[:budget], encoding="utf-8")
    return out


def find_max_ctx(config: dict, probe_prompt: Path, result_file, summary: dict,
                 launch_counter: int = 0):
    '''Exponential probe + bisection. Returns (launch_counter, max_safe_ctx,
    first_bad_ctx, reached_native_max, last_good_result).'''
    native = int(config["native_ctx"])
    granularity = int(config.get("ctx_granularity", 1024))
    multiplier = int(config.get("ctx_multiplier", 2))
    ctx = align(int(config.get("min_ctx", 4096)), granularity)

    last_good_ctx, first_bad_ctx = None, None
    last_good_result = None
    reached_native = False

    def record(phase, this_ctx, status, result, reason=""):
        nonlocal launch_counter
        launch_counter += 1
        parsed = LLM.parse_server_log(result.get("server_log") or "") \
            if status == "ok" else None
        row = {
            "model": config["model_name"], "phase": phase,
            "launch": launch_counter, "ctx": this_ctx, "status": status,
        }
        if parsed:
            row.update({
                "n_tokens_prompt": parsed["prompt_eval"]["n_tokens"]
                                   if parsed["prompt_eval"] else None,
                "pp_tps": parsed["prompt_eval"]["tps"]
                          if parsed["prompt_eval"] else None,
                "pp_auc": round(sum(p["tps_seg"] for p in parsed["points"]), 3),
                "curve": [[p["n_tokens_cum"], p["tps_seg"]]
                          for p in parsed["points"]],
            })
        if reason:
            row["reason"] = reason
        if status != "ok":
            row["error"] = str(result.get("error") or "")[:2000]
            row["server_log_tail"] = (result.get("server_log") or "")[-2000:]
        result_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        result_file.flush()
        print(f"  [{phase:11}] ctx={this_ctx:<8} {status}"
              + (f"  ({reason})" if reason else "")
              + (f"  pp_auc={row['pp_auc']}" if "pp_auc" in row else ""))
        return row

    # ---- exponential probe ----
    while True:
        result = launch_ctx(config, ctx, str(probe_prompt), launch_counter + 1)
        status, reason = classify(result)
        record("probe", ctx, status, result, reason)
        if status == "ok":
            last_good_ctx, last_good_result = ctx, result
            if ctx >= native:
                reached_native = True
                break
            ctx = min(ctx * multiplier, native)
            ctx = align(ctx, granularity) if ctx < native else native
            continue
        if status == "oom":
            first_bad_ctx = ctx
            break
        raise RuntimeError(f"launch at ctx={ctx} failed for a non-OOM reason: "
                           f"{reason}")

    # ---- bisection ----
    while (first_bad_ctx is not None and last_good_ctx is not None
           and first_bad_ctx - last_good_ctx > granularity):
        mid = align((last_good_ctx + first_bad_ctx) // 2, granularity)
        if mid <= last_good_ctx or mid >= first_bad_ctx:
            break
        result = launch_ctx(config, mid, str(probe_prompt), launch_counter + 1)
        status, reason = classify(result)
        record("bisect", mid, status, result, reason)
        if status == "ok":
            last_good_ctx, last_good_result = mid, result
        elif status == "oom":
            first_bad_ctx = mid
        else:
            raise RuntimeError(f"launch at ctx={mid} failed for a non-OOM "
                               f"reason: {reason}")

    summary.update({
        "native_ctx": native,
        "max_safe_ctx": last_good_ctx,
        "first_bad_ctx": first_bad_ctx,
        "reached_native_max": reached_native,
        "ctx_granularity": granularity,
    })
    return launch_counter, last_good_ctx, first_bad_ctx, reached_native, \
        last_good_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the max safe --ctx-size for a model")
    parser.add_argument("--config", required=True,
                        help="model-specific JSON config")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs, print the probe ladder, "
                             "no container launches")
    parser.add_argument("--no-stop-production", action="store_true",
                        help="do not stop/restart the production containers")
    parser.add_argument("--min-ctx", type=int, default=None,
                        help="override config min_ctx")
    parser.add_argument("--native-ctx", type=int, default=None,
                        help="override config native_ctx")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    if args.min_ctx is not None:
        config["min_ctx"] = args.min_ctx
    if args.native_ctx is not None:
        config["native_ctx"] = args.native_ctx

    for key in ("model_name", "model_file", "native_ctx", "server_parameters"):
        if key not in config:
            parser.error(f"config is missing required key {key!r}")

    config.setdefault("models_path", str(ROOT / "models"))
    config.setdefault("image", LLM.IMAGE_DEFAULT)
    config.setdefault("ntg", 1)
    # We are deliberately probing the VRAM boundary; the gate must not refuse
    # launches merely because free VRAM is lower than a tuning-oriented floor.
    config.setdefault("min_free_vram_gib", 1)
    if args.no_stop_production:
        config["stop_production"] = False

    model_host = Path(config["models_path"]) / config["model_file"]
    prompt_cfg = Path(config["prompt_file"]).expanduser() \
        if config.get("prompt_file") else None
    long_prompt = (prompt_cfg if prompt_cfg.is_absolute() else ROOT / prompt_cfg
                   ).resolve() if prompt_cfg else None
    if not model_host.is_file():
        parser.error(f"model file not found: {model_host}")
    if long_prompt and not long_prompt.is_file():
        parser.error(f"prompt file not found: {long_prompt}")

    native = int(config["native_ctx"])
    min_ctx = int(config.get("min_ctx", 4096))
    granularity = int(config.get("ctx_granularity", 1024))
    multiplier = int(config.get("ctx_multiplier", 2))
    if min_ctx < granularity or min_ctx > native:
        parser.error(f"min_ctx must be between {granularity} and native_ctx "
                     f"({native}), got {min_ctx}")
    if multiplier < 2:
        parser.error("ctx_multiplier must be >= 2")

    print(f"llama-context-tuner: {config['model_name']}")
    print(f"  model     : {model_host}")
    print(f"  server    : {config['server_parameters']}")
    print(f"  native ctx: {native}  min_ctx={min_ctx}  "
          f"granularity={granularity}  multiplier={multiplier}")
    print(f"  probe ladder: {probe_sequence(min_ctx, native, multiplier, granularity)}")
    print(f"  long prompt: {long_prompt} (ntg={config['ntg']})")
    if config.get("stop_production", True):
        print(f"  ⚠️ stopping production containers "
              f"{LLM.production_container_names(config)} for the search")
    if args.dry_run:
        print("  (dry run: no launches)")
        return 0

    output_dir = ROOT / "tests/results/llama-context-tuner"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_file = output_dir / f"{config['model_name']}_{timestamp}.jsonl"
    summary = {
        "model": config["model_name"],
        "model_file": config["model_file"],
        "config": config,
        "result_file": str(result_file),
        "launches": [],
    }
    print(f"  results   : {result_file}")

    workdir = Path(tempfile.mkdtemp(prefix="ctxprobe_"))
    probe_prompt = workdir / "probe_prompt.txt"
    probe_prompt.write_text(str(config.get("probe_prompt",
                                           DEFAULT_PROBE_PROMPT)),
                            encoding="utf-8")

    LLM.cleanup_stale_bench()
    stopped = []
    try:
        stopped = LLM.stop_production_containers(config)
        with open(result_file, "a", encoding="utf-8") as f:
            n_launches, max_safe, first_bad, reached_native, last_good = \
                find_max_ctx(config, probe_prompt, f, summary)
            summary["n_launches"] = n_launches

            # ---- final characterization at the winning context ----
            curve, curve_ctx, curve_status = [], max_safe, "no-confirm"
            if max_safe and long_prompt:
                chars_per_token = float(config.get("chars_per_token", 3.07))
                confirm_prompt = truncated_prompt(
                    long_prompt, workdir, max_safe,
                    int(config["ntg"]), chars_per_token)
                result = launch_ctx(config, max_safe, str(confirm_prompt),
                                    n_launches + 1)
                status, reason = classify(result)
                curve = curve_from_log(result)
                curve_status = status
                parsed = LLM.parse_server_log(result.get("server_log") or "")
                summary["confirm"] = {
                    "ctx": max_safe, "status": status, "reason": reason,
                    "n_tokens_prompt": parsed["prompt_eval"]["n_tokens"]
                                       if parsed and parsed["prompt_eval"]
                                       else None,
                    "pp_auc": round(sum(p["tps_seg"] for p in parsed["points"]), 3)
                              if parsed else None,
                    "error": str(result.get("error") or "")[:2000],
                }
                f.write(json.dumps({
                    "model": config["model_name"], "phase": "confirm",
                    "launch": n_launches + 1,
                    "ctx": max_safe, "status": status,
                    "curve": curve, "reason": reason,
                    "error": str(result.get("error") or "")[:2000],
                }, separators=(",", ":")) + "\n")

            summary["pp_curve"] = curve
            summary["pp_curve_ctx"] = curve_ctx
            summary["pp_curve_status"] = curve_status
            if not curve and last_good:
                # Fall back to the last successful probe's curve.
                summary["pp_curve"] = curve_from_log(last_good)
                summary["pp_curve_status"] = "probe-fallback"
    finally:
        LLM.restore_production_containers(stopped)
        shutil.rmtree(workdir, ignore_errors=True)

    summary_file = result_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMax safe context: {summary.get('max_safe_ctx')} "
          f"(native={native}, "
          f"{'native limit reached' if summary.get('reached_native_max') else 'OOM boundary'})")
    print(f"Wrote results: {result_file}")
    print(f"Wrote summary: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
