#!/usr/bin/env python3
"""
context_fit.py

Targeted context-fit benchmark runner for llama.cpp / llama-bench.

Purpose:
1. Accept model and GPUs as arguments.
2. Run one benchmark per context size, while tracking peak VRAM usage.
3. Save structured output and full logs.

Workflow:
- Coarse phase: fixed context scan through COARSE_CONTEXT_SIZES until first failure.
  If the largest coarse size succeeds, that is the maximum and no bisection runs.
- Bisection phase: bracketed halving search between last success and first failure,
  using step size --refine-step (default 1024), until the bracket narrows to one step.
- Verification phase: the candidate max context is re-run --verify-runs times.
  Any single failure marks that context as unstable and unfit for production use.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from helper_funcs.context_fit_io import (
    append_log,
    append_warning,
    service_is_active,
    start_service,
    stop_service,
    write_csv,
    write_plot_png,
    write_summary_json,
)
from helper_funcs.context_fit_runner import run_one_context, verify_boundary
from helper_funcs.context_fit_types import QuantRunSummary, RunResult
from helper_funcs.context_fit_utils import (
    coarse_sizes_for_max,
    get_stable_success_contexts,
    kv_label_for_type,
    midpoint_in_bracket,
    normalize_score_breakpoints,
    parse_devices,
    read_total_vram_mib,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_BIN = Path("/opt/llama.cpp/build/bin/llama-bench")
DEFAULT_RESULTS_DIR = REPO_ROOT / "tests" / "results" / "context_fit"
DEFAULT_CONTEXT_FIT_CONFIG = REPO_ROOT / "tests" / "config" / "context_fit" / "context_fit.yaml"
DEFAULT_SCORE_BREAKPOINTS: list[tuple[str, float]] = [
    ("interactive", 4.0),
    ("batch", 0.5),
    ("exclude", 0.0),
]
COARSE_CONTEXT_SIZES = [32768, 65536, 131072, 262144]


def load_context_fit_config(
    config_path: Optional[Path]
) -> tuple[dict[str, object], list[tuple[str, float]], Optional[Path]]:
    '''Loads the optional YAML config for benchmark defaults and score thresholds.'''

    if config_path is None:
        return {}, list(DEFAULT_SCORE_BREAKPOINTS), None

    if not config_path.exists():
        if config_path == DEFAULT_CONTEXT_FIT_CONFIG:
            return {}, list(DEFAULT_SCORE_BREAKPOINTS), None

        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw, dict):
        print(f"ERROR: config file must contain a YAML mapping: {config_path}")
        sys.exit(1)

    run_config = raw.get("run") if isinstance(raw.get("run"), dict) else raw
    if not isinstance(run_config, dict):
        print(f"ERROR: config run section must be a YAML mapping: {config_path}")
        sys.exit(1)

    breakpoints_raw = raw.get("score_breakpoints", run_config.get("score_breakpoints"))

    try:
        breakpoints = normalize_score_breakpoints(breakpoints_raw, DEFAULT_SCORE_BREAKPOINTS)

    except (TypeError, ValueError) as exc:
        print(f"ERROR: invalid score_breakpoints in {config_path}: {exc}")
        sys.exit(1)

    return run_config, breakpoints, config_path


def resolve_model_path(model_arg: str) -> Path:
    '''Resolves a model path argument to an absolute Path, checking various locations.'''

    model_path = Path(model_arg).expanduser()

    if not model_path.is_absolute():
        if len(model_path.parts) == 1:

            # Bare filename: check repo models/ first, then /opt/models.
            local = REPO_ROOT / "models" / model_path

            if local.exists():
                return local

            model_path = Path("/opt/models") / model_path

        else:
            model_path = (REPO_ROOT / model_path).resolve()

    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        print(
            "Hint: pass an absolute path, a bare filename (checked " +
            "in <repo>/models/ then /opt/models/), or a relative path from the repo root."
        )

        sys.exit(1)

    return model_path


def load_model_list(path: Path) -> list[tuple[Path, int]]:
    '''Loads (model_path, max_context) pairs from a two-column CSV (model,max_context).
    The header row is detected and skipped automatically.
    Empty lines and lines starting with # are ignored.'''

    models: list[tuple[Path, int]] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                print(f"WARNING: skipping malformed line in {path}: {line!r}")
                continue

            model_name, max_ctx_str = parts[0].strip(), parts[1].strip()

            try:
                max_ctx = int(max_ctx_str)
            except ValueError:
                # Header row or non-integer value; skip
                continue

            models.append((resolve_model_path(model_name), max_ctx))

    if not models:
        print(f"ERROR: no models found in {path}")
        sys.exit(1)

    return models


def parse_args() -> argparse.Namespace:
    '''Parses command-line arguments for the context-fit benchmark runner.'''

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file with run settings and score breakpoints",
    )

    bootstrap_args, _ = bootstrap.parse_known_args()
    config_path = bootstrap_args.config or (
        DEFAULT_CONTEXT_FIT_CONFIG if DEFAULT_CONTEXT_FIT_CONFIG.exists() else None
    )
    run_config, score_breakpoints, config_path = load_context_fit_config(config_path)

    parser = argparse.ArgumentParser(
        parents=[bootstrap],
        description=(
            "Run context-size fit benchmarks one context at a time, track peak VRAM, "
            "and refine the fail boundary with bisection probes."
        )
    )
    parser.add_argument(
        "--model",
        default=run_config.get("model"),
        help="Model path or filename (mutually exclusive with --model-list)"
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=Path(run_config["model_list"]) if run_config.get("model_list") else None,
        help=(
            "Path to a text file listing one model path or filename per line "
            "(mutually exclusive with --model)"
        )
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=int(run_config.get("max_context", 262144)),
        help=(
            "Maximum context for single-model runs (no --model-list). "
            "The coarse sweep is derived as max//8, max//4, max//2, max"
        ),
    )
    parser.add_argument(
        "--gpus",
        default=run_config.get("gpus"),
        help="CUDA_VISIBLE_DEVICES string, e.g. '1,2'"
    )
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=Path(run_config.get("bench_bin", DEFAULT_BENCH_BIN)),
        help="Path to llama-bench binary"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(run_config.get("results_dir", DEFAULT_RESULTS_DIR)),
        help="Output directory"
    )
    parser.add_argument(
        "--run-name",
        default=run_config.get("run_name"),
        help="Output run label (default: timestamp + model name)"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=int(run_config.get("n_gpu_layers", 99))
    )
    parser.add_argument(
        "--split-mode",
        default=str(run_config.get("split_mode", "layer"))
    )
    parser.add_argument(
        "--tensor-split",
        default=str(run_config.get("tensor_split", "1/1")),
        help=(
            "Per-device split ratio passed to llama-bench -ts. Use '/' between "
            "device ratios (e.g. '1/1'); llama-bench treats ',' as a delimiter "
            "between separate sweep configs, so '1,1' silently puts everything "
            "on device 0 instead of splitting across devices."
        ),
    )
    parser.add_argument(
        "--allow-host",
        action="store_true",
        default=bool(run_config.get("allow_host", False)),
        help=(
            "Allow llama-bench to spill beyond the GPUs into host RAM. "
            "By default this benchmark uses strict GPU-only fitting."
        ),
    )
    parser.add_argument(
        "--fit-target",
        type=int,
        default=int(run_config.get("fit_target", 0))
    )
    parser.add_argument(
        "--fit-ctx",
        type=int,
        default=int(run_config.get("fit_ctx", 0))
    )
    parser.add_argument(
        "--n-prompt",
        type=int,
        default=int(run_config.get("n_prompt", 512))
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=int(run_config.get("n_gen", 128))
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=int(run_config.get("repetitions", 3))
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(run_config.get("poll_interval", 0.25)),
        help="nvidia-smi poll interval in seconds"
    )
    parser.add_argument(
        "--flash-attn",
        choices=["on", "off", "auto"],
        default=str(run_config.get("flash_attn", "on")),
        help="Flash attention mode passed to llama-bench (default: on, matching llamacpp.service)",
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=int(run_config.get("refine_step", 1024)),
        help="Step size for bisection refinement contexts"
    )
    parser.add_argument(
        "--verify-runs",
        type=int,
        default=int(run_config.get("verify_runs", 3)),
        help=(
            "Number of confirmation runs at the final max "
            "context; any failure marks that context as unstable"
        )
    )
    parser.add_argument(
        "--max-run-seconds",
        type=int,
        default=int(run_config.get("max_run_seconds", 21600)),
        help=(
            "Hard wall-clock timeout per llama-bench invocation. "
            "Timed-out runs are marked failed (default: 21600)"
        )
    )
    parser.add_argument(
        "--kv-cache-types",
        default=str(run_config.get("kv_cache_types", "q4_0,q8_0,f16")),
        help=(
            "Comma-separated KV cache types for iterative runs. "
            "Defaults to q4_0,q8_0,f16"
        )
    )
    parser.add_argument(
        "--service-name",
        default=str(run_config.get("service_name", "llamacpp.service")),
        help="Systemd service to stop before run and restore after run"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=bool(run_config.get("skip_completed", False)),
        help=(
            "Skip models with an existing summary.json in the output directory. "
            "Useful for resuming interrupted multi-model runs."
        ),
    )
    parser.add_argument(
        "--stop-after-model",
        type=int,
        default=int(run_config.get("stop_after_model", 0)),
        help="Stop after N models in this invocation (0 means all models)",
    )
    parser.add_argument(
        "--no-manage-service",
        action="store_true",
        default=bool(run_config.get("no_manage_service", False)),
        help="Disable automatic stop/start of the llama.cpp service around the benchmark run",
    )

    args = parser.parse_args()

    if not args.gpus:
        print("ERROR: --gpus must be provided either on the command line or in the YAML config")
        sys.exit(1)

    if not args.model and not args.model_list:
        print(
            "ERROR: one of --model or --model-list is required, "
            "either on the command line or in the YAML config"
        )
        sys.exit(1)

    if args.max_context <= 0:
        print("ERROR: --max-context must be > 0")
        sys.exit(1)

    if args.stop_after_model < 0:
        print("ERROR: --stop-after-model must be >= 0")
        sys.exit(1)

    args.config = config_path
    args.score_breakpoints = score_breakpoints
    args.no_host = not args.allow_host

    return args


def _run_for_model(
    model_path: Path,
    run_name: str,
    kv_runs: list[tuple[str, str]],
    devices: list[int],
    args: argparse.Namespace,
    env: dict[str, str],
    coarse_sizes: list[int],
) -> None:
    '''Runs the context-fit benchmark for a single model, saving results and logs.'''

    args.model = model_path  # build_command reads args.model

    # Read total available VRAM once; used to decide whether stability verification is needed.
    total_vram_mib = read_total_vram_mib(devices)
    # verify only when within 1 GiB of limit
    verify_vram_threshold_mib = max(0, total_vram_mib - 1024)

    out_dir = args.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    log_path = out_dir / "run.log"

    rows: list[RunResult] = []
    quant_summaries: list[QuantRunSummary] = []
    warnings: list[str] = []

    print(f"\n{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Results: {out_dir}")

    for kv_label, kv_type in kv_runs:
        print(f"\n=== KV cache run: {kv_label} ({kv_type}) ===")

        tested_contexts: set[int] = set()
        first_fail_ctx: Optional[int] = None
        bisect_high: Optional[int] = None
        boundary_stable: Optional[bool] = None
        quant_warnings: list[str] = []

        # Phase 1: coarse scan
        for ctx in coarse_sizes:
            print(f"[{kv_label} coarse] ctx={ctx}")
            result = run_one_context(kv_label, kv_type, "coarse", ctx, args, env, devices)
            rows.append(result)
            tested_contexts.add(ctx)
            append_log(log_path, result, env)

            if result.status in ("failed_oom", "failed"):
                first_fail_ctx = ctx
                print(f"  Failure at ctx={ctx}; entering bisection refinement")
                break

        # If all coarse sizes succeeded, the largest is the max; run stability verification.
        if first_fail_ctx is None and coarse_sizes and coarse_sizes[-1] in tested_contexts:
            max_coarse = coarse_sizes[-1]

            print(
                f"[{kv_label}] Reached maximum tested context "
                f"({max_coarse}) successfully; skipping bisection")

            boundary_stable = verify_boundary(
                kv_label=kv_label,
                kv_type=kv_type,
                ctx=max_coarse,
                label="Max context",
                args=args,
                env=env,
                devices=devices,
                rows=rows,
                log_path=log_path,
                warnings=warnings,
                quant_warnings=quant_warnings,
                total_vram_mib=total_vram_mib,
                verify_vram_threshold_mib=verify_vram_threshold_mib,
            )

        # Phase 2: bisection refinement between last success and first fail
        quant_rows = [row for row in rows if row.kv_cache_type == kv_type]
        successful = [row for row in quant_rows if row.status == "ok"]

        if first_fail_ctx is not None and successful:
            low = max(row.context_size for row in successful)
            high = first_fail_ctx

            if low >= high:
                msg = (
                    f"[{kv_label}] Invalid refinement bracket: "
                    f"low={low}, high={high}. Skipping refinement."
                )

                print(f"WARNING: {msg}")
                warnings.append(msg)
                quant_warnings.append(msg)
                append_warning(log_path, msg)

            else:
                print(
                    f"[{kv_label}] Bisection refinement bracket: "
                    f"low={low} high={high} step={args.refine_step}"
                )

                while (high - low) > args.refine_step:
                    ctx = midpoint_in_bracket(low, high, args.refine_step)

                    if ctx in tested_contexts:
                        candidates = [
                            c
                            for c in range(low + args.refine_step, high, args.refine_step)
                            if c not in tested_contexts
                        ]

                        if not candidates:
                            break

                        ctx = candidates[len(candidates) // 2]

                    print(f"[{kv_label} refine] ctx={ctx}")
                    result = run_one_context(kv_label, kv_type, "refine", ctx, args, env, devices)
                    rows.append(result)
                    tested_contexts.add(ctx)
                    append_log(log_path, result, env)

                    if result.status == "ok":
                        low = ctx

                    else:
                        high = ctx

                bisect_high = high

                boundary_stable = verify_boundary(
                    kv_label=kv_label,
                    kv_type=kv_type,
                    ctx=low,
                    label="Boundary context",
                    args=args,
                    env=env,
                    devices=devices,
                    rows=rows,
                    log_path=log_path,
                    warnings=warnings,
                    quant_warnings=quant_warnings,
                    total_vram_mib=total_vram_mib,
                    verify_vram_threshold_mib=verify_vram_threshold_mib,
                )

        elif first_fail_ctx is not None and not successful:
            msg = (
                f"[{kv_label}] Coarse failure at first tested context={first_fail_ctx}; "
                "no successful lower bound is available, so bisection refinement is skipped."
            )

            print(f"WARNING: {msg}")
            warnings.append(msg)
            quant_warnings.append(msg)
            append_warning(log_path, msg)

        elif first_fail_ctx is not None:
            msg = (
                f"[{kv_label}] Refinement skipped: first tested context {first_fail_ctx} failed, "
                "so there is no successful lower bracket to bisect from."
            )

            print(f"WARNING: {msg}")
            warnings.append(msg)
            quant_warnings.append(msg)
            append_warning(log_path, msg)

        quant_rows = [row for row in rows if row.kv_cache_type == kv_type]
        successful = get_stable_success_contexts(quant_rows)

        quant_summaries.append(
            QuantRunSummary(
                kv_cache_label=kv_label,
                kv_cache_type=kv_type,
                max_success_ctx=successful[-1] if successful else None,
                first_fail_ctx=first_fail_ctx,
                bisect_fail_ctx=bisect_high,
                boundary_stable=boundary_stable,
                warnings=quant_warnings,
            )
        )

    write_csv(csv_path, rows, str(model_path), args.gpus)

    summary_path = out_dir / "summary.json"
    plot_path = out_dir / "plot.png"

    write_summary_json(
        summary_path,
        model=model_path.name,
        gpu_devices=args.gpus,
        coarse_sizes=coarse_sizes,
        score_breakpoints=args.score_breakpoints,
        rows=rows,
        quant_summaries=quant_summaries,
        warnings=warnings,
    )

    write_plot_png(plot_path, rows)

    print(f"\nDone: {model_path.name}")
    print(f"  CSV : {csv_path}")
    print(f"  Log : {log_path}")
    print(f"  JSON: {summary_path}")
    print(f"  Plot: {plot_path}")

    for summary in quant_summaries:
        stable_str = {
            True: "stable",
            False: "UNSTABLE",
            None: "not verified"
        }.get(summary.boundary_stable, "?")

        print(
            "  "
            f"{summary.kv_cache_label}: max_success_ctx={summary.max_success_ctx} "
            f"first_fail_ctx={summary.first_fail_ctx} bisect_fail_ctx={summary.bisect_fail_ctx} "
            f"boundary={stable_str}"
        )


def main() -> None:
    '''Main entry point for the context-fit benchmark runner. Parses arguments,
    manages service state, and runs benchmarks for specified models.'''

    args = parse_args()

    if not args.bench_bin.exists():
        print(f"ERROR: llama-bench not found: {args.bench_bin}")
        sys.exit(1)

    devices = parse_devices(args.gpus)

    kv_type_tokens = [token.strip() for token in args.kv_cache_types.split(",") if token.strip()]

    if not kv_type_tokens:
        print("ERROR: --kv-cache-types must include at least one cache type")
        sys.exit(1)

    kv_runs: list[tuple[str, str]] = [
        (kv_label_for_type(kv_type), kv_type) for kv_type in kv_type_tokens
    ]

    if args.model and args.model_list:
        print("ERROR: --model and --model-list are mutually exclusive")
        sys.exit(1)

    if not args.model and not args.model_list:
        print("ERROR: one of --model or --model-list is required")
        sys.exit(1)

    if args.model_list:
        models = load_model_list(args.model_list)

    else:
        models = [(resolve_model_path(args.model), args.max_context)]

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(f"CUDA_VISIBLE_DEVICES={args.gpus}")
    print("KV cache runs: " + ", ".join(f"{label}({kv_type})" for label, kv_type in kv_runs))
    print(f"Models to benchmark: {len(models)}")

    stopped_service = False
    should_restore_service = False

    try:
        if not args.no_manage_service:
            if service_is_active(args.service_name):
                print(f"Stopping active service: {args.service_name}")
                stop_service(args.service_name)
                stopped_service = True
                should_restore_service = True
            else:
                print(f"Service not active, no stop needed: {args.service_name}")

        ran_models = 0

        for model_path, max_ctx in models:
            coarse_sizes = (
                coarse_sizes_for_max(max_ctx) if max_ctx is not None else COARSE_CONTEXT_SIZES
            )
            run_name = (
                model_path.stem
                if args.model_list
                else (
                    args.run_name
                    or f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_path.stem}"
                )
            )

            out_dir = args.results_dir / run_name
            summary_path = out_dir / "summary.json"

            if args.skip_completed and summary_path.exists():
                print(
                    f"Skipping completed model: {model_path.name} "
                    f"(found {summary_path})"
                )
                continue

            _run_for_model(model_path, run_name, kv_runs, devices, args, env, coarse_sizes)
            ran_models += 1

            if args.stop_after_model > 0 and ran_models >= args.stop_after_model:
                print(
                    f"Stopping early after {ran_models} model(s) "
                    f"due to --stop-after-model={args.stop_after_model}"
                )
                break

    finally:
        if should_restore_service:
            print(f"Restoring service: {args.service_name}")

            try:
                start_service(args.service_name)

                if stopped_service:
                    print(f"Service restored successfully: {args.service_name}")

            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Failed to restore service {args.service_name}: {exc}")


if __name__ == "__main__":
    main()
