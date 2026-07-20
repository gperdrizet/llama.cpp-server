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
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import yaml

from helper_funcs.context_fit_utils import (
    aggregate_runtime_by_context,
    aggregate_runtime_by_phase,
    best_error_line,
    deployment_tier_for_score,
    detect_oom_like_failure,
    get_stable_success_contexts,
    mean_optional,
    midpoint_in_bracket,
    normalize_score_breakpoints,
    parse_llama_bench_csv,
    score_breakpoints_to_dict,
    select_deployment_rows,
    weighted_harmonic_mean,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_BIN = Path("/opt/llama.cpp/build/bin/llama-bench")
DEFAULT_RESULTS_DIR = REPO_ROOT / "tests" / "results" / "context-size"
DEFAULT_CONTEXT_FIT_CONFIG = REPO_ROOT / "tests" / "config" / "context_fit" / "context_fit.yaml"
DEFAULT_SCORE_BREAKPOINTS: list[tuple[str, float]] = [
    ("interactive", 4.0),
    ("batch", 0.5),
    ("exclude", 0.0),
]
COARSE_CONTEXT_SIZES = [32768, 65536, 131072, 262144]


def coarse_sizes_for_max(max_ctx: int) -> list[int]:
    '''Returns 4 context sizes for the coarse sweep: max//8, max//4, max//2, max.'''
    return [max_ctx >> 3, max_ctx >> 2, max_ctx >> 1, max_ctx]


def load_context_fit_config(config_path: Optional[Path]) -> tuple[dict[str, object], list[tuple[str, float]], Optional[Path]]:
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




@dataclass
class RunResult:
    '''Dataclass to hold the result of a single context-size run.'''
    kv_cache_label: str
    kv_cache_type: str
    phase: str
    context_size: int
    status: str
    return_code: int
    elapsed_s: float
    peak_vram_total_mib: int
    peak_vram_per_device: dict[int, int]
    command: list[str]
    pp_ts: Optional[float]
    pp_stddev_ts: Optional[float]
    tg_ts: Optional[float]
    tg_stddev_ts: Optional[float]
    stdout: str
    stderr: str


@dataclass
class QuantRunSummary:
    '''Dataclass to hold the summary of a all trials from a single quantization run.'''
    kv_cache_label: str
    kv_cache_type: str
    max_success_ctx: Optional[int]
    first_fail_ctx: Optional[int]
    bisect_fail_ctx: Optional[int]
    boundary_stable: Optional[bool]
    warnings: list[str]


class GpuPeakPoller:
    '''Polls nvidia-smi for peak VRAM usage on the specified devices at a given interval.'''

    def __init__(self, devices: list[int], interval_s: float):
        self.devices = devices
        self.interval_s = interval_s
        self.peak_used: dict[int, int] = {d: 0 for d in devices}
        self.total_mem: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_snapshot(self) -> None:
        '''Captures snapshot of current VRAM usage and updates peak values.'''
    
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0:
            return

        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]

            if len(parts) != 3:
                continue

            try:
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])

            except ValueError:
                continue

            if idx not in self.devices:
                continue

            if used > self.peak_used.get(idx, 0):
                self.peak_used[idx] = used

            self.total_mem[idx] = total

    def _loop(self) -> None:
        '''Thread loop to poll VRAM usage until stopped.'''

        while not self._stop.is_set():
            self._read_snapshot()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        '''Starts the polling thread.'''

        self._read_snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        '''Stops the polling thread and waits for it to finish.'''

        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._read_snapshot()


def parse_devices(text: str) -> list[int]:
    '''Parses a comma-separated list of GPU device indices into a list of integers.
    Raises ValueError if no valid indices are found.'''

    values = []

    for part in text.split(","):
        part = part.strip()

        if not part:
            continue

        values.append(int(part))

    if not values:
        raise ValueError("At least one GPU device index must be provided")

    return values




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


def build_command(args: argparse.Namespace, context_size: int, kv_cache_type: str) -> list[str]:
    '''Builds the command line for llama-bench based on the provided arguments,
    context size, and KV cache type.'''

    cmd = [
        str(args.bench_bin),
        "-m", str(args.model),
        "-ngl", str(args.n_gpu_layers),
        "-sm", args.split_mode,
        "--fit-target", str(args.fit_target),
        "--fit-ctx", str(args.fit_ctx),
        "-p", str(args.n_prompt),
        "-n", str(args.n_gen),
        "-d", str(context_size),
        "-r", str(args.repetitions),
        "-ctk", kv_cache_type,
        "-ctv", kv_cache_type,
        "-o", "csv",
    ]

    if args.tensor_split:
        cmd.extend(["-ts", args.tensor_split])

    return cmd


def run_one_context(
    kv_cache_label: str,
    kv_cache_type: str,
    phase: str,
    context_size: int,
    args: argparse.Namespace,
    env: dict[str, str],
    devices: list[int],
) -> RunResult:
    '''Runs a single context-size benchmark using llama-bench and
    returns the result as a RunResult dataclass.'''

    cmd = build_command(args, context_size, kv_cache_type)

    poller = GpuPeakPoller(devices=devices, interval_s=args.poll_interval)
    t0 = time.perf_counter()
    poller.start()

    timeout_hit = False

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            check=False,
            timeout=args.max_run_seconds,
        )

    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")

        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        timeout_line = (
            f"TIMEOUT: run exceeded --max-run-seconds={args.max_run_seconds} "
            f"(phase={phase}, ctx={context_size}, kv={kv_cache_type})"
        )
        stderr = (stderr + "\n" + timeout_line).strip() + "\n"
        proc = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )

    finally:
        poller.stop()

    elapsed = time.perf_counter() - t0
    peak_total = sum(poller.peak_used.values())

    pp_ts, pp_stddev_ts, tg_ts, tg_stddev_ts = parse_llama_bench_csv(proc.stdout)

    if timeout_hit:
        status = "failed"

    elif proc.returncode == 0:
        status = "ok"

    elif detect_oom_like_failure(proc.returncode, proc.stdout, proc.stderr):
        status = "failed_oom"

    else:
        status = "failed"

    return RunResult(
        kv_cache_label=kv_cache_label,
        kv_cache_type=kv_cache_type,
        phase=phase,
        context_size=context_size,
        status=status,
        return_code=proc.returncode,
        elapsed_s=elapsed,
        peak_vram_total_mib=peak_total,
        peak_vram_per_device=poller.peak_used,
        command=cmd,
        pp_ts=pp_ts,
        pp_stddev_ts=pp_stddev_ts,
        tg_ts=tg_ts,
        tg_stddev_ts=tg_stddev_ts,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def write_csv(path: Path, rows: list[RunResult], model: str, devices_text: str) -> None:
    '''Writes the benchmark results to a CSV file with structured columns.'''

    def make_excerpt(text: str, limit: int = 1200) -> str:
        '''Returns a compact excerpt preserving both header and tail context.'''

        if len(text) <= limit:
            return text.replace("\n", "\\n")

        head = text[: limit // 2]
        tail = text[-(limit // 2):]
        combined = head + "\n...[truncated]...\n" + tail
        return combined.replace("\n", "\\n")

    fieldnames = [
        "timestamp",
        "model",
        "gpu_devices",
        "kv_cache_label",
        "kv_cache_type",
        "phase",
        "context_size",
        "status",
        "return_code",
        "elapsed_s",
        "peak_vram_total_mib",
        "peak_vram_per_device",
        "pp_ts",
        "pp_stddev_ts",
        "tg_ts",
        "tg_stddev_ts",
        "command",
        "stdout_excerpt",
        "stderr_excerpt",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": model,
                    "gpu_devices": devices_text,
                    "kv_cache_label": row.kv_cache_label,
                    "kv_cache_type": row.kv_cache_type,
                    "phase": row.phase,
                    "context_size": row.context_size,
                    "status": row.status,
                    "return_code": row.return_code,
                    "elapsed_s": f"{row.elapsed_s:.3f}",
                    "peak_vram_total_mib": row.peak_vram_total_mib,
                    "peak_vram_per_device": json.dumps(row.peak_vram_per_device, sort_keys=True),
                    "pp_ts": "" if row.pp_ts is None else f"{row.pp_ts:.6f}",
                    "pp_stddev_ts": "" if row.pp_stddev_ts is None else f"{row.pp_stddev_ts:.6f}",
                    "tg_ts": "" if row.tg_ts is None else f"{row.tg_ts:.6f}",
                    "tg_stddev_ts": "" if row.tg_stddev_ts is None else f"{row.tg_stddev_ts:.6f}",
                    "command": " ".join(row.command),
                    "stdout_excerpt": make_excerpt(row.stdout),
                    "stderr_excerpt": make_excerpt(row.stderr),
                }
            )


def append_log(path: Path, row: RunResult, env: dict[str, str]) -> None:
    '''Appends a single benchmark run result to the log file.'''

    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")
        f.write(
            f"kv_cache={row.kv_cache_label} ({row.kv_cache_type}) "
            f"phase={row.phase} context_size={row.context_size} status={row.status}\n"
        )
        f.write(f"return_code={row.return_code} elapsed_s={row.elapsed_s:.3f}\n")
        f.write(f"peak_vram_total_mib={row.peak_vram_total_mib}\n")
        f.write(f"peak_vram_per_device={json.dumps(row.peak_vram_per_device, sort_keys=True)}\n")
        f.write(f"pp_ts={row.pp_ts} pp_stddev_ts={row.pp_stddev_ts} tg_ts={row.tg_ts} tg_stddev_ts={row.tg_stddev_ts}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        f.write("CMD: " + " ".join(row.command) + "\n")

        if row.stdout:
            f.write("--- stdout ---\n")
            f.write(row.stdout)

            if not row.stdout.endswith("\n"):
                f.write("\n")

        if row.stderr:
            f.write("--- stderr ---\n")
            f.write(row.stderr)

            if not row.stderr.endswith("\n"):
                f.write("\n")


def append_warning(log_path: Path, message: str) -> None:
    '''Appends a warning message to the log file, clearly marked.'''

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "!" * 88 + "\n")
        f.write("WARNING\n")
        f.write(message + "\n")
        f.write("!" * 88 + "\n")


def write_summary_json(
    path: Path,
    *,
    model: str,
    gpu_devices: str,
    coarse_sizes: list[int],
    score_breakpoints: list[tuple[str, float]],
    rows: list[RunResult],
    quant_summaries: list[QuantRunSummary],
    warnings: list[str],
) -> None:
    '''Writes a summary of the benchmark results to a JSON file.'''

    runs: dict[str, dict[str, object]] = {}
    overall_errors: list[str] = list(warnings)
    overall_score_candidates: list[tuple[float, str, Optional[int]]] = []
    breakpoint_dict = score_breakpoints_to_dict(score_breakpoints)
    runtime_by_kv_cache_s: dict[str, float] = {}

    for summary in quant_summaries:
        quant_rows = [r for r in rows if r.kv_cache_type == summary.kv_cache_type]
        run_errors: list[str] = list(summary.warnings)
        deployment_rows = select_deployment_rows(rows, summary.kv_cache_type, summary.max_success_ctx)

        deployment_pp_ts = mean_optional([r.pp_ts for r in deployment_rows])
        deployment_tg_ts = mean_optional([r.tg_ts for r in deployment_rows])
        deployment_score = weighted_harmonic_mean(deployment_pp_ts, deployment_tg_ts)
        deployment_tier, deployment_tier_threshold = deployment_tier_for_score(deployment_score, score_breakpoints)
        runtime_total_s = sum(r.elapsed_s for r in quant_rows)
        runtime_by_context_s = aggregate_runtime_by_context(quant_rows)
        runtime_by_phase_s = aggregate_runtime_by_phase(quant_rows)
        runtime_by_kv_cache_s[summary.kv_cache_label] = round(runtime_total_s, 3)

        if deployment_score is not None:
            overall_score_candidates.append((deployment_score, summary.kv_cache_label, summary.max_success_ctx))

        for row in quant_rows:
            if row.status == "failed":
                err_line = best_error_line(row.stderr)

                run_errors.append(f"ctx={row.context_size}: {err_line}")

        run_errors = list(dict.fromkeys(run_errors))
        overall_errors.extend(run_errors)

        runs[summary.kv_cache_label] = {
            "kv_cache_label": summary.kv_cache_label,
            "kv_cache_type": summary.kv_cache_type,
            "max_context": summary.max_success_ctx,
            "max_context_stable": summary.boundary_stable,
            "first_fail_ctx": summary.first_fail_ctx,
            "bisect_fail_ctx": summary.bisect_fail_ctx,
            "runs_total": len(quant_rows),
            "runs_ok": sum(1 for r in quant_rows if r.status == "ok"),
            "runs_failed_oom": sum(1 for r in quant_rows if r.status == "failed_oom"),
            "runs_failed_other": sum(1 for r in quant_rows if r.status == "failed"),
            "runtime_total_s": round(runtime_total_s, 3),
            "runtime_by_context_s": runtime_by_context_s,
            "runtime_by_phase_s": runtime_by_phase_s,
            "deployment_score": deployment_score,
            "deployment_tier": deployment_tier,
            "deployment_tier_threshold": deployment_tier_threshold,
            "deployment_score_pp_ts_mean": deployment_pp_ts,
            "deployment_score_tg_ts_mean": deployment_tg_ts,
            "deployment_score_source_context": summary.max_success_ctx,
            "deployment_score_formula": "weighted_harmonic_mean(pp_ts_mean, tg_ts_mean; prompt_weight=0.35, generation_weight=0.65)",
            "deployment_score_breakpoints": breakpoint_dict,
            "errors": run_errors,
        }

    overall_errors = list(dict.fromkeys(overall_errors))

    overall_summary: dict[str, object] = {
        "runs_total": len(rows),
        "runs_ok": sum(1 for r in rows if r.status == "ok"),
        "runs_failed_oom": sum(1 for r in rows if r.status == "failed_oom"),
        "runs_failed_other": sum(1 for r in rows if r.status == "failed"),
        "runtime_total_s": round(sum(r.elapsed_s for r in rows), 3),
        "runtime_by_kv_cache_s": runtime_by_kv_cache_s,
        "runtime_by_context_s": aggregate_runtime_by_context(rows),
        "runtime_by_phase_s": aggregate_runtime_by_phase(rows),
        "errors": overall_errors,
    }

    for summary in quant_summaries:
        overall_summary[f"{summary.kv_cache_label}_max_context"] = summary.max_success_ctx
        overall_summary[f"{summary.kv_cache_label}_max_context_stable"] = summary.boundary_stable

    if overall_score_candidates:
        best_score, best_label, best_context = max(overall_score_candidates, key=lambda item: item[0])
        best_tier, best_tier_threshold = deployment_tier_for_score(best_score, score_breakpoints)
        overall_summary["deployment_score"] = best_score
        overall_summary["deployment_score_kv_cache_label"] = best_label
        overall_summary["deployment_score_context"] = best_context
        overall_summary["deployment_tier"] = best_tier
        overall_summary["deployment_tier_threshold"] = best_tier_threshold
        overall_summary["deployment_score_formula"] = "weighted_harmonic_mean(pp_ts_mean, tg_ts_mean; prompt_weight=0.35, generation_weight=0.65)"
    else:
        overall_summary["deployment_score"] = None
        overall_summary["deployment_score_kv_cache_label"] = None
        overall_summary["deployment_score_context"] = None
        overall_summary["deployment_tier"] = None
        overall_summary["deployment_tier_threshold"] = None
        overall_summary["deployment_score_formula"] = "weighted_harmonic_mean(pp_ts_mean, tg_ts_mean; prompt_weight=0.35, generation_weight=0.65)"

    overall_summary["deployment_score_breakpoints"] = breakpoint_dict

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "gpu_devices": gpu_devices,
        "context_sizes": coarse_sizes,
        "overall_summary": overall_summary,
        "runs": runs,
    }

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_plot_png(
    path: Path,
    rows: list[RunResult],
) -> None:
    '''Generates a scatter plot of context size vs peak VRAM usage,
    color-coded by KV cache type and status.'''

    fig, ax = plt.subplots(figsize=(10, 6))

    quant_order = ["q4", "q8", "f16"]
    color_map = {
        "f16": "#1f77b4",
        "q8": "#ff7f0e",
        "q4": "#2ca02c",
    }

    labels = sorted(
        {r.kv_cache_label for r in rows}, key=lambda x: quant_order.index(x) if x in quant_order else x
    )

    for label in labels:
        color = color_map.get(label, "#7f7f7f")
        quant_rows = [r for r in rows if r.kv_cache_label == label]
        ok_rows = sorted(
            [r for r in quant_rows if r.status == "ok"],
            key=lambda r: r.context_size,
        )
        oom_rows = sorted(
            [r for r in quant_rows if r.status == "failed_oom"],
            key=lambda r: r.context_size,
        )

        if ok_rows:
            x_ok = [r.context_size for r in ok_rows]
            y_ok = [r.peak_vram_total_mib for r in ok_rows]
            ax.scatter(x_ok, y_ok, color=color, s=45, label=f"{label} ok")

        if oom_rows:
            x_oom = [r.context_size for r in oom_rows]
            y_oom = [r.peak_vram_total_mib for r in oom_rows]
            ax.scatter(x_oom, y_oom, color=color, marker="x", s=60, label=f"{label} oom")

    ax.set_xlabel("Context size")
    ax.set_ylabel("Peak VRAM used (MiB, summed selected GPUs)")
    ax.set_title("Context size vs peak VRAM by KV cache quantization")
    ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def service_is_active(service_name: str) -> bool:
    '''Checks if a systemd service is active by invoking 'systemctl is-active'.'''

    proc = subprocess.run(
        ["systemctl", "is-active", service_name],
        capture_output=True,
        text=True,
        check=False,
    )

    return proc.returncode == 0 and proc.stdout.strip() == "active"


def stop_service(service_name: str) -> None:
    '''Stops a systemd service by invoking 'systemctl stop'.
    Raises CalledProcessError on failure.'''

    subprocess.run(["sudo", "systemctl", "stop", service_name], check=True)


def start_service(service_name: str) -> None:
    '''Starts a systemd service by invoking 'systemctl start'.
    Raises CalledProcessError on failure.'''

    subprocess.run(["sudo", "systemctl", "start", service_name], check=True)


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
    config_path = bootstrap_args.config or (DEFAULT_CONTEXT_FIT_CONFIG if DEFAULT_CONTEXT_FIT_CONFIG.exists() else None)
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
        help="Path to a text file listing one model path or filename per line (mutually exclusive with --model)"
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
        default=str(run_config.get("tensor_split", "1,1"))
    )
    parser.add_argument(
        "--fit-target",
        type=int,
        default=int(run_config.get("fit_target", 512))
    )
    parser.add_argument(
        "--fit-ctx",
        type=int,
        default=int(run_config.get("fit_ctx", 2048))
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
        print("ERROR: one of --model or --model-list is required, either on the command line or in the YAML config")
        sys.exit(1)

    if args.max_context <= 0:
        print("ERROR: --max-context must be > 0")
        sys.exit(1)

    args.config = config_path
    args.score_breakpoints = score_breakpoints

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
    _vram_probe = GpuPeakPoller(devices=devices, interval_s=1.0)
    _vram_probe._read_snapshot()
    total_vram_mib = sum(_vram_probe.total_mem.values())
    verify_vram_threshold_mib = max(0, total_vram_mib - 1024)  # verify only when within 1 GiB of limit

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

            if args.verify_runs > 0:
                max_coarse_result = next(
                    (r for r in reversed(rows) if r.context_size == max_coarse and r.kv_cache_type == kv_type and r.status == "ok"),
                    None,
                )
                peak = max_coarse_result.peak_vram_total_mib if max_coarse_result else 0
                if peak >= verify_vram_threshold_mib:
                    print(
                        f"[{kv_label}] Verifying max context={max_coarse} "
                        f"with {args.verify_runs} run(s) (peak {peak} MiB within 1 GiB of {total_vram_mib} MiB)"
                    )

                    verify_failed = False

                    for i in range(args.verify_runs):
                        print(f"[{kv_label} verify] ctx={max_coarse} run={i + 1}/{args.verify_runs}")

                        result = run_one_context(
                            kv_label,
                            kv_type,
                            "verify",
                            max_coarse,
                            args,
                            env,
                            devices
                        )

                        rows.append(result)
                        append_log(log_path, result, env)

                        if result.status != "ok":
                            verify_failed = True

                    boundary_stable = not verify_failed

                    if verify_failed:
                        msg = (
                            f"[{kv_label}] Max context {max_coarse} is unstable: "
                            "at least one verification run failed. "
                            "A single failure disqualifies this context for production use."
                        )

                        print(f"WARNING: {msg}")
                        warnings.append(msg)
                        quant_warnings.append(msg)
                        append_warning(log_path, msg)
                else:
                    print(
                        f"[{kv_label}] Skipping stability verification: peak VRAM {peak} MiB "
                        f"has {total_vram_mib - peak} MiB headroom (threshold 1 GiB)"
                    )
                    boundary_stable = True

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

                if args.verify_runs > 0:
                    low_result = next(
                        (r for r in reversed(rows) if r.context_size == low and r.kv_cache_type == kv_type and r.status == "ok"),
                        None,
                    )
                    peak = low_result.peak_vram_total_mib if low_result else 0
                    if peak >= verify_vram_threshold_mib:
                        print(
                            f"[{kv_label}] Verifying boundary "
                            f"context={low} with {args.verify_runs} run(s) (peak {peak} MiB within 1 GiB of {total_vram_mib} MiB)"
                        )

                        verify_failed = False

                        for i in range(args.verify_runs):
                            print(f"[{kv_label} verify] ctx={low} run={i + 1}/{args.verify_runs}")

                            result = run_one_context(
                                kv_label,
                                kv_type,
                                "verify",
                                low,
                                args,
                                env,
                                devices
                            )

                            rows.append(result)
                            append_log(log_path, result, env)

                            if result.status != "ok":
                                verify_failed = True

                        boundary_stable = not verify_failed

                        if verify_failed:
                            msg = (
                                f"[{kv_label}] Boundary context {low} is unstable: at "
                                "least one verification run failed. A single failure "
                                "disqualifies this context for production use."
                            )

                            print(f"WARNING: {msg}")
                            warnings.append(msg)
                            quant_warnings.append(msg)
                            append_warning(log_path, msg)
                    else:
                        print(
                            f"[{kv_label}] Skipping stability verification: peak VRAM {peak} MiB "
                            f"has {total_vram_mib - peak} MiB headroom (threshold 1 GiB)"
                        )
                        boundary_stable = True

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

    label_map = {
        "f16": "f16",
        "q8_0": "q8",
        "q8": "q8",
        "q4_0": "q4",
        "q4": "q4",
    }

    kv_runs: list[tuple[str, str]] = []

    for kv_type in kv_type_tokens:
        label = label_map.get(kv_type, kv_type)
        kv_runs.append((label, kv_type))

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

        for model_path, max_ctx in models:
            coarse_sizes = coarse_sizes_for_max(max_ctx) if max_ctx is not None else COARSE_CONTEXT_SIZES
            run_name = (
                model_path.stem
                if args.model_list
                else (args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_path.stem}")
            )

            _run_for_model(model_path, run_name, kv_runs, devices, args, env, coarse_sizes)

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
