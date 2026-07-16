#!/usr/bin/env python3
"""
run_fit_context_size.py

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
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_BIN = Path("/opt/llama.cpp/build/bin/llama-bench")
DEFAULT_RESULTS_DIR = REPO_ROOT / "tests" / "results" / "context_fit"
COARSE_CONTEXT_SIZES = [32768, 65536, 131072, 262144]


def coarse_sizes_for_max(max_ctx: int) -> list[int]:
    '''Returns 4 context sizes for the coarse sweep: max//8, max//4, max//2, max.'''
    return [max_ctx >> 3, max_ctx >> 2, max_ctx >> 1, max_ctx]


OOM_PATTERNS = [
    re.compile(r"cuda.*out of memory", re.IGNORECASE),
    re.compile(r"cudamalloc failed", re.IGNORECASE),
    re.compile(r"failed to allocate cuda", re.IGNORECASE),
    re.compile(r"unable to allocate", re.IGNORECASE),
    re.compile(r"cuda_error_out_of_memory", re.IGNORECASE),
    re.compile(r"memory allocation of .* failed", re.IGNORECASE),
    re.compile(r"resource temporarily unavailable", re.IGNORECASE),
]

RESOURCE_FAILURE_PATTERNS = [
    re.compile(r"ggml_cuda_error", re.IGNORECASE),
    re.compile(r"ggml_cuda_pool_vmm::alloc", re.IGNORECASE),
    re.compile(r"ggml-cuda\\.cu:\\d+:\\s*CUDA error", re.IGNORECASE),
    re.compile(r"cuda error", re.IGNORECASE),
    re.compile(r"abort|aborted", re.IGNORECASE),
    re.compile(r"signal\\s+6|SIGABRT", re.IGNORECASE),
]


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


def detect_oom_like_failure(return_code: int, stdout: str, stderr: str) -> bool:
    '''Detects if the process likely failed due to an OOM or other resource
    type failure based on return code and output.'''

    combined = f"{stdout}\n{stderr}"

    if any(pattern.search(combined) for pattern in OOM_PATTERNS):
        return True

    # SIGABRT/SIGKILL/SIGSEGV often appear as negative return codes when
    # the backend hard-aborts under memory/resource pressure.
    if return_code in (-6, -9, -11):
        if any(pattern.search(combined) for pattern in RESOURCE_FAILURE_PATTERNS):
            return True

    return any(pattern.search(combined) for pattern in RESOURCE_FAILURE_PATTERNS)


def parse_llama_bench_csv(stdout: str) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    '''Parses the CSV output from llama-bench and returns (pp_ts, pp_stddev_ts, tg_ts, tg_stddev_ts).
    Rows are matched by the ``type`` column ("pp" / "tg").'''

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]

    if len(lines) < 2:
        return None, None, None, None

    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        rows = list(reader)

    except Exception:
        return None, None, None, None

    def as_float(value: str | None) -> Optional[float]:
        '''Converts a string to float, returning None if conversion fails or value is empty.'''

        if value is None or value == "":
            return None

        try:
            return float(value)

        except ValueError:
            return None

    def row_for_type(t: str) -> Optional[dict]:
        return next((r for r in rows if (r.get("type") or "").strip() == t), None)

    pp = row_for_type("pp")
    tg = row_for_type("tg")

    return (
        as_float(pp.get("avg_ts") if pp else None),
        as_float(pp.get("stddev_ts") if pp else None),
        as_float(tg.get("avg_ts") if tg else None),
        as_float(tg.get("stddev_ts") if tg else None),
    )


def round_to_step(value: int, step: int) -> int:
    '''Rounds a value to the nearest multiple of step. If step <= 1, returns the original value.'''

    if step <= 1:
        return value

    return int(round(value / step) * step)


def midpoint_in_bracket(low: int, high: int, step: int) -> int:
    '''Returns the midpoint between low and high, rounded to the nearest multiple of step.'''

    mid = round_to_step((low + high) // 2, step)

    if mid <= low:
        mid = low + step

    if mid >= high:
        mid = high - step

    return mid


def get_stable_success_contexts(rows: list[RunResult]) -> list[int]:
    '''Returns a sorted list of context sizes that succeeded without any failures.'''

    ok_ctx = {r.context_size for r in rows if r.status == "ok"}
    failed_ctx = {r.context_size for r in rows if r.status in ("failed", "failed_oom")}

    return sorted(ctx for ctx in ok_ctx if ctx not in failed_ctx)


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

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            check=False,
        )

    finally:
        poller.stop()

    elapsed = time.perf_counter() - t0
    peak_total = sum(poller.peak_used.values())

    pp_ts, pp_stddev_ts, tg_ts, tg_stddev_ts = parse_llama_bench_csv(proc.stdout)

    if proc.returncode == 0:
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
                    "stdout_excerpt": row.stdout[:400].replace("\n", "\\n"),
                    "stderr_excerpt": row.stderr[:400].replace("\n", "\\n"),
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
    rows: list[RunResult],
    quant_summaries: list[QuantRunSummary],
    warnings: list[str],
) -> None:
    '''Writes a summary of the benchmark results to a JSON file.'''

    runs: dict[str, dict[str, object]] = {}
    overall_errors: list[str] = list(warnings)

    for summary in quant_summaries:
        quant_rows = [r for r in rows if r.kv_cache_type == summary.kv_cache_type]
        run_errors: list[str] = list(summary.warnings)

        for row in quant_rows:
            if row.status == "failed":
                err_line = "unknown error"

                for line in row.stderr.splitlines():
                    if line.strip():
                        err_line = line.strip()

                        break

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
            "errors": run_errors,
        }

    overall_errors = list(dict.fromkeys(overall_errors))

    overall_summary: dict[str, object] = {
        "runs_total": len(rows),
        "runs_ok": sum(1 for r in rows if r.status == "ok"),
        "runs_failed_oom": sum(1 for r in rows if r.status == "failed_oom"),
        "runs_failed_other": sum(1 for r in rows if r.status == "failed"),
        "errors": overall_errors,
    }

    for summary in quant_summaries:
        overall_summary[f"{summary.kv_cache_label}_max_context"] = summary.max_success_ctx
        overall_summary[f"{summary.kv_cache_label}_max_context_stable"] = summary.boundary_stable

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

    parser = argparse.ArgumentParser(
        description=(
            "Run context-size fit benchmarks one context at a time, track peak VRAM, "
            "and refine the fail boundary with bisection probes."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path or filename (mutually exclusive with --model-list)"
    )
    parser.add_argument(
        "--model-list",
        type=Path,
        default=None,
        help="Path to a text file listing one model path or filename per line (mutually exclusive with --model)"
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="CUDA_VISIBLE_DEVICES string, e.g. '1,2'"
    )
    parser.add_argument(
        "--bench-bin",
        type=Path,
        default=DEFAULT_BENCH_BIN,
        help="Path to llama-bench binary"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output run label (default: timestamp + model name)"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=99
    )
    parser.add_argument(
        "--split-mode",
        default="layer"
    )
    parser.add_argument(
        "--tensor-split",
        default="1,1"
    )
    parser.add_argument(
        "--fit-target",
        type=int,
        default=512
    )
    parser.add_argument(
        "--fit-ctx",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--n-prompt",
        type=int,
        default=512
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=128
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="nvidia-smi poll interval in seconds"
    )
    parser.add_argument(
        "--refine-step",
        type=int,
        default=1024,
        help="Step size for bisection refinement contexts"
    )
    parser.add_argument(
        "--verify-runs",
        type=int,
        default=3,
        help=(
            "Number of confirmation runs at the final max "
            "context; any failure marks that context as unstable"
        )
    )
    parser.add_argument(
        "--kv-cache-types",
        default="q4_0,q8_0,f16",
        help=(
            "Comma-separated KV cache types for iterative runs. "
            "Defaults to q4_0,q8_0,f16"
        )
    )
    parser.add_argument(
        "--service-name",
        default="llamacpp.service",
        help="Systemd service to stop before run and restore after run"
    )
    parser.add_argument(
        "--no-manage-service",
        action="store_true",
        help="Disable automatic stop/start of the llama.cpp service around the benchmark run",
    )

    return parser.parse_args()


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

    csv_path = out_dir / "context_fit.csv"
    log_path = out_dir / "context_fit.log"

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

    summary_path = out_dir / "context_fit_summary.json"
    plot_path = out_dir / "context_fit_plot.png"

    write_summary_json(
        summary_path,
        model=model_path.name,
        gpu_devices=args.gpus,
        coarse_sizes=coarse_sizes,
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
        models = [(resolve_model_path(args.model), None)]

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
