#!/usr/bin/env python3
"""
run_fit_context_size.py

Targeted context-fit benchmark runner for llama.cpp / llama-bench.

Purpose:
1. Accept model and GPUs as arguments.
2. Run one benchmark per context size, while tracking peak VRAM usage.
3. Save structured output and full logs.

Workflow:
- Coarse phase: fixed context scan until first CUDA OOM.
- Refinement phase: fit linear regression on successful points (peak_vram vs ctx)
  then probe around predicted limit for finer boundary estimation.
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
COARSE_CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
OOM_PATTERNS = [
    re.compile(r"cuda.*out of memory", re.IGNORECASE),
    re.compile(r"cudamalloc failed", re.IGNORECASE),
    re.compile(r"failed to allocate cuda", re.IGNORECASE),
    re.compile(r"unable to allocate", re.IGNORECASE),
]


@dataclass
class RunResult:
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
    avg_ts: Optional[float]
    stddev_ts: Optional[float]
    samples_ns: Optional[int]
    stdout: str
    stderr: str


@dataclass
class RegressionFit:
    slope: float
    intercept: float
    r2: float
    rmse_mib: float


@dataclass
class QuantRunSummary:
    kv_cache_label: str
    kv_cache_type: str
    memory_cap_mib: Optional[int]
    regression_slope: Optional[float]
    regression_intercept: Optional[float]
    predicted_ctx: Optional[int]
    max_success_ctx: Optional[int]
    first_oom_ctx: Optional[int]
    refined_oom_ctx: Optional[int]
    warnings: list[str]


class GpuPeakPoller:
    def __init__(self, devices: list[int], interval_s: float):
        self.devices = devices
        self.interval_s = interval_s
        self.peak_used: dict[int, int] = {d: 0 for d in devices}
        self.total_mem: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_snapshot(self) -> None:
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
        while not self._stop.is_set():
            self._read_snapshot()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._read_snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._read_snapshot()


def parse_devices(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one GPU device index must be provided")
    return values


def detect_oom(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}"
    return any(pattern.search(combined) for pattern in OOM_PATTERNS)


def parse_llama_bench_csv(stdout: str) -> tuple[Optional[float], Optional[float], Optional[int]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None, None, None

    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        first = next(reader)
    except Exception:
        return None, None, None

    def as_float(value: str | None) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def as_int(value: str | None) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    return as_float(first.get("avg_ts")), as_float(first.get("stddev_ts")), as_int(first.get("samples_ns"))


def linear_fit(successful_rows: list[RunResult]) -> RegressionFit | None:
    if len(successful_rows) < 2:
        return None

    xs = [float(row.context_size) for row in successful_rows]
    ys = [float(row.peak_vram_total_mib) for row in successful_rows]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None

    slope = num / den
    intercept = y_mean - slope * x_mean
    if slope <= 0:
        return None

    y_pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = (ss_res / len(xs)) ** 0.5

    return RegressionFit(slope=slope, intercept=intercept, r2=r2, rmse_mib=rmse)


def round_to_step(value: int, step: int) -> int:
    if step <= 1:
        return value
    return int(round(value / step) * step)


def get_memory_cap_mib(
    poller: GpuPeakPoller,
    override_cap_mib: Optional[int],
    headroom_mib: int,
) -> int:
    if override_cap_mib is not None:
        return override_cap_mib

    total = sum(poller.total_mem.values())
    if total <= 0:
        raise RuntimeError("Could not read total VRAM from nvidia-smi")

    cap = total - max(0, headroom_mib)
    if cap <= 0:
        raise RuntimeError("Computed memory cap is non-positive; reduce headroom")

    return cap


def build_command(args: argparse.Namespace, context_size: int, kv_cache_type: str) -> list[str]:
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

    avg_ts, stddev_ts, samples_ns = parse_llama_bench_csv(proc.stdout)

    if proc.returncode == 0:
        status = "ok"
    elif detect_oom(proc.stdout, proc.stderr):
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
        avg_ts=avg_ts,
        stddev_ts=stddev_ts,
        samples_ns=samples_ns,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def write_csv(path: Path, rows: list[RunResult], model: str, devices_text: str) -> None:
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
        "avg_ts",
        "stddev_ts",
        "samples_ns",
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
                    "avg_ts": "" if row.avg_ts is None else f"{row.avg_ts:.6f}",
                    "stddev_ts": "" if row.stddev_ts is None else f"{row.stddev_ts:.6f}",
                    "samples_ns": "" if row.samples_ns is None else row.samples_ns,
                    "command": " ".join(row.command),
                    "stdout_excerpt": row.stdout[:400].replace("\n", "\\n"),
                    "stderr_excerpt": row.stderr[:400].replace("\n", "\\n"),
                }
            )


def append_log(path: Path, row: RunResult, env: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")
        f.write(
            f"kv_cache={row.kv_cache_label} ({row.kv_cache_type}) "
            f"phase={row.phase} context_size={row.context_size} status={row.status}\n"
        )
        f.write(f"return_code={row.return_code} elapsed_s={row.elapsed_s:.3f}\n")
        f.write(f"peak_vram_total_mib={row.peak_vram_total_mib}\n")
        f.write(f"peak_vram_per_device={json.dumps(row.peak_vram_per_device, sort_keys=True)}\n")
        f.write(f"avg_ts={row.avg_ts} stddev_ts={row.stddev_ts} samples_ns={row.samples_ns}\n")
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
    rows: list[RunResult],
    quant_summaries: list[QuantRunSummary],
    warnings: list[str],
) -> None:
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
            "memory_cap_mib": summary.memory_cap_mib,
            "max_context": summary.max_success_ctx,
            "predicted_max_context": summary.predicted_ctx,
            "model_slope": summary.regression_slope,
            "model_intercept": summary.regression_intercept,
            "runs_total": len(quant_rows),
            "runs_ok": sum(1 for r in quant_rows if r.status == "ok"),
            "runs_failed_oom": sum(1 for r in quant_rows if r.status == "failed_oom"),
            "runs_failed_other": sum(1 for r in quant_rows if r.status == "failed"),
            "max_success_ctx": summary.max_success_ctx,
            "first_oom_ctx": summary.first_oom_ctx,
            "refined_oom_ctx": summary.refined_oom_ctx,
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

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "gpu_devices": gpu_devices,
        "context_sizes": COARSE_CONTEXT_SIZES,
        "overall_summary": overall_summary,
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_plot_png(
    path: Path,
    rows: list[RunResult],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    quant_order = ["q4", "q8", "f16"]
    color_map = {
        "f16": "#1f77b4",
        "q8": "#ff7f0e",
        "q4": "#2ca02c",
    }

    labels = sorted({r.kv_cache_label for r in rows}, key=lambda x: quant_order.index(x) if x in quant_order else x)
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

        fit = linear_fit(ok_rows) if len(ok_rows) >= 2 else None
        if fit is not None:
            x_min = min(r.context_size for r in quant_rows)
            x_max = max(r.context_size for r in quant_rows)
            if x_max == x_min:
                x_line = [float(x_min)]
            else:
                n_points = 100
                step = (x_max - x_min) / (n_points - 1)
                x_line = [x_min + i * step for i in range(n_points)]
            y_line = [fit.slope * x + fit.intercept for x in x_line]
            ax.plot(x_line, y_line, color=color, linewidth=2.0, alpha=0.9, label=f"{label} regression")

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
    proc = subprocess.run(
        ["systemctl", "is-active", service_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "active"


def stop_service(service_name: str) -> None:
    subprocess.run(["sudo", "systemctl", "stop", service_name], check=True)


def start_service(service_name: str) -> None:
    subprocess.run(["sudo", "systemctl", "start", service_name], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run context-size fit benchmarks one context at a time, track peak VRAM, "
            "and refine the OOM boundary with regression-guided probes."
        )
    )
    parser.add_argument("--model", required=True, help="Model path or filename")
    parser.add_argument("--gpus", required=True, help="CUDA_VISIBLE_DEVICES string, e.g. '1,2'")
    parser.add_argument("--bench-bin", type=Path, default=DEFAULT_BENCH_BIN, help="Path to llama-bench binary")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Output directory")
    parser.add_argument("--run-name", default=None, help="Output run label (default: timestamp + model name)")

    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--split-mode", default="layer")
    parser.add_argument("--tensor-split", default="1,1")
    parser.add_argument("--fit-target", type=int, default=512)
    parser.add_argument("--fit-ctx", type=int, default=2048)
    parser.add_argument("--n-prompt", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)

    parser.add_argument("--poll-interval", type=float, default=0.25, help="nvidia-smi poll interval in seconds")
    parser.add_argument("--memory-cap-mib", type=int, default=None, help="Override memory ceiling for regression target")
    parser.add_argument("--memory-headroom-mib", type=int, default=1024, help="Headroom subtracted from total selected VRAM")
    parser.add_argument("--refine-step", type=int, default=512, help="Rounding step for refinement contexts")
    parser.add_argument(
        "--kv-cache-types",
        default="q4_0,q8_0,f16",
        help=(
            "Comma-separated KV cache types for iterative runs. "
            "Defaults to q4_0,q8_0,f16"
        ),
    )
    parser.add_argument("--service-name", default="llamacpp.service", help="Systemd service to stop before run and restore after run")
    parser.add_argument(
        "--no-manage-service",
        action="store_true",
        help="Disable automatic stop/start of the llama.cpp service around the benchmark run",
    )

    return parser.parse_args()


def main() -> None:
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

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        # Bare filename means model is expected under /opt/models.
        # Relative paths with directories are resolved from the repo root.
        if len(model_path.parts) == 1:
            model_path = Path("/opt/models") / model_path
        else:
            model_path = (REPO_ROOT / model_path).resolve()

    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}")
        print("Hint: pass an absolute model path, a bare filename in /opt/models, or a valid relative path from the repo root.")
        sys.exit(1)

    args.model = model_path

    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_path.stem}"
    out_dir = args.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "context_fit.csv"
    log_path = out_dir / "context_fit.log"

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

    rows: list[RunResult] = []
    quant_summaries: list[QuantRunSummary] = []
    warnings: list[str] = []

    print(f"Model: {model_path}")
    print(f"CUDA_VISIBLE_DEVICES={args.gpus}")
    print("KV cache runs: " + ", ".join(f"{label}({kv_type})" for label, kv_type in kv_runs))
    print(f"Results: {out_dir}")

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

        for kv_label, kv_type in kv_runs:
            print(f"\n=== KV cache run: {kv_label} ({kv_type}) ===")

            tested_contexts: set[int] = set()
            first_oom_ctx: Optional[int] = None
            mem_cap_mib: Optional[int] = None
            regression_slope: Optional[float] = None
            regression_intercept: Optional[float] = None
            predicted_ctx: Optional[int] = None
            quant_warnings: list[str] = []

            # Phase 1: coarse scan
            for ctx in COARSE_CONTEXT_SIZES:
                print(f"[{kv_label} coarse] ctx={ctx}")
                result = run_one_context(kv_label, kv_type, "coarse", ctx, args, env, devices)
                rows.append(result)
                tested_contexts.add(ctx)
                append_log(log_path, result, env)

                if result.status == "failed_oom":
                    first_oom_ctx = ctx
                    print(f"  OOM at ctx={ctx}; entering refinement phase")
                    break

                if result.status == "failed":
                    print(f"  Non-OOM failure at ctx={ctx}; stopping")
                    break

            # Phase 2: refinement around predicted limit
            quant_rows = [row for row in rows if row.kv_cache_type == kv_type]
            successful = [row for row in quant_rows if row.status == "ok"]
            if first_oom_ctx is not None and len(successful) >= 2:
                fit = linear_fit(successful)

                if fit is not None:
                    regression_slope = fit.slope
                    regression_intercept = fit.intercept

                    cap_probe = GpuPeakPoller(devices=devices, interval_s=max(args.poll_interval, 0.25))
                    cap_probe._read_snapshot()
                    mem_cap = get_memory_cap_mib(cap_probe, args.memory_cap_mib, args.memory_headroom_mib)
                    predicted = int((mem_cap - fit.intercept) / fit.slope)
                    mem_cap_mib = mem_cap
                    predicted_ctx = predicted

                    last_success_ctx = max(row.context_size for row in successful)

                    # Internal sanity checks before refinement sweep
                    fit_ok = (fit.r2 >= 0.95)
                    predicted_ok = (predicted > last_success_ctx and predicted < first_oom_ctx)

                    if not fit_ok:
                        msg = (
                            f"[{kv_label}] Regression fit quality check failed (r2={fit.r2:.4f}, "
                            f"threshold=0.95). Skipping fine-grained sweep."
                        )
                        print(f"WARNING: {msg}")
                        warnings.append(msg)
                        quant_warnings.append(msg)
                        append_warning(log_path, msg)

                    if not predicted_ok:
                        msg = (
                            f"[{kv_label}] Predicted context check failed (predicted={predicted}, "
                            f"last_success={last_success_ctx}, coarse_oom={first_oom_ctx}). "
                            "Skipping fine-grained sweep."
                        )
                        print(f"WARNING: {msg}")
                        warnings.append(msg)
                        quant_warnings.append(msg)
                        append_warning(log_path, msg)

                    print(
                        f"[{kv_label}] Refinement: slope={fit.slope:.6f} intercept={fit.intercept:.2f} "
                        f"r2={fit.r2:.4f} rmse_mib={fit.rmse_mib:.2f} "
                        f"mem_cap_mib={mem_cap} predicted_ctx={predicted}"
                    )

                    if fit_ok and predicted_ok:
                        # Fine sweep: start at 90% of prediction, increase by 5% until OOM
                        ctx = round_to_step(int(predicted * 0.90), args.refine_step)
                        if ctx <= 0:
                            ctx = args.refine_step

                        print(f"[{kv_label}] Fine sweep start ctx={ctx} (90% of predicted {predicted})")
                        while True:
                            if ctx in tested_contexts:
                                # Ensure progress in case rounding repeats a previous value.
                                ctx += args.refine_step

                            print(f"[{kv_label} refine] ctx={ctx}")
                            result = run_one_context(kv_label, kv_type, "refine", ctx, args, env, devices)
                            rows.append(result)
                            tested_contexts.add(ctx)
                            append_log(log_path, result, env)

                            if result.status == "failed_oom":
                                print(f"  OOM at refined ctx={ctx}; stopping refinement")
                                break
                            if result.status == "failed":
                                print(f"  Non-OOM failure at refined ctx={ctx}; stopping refinement")
                                break

                            next_ctx = round_to_step(int(ctx * 1.05), args.refine_step)
                            if next_ctx <= ctx:
                                next_ctx = ctx + args.refine_step
                            ctx = next_ctx
                else:
                    msg = f"[{kv_label}] Regression fit could not be computed (insufficient or degenerate data). Skipping fine-grained sweep."
                    print(f"WARNING: {msg}")
                    warnings.append(msg)
                    quant_warnings.append(msg)
                    append_warning(log_path, msg)

            quant_rows = [row for row in rows if row.kv_cache_type == kv_type]
            successful = sorted([row.context_size for row in quant_rows if row.status == "ok"])
            failed_oom = sorted([row.context_size for row in quant_rows if row.status == "failed_oom"])
            refined_failed_oom = sorted(
                [row.context_size for row in quant_rows if row.status == "failed_oom" and row.phase == "refine"]
            )

            quant_summaries.append(
                QuantRunSummary(
                    kv_cache_label=kv_label,
                    kv_cache_type=kv_type,
                    memory_cap_mib=mem_cap_mib,
                    regression_slope=regression_slope,
                    regression_intercept=regression_intercept,
                    predicted_ctx=predicted_ctx,
                    max_success_ctx=successful[-1] if successful else None,
                    first_oom_ctx=failed_oom[0] if failed_oom else None,
                    refined_oom_ctx=refined_failed_oom[0] if refined_failed_oom else None,
                    warnings=quant_warnings,
                )
            )
    finally:
        if should_restore_service:
            print(f"Restoring service: {args.service_name}")
            try:
                start_service(args.service_name)
                if stopped_service:
                    print(f"Service restored successfully: {args.service_name}")
            except subprocess.CalledProcessError as exc:
                msg = f"Failed to restore service {args.service_name}: {exc}"
                print(f"WARNING: {msg}")
                warnings.append(msg)
                append_warning(log_path, msg)

    write_csv(csv_path, rows, str(model_path), args.gpus)

    summary_path = out_dir / "context_fit_summary.json"
    plot_path = out_dir / "context_fit_plot.png"

    write_summary_json(
        summary_path,
        model=model_path.name,
        gpu_devices=args.gpus,
        rows=rows,
        quant_summaries=quant_summaries,
        warnings=warnings,
    )
    write_plot_png(plot_path, rows)

    print("\nDone")
    print(f"  CSV : {csv_path}")
    print(f"  Log : {log_path}")
    print(f"  JSON: {summary_path}")
    print(f"  Plot: {plot_path}")
    for summary in quant_summaries:
        print(
            "  "
            f"{summary.kv_cache_label}: max_success_ctx={summary.max_success_ctx} "
            f"first_oom_ctx={summary.first_oom_ctx} refined_oom_ctx={summary.refined_oom_ctx}"
        )


if __name__ == "__main__":
    main()
