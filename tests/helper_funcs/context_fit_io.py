"""Result serialization (CSV, log, summary JSON, plot) and service control
for the context-fit benchmark."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from helper_funcs.context_fit_types import QuantRunSummary, RunResult
from helper_funcs.context_fit_utils import (
    aggregate_runtime_by_context,
    aggregate_runtime_by_phase,
    best_error_line,
    deployment_tier_for_score,
    mean_optional,
    score_breakpoints_to_dict,
    select_deployment_rows,
    weighted_harmonic_mean,
)

DEPLOYMENT_SCORE_FORMULA = (
    "weighted_harmonic_mean(pp_ts_mean, tg_ts_mean; "
    "prompt_weight=0.35, generation_weight=0.65)"
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
    '''Appends a single benchmark run result, with full stdout/stderr, to the log file.'''

    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 88 + "\n")
        f.write(
            f"kv_cache={row.kv_cache_label} ({row.kv_cache_type}) "
            f"phase={row.phase} context_size={row.context_size} status={row.status}\n"
        )
        f.write(f"return_code={row.return_code} elapsed_s={row.elapsed_s:.3f}\n")
        f.write(f"peak_vram_total_mib={row.peak_vram_total_mib}\n")
        f.write(f"peak_vram_per_device={json.dumps(row.peak_vram_per_device, sort_keys=True)}\n")
        f.write(
            f"pp_ts={row.pp_ts} pp_stddev_ts={row.pp_stddev_ts} "
            f"tg_ts={row.tg_ts} tg_stddev_ts={row.tg_stddev_ts}\n"
        )
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
    '''Appends a clearly marked warning message to the log file.'''

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "!" * 88 + "\n")
        f.write("WARNING\n")
        f.write(message + "\n")
        f.write("!" * 88 + "\n")


def _build_quant_run_dict(
    summary: QuantRunSummary,
    quant_rows: list[RunResult],
    score_breakpoints: list[tuple[str, float]],
    breakpoint_dict: dict[str, float],
) -> tuple[dict[str, object], list[str], Optional[float]]:
    '''Builds the per-KV-cache summary dict, its error list, and its deployment score.'''

    run_errors: list[str] = list(summary.warnings)
    deployment_rows = select_deployment_rows(
        quant_rows, summary.kv_cache_type, summary.max_success_ctx
    )

    deployment_pp_ts = mean_optional([r.pp_ts for r in deployment_rows])
    deployment_tg_ts = mean_optional([r.tg_ts for r in deployment_rows])
    deployment_score = weighted_harmonic_mean(deployment_pp_ts, deployment_tg_ts)
    deployment_tier, deployment_tier_threshold = deployment_tier_for_score(
        deployment_score, score_breakpoints
    )
    runtime_total_s = sum(r.elapsed_s for r in quant_rows)

    for row in quant_rows:
        if row.status == "failed":
            run_errors.append(f"ctx={row.context_size}: {best_error_line(row.stderr)}")

    run_errors = list(dict.fromkeys(run_errors))

    run_dict: dict[str, object] = {
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
        "runtime_by_context_s": aggregate_runtime_by_context(quant_rows),
        "runtime_by_phase_s": aggregate_runtime_by_phase(quant_rows),
        "deployment_score": deployment_score,
        "deployment_tier": deployment_tier,
        "deployment_tier_threshold": deployment_tier_threshold,
        "deployment_score_pp_ts_mean": deployment_pp_ts,
        "deployment_score_tg_ts_mean": deployment_tg_ts,
        "deployment_score_source_context": summary.max_success_ctx,
        "deployment_score_formula": DEPLOYMENT_SCORE_FORMULA,
        "deployment_score_breakpoints": breakpoint_dict,
        "errors": run_errors,
    }

    return run_dict, run_errors, deployment_score


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
    '''Writes the aggregated per-KV-cache and overall benchmark summary to JSON.'''

    runs: dict[str, dict[str, object]] = {}
    overall_errors: list[str] = list(warnings)
    overall_score_candidates: list[tuple[float, str, Optional[int]]] = []
    breakpoint_dict = score_breakpoints_to_dict(score_breakpoints)
    runtime_by_kv_cache_s: dict[str, float] = {}

    for summary in quant_summaries:
        quant_rows = [r for r in rows if r.kv_cache_type == summary.kv_cache_type]
        run_dict, run_errors, deployment_score = _build_quant_run_dict(
            summary, quant_rows, score_breakpoints, breakpoint_dict
        )

        runtime_by_kv_cache_s[summary.kv_cache_label] = round(
            sum(r.elapsed_s for r in quant_rows), 3
        )
        overall_errors.extend(run_errors)

        if deployment_score is not None:
            overall_score_candidates.append(
                (deployment_score, summary.kv_cache_label, summary.max_success_ctx)
            )

        runs[summary.kv_cache_label] = run_dict

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
        best_score, best_label, best_context = max(
            overall_score_candidates, key=lambda item: item[0]
        )
        best_tier, best_tier_threshold = deployment_tier_for_score(best_score, score_breakpoints)
        overall_summary["deployment_score"] = best_score
        overall_summary["deployment_score_kv_cache_label"] = best_label
        overall_summary["deployment_score_context"] = best_context
        overall_summary["deployment_tier"] = best_tier
        overall_summary["deployment_tier_threshold"] = best_tier_threshold
    else:
        overall_summary["deployment_score"] = None
        overall_summary["deployment_score_kv_cache_label"] = None
        overall_summary["deployment_score_context"] = None
        overall_summary["deployment_tier"] = None
        overall_summary["deployment_tier_threshold"] = None

    overall_summary["deployment_score_formula"] = DEPLOYMENT_SCORE_FORMULA
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


def write_plot_png(path: Path, rows: list[RunResult]) -> None:
    '''Generates a scatter plot of context size vs peak VRAM, keyed by KV cache and status.'''

    fig, ax = plt.subplots(figsize=(10, 6))

    quant_order = ["q4", "q8", "f16"]
    color_map = {
        "f16": "#1f77b4",
        "q8": "#ff7f0e",
        "q4": "#2ca02c",
    }

    labels = sorted(
        {r.kv_cache_label for r in rows},
        key=lambda x: quant_order.index(x) if x in quant_order else x,
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

    handles, _ = ax.get_legend_handles_labels()

    if handles:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def service_is_active(service_name: str) -> bool:
    '''Returns True if the given systemd service reports as active.'''

    proc = subprocess.run(
        ["systemctl", "is-active", service_name],
        capture_output=True,
        text=True,
        check=False,
    )

    return proc.returncode == 0 and proc.stdout.strip() == "active"


def stop_service(service_name: str) -> None:
    '''Stops a systemd service; raises CalledProcessError on failure.'''

    subprocess.run(["sudo", "systemctl", "stop", service_name], check=True)


def start_service(service_name: str) -> None:
    '''Starts a systemd service; raises CalledProcessError on failure.'''

    subprocess.run(["sudo", "systemctl", "start", service_name], check=True)
