"""llama-bench command construction and single-context execution for the
context-fit benchmark."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional

from helper_funcs.context_fit_gpu import GpuPeakPoller
from helper_funcs.context_fit_io import append_log, append_warning
from helper_funcs.context_fit_types import RunResult
from helper_funcs.context_fit_utils import detect_oom_like_failure, parse_llama_bench_csv

# helper_funcs -> tests -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def build_command(args: argparse.Namespace, context_size: int, kv_cache_type: str) -> list[str]:
    '''Builds the llama-bench command line for one context size and KV cache type.

    Default behavior is strict GPU-only fitting: no host spill and no fit-target
    padding. Host RAM spill is only allowed when the user explicitly opts in.
    '''

    cmd = [
        str(args.bench_bin),
        "-m", str(args.model),
        "-ngl", str(args.n_gpu_layers),
        "-sm", args.split_mode,
        "-p", str(args.n_prompt),
        "-n", str(args.n_gen),
        "-d", str(context_size),
        "-r", str(args.repetitions),
        "-ctk", kv_cache_type,
        "-ctv", kv_cache_type,
        "-fa", args.flash_attn,
        "-o", "csv",
    ]

    if args.no_host:
        cmd.extend(["--no-host", "1"])
    elif args.fit_target > 0 or args.fit_ctx > 0:
        cmd.extend(["--fit-target", str(args.fit_target), "--fit-ctx", str(args.fit_ctx)])

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
    '''Runs one llama-bench invocation while polling peak VRAM, and returns a RunResult.

    Timeouts are recorded as failed runs; non-zero exits are classified as
    failed_oom when the output matches known out-of-memory signatures.
    '''

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


def verify_boundary(
    *,
    kv_label: str,
    kv_type: str,
    ctx: int,
    label: str,
    args: argparse.Namespace,
    env: dict[str, str],
    devices: list[int],
    rows: list[RunResult],
    log_path: Path,
    warnings: list[str],
    quant_warnings: list[str],
    total_vram_mib: int,
    verify_vram_threshold_mib: int,
) -> Optional[bool]:
    '''Re-runs a candidate boundary context to confirm stability. Verification is
    skipped when peak VRAM leaves more than 1 GiB of headroom. Returns True/False
    stability, or None when verification does not run.'''

    if args.verify_runs <= 0:
        return None

    last_ok = next(
        (
            r for r in reversed(rows)
            if r.context_size == ctx
            and r.kv_cache_type == kv_type
            and r.status == "ok"
        ),
        None,
    )
    peak = last_ok.peak_vram_total_mib if last_ok else 0

    if peak < verify_vram_threshold_mib:
        print(
            f"[{kv_label}] Skipping stability verification: peak VRAM {peak} MiB "
            f"has {total_vram_mib - peak} MiB headroom (threshold 1 GiB)"
        )
        return True

    print(
        f"[{kv_label}] Verifying {label.lower()}={ctx} with {args.verify_runs} run(s) "
        f"(peak {peak} MiB within 1 GiB of {total_vram_mib} MiB)"
    )

    verify_failed = False

    for i in range(args.verify_runs):
        print(f"[{kv_label} verify] ctx={ctx} run={i + 1}/{args.verify_runs}")
        result = run_one_context(kv_label, kv_type, "verify", ctx, args, env, devices)
        rows.append(result)
        append_log(log_path, result, env)

        if result.status != "ok":
            verify_failed = True

    if verify_failed:
        msg = (
            f"[{kv_label}] {label} {ctx} is unstable: at least one verification run "
            "failed. A single failure disqualifies this context for production use."
        )
        print(f"WARNING: {msg}")
        warnings.append(msg)
        quant_warnings.append(msg)
        append_warning(log_path, msg)

    return not verify_failed
