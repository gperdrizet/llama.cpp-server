"""Pure helper functions for the context-fit benchmark: output parsing, failure
classification, scoring, and runtime aggregation."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from typing import Optional

KV_LABEL_MAP = {
    "f16": "f16",
    "q8_0": "q8",
    "q8": "q8",
    "q4_0": "q4",
    "q4": "q4",
}

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


def normalize_score_breakpoints(
    raw_breakpoints: object,
    default_breakpoints: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    '''Normalizes score breakpoints to a list of (label, min_score) sorted high to low.'''
    if raw_breakpoints is None:
        breakpoints = list(default_breakpoints)
    elif isinstance(raw_breakpoints, dict):
        breakpoints = [
            (str(label), float(min_score)) for label, min_score in raw_breakpoints.items()
        ]
    else:
        raise ValueError("score_breakpoints must be a mapping of label -> minimum_score")

    return sorted(breakpoints, key=lambda item: item[1], reverse=True)


def score_breakpoints_to_dict(breakpoints: list[tuple[str, float]]) -> dict[str, float]:
    '''Converts a list of (label, min_score) breakpoints to a label -> min_score dict.'''
    return {label: minimum_score for label, minimum_score in breakpoints}


def detect_oom_like_failure(return_code: int, stdout: str, stderr: str) -> bool:
    '''Returns True if the output or return code matches known out-of-memory signatures.'''
    combined = f"{stdout}\n{stderr}"

    if any(pattern.search(combined) for pattern in OOM_PATTERNS):
        return True

    # SIGABRT/SIGKILL/SIGSEGV often appear as negative return codes when
    # the backend hard-aborts under memory/resource pressure.
    if return_code in (-6, -9, -11):
        if any(pattern.search(combined) for pattern in RESOURCE_FAILURE_PATTERNS):
            return True

    return any(pattern.search(combined) for pattern in RESOURCE_FAILURE_PATTERNS)


def parse_llama_bench_csv(
    stdout: str,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    '''Parses llama-bench CSV output into mean (pp_ts, pp_stddev, tg_ts, tg_stddev).'''
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]

    if len(lines) < 2:
        return None, None, None, None

    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        rows = list(reader)

    except (csv.Error, ValueError, TypeError):
        return None, None, None, None

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

    def classify_row(row: dict[str, str]) -> Optional[str]:
        explicit_type = (row.get("type") or "").strip().lower()
        if explicit_type in ("pp", "tg"):
            return explicit_type

        n_prompt = as_int(row.get("n_prompt"))
        n_gen = as_int(row.get("n_gen"))

        if (n_prompt or 0) > 0 and (n_gen or 0) == 0:
            return "pp"

        if (n_gen or 0) > 0 and (n_prompt or 0) == 0:
            return "tg"

        return None

    def mean_of_type(kind: str, key: str) -> Optional[float]:
        values: list[float] = []

        for row in rows:
            if classify_row(row) != kind:
                continue

            parsed = as_float(row.get(key))
            if parsed is not None:
                values.append(parsed)

        if not values:
            return None

        return sum(values) / len(values)

    return (
        mean_of_type("pp", "avg_ts"),
        mean_of_type("pp", "stddev_ts"),
        mean_of_type("tg", "avg_ts"),
        mean_of_type("tg", "stddev_ts"),
    )


def round_to_step(value: int, step: int) -> int:
    '''Rounds value to the nearest multiple of step (no-op when step <= 1).'''
    if step <= 1:
        return value

    return int(round(value / step) * step)


def midpoint_in_bracket(low: int, high: int, step: int) -> int:
    '''Returns a step-aligned midpoint strictly between low and high.'''
    mid = round_to_step((low + high) // 2, step)

    if mid <= low:
        mid = low + step

    if mid >= high:
        mid = high - step

    return mid


def get_stable_success_contexts(rows: list[object]) -> list[int]:
    '''Returns sorted contexts that succeeded and never failed at the same size.'''
    ok_ctx = {getattr(r, "context_size") for r in rows if getattr(r, "status") == "ok"}
    failed_ctx = {
        getattr(r, "context_size") for r in rows if getattr(r, "status") in ("failed", "failed_oom")
    }

    return sorted(ctx for ctx in ok_ctx if ctx not in failed_ctx)


def best_error_line(stderr: str) -> str:
    '''Picks the most informative error line from stderr, skipping known noise.'''
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "unknown error"

    def is_noise(line: str) -> bool:
        lower = line.lower()
        return (
            lower.startswith("ggml_cuda_init:")
            or lower.startswith("device ")
            or lower.startswith("warning:")
        )

    informative_patterns = [
        re.compile(r"failed to load model", re.IGNORECASE),
        re.compile(r"failed to create context", re.IGNORECASE),
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"out of memory", re.IGNORECASE),
        re.compile(r"common_fit_params.*error", re.IGNORECASE),
        re.compile(r"ggml_cuda_pool_vmm::alloc", re.IGNORECASE),
        re.compile(r"cuda error", re.IGNORECASE),
        re.compile(r"abort|aborted", re.IGNORECASE),
    ]

    non_noise = [line for line in lines if not is_noise(line)]
    for pattern in informative_patterns:
        for line in non_noise:
            if pattern.search(line):
                return line

    if non_noise:
        return non_noise[0]

    return lines[0]


def weighted_harmonic_mean(
    pp_ts: Optional[float],
    tg_ts: Optional[float],
    *,
    prompt_weight: float = 0.35,
    generation_weight: float = 0.65,
) -> Optional[float]:
    '''Returns the prompt/generation weighted harmonic mean, or None if inputs invalid.'''
    if pp_ts is None or tg_ts is None:
        return None

    if pp_ts <= 0 or tg_ts <= 0:
        return None

    total_weight = prompt_weight + generation_weight
    if total_weight <= 0:
        return None

    return total_weight / ((prompt_weight / pp_ts) + (generation_weight / tg_ts))


def mean_optional(values: list[Optional[float]]) -> Optional[float]:
    '''Returns the mean of the non-None values, or None if there are none.'''
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None

    return sum(filtered) / len(filtered)


def deployment_tier_for_score(
    score: Optional[float],
    breakpoints: list[tuple[str, float]],
) -> tuple[Optional[str], Optional[float]]:
    '''Returns the (tier label, threshold) for the highest breakpoint the score meets.'''
    if score is None:
        return None, None

    for label, minimum_score in breakpoints:
        if score >= minimum_score:
            return label, minimum_score

    return None, None


def select_deployment_rows(
    rows: list[object], kv_cache_type: str, max_context: Optional[int]
) -> list[object]:
    '''Returns the successful rows at the given KV cache type and max context.'''
    if max_context is None:
        return []

    return [
        row
        for row in rows
        if getattr(row, "kv_cache_type") == kv_cache_type
        and getattr(row, "context_size") == max_context
        and getattr(row, "status") == "ok"
    ]


def aggregate_runtime_by_context(rows: list[object]) -> dict[str, float]:
    '''Sums elapsed seconds per context size, keyed by context as a string.'''
    totals: dict[int, float] = {}

    for row in rows:
        context_size = int(getattr(row, "context_size"))
        elapsed = float(getattr(row, "elapsed_s"))
        totals[context_size] = totals.get(context_size, 0.0) + elapsed

    return {
        str(ctx): round(total_s, 3)
        for ctx, total_s in sorted(totals.items())
    }


def aggregate_runtime_by_phase(rows: list[object]) -> dict[str, float]:
    '''Sums elapsed seconds per phase (coarse, refine, verify).'''
    totals: dict[str, float] = {}

    for row in rows:
        phase = str(getattr(row, "phase"))
        elapsed = float(getattr(row, "elapsed_s"))
        totals[phase] = totals.get(phase, 0.0) + elapsed

    return {
        phase: round(total_s, 3)
        for phase, total_s in sorted(totals.items())
    }


def coarse_sizes_for_max(max_ctx: int) -> list[int]:
    '''Returns 4 context sizes for the coarse sweep: max//8, max//4, max//2, max.'''
    return [max_ctx >> 3, max_ctx >> 2, max_ctx >> 1, max_ctx]


def parse_devices(text: str) -> list[int]:
    '''Parses a comma-separated list of GPU device indices into a list of integers.
    Raises ValueError if no valid indices are found.'''

    values = [int(part.strip()) for part in text.split(",") if part.strip()]

    if not values:
        raise ValueError("At least one GPU device index must be provided")

    return values


def kv_label_for_type(kv_cache_type: str) -> str:
    '''Maps a KV cache type token to its short label (e.g. q8_0 -> q8).'''
    return KV_LABEL_MAP.get(kv_cache_type, kv_cache_type)


def read_total_vram_mib(devices: list[int]) -> int:
    '''Returns the summed total VRAM (MiB) of the given devices via nvidia-smi.
    Returns 0 if nvidia-smi is unavailable or returns no parseable rows.'''

    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        return 0

    total = 0

    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]

        if len(parts) != 2:
            continue

        try:
            idx = int(parts[0])
            mem_total = int(parts[1])

        except ValueError:
            continue

        if idx in devices:
            total += mem_total

    return total

