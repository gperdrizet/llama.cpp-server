from __future__ import annotations

import csv
import io
import re
from typing import Optional

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
    if raw_breakpoints is None:
        breakpoints = list(default_breakpoints)
    elif isinstance(raw_breakpoints, dict):
        breakpoints = [(str(label), float(min_score)) for label, min_score in raw_breakpoints.items()]
    else:
        raise ValueError("score_breakpoints must be a mapping of label -> minimum_score")

    return sorted(breakpoints, key=lambda item: item[1], reverse=True)


def score_breakpoints_to_dict(breakpoints: list[tuple[str, float]]) -> dict[str, float]:
    return {label: minimum_score for label, minimum_score in breakpoints}


def detect_oom_like_failure(return_code: int, stdout: str, stderr: str) -> bool:
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
    if step <= 1:
        return value

    return int(round(value / step) * step)


def midpoint_in_bracket(low: int, high: int, step: int) -> int:
    mid = round_to_step((low + high) // 2, step)

    if mid <= low:
        mid = low + step

    if mid >= high:
        mid = high - step

    return mid


def get_stable_success_contexts(rows: list[object]) -> list[int]:
    ok_ctx = {getattr(r, "context_size") for r in rows if getattr(r, "status") == "ok"}
    failed_ctx = {getattr(r, "context_size") for r in rows if getattr(r, "status") in ("failed", "failed_oom")}

    return sorted(ctx for ctx in ok_ctx if ctx not in failed_ctx)


def best_error_line(stderr: str) -> str:
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
    if pp_ts is None or tg_ts is None:
        return None

    if pp_ts <= 0 or tg_ts <= 0:
        return None

    total_weight = prompt_weight + generation_weight
    if total_weight <= 0:
        return None

    return total_weight / ((prompt_weight / pp_ts) + (generation_weight / tg_ts))


def mean_optional(values: list[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None

    return sum(filtered) / len(filtered)


def deployment_tier_for_score(
    score: Optional[float],
    breakpoints: list[tuple[str, float]],
) -> tuple[Optional[str], Optional[float]]:
    if score is None:
        return None, None

    for label, minimum_score in breakpoints:
        if score >= minimum_score:
            return label, minimum_score

    return None, None


def select_deployment_rows(rows: list[object], kv_cache_type: str, max_context: Optional[int]) -> list[object]:
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
    totals: dict[str, float] = {}

    for row in rows:
        phase = str(getattr(row, "phase"))
        elapsed = float(getattr(row, "elapsed_s"))
        totals[phase] = totals.get(phase, 0.0) + elapsed

    return {
        phase: round(total_s, 3)
        for phase, total_s in sorted(totals.items())
    }
