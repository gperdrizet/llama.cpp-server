"""Dataclasses shared across the context-fit benchmark modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunResult:
    '''Holds the outcome and measurements of a single context-size run.'''
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
    '''Holds the aggregated boundary result for one KV-cache quantization run.'''
    kv_cache_label: str
    kv_cache_type: str
    max_success_ctx: Optional[int]
    first_fail_ctx: Optional[int]
    bisect_fail_ctx: Optional[int]
    boundary_stable: Optional[bool]
    warnings: list[str]
