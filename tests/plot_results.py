#!/usr/bin/env python3
"""
plot_results.py

Regenerate the benchmark figures embedded in the repo README from the raw
result artifacts. Reads the latest generation-benchmark CSVs and the
context-fit dual_gpu summaries, and writes PNGs into <repo>/assets/.

Usage:
    .venv/bin/python tests/plot_results.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"
GEN_DIR = REPO_ROOT / "tests" / "results" / "generation-benchmark"
CFIT_DUAL = REPO_ROOT / "tests" / "results" / "context_fit" / "dual_gpu"

SHORT_NAMES = {
    "Qwen3.8-27B-UD-Q4_K_M": "Qwen 27B Q4_K_M",
    "Qwen3.8-27B-UD-IQ4_XS": "Qwen 27B IQ4_XS",
    "Qwen3.8-27B-UD-Q3_K_XL": "Qwen 27B Q3_K_XL",
    "Qwen3.8-27B-UD-IQ3_S": "Qwen 27B IQ3_S",
    "gemma-4-31B-it-Q4_K_M": "gemma 31B Q4_K_M",
    "gemma-4-26B-A4B-it-UD-IQ4_XS": "gemma 26B-A4B IQ4_XS",
    "gemma-4-26B-A4B-it-UD-Q3_K_M": "gemma 26B-A4B Q3_K_M",
    "GLM-4.7-Flash-Q4_K_M": "GLM-4.7-Flash Q4_K_M",
    "GLM-4.7-Flash-UD-IQ3_XXS": "GLM-4.7-Flash IQ3_XXS",
    "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M": "Mistral 24B Q4_K_M",
    "Mistral-Small-3.2-24B-Instruct-2506-UD-Q3_K_XL": "Mistral 24B Q3_K_XL",
    "gpt-oss-20b-Q4_K_M": "gpt-oss-20b",
}


def short(model: str) -> str:
    '''Maps a GGUF filename to a compact label for plot legends.'''
    return SHORT_NAMES.get(model.replace(".gguf", ""), model.replace(".gguf", ""))


def latest(pattern: str) -> Path:
    '''Returns the most recently modified generation-benchmark CSV matching pattern.'''
    matches = sorted(GEN_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        print(f"ERROR: no generation-benchmark CSV matches {pattern!r} in {GEN_DIR}")
        sys.exit(1)
    return matches[0]


def ctx_label(n: int) -> str:
    '''Formats a context size as a short 'k' token count.'''
    return f"{n // 1024}k" if n >= 1024 else str(n)


def steady_tps(csv_path: Path) -> dict[str, list[tuple[int, float]]]:
    '''Returns {model: [(context, steady_mean_tok_per_sec), ...]}, averaging the
    repetitions after the cold first one (matching the benchmark's steady mean).'''
    reps: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["success"] != "True":
                continue
            key = (row["model"], int(row["context_size"]), int(row["repetition"]))
            reps[key].append(float(row["tok_per_sec"]))

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (model, ctx, rep), values in reps.items():
        if rep > 1:  # drop the cold first repetition
            grouped[model][ctx].extend(values)

    return {
        model: [(ctx, statistics.mean(vals)) for ctx, vals in sorted(per_ctx.items())]
        for model, per_ctx in grouped.items()
    }


def plot_generation(csv_path: Path, title: str, out: Path) -> None:
    '''Line plot of steady generation rate vs context depth, one series per model.'''
    data = steady_tps(csv_path)
    order = sorted(data, key=lambda m: -data[m][0][1])  # fastest at low context first
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, model in enumerate(order):
        xs = [c for c, _ in data[model]]
        ys = [t for _, t in data[model]]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=cmap(i % 10), label=short(model))

    ax.set_xscale("log", base=2)
    xticks = sorted({c for series in data.values() for c, _ in series})
    ax.set_xticks(xticks)
    ax.set_xticklabels([ctx_label(c) for c in xticks])
    ax.set_xlabel("Context depth (tokens)")
    ax.set_ylabel("Generation rate (tokens/s, single stream)")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(REPO_ROOT))


def plot_max_context(out: Path) -> None:
    '''Grouped bar chart of the max verified context per model and KV-cache type.'''
    rows = []
    for model_dir in sorted(CFIT_DUAL.iterdir()):
        summary = model_dir / "summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        f16 = data["runs"].get("f16", {}).get("max_context")
        q8 = data["runs"].get("q8", {}).get("max_context")
        if f16 is None or q8 is None:
            continue
        rows.append((short(data["model"]), f16 / 1024, q8 / 1024))

    if not rows:
        print(f"ERROR: no context-fit summaries found in {CFIT_DUAL}")
        sys.exit(1)

    rows.sort(key=lambda r: -max(r[1], r[2]))
    labels = [r[0] for r in rows]
    f16_vals = [r[1] for r in rows]
    q8_vals = [r[2] for r in rows]
    x = range(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_f16 = ax.bar([i - width / 2 for i in x], f16_vals, width, label="f16 KV", color="#1f77b4")
    bars_q8 = ax.bar([i + width / 2 for i in x], q8_vals, width, label="q8_0 KV", color="#ff7f0e")
    ax.bar_label(bars_f16, fmt="%.0fk", fontsize=8, padding=2)
    ax.bar_label(bars_q8, fmt="%.0fk", fontsize=8, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Max verified context (k tokens)")
    ax.set_title("Max GPU-resident context, 2x P100 (dual GPU, Q4_K_M weights)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(REPO_ROOT))


def main() -> None:
    '''Regenerates all README figures into <repo>/assets/.'''
    ASSETS.mkdir(exist_ok=True)
    plot_max_context(ASSETS / "context-fit-max-context.png")
    plot_generation(
        latest("*dual_gpu*slots1*.csv"),
        "Generation rate vs context - dual GPU (row split, Q4_K_M, f16 KV)",
        ASSETS / "generation-rate-dual-gpu.png",
    )
    plot_generation(
        latest("*single_gpu*slots1*.csv"),
        "Generation rate vs context - single GPU (f16 KV)",
        ASSETS / "generation-rate-single-gpu.png",
    )


if __name__ == "__main__":
    main()
