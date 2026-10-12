#!/usr/bin/env python3
"""
router_benchmark.py

Benchmarks every model preset defined in the GPU router's ini file
(utils/agent-router.ini by default) by sending each one a long text
generation request, twice in a row.

For each preset:
1. "cold" run - first request this session. Since the router loads models
   on demand and evicts the LRU model when --models-max is exceeded, this
   captures the full model-swap + load + prefill + first-decode latency.
2. "hot" run - immediate repeat. The model is already resident, so this
   isolates steady-state prefill/decode performance from load overhead.

Measures per run:
- Time to first token (TTFT), wall-clock from request send to first
  streamed token (reasoning or content).
- Prompt processing rate (tokens/sec) and prompt token count, from the
  server's own "timings" block in the final streamed chunk.
- Generation rate (tokens/sec) and generated token count, same source.
- VRAM used on both GPUs (nvidia-smi) and system RAM (/proc/meminfo),
  sampled before the model's first run and after each run. The "after"
  values are each model's absolute resident footprint while it is loaded -
  that is the headline number. We deliberately do NOT report before/after
  deltas: the router runs with --models-max 1 and evicts the previously
  resident preset lazily, only once the next preset's request actually
  arrives. That means the "before" sample for preset N still reflects
  preset N-1 (whatever was last resident), not an empty baseline, so a
  naive after-minus-before delta is really (size of N) - (size of N-1)
  and goes negative whenever N is smaller than N-1. Absolute "after"
  values avoid that trap entirely.

A model that fails to load or errors out is logged and skipped; the
benchmark continues with the next preset in the ini file.

Usage:
    python tests/router_benchmark.py
    python tests/router_benchmark.py --models qwen gemma-4
    python tests/router_benchmark.py --max-tokens 64 --target-prompt-tokens 6000
    python tests/router_benchmark.py --url http://localhost:8503 --api-key ...

Environment:
    LLAMA_ROUTER_URL: Router base URL (default: http://localhost:8503)
    LLAMA_ROUTER_KEY: ****** for the router (if --api-key not given)
"""

import argparse
import configparser
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv

except ImportError:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

DEFAULT_INI = REPO_ROOT / "utils" / "agent-router.ini"
DEFAULT_URL = os.getenv("LLAMA_ROUTER_URL", "http://localhost:8503")
DEFAULT_API_KEY = os.getenv("API_KEY", "")
RESULTS_DIR = REPO_ROOT / "tests" / "results" / "router-benchmark"

DEFAULT_MAX_TOKENS = 128
DEFAULT_TARGET_PROMPT_TOKENS = 4000
# Cold runs may need to evict a resident model, mmap a large (possibly
# multi-hundred-GB) GGUF, and page it in before the first token streams.
DEFAULT_COLD_TIMEOUT_S = 900
DEFAULT_HOT_TIMEOUT_S = 180
DEFAULT_CONNECT_TIMEOUT_S = 10

GPU_INDICES = [0, 1]

# Original passage, repeated to reach the target prompt length. Varied
# technical content avoids degenerate repetition artifacts in tokenization.
CORPUS = """
Serving several large language models from one GPU router means balancing
VRAM, system RAM, and scheduling between very different model shapes. A
small dense model fits entirely on one GPU and answers in well under a
second. A mixture-of-experts model many times larger can still run
acceptably if only its always-active layers live in VRAM while the bulk
of its expert weights are mapped from system memory and paged in as
needed, trading some generation speed for the ability to run at all.

Speculative decoding complicates this picture further. A small draft
model proposes several tokens ahead, and the large target model verifies
them in a single batched forward pass. When the draft model's
distribution matches the target closely, most proposals are accepted and
effective generation speed can be several times higher than naive
autoregressive decoding. When they diverge, the extra verification work
is pure overhead with no benefit. Matching quality depends on how the
draft model was trained relative to the target, not merely on
architecture family or parameter count.

Prefill and decode have very different performance characteristics.
Prefill processes the entire prompt as a single large batch, so it is
compute-bound and benefits enormously from bigger batch and micro-batch
sizes as well as tensor-core throughput. Decode processes one token at a
time per sequence, so it is memory-bandwidth-bound: the bottleneck is
how fast the accelerator can stream weights and KV-cache entries, not how
many floating point operations it can issue. This is why quantizing the
KV cache, not just the model weights, has an outsized effect on decode
throughput at long context lengths.

Time to first token is a composite metric. For a model that must be
loaded from disk or evicted from VRAM to make room for another, most of
that latency comes from mapping and paging in gigabytes of weights, not
from the prompt itself. For a model that is already resident, time to
first token is dominated by prefill time, which scales with prompt
length and batch size. Reporting both a cold and a hot measurement for
the same model separates these two very different costs, which a single
average would otherwise blur together.

Context window size interacts with all of the above. A longer context
means more KV-cache memory is reserved per sequence, which can force a
model to spill more of its own structure off the accelerator to make
room, independent of how long any individual prompt actually is. This is
one reason a model configured for a very large maximum context can be
slower in practice than its raw compute budget would suggest, even on a
short prompt, since the reservation itself consumes device memory before
a single token is processed.
""".strip()


def estimate_tokens(text: str) -> int:
    """Rough heuristic (~4 characters per token) for sizing the prompt."""
    return max(1, len(text) // 4)


def build_prompt(target_tokens: int) -> str:
    """Repeat CORPUS until it is at least target_tokens long (by estimate),
    then append a short instruction so every model is asked to do the same
    bounded amount of generation work regardless of prompt length."""
    parts = []
    total_chars = 0
    target_chars = target_tokens * 4

    while total_chars < target_chars:
        parts.append(CORPUS)
        total_chars += len(CORPUS) + 2

    body = "\n\n".join(parts)
    instruction = (
        "\n\nIn exactly three sentences, summarize the single most important "
        "idea from the text above."
    )
    return body + instruction


def with_cache_busting_nonce(body: str, nonce: str) -> str:
    """Prepend a short unique line so the server's prompt-prefix cache can't
    reuse an earlier run's KV cache. Without this, a "hot" run against the
    same prompt reprocesses only a handful of new tokens instead of the
    full prompt, making prompt_per_second meaningless for that run. The
    nonce goes at the very start, since prefix caching matches from the
    beginning of the prompt - any difference there forces full
    reprocessing."""
    return f"[benchmark run {nonce}]\n\n{body}"


def parse_presets(ini_path: Path) -> list[dict]:
    """Parse model preset sections from the router ini, in file order,
    skipping the global [*] defaults section."""
    parser = configparser.ConfigParser(strict=False)
    # Preserve key case so we can read hyphenated ini keys as-is.
    parser.optionxform = str
    with open(ini_path, encoding="utf-8") as f:
        parser.read_file(f)

    presets = []
    for section in parser.sections():
        if section == "*":
            continue
        values = dict(parser.items(section))
        request_model = values.get("alias", section)
        presets.append({
            "section": section,
            "request_model": request_model,
            "ctx_size": values.get("ctx-size"),
            "device": values.get("device", ""),
            "cpu_moe": values.get("cpu-moe", "false"),
        })
    return presets


def query_gpu_memory_mib() -> dict[int, Optional[int]]:
    """Return {gpu_index: used_MiB} via nvidia-smi, or None per GPU on failure."""
    result: dict[int, Optional[int]] = {i: None for i in GPU_INDICES}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError, FileNotFoundError) as exc:
        print(f"  WARNING: nvidia-smi query failed: {exc}")
        return result

    for line in proc.stdout.strip().splitlines():
        try:
            idx_str, used_str = [p.strip() for p in line.split(",")]
            idx = int(idx_str)
            if idx in result:
                result[idx] = int(used_str)
        except ValueError:
            continue
    return result


def query_system_ram_used_mib() -> Optional[int]:
    """Return used system RAM in MiB (MemTotal - MemAvailable) from /proc/meminfo."""
    try:
        meminfo = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                value_kib = rest.strip().split()[0]
                meminfo[key] = int(value_kib)
        total_kib = meminfo["MemTotal"]
        avail_kib = meminfo.get("MemAvailable", 0)
        return (total_kib - avail_kib) // 1024
    except (OSError, KeyError, ValueError, IndexError) as exc:
        print(f"  WARNING: /proc/meminfo read failed: {exc}")
        return None


def run_generation(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
    connect_timeout_s: int,
) -> dict:
    """Send one streaming chat completion request and return timing/result info."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    out = {
        "success": False,
        "error": None,
        "ttft_s": None,
        "wall_s": None,
        "prompt_n": None,
        "prompt_ms": None,
        "prompt_per_second": None,
        "predicted_n": None,
        "predicted_ms": None,
        "predicted_per_second": None,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=(connect_timeout_s, timeout_s),
        )

        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
            return out

        final_timings = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data_str = raw_line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}) if choices else {}
            token_text = delta.get("content") or delta.get("reasoning_content") or ""
            if token_text and out["ttft_s"] is None:
                out["ttft_s"] = time.perf_counter() - t0

            if "timings" in chunk:
                final_timings = chunk["timings"]

        out["wall_s"] = time.perf_counter() - t0

        if final_timings is not None:
            out["prompt_n"] = final_timings.get("prompt_n")
            out["prompt_ms"] = final_timings.get("prompt_ms")
            out["prompt_per_second"] = final_timings.get("prompt_per_second")
            out["predicted_n"] = final_timings.get("predicted_n")
            out["predicted_ms"] = final_timings.get("predicted_ms")
            out["predicted_per_second"] = final_timings.get("predicted_per_second")

        out["success"] = out["ttft_s"] is not None
        if not out["success"] and out["error"] is None:
            out["error"] = "no tokens streamed before connection closed"

    except requests.exceptions.RequestException as exc:
        out["error"] = str(exc) or exc.__class__.__name__

    return out


def benchmark_model(
    preset: dict,
    url: str,
    api_key: str,
    prompt_body: str,
    max_tokens: int,
    cold_timeout_s: int,
    hot_timeout_s: int,
    connect_timeout_s: int,
) -> list[dict]:
    """Run the cold then hot request for one preset. Returns one result row
    per attempted run (1 row if cold fails, 2 rows otherwise)."""
    rows = []
    model = preset["request_model"]
    section = preset["section"]

    baseline_gpu = query_gpu_memory_mib()
    baseline_ram = query_system_ram_used_mib()

    runs = [("cold", cold_timeout_s), ("hot", hot_timeout_s)]
    for label, timeout_s in runs:
        print(f"  [{section}] {label} run...", end=" ", flush=True)
        # Unique nonce per run so neither run can reuse the other's (or a
        # prior session's) cached prompt prefix - see with_cache_busting_nonce.
        prompt = with_cache_busting_nonce(prompt_body, f"{section}-{label}-{uuid.uuid4().hex[:8]}")
        result = run_generation(
            url, api_key, model, prompt, max_tokens, timeout_s, connect_timeout_s
        )

        after_gpu = query_gpu_memory_mib()
        after_ram = query_system_ram_used_mib()


        row = {
            "section": section,
            "request_model": model,
            "run": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **result,
        }
        # "before" is retained only for context/debugging - it reflects
        # whatever preset was last resident, not an empty baseline, since
        # the router evicts lazily (see module docstring). "after" is the
        # number that actually matters: this model's absolute resident
        # footprint while loaded.
        for i in GPU_INDICES:
            row[f"gpu{i}_used_mib_before"] = baseline_gpu.get(i)
            row[f"gpu{i}_used_mib_after"] = after_gpu.get(i)
        row["ram_used_mib_before"] = baseline_ram
        row["ram_used_mib_after"] = after_ram

        rows.append(row)

        if result["success"]:
            ttft = result["ttft_s"]
            pp = result["prompt_per_second"]
            tg = result["predicted_per_second"]
            print(f"OK (ttft={ttft:.2f}s pp={pp:.1f}tok/s tg={tg:.1f}tok/s)")
        else:
            print(f"FAILED ({result['error']})")
            print(f"  [{section}] skipping remaining runs for this model")
            break

    return rows


CSV_FIELDS = [
    "section", "request_model", "run", "timestamp",
    "success", "error",
    "ttft_s", "wall_s",
    "prompt_n", "prompt_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_second",
    "gpu0_used_mib_before", "gpu0_used_mib_after",
    "gpu1_used_mib_before", "gpu1_used_mib_after",
    "ram_used_mib_before", "ram_used_mib_after",
]


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (
        f"{'model':<20} {'run':<5} {'ok':<4} {'ttft_s':>8} {'pp_tok_s':>10} "
        f"{'tg_tok_s':>10} {'gpu0_mib':>10} {'gpu1_mib':>10} {'ram_mib':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ok = "yes" if row["success"] else "no"
        ttft = f"{row['ttft_s']:.2f}" if row.get("ttft_s") is not None else "-"
        pp = f"{row['prompt_per_second']:.1f}" if row.get("prompt_per_second") is not None else "-"
        tg = f"{row['predicted_per_second']:.1f}" if row.get("predicted_per_second") is not None else "-"
        # Absolute resident footprint while this model was loaded - see
        # module docstring for why this is reported instead of a delta.
        gpu0 = row.get("gpu0_used_mib_after")
        gpu1 = row.get("gpu1_used_mib_after")
        ram = row.get("ram_used_mib_after")
        print(
            f"{row['section']:<20} {row['run']:<5} {ok:<4} {ttft:>8} {pp:>10} "
            f"{tg:>10} {str(gpu0):>10} {str(gpu1):>10} {str(ram):>10}"
        )
        if not row["success"] and row.get("error"):
            print(f"    error: {row['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark every model preset in the router's ini file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ini", type=Path, default=DEFAULT_INI,
                         help=f"Router preset ini file (default: {DEFAULT_INI})")
    parser.add_argument("--url", default=DEFAULT_URL,
                         help=f"Router base URL (default: {DEFAULT_URL})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                         help="Router API key (default: $LLAMA_ROUTER_KEY)")
    parser.add_argument("--models", nargs="*", default=None,
                         help="Only benchmark these preset section names (default: all)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                         help=f"Max tokens to generate per run (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--target-prompt-tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS,
                         help=f"Approx. prompt length in tokens (default: {DEFAULT_TARGET_PROMPT_TOKENS})")
    parser.add_argument("--cold-timeout", type=int, default=DEFAULT_COLD_TIMEOUT_S,
                         help=f"Read timeout in seconds for the cold run (default: {DEFAULT_COLD_TIMEOUT_S})")
    parser.add_argument("--hot-timeout", type=int, default=DEFAULT_HOT_TIMEOUT_S,
                         help=f"Read timeout in seconds for the hot run (default: {DEFAULT_HOT_TIMEOUT_S})")
    parser.add_argument("--output", type=Path, default=None,
                         help="Output CSV path (default: tests/results/router-benchmark/TIMESTAMP.csv)")
    args = parser.parse_args()

    if not args.ini.exists():
        print(f"ERROR: ini file not found: {args.ini}")
        return 1

    presets = parse_presets(args.ini)
    if args.models:
        presets = [p for p in presets if p["section"] in args.models]
        if not presets:
            print(f"ERROR: none of {args.models} found in {args.ini}")
            return 1

    if not args.api_key:
        print("WARNING: no API key set (--api-key / $LLAMA_ROUTER_KEY) - "
              "requests will fail if the router requires auth.")

    prompt_body = build_prompt(args.target_prompt_tokens)
    print(f"Prompt built: ~{estimate_tokens(prompt_body)} estimated tokens "
          f"({len(prompt_body)} chars)")
    print(f"Router URL: {args.url}")
    print(f"Presets to test ({len(presets)}): "
          f"{', '.join(p['section'] for p in presets)}")
    print()

    all_rows = []
    for preset in presets:
        print(f"=== {preset['section']} (model={preset['request_model']}, "
              f"ctx={preset['ctx_size']}, device={preset['device'] or 'default'}, "
              f"cpu-moe={preset['cpu_moe']}) ===")
        try:
            rows = benchmark_model(
                preset,
                args.url,
                args.api_key,
                prompt_body,
                args.max_tokens,
                args.cold_timeout,
                args.hot_timeout,
                DEFAULT_CONNECT_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - keep the loop alive no matter what
            print(f"  UNEXPECTED ERROR benchmarking {preset['section']}: {exc}")
            rows = [{
                "section": preset["section"],
                "request_model": preset["request_model"],
                "run": "cold",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "success": False,
                "error": f"unexpected exception: {exc}",
            }]
        all_rows.extend(rows)
        print()

    output_path = args.output or (
        RESULTS_DIR / f"router-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    write_csv(all_rows, output_path)
    print(f"Results written to {output_path}")

    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
