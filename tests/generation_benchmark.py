#!/usr/bin/env python3
"""
generation_benchmark.py

Realistic generation-rate benchmark for llama.cpp server.

Simulates a growing conversation, testing latency and generation rate
as context accumulates. Tests multiple slot configurations and KV cache
quantization strategies.

Usage:
    python tests/generation_benchmark.py [options]

    python tests/generation_benchmark.py \
        --config-csv tests/config/generation_benchmark/models.csv
    python tests/generation_benchmark.py \
        --model Qwen3.8-27B-UD-IQ4_XS.gguf --cache-k q8_0 --cache-v q8_0
    python tests/generation_benchmark.py \
        --slot-counts 1 2 4 --context-sizes 256 512 1024 4096

Environment:
    LLAMA_API_KEY: Bearer token for server (if --api-key set)
    LLAMA_BASE_URL: Server base URL (default: http://localhost:8502)
"""

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv required. Install with: pip install python-dotenv")
    sys.exit(1)

# Load .env
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

DEFAULT_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8502")
DEFAULT_API_KEY = os.getenv("LLAMA_API_KEY") or os.getenv("API_KEY", "")
DEFAULT_CONFIG_CSV = REPO_ROOT / "tests" / "config" / "generation_benchmark" / "models.csv"
DEFAULT_CONTEXT_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_SLOT_COUNTS = [1, 2, 4]
DEFAULT_REPETITIONS = 5
DEFAULT_MAX_TOKENS = 128
DEPLOY_SCRIPT = REPO_ROOT / "utils" / "deploy_service.sh"
RESULTS_DIR = REPO_ROOT / "tests" / "results" / "generation-benchmark"

# Real text corpus (from conversation/docs)
CORPUS = """
This is a focused discussion on optimization techniques for inference servers.

The main challenge with running large language models on consumer-grade GPUs like the P100
is balancing throughput, latency, and power consumption. Modern attention mechanisms
scale with context length, and even sparse or hybrid approaches still pay a real computational
cost as the sequence grows.

When deploying a model across multiple GPUs without NVLink, the cross-GPU synchronization
overhead becomes significant. Each layer must communicate between devices, adding latency
that compounds over 64 layers. This is fundamentally different from a single-GPU deployment,
where data stays local.

Quantization helps reduce VRAM usage and memory bandwidth requirements. Moving from 8-bit
weights to 4-bit, or further to 3-bit representations, trades model accuracy for speed and
memory efficiency. The hybrid attention architecture, where only a fraction of layers use
full attention while others use sliding-window approximations, naturally reduces the context
cost at deeper depths.

For interactive workloads like code assistance or technical writing, response latency matters
more than absolute throughput. Users expect sub-second interaction, which means 5 tokens per
second feels usable for a 500-token response (100 seconds total), but 0.2 tokens per second
(2500 seconds) is frustratingly slow.

Prompt caching can recover some performance by reusing KV cache across similar requests.
Multi-slot serving allows concurrent requests to queue and batch efficiently, amortizing
fixed costs like model loading across multiple users.

The sweet spot for most conversational deployments is 5-15 tokens per second generation
speed, with a context window that grows over time but doesn't exceed 64K tokens for any
single session. This requires careful benchmarking with realistic workloads.
""".strip()


def wait_for_health(url: str, max_retries: int = 30, api_key: str = "") -> bool:
    """Wait for server to report healthy."""
    for i in range(max_retries):
        try:
            req = urllib.request.Request(f"{url}/health")
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    print(f"  Server healthy after {i*5} seconds")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def deploy_model(
    model_file: str,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    slot_count: int = 1,
    api_key: str = "",
    cuda_device: Optional[str] = None,
    max_context_size: Optional[int] = None,
) -> bool:
    """Deploy the server with the specified model and cache settings."""
    gpu_note = f", cuda_device={cuda_device}" if cuda_device is not None else ""
    ctx_note = f", ctx_size={max_context_size}" if max_context_size is not None else ""
    print(
        f"  Deploying {model_file} with {slot_count} slots, "
        f"cache={cache_type_k}/{cache_type_v}{gpu_note}{ctx_note}"
    )
    
    try:
        # Update .env with new values
        env_content = ENV_PATH.read_text()
        
        # Replace only the MODEL, SLOTS, KV_CACHE_TYPE, and (optionally)
        # CUDA_DEVICE/TENSOR_SPLIT/SPLIT_MODE lines
        env_content = re.sub(
            r'^MODEL=.*$', f'MODEL={model_file}', env_content, flags=re.MULTILINE
        )
        env_content = re.sub(
            r'^SLOTS=.*$', f'SLOTS={slot_count}', env_content, flags=re.MULTILINE
        )
        env_content = re.sub(
            r'^KV_CACHE_TYPE=.*$', f'KV_CACHE_TYPE={cache_type_k}',
            env_content, flags=re.MULTILINE
        )
        if cuda_device is not None:
            env_content = re.sub(
                r'^CUDA_DEVICE=.*$', f'CUDA_DEVICE={cuda_device}',
                env_content, flags=re.MULTILINE
            )
            # Leave TENSOR_SPLIT empty so --fit auto-balances; setting it explicitly
            # disables --fit and can OOM at max context (see .env comment).
            env_content = re.sub(
                r'^TENSOR_SPLIT=.*$', 'TENSOR_SPLIT=', env_content, flags=re.MULTILINE
            )
            # 'layer' split-mode with a single visible device can take a slower code
            # path than 'none' - use 'none' for single-GPU so the comparison is fair.
            split_mode = 'none' if ',' not in cuda_device else 'layer'
            env_content = re.sub(
                r'^SPLIT_MODE=.*$', f'SPLIT_MODE={split_mode}',
                env_content, flags=re.MULTILINE
            )
        if max_context_size is not None:
            # --fit sizes its KV-cache budget off CTX_SIZE, not what we actually send -
            # leaving a stale (e.g. 262144) value here over-provisions the fit
            # calculation and can cause spurious CPU offload at much shallower depths.
            env_content = re.sub(
                r'^CTX_SIZE=.*$', f'CTX_SIZE={max_context_size}', env_content, flags=re.MULTILINE
            )

        ENV_PATH.write_text(env_content)

        # Run deploy script
        result = subprocess.run(
            [str(DEPLOY_SCRIPT), "--restart"],
            capture_output=True,
            timeout=120,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"    ERROR: deploy failed: {result.stderr}")
            return False
        print("  Deployment successful, waiting for health...")
        return wait_for_health(DEFAULT_URL, api_key=api_key)
    except subprocess.TimeoutExpired:
        print("  ERROR: deployment timeout")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def generate(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    url: str = DEFAULT_URL,
    api_key: str = "",
) -> dict:
    """
    Send a single generation request and return latency, token count, and tok/s.
    Returns latency (full round-trip), number of tokens generated, and tok/s rate.
    """
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
    }
    
    start = time.time()
    try:
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            latency = time.time() - start
            
            # Extract token count from response
            tokens_generated = data.get("usage", {}).get("completion_tokens", 0)
            tok_per_sec = tokens_generated / latency if latency > 0 else 0
            
            return {
                "latency": latency,
                "tokens": tokens_generated,
                "tok_per_sec": tok_per_sec,
                "success": True,
            }
    except urllib.error.HTTPError as e:
        return {
            "latency": time.time() - start,
            "tokens": 0,
            "tok_per_sec": 0,
            "success": False,
            "error": f"HTTP {e.code}",
        }
    except Exception as e:
        return {
            "latency": time.time() - start,
            "tokens": 0,
            "tok_per_sec": 0,
            "success": False,
            "error": str(e),
        }


CSV_FIELDNAMES = [
    "model", "cache_k", "cache_v", "slot_count", "context_size",
    "repetition", "latency", "tokens", "tok_per_sec", "success",
    "error", "timestamp", "gpu_resident", "layers_offloaded", "layers_total",
    "cpu_buffer_mib", "wrong_gpu",
]


def build_run_output_path(
    config_csv: Path, model_override: Optional[str], slot_counts: list[int],
    context_sizes: list[int], repetitions: int,
) -> Path:
    """
    Build a filename that identifies a run by its settings (not a timestamp), so
    re-running with the same settings appends/resumes into the same file instead
    of starting a new one.
    """
    config_label = model_override.replace(".gguf", "") if model_override else config_csv.stem
    slots_label = "-".join(str(s) for s in slot_counts)
    ctx_label = f"{min(context_sizes)}-{max(context_sizes)}"
    return RESULTS_DIR / (
        f"generation-benchmark_{config_label}_slots{slots_label}"
        f"_ctx{ctx_label}_reps{repetitions}.csv"
    )


def build_run_config_path(output_file: Path) -> Path:
    """Config file path paired with the output CSV, e.g. foo.csv -> foo.config.json."""
    return output_file.with_name(output_file.stem + ".config.json")


def write_run_config(
    config_path: Path, models_to_test: list[tuple[str, str, str]], args: argparse.Namespace,
) -> None:
    """
    Record the exact settings for this run next to its output CSV, so the run can
    be identified/reproduced later. Preserves the original 'created' timestamp
    across resumes; 'last_updated' reflects the most recent invocation.
    """
    created = datetime.now().isoformat(timespec="seconds")
    if config_path.exists():
        try:
            created = json.loads(config_path.read_text(encoding="utf-8")).get("created", created)
        except (json.JSONDecodeError, OSError):
            pass
    
    config = {
        "created": created,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "config_csv": str(args.config_csv) if not args.model else None,
        "model_override": args.model,
        "models": [
            {"model": m, "cache_type_k": ck, "cache_type_v": cv} for m, ck, cv in models_to_test
        ],
        "slot_counts": args.slot_counts,
        "context_sizes": args.context_sizes,
        "repetitions": args.repetitions,
        "cuda_device": args.cuda_device,
        "url": args.url,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_existing_rows(output_file: Path) -> list[dict]:
    """Load previously written rows from a prior (possibly interrupted) run."""
    if not output_file.exists():
        return []
    with open(output_file, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def condition_is_complete(
    existing_rows: list[dict], model: str, cache_k: str, cache_v: str,
    slot_count: int, context_sizes: list[int], repetitions: int,
) -> bool:
    """A model/cache/slot condition is complete once every context size has
    `repetitions` rows recorded for it (successful or not - a failed row still
    means that rep was attempted)."""
    matching = [
        r for r in existing_rows
        if r["model"] == model and r["cache_k"] == cache_k and r["cache_v"] == cache_v
        and r["slot_count"] == str(slot_count)
    ]
    for ctx in context_sizes:
        count = sum(1 for r in matching if r["context_size"] == str(ctx))
        if count < repetitions:
            return False
    return True


def append_rows(output_file: Path, rows: list[dict]) -> None:
    """Append rows to the run's CSV, writing the header only if the file is new."""
    is_new = not output_file.exists() or output_file.stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def check_gpu_residency(deploy_time: datetime) -> dict:
    """
    Parse the systemd journal for llama.cpp's own "load_tensors: offloaded X/Y
    layers to GPU" line, logged at LLAMA_ARG_LOG_VERBOSITY=4 (set in the unit
    file). This is ground truth from the loader itself - unlike comparing
    nvidia-smi memory against the model file size, it can't miss a partial
    offload of just a few layers, and it isn't fooled by KV cache/compute
    buffers padding out VRAM usage on top of a partially-offloaded model.
    """
    since = (deploy_time - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        log = subprocess.run(
            ["journalctl", "-u", "llamacpp.service", "--no-pager", "--since", since],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as e:
        print(f"    WARN: could not read journal for GPU offload check: {e}")
        return {"layers_offloaded": None, "layers_total": None, "gpu_resident": None,
                "cpu_buffer_mib": None, "wrong_gpu": None}
    
    offload_matches = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    cpu_buffer_matches = re.findall(r"CPU_Mapped model buffer size\s*=\s*([\d.]+) MiB", log)
    
    # Guard against ever silently landing on the wrong physical GPU (e.g. the small
    # 8GB GTX 1070 used for display, instead of the P100s) - a device mismatch here
    # invalidates any timing regardless of layer offload status.
    device_names = re.findall(r"-\s+CUDA\d+\s*:\s*(.+?)\s*\(\d+ MiB", log)
    unexpected_devices = [d for d in device_names if "P100" not in d]
    wrong_gpu = bool(unexpected_devices)
    if wrong_gpu:
        print(f"    !!! WRONG GPU WARNING: deployed on non-P100 device(s): {unexpected_devices}")
    
    if not offload_matches:
        print("    WARN: no 'offloaded X/Y layers to GPU' line found in journal "
              "since deploy - cannot confirm GPU residency")
        return {"layers_offloaded": None, "layers_total": None, "gpu_resident": None,
                "cpu_buffer_mib": None, "wrong_gpu": wrong_gpu}
    
    # Most recent match wins, in case the journal window caught an earlier restart too
    offloaded, total = (int(x) for x in offload_matches[-1])
    cpu_buffer_mib = float(cpu_buffer_matches[-1]) if cpu_buffer_matches else 0.0
    gpu_resident = offloaded == total
    
    status = "OK (fully GPU-resident)" if gpu_resident else "WARNING (CPU offload confirmed)"
    print(
        f"    GPU residency check: {status} - "
        f"offloaded {offloaded}/{total} layers, CPU buffer={cpu_buffer_mib:.0f} MiB"
    )
    
    return {
        "layers_offloaded": offloaded,
        "layers_total": total,
        "gpu_resident": gpu_resident,
        "cpu_buffer_mib": round(cpu_buffer_mib, 1),
        "wrong_gpu": wrong_gpu,
    }


def test_condition(
    model: str,
    cache_k: str,
    cache_v: str,
    slot_count: int,
    context_sizes: list[int],
    repetitions: int,
    api_key: str = "",
    cuda_device: Optional[str] = None,
) -> list[dict]:
    """
    Test a single condition: model + cache + slot count.
    Returns list of result dicts.
    """
    results = []
    
    # Deploy once for this condition, sizing CTX_SIZE to the deepest context this
    # run actually tests (plus headroom for the prompt suffix and generated tokens),
    # so --fit's KV-cache budget reflects reality instead of a stale prior value.
    deploy_ctx_size = max(context_sizes) + DEFAULT_MAX_TOKENS + 512
    deploy_time = datetime.now()
    if not deploy_model(
        model, cache_k, cache_v, slot_count, api_key, cuda_device, deploy_ctx_size
    ):
        print(f"  WARN: Failed to deploy {model}, skipping")
        return results
    
    # Warm up
    print(f"    Warming up...")
    generate("Hello", url=DEFAULT_URL, api_key=api_key)
    time.sleep(2)
    
    # Layer-to-GPU placement is decided once at model load time (llama.cpp's --fit
    # can silently reduce n_gpu_layers to stay within its memory-fit margin), so
    # one check per deploy is sufficient - it won't change mid-session.
    residency = check_gpu_residency(deploy_time)
    if residency["gpu_resident"] is False:
        print(
            f"    !!! SPILLOVER WARNING: {model} is NOT fully GPU-resident "
            f"({residency['layers_offloaded']}/{residency['layers_total']} layers on GPU, "
            f"{residency['cpu_buffer_mib']} MiB on CPU)"
        )
    
    # Test growing context
    conversation = ""
    for context_size in context_sizes:
        print(f"    Testing context_size={context_size}")
        
        # Add text until we reach target context size
        while len(conversation.split()) < context_size:
            lines = CORPUS.split(".")
            for line in lines:
                conversation += line.strip() + ". "
                if len(conversation.split()) >= context_size:
                    break
        
        # Trim to exact size (roughly)
        words = conversation.split()[:context_size]
        truncated_context = " ".join(words)
        
        # Run repetitions
        for rep in range(repetitions):
            prompt = truncated_context + "\n\nBriefly describe the key concepts mentioned above."
            result = generate(prompt, url=DEFAULT_URL, api_key=api_key)
            result.update({
                "model": model,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "slot_count": slot_count,
                "context_size": context_size,
                "repetition": rep + 1,
                "timestamp": datetime.now().isoformat(),
                **residency,
            })
            results.append(result)
            
            # Small delay between repetitions
            time.sleep(0.5)
    
    return results


def main():
    '''Main entry point for the generation benchmark script.'''

    parser = argparse.ArgumentParser(description="Realistic generation benchmark for llama.cpp")

    parser.add_argument(
        "--config-csv",
        type=Path,
        default=DEFAULT_CONFIG_CSV,
        help=f"Model config CSV (default: {DEFAULT_CONFIG_CSV})",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Override: specific model file to test",
    )

    parser.add_argument(
        "--cache-k",
        type=str,
        default="f16",
        help="Override: KV cache type for K (default: f16)",
    )

    parser.add_argument(
        "--cache-v",
        type=str,
        default="f16",
        help="Override: KV cache type for V (default: f16)",
    )

    parser.add_argument(
        "--slot-counts",
        type=int,
        nargs="+",
        default=DEFAULT_SLOT_COUNTS,
        help=f"Slot counts to test (default: {DEFAULT_SLOT_COUNTS})",
    )

    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_CONTEXT_SIZES,
        help=f"Context sizes to test (default: {DEFAULT_CONTEXT_SIZES})",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Repetitions per condition (default: {DEFAULT_REPETITIONS})",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"Server URL (default: {DEFAULT_URL})",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="API key if server has --api-key set",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: derived from run settings in the results dir, "
             "so re-running the same settings resumes/appends instead of starting over)",
    )

    parser.add_argument(
        "--cuda-device",
        type=str,
        default=None,
        help="Override CUDA_DEVICE in .env, e.g. '1' for single-GPU or '1,2' for dual-GPU. "
             "Leave unset to use whatever is already configured in .env.",
    )

    args = parser.parse_args()

    # Load model configs
    models_to_test = []

    if args.model:

        # Single model override
        models_to_test = [(args.model, args.cache_k, args.cache_v)] 

    else:

        # Load from CSV
        if not args.config_csv.exists():
            print(f"ERROR: config CSV not found: {args.config_csv}")
            sys.exit(1)

        with open(args.config_csv, "r", encoding="utf-8") as f:

            reader = csv.DictReader(f)

            for row in reader:
                models_to_test.append((
                    row["model"],
                    row.get("cache_type_k", "f16"),
                    row.get("cache_type_v", "f16"),
                ))

    # Prepare output
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = args.output or build_run_output_path(
        args.config_csv, args.model, args.slot_counts, args.context_sizes, args.repetitions
    )
    existing_rows = load_existing_rows(output_file)
    if existing_rows:
        print(f"Found existing results at {output_file} ({len(existing_rows)} rows) - resuming\n")

    print("\n" + "="*70)
    print("Realistic Generation Benchmark")
    print(f"  Models: {len(models_to_test)}")
    print(f"  Slot counts: {args.slot_counts}")
    print(f"  Context sizes: {args.context_sizes}")
    print(f"  Repetitions: {args.repetitions}")
    cuda_device_note = args.cuda_device if args.cuda_device is not None else "(unchanged from .env)"
    print(f"  CUDA device: {cuda_device_note}")
    print(f"  Output: {output_file}")

    config_path = build_run_config_path(output_file)
    write_run_config(config_path, models_to_test, args)
    print(f"  Run config: {config_path}")
    print(f"{'='*70}\n")

    # Test each model x slot count combination. Results are appended to disk after
    # each condition finishes, and conditions already fully recorded are skipped,
    # so an interrupted run can be resumed by invoking with the same settings.
    for model, cache_k, cache_v in models_to_test:
        for slot_count in args.slot_counts:
            if condition_is_complete(
                existing_rows, model, cache_k, cache_v, slot_count,
                args.context_sizes, args.repetitions,
            ):
                print(f"\nSkipping {model} (slots={slot_count}, cache={cache_k}/{cache_v}) "
                      f"- already complete")
                continue

            print(f"\nTesting {model} (slots={slot_count}, cache={cache_k}/{cache_v})")

            results = test_condition(
                model,
                cache_k,
                cache_v,
                slot_count,
                args.context_sizes,
                args.repetitions,
                args.api_key,
                args.cuda_device,
            )

            if results:
                append_rows(output_file, results)
                existing_rows.extend(
                    {k: str(v) for k, v in row.items()} for row in results
                )

    # Reload the full accumulated file (covers both resumed and freshly-run rows)
    all_results = load_existing_rows(output_file)

    if all_results:
        print(f"\n✓ Results at {output_file}")

        # Print summary stats
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}")

        # Group by model/cache/slot/context. Repetition 1 pays full prefill cost for that
        # context depth; reps 2+ hit the cached prefix (--cache-ram), so they isolate
        # steady-state decode throughput. Report both since they answer different questions.
        grouped = defaultdict(list)

        for r in all_results:
            if r.get("success") == "True":
                key = (r["model"], r["cache_k"], r["cache_v"], r["slot_count"], r["context_size"])
                grouped[key].append((int(r["repetition"]), float(r["tok_per_sec"])))

        print(
            f"{'model':40s} {'cfg':30s} {'cold(1st)':>10s} " + 
            f"{'steady mean':>12s} {'steady ±':>10s}"
        )

        def _sort_key(kv):
            model, cache_k, cache_v, slots, ctx = kv[0]
            return (model, cache_k, cache_v, int(slots), int(ctx))

        for (model, cache_k, cache_v, slots, ctx), reps in sorted(grouped.items(), key=_sort_key):
            reps.sort()
            cold_rate = reps[0][1] if reps else 0
            steady_rates = [v for n, v in reps if n != 1]
            mean_rate = statistics.mean(steady_rates) if steady_rates else 0
            stdev = statistics.stdev(steady_rates) if len(steady_rates) > 1 else 0
            cfg = f"slots={slots} ctx={ctx} cache={cache_k}/{cache_v}"
            print(f"{model:40s} {cfg:30s} {cold_rate:10.2f} {mean_rate:12.2f} {stdev:10.2f}")

        # Flag any model/slot condition where the model was not fully GPU-resident -
        # those rows' timings reflect partial CPU offload, not pure GPU throughput.
        # Residency is decided once at model load, so this is per model+slots, not per context.
        spillover = sorted({
            (r["model"], r["cache_k"], r["cache_v"], r["slot_count"],
             r["layers_offloaded"], r["layers_total"], r["cpu_buffer_mib"])
            for r in all_results if r.get("gpu_resident") == "False"
        })
        print(f"\n{'='*70}")
        if spillover:
            print(f"!!! GPU MEMORY SPILLOVER DETECTED in {len(spillover)} condition(s):")
            for model, cache_k, cache_v, slots, offloaded, total, cpu_mib in spillover:
                print(
                    f"    {model} cache={cache_k}/{cache_v} slots={slots} "
                    f"- only {offloaded}/{total} layers on GPU, {cpu_mib} MiB on CPU"
                )
        else:
            print("GPU residency check: no spillover detected - every condition ran fully on GPU.")

        wrong_gpu_models = sorted({
            (r["model"], r["cache_k"], r["cache_v"], r["slot_count"])
            for r in all_results if r.get("wrong_gpu") == "True"
        })
        if wrong_gpu_models:
            print(f"\n!!! WRONG GPU DETECTED in {len(wrong_gpu_models)} condition(s) "
                  f"(non-P100 device in use):")
            for model, cache_k, cache_v, slots in wrong_gpu_models:
                print(f"    {model} cache={cache_k}/{cache_v} slots={slots}")
        print(f"{'='*70}")

    else:
        print("\nERROR: No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
