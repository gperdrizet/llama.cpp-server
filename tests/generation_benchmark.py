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
from datetime import datetime
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
) -> bool:
    """Deploy the server with the specified model and cache settings."""
    gpu_note = f", cuda_device={cuda_device}" if cuda_device is not None else ""
    print(
        f"  Deploying {model_file} with {slot_count} slots, "
        f"cache={cache_type_k}/{cache_type_v}{gpu_note}"
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
    
    # Deploy once for this condition
    if not deploy_model(model, cache_k, cache_v, slot_count, api_key, cuda_device):
        print(f"  WARN: Failed to deploy {model}, skipping")
        return results
    
    # Warm up
    print(f"    Warming up...")
    generate("Hello", url=DEFAULT_URL, api_key=api_key)
    time.sleep(2)
    
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
        help="Output CSV (default: auto-generated in results dir)",
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

    output_file = args.output or (
        RESULTS_DIR / f"generation-benchmark-{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    print("\n" + "="*70)
    print("Realistic Generation Benchmark")
    print(f"  Models: {len(models_to_test)}")
    print(f"  Slot counts: {args.slot_counts}")
    print(f"  Context sizes: {args.context_sizes}")
    print(f"  Repetitions: {args.repetitions}")
    cuda_device_note = args.cuda_device if args.cuda_device is not None else "(unchanged from .env)"
    print(f"  CUDA device: {cuda_device_note}")
    print(f"  Output: {output_file}")
    print(f"{'='*70}\n")

    all_results = []

    # Test each model x slot count combination
    for model, cache_k, cache_v in models_to_test:
        for slot_count in args.slot_counts:
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

            all_results.extend(results)

    # Write results
    if all_results:
        fieldnames = [
            "model", "cache_k", "cache_v", "slot_count", "context_size",
            "repetition", "latency", "tokens", "tok_per_sec", "success",
            "error", "timestamp"
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n Results written to {output_file}")

        # Print summary stats
        print(f"\n{'='*70}")
        print("Summary Statistics")
        print(f"{'='*70}")

        # Group by model/cache/slot/context. Repetition 1 pays full prefill cost for that
        # context depth; reps 2+ hit the cached prefix (--cache-ram), so they isolate
        # steady-state decode throughput. Report both since they answer different questions.
        grouped = defaultdict(list)

        for r in all_results:
            if r.get("success"):
                key = (r["model"], r["cache_k"], r["cache_v"], r["slot_count"], r["context_size"])
                grouped[key].append((r["repetition"], r["tok_per_sec"]))

        print(
            f"{'model':40s} {'cfg':30s} {'cold(1st)':>10s} " + 
            f"{'steady mean':>12s} {'steady ±':>10s}"
        )

        for (model, cache_k, cache_v, slots, ctx), reps in sorted(grouped.items()):
            reps.sort()
            cold_rate = reps[0][1] if reps else 0
            steady_rates = [v for n, v in reps if n != 1]
            mean_rate = statistics.mean(steady_rates) if steady_rates else 0
            stdev = statistics.stdev(steady_rates) if len(steady_rates) > 1 else 0
            cfg = f"slots={slots} ctx={ctx} cache={cache_k}/{cache_v}"
            print(f"{model:40s} {cfg:30s} {cold_rate:10.2f} {mean_rate:12.2f} {stdev:10.2f}")

    else:
        print("\nERROR: No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
