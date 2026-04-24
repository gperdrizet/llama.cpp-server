#!/usr/bin/env python3
"""
load_test.py: llama.cpp server latency test

Measures response latency as a function of concurrent callers.
For each concurrency level, fires a batch of requests simultaneously
and reports timing statistics.

Usage:
    python tests/load_test.py [options]

Environment:
    LLAMA_API_KEY   Bearer token (required if server has --api-key set)
    LLAMA_BASE_URL  Server base URL (default: http://localhost:8502)

Examples:
    LLAMA_API_KEY=your_key python tests/load_test.py
    LLAMA_API_KEY=your_key python tests/load_test.py --levels 1 2 4 8 --requests 5
    LLAMA_API_KEY=your_key python tests/load_test.py --url http://pyrite:8502 --stream
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required.  Install it with:  pip install aiohttp")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://localhost:8502"
DEFAULT_LEVELS = [1, 2, 4, 8]
DEFAULT_REQUESTS_PER_LEVEL = 4  # concurrent requests fired simultaneously
DEFAULT_PROMPT = "In one sentence, explain what a transformer neural network is."
DEFAULT_MAX_TOKENS = 128


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

async def chat_request(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: Optional[str],
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> dict:
    """Send one chat completion request and return timing info."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }

    url = f"{base_url}/v1/chat/completions"
    t_start = time.perf_counter()
    ttft: Optional[float] = None
    total_tokens = 0

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"error": f"HTTP {resp.status}: {body}", "latency": None, "ttft": None}

            if stream:
                # Read SSE stream line by line
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # First chunk containing a token → TTFT
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft is None:
                        ttft = time.perf_counter() - t_start
                    usage = chunk.get("usage") or {}
                    total_tokens = max(total_tokens, usage.get("completion_tokens", 0))
            else:
                body = await resp.json()
                ttft = None  # not measurable without streaming
                usage = body.get("usage") or {}
                total_tokens = usage.get("completion_tokens", 0)

        latency = time.perf_counter() - t_start
        return {"latency": latency, "ttft": ttft, "tokens": total_tokens, "error": None}

    except aiohttp.ClientError as exc:
        return {"error": str(exc), "latency": None, "ttft": None, "tokens": 0}


# ---------------------------------------------------------------------------
# One concurrency level
# ---------------------------------------------------------------------------

async def run_level(
    concurrency: int,
    n_requests: int,
    base_url: str,
    api_key: Optional[str],
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> list[dict]:
    """Fire `n_requests` requests concurrently and collect results."""
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            chat_request(session, base_url, api_key, prompt, max_tokens, stream)
            for _ in range(n_requests)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1
    if hi >= len(data):
        return data[lo]
    frac = idx - lo
    return data[lo] + frac * (data[hi] - data[lo])


def print_stats(label: str, values: list[float], unit: str = "s") -> None:
    if not values:
        print(f"  {label}: no data")
        return
    print(
        f"  {label}: "
        f"min={min(values):.3f}{unit}  "
        f"mean={statistics.mean(values):.3f}{unit}  "
        f"median={statistics.median(values):.3f}{unit}  "
        f"p95={percentile(values, 95):.3f}{unit}  "
        f"max={max(values):.3f}{unit}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    base_url = args.url.rstrip("/")
    api_key = args.api_key  # may be None if server has no --api-key

    print(f"Target : {base_url}")
    print(f"Prompt : {args.prompt!r}")
    print(f"Tokens : up to {args.max_tokens}")
    print(f"Stream : {args.stream}")
    print(f"Levels : {args.levels}")
    print(f"Requests per level: {args.requests}")
    print()

    separator = "─" * 72

    for concurrency in args.levels:
        print(separator)
        print(f"Concurrency = {concurrency}  ({args.requests} requests fired simultaneously)")

        t_wall_start = time.perf_counter()
        results = await run_level(
            concurrency=concurrency,
            n_requests=args.requests,
            base_url=base_url,
            api_key=api_key,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )
        wall_time = time.perf_counter() - t_wall_start

        errors = [r for r in results if r["error"]]
        successes = [r for r in results if not r["error"]]

        print(f"  Success: {len(successes)}/{args.requests}   Wall time: {wall_time:.2f}s")

        if errors:
            for e in errors:
                print(f"  ERROR: {e['error']}")

        if successes:
            latencies = [r["latency"] for r in successes]
            print_stats("Latency (total)", latencies)

            if args.stream:
                ttfts = [r["ttft"] for r in successes if r["ttft"] is not None]
                if ttfts:
                    print_stats("TTFT           ", ttfts)

            token_counts = [r["tokens"] for r in successes if r["tokens"]]
            if token_counts:
                avg_tokens = statistics.mean(token_counts)
                avg_latency = statistics.mean(latencies)
                tps = avg_tokens / avg_latency if avg_latency > 0 else 0
                print(f"  Avg tokens/response: {avg_tokens:.1f}   Throughput: {tps:.1f} tok/s (aggregate)")

    print(separator)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure llama.cpp server latency vs. concurrency."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LLAMA_BASE_URL", DEFAULT_URL),
        help=f"Server base URL (default: {DEFAULT_URL}, or $LLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMA_API_KEY"),
        dest="api_key",
        help="API key (default: $LLAMA_API_KEY)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=DEFAULT_LEVELS,
        metavar="N",
        help=f"Concurrency levels to test (default: {DEFAULT_LEVELS})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        metavar="N",
        help=f"Number of simultaneous requests per level (default: {DEFAULT_REQUESTS_PER_LEVEL})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest="max_tokens",
        help=f"Max completion tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses (enables TTFT measurement)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
