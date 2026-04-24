#!/usr/bin/env python3
'''
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
    python tests/load_test.py
    python tests/load_test.py --levels 1 2 4 8 --requests 5
    python tests/load_test.py --url http://pyrite:8502 --stream
'''

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

except ImportError:
    print('ERROR: python-dotenv is required.  Install it with:  pip install python-dotenv')
    sys.exit(1)

try:
    import aiohttp

except ImportError:
    print('ERROR: aiohttp is required.  Install it with:  pip install aiohttp')
    sys.exit(1)

# Load .env from the repo root (one level above tests/)
_ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_URL = 'http://localhost:8502'
DEFAULT_LEVELS = [1, 2, 4, 8]
DEFAULT_REQUESTS_PER_LEVEL = 3  # repetitions per concurrency level (for averaging)
DEFAULT_PROMPT = 'In one sentence, explain what a transformer neural network is.'
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
    '''Send one chat completion request and return timing info.'''

    headers = {'Content-Type': 'application/json'}

    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': 'local',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'stream': stream,
    }

    url = f'{base_url}/v1/chat/completions'
    t_start = time.perf_counter()
    ttft: Optional[float] = None
    total_tokens = 0

    try:
        async with session.post(url, headers=headers, json=payload) as resp:

            if resp.status != 200:
                body = await resp.text()
                return {'error': f'HTTP {resp.status}: {body}', 'latency': None, 'ttft': None}

            if stream:

                # Read SSE stream line by line
                async for raw_line in resp.content:

                    line = raw_line.decode('utf-8').strip()

                    if not line or not line.startswith('data:'):
                        continue

                    data_str = line[len('data:'):].strip()

                    if data_str == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data_str)

                    except json.JSONDecodeError:
                        continue

                    # First chunk containing a token → TTFT
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')

                    if content and ttft is None:
                        ttft = time.perf_counter() - t_start

                    usage = chunk.get('usage') or {}
                    total_tokens = max(total_tokens, usage.get('completion_tokens', 0))

            else:
                body = await resp.json()
                ttft = None  # not measurable without streaming
                usage = body.get('usage') or {}
                total_tokens = usage.get('completion_tokens', 0)

        latency = time.perf_counter() - t_start
        return {'latency': latency, 'ttft': ttft, 'tokens': total_tokens, 'error': None}

    except aiohttp.ClientError as exc:
        return {'error': str(exc), 'latency': None, 'ttft': None, 'tokens': 0}


# ---------------------------------------------------------------------------
# One concurrency level
# ---------------------------------------------------------------------------

async def run_level(
    concurrency: int,
    n_reps: int,
    base_url: str,
    api_key: Optional[str],
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> list[dict]:
    '''
    Fire `concurrency` simultaneous requests, repeated `n_reps` times.

    Each repetition sends exactly `concurrency` requests at the same instant,
    waits for all to complete, then starts the next repetition. This ensures
    the concurrency level accurately reflects the number of parallel callers.
    '''

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=300)
    all_results = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        for _ in range(n_reps):

            tasks = [
                chat_request(session, base_url, api_key, prompt, max_tokens, stream)
                for _ in range(concurrency)
            ]

            results = await asyncio.gather(*tasks)
            all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    '''Return the p-th percentile of data (0-100).'''

    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = lo + 1

    if hi >= len(data):
        return data[lo]

    frac = idx - lo
    return data[lo] + frac * (data[hi] - data[lo])


def print_stats(label: str, values: list[float], unit: str = 's') -> None:
    '''Print min, mean, median, p95, and max for a list of float measurements.'''

    if not values:
        print(f'  {label}: no data')
        return

    print(
        f'  {label}: '
        f'min={min(values):.3f}{unit}  '
        f'mean={statistics.mean(values):.3f}{unit}  '
        f'median={statistics.median(values):.3f}{unit}  '
        f'p95={percentile(values, 95):.3f}{unit}  '
        f'max={max(values):.3f}{unit}'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    '''Run the load test across all configured concurrency levels and print results.'''

    base_url = args.url.rstrip('/')
    api_key = args.api_key  # may be None if server has no --api-key

    # All raw rows accumulated here for CSV output
    csv_rows: list[dict] = []
    run_ts = datetime.now().isoformat(timespec='seconds')

    print(f'Target : {base_url}')
    print(f'Prompt : {args.prompt!r}')
    print(f'Tokens : up to {args.max_tokens}')
    print(f'Stream : {args.stream}')
    print(f'Levels : {args.levels}')
    print(f'Repetitions per level: {args.requests}')
    print()

    separator = '─' * 72

    for concurrency in args.levels:

        print(separator)
        print(
            f'Concurrency = {concurrency}  ({concurrency} simultaneous ' +
            f'requests x {args.requests} repetitions)')

        t_wall_start = time.perf_counter()

        results = await run_level(
            concurrency=concurrency,
            n_reps=args.requests,
            base_url=base_url,
            api_key=api_key,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )

        wall_time = time.perf_counter() - t_wall_start
        errors = [r for r in results if r['error']]
        successes = [r for r in results if not r['error']]
        total = concurrency * args.requests

        # Accumulate raw rows for CSV
        for r in results:
            csv_rows.append({
                'timestamp': run_ts,
                'concurrency': concurrency,
                'latency_s': round(r['latency'], 4) if r['latency'] is not None else 'NaN',
                'ttft_s': round(r['ttft'], 4) if r['ttft'] is not None else 'NaN',
                'tokens': r['tokens'],
                'error': r['error'] if r['error'] else 'NaN',
            })

        print(f'  Success: {len(successes)}/{total}   Wall time: {wall_time:.2f}s')

        if errors:
            for e in errors:
                print(f"  ERROR: {e['error']}")

        if successes:

            latencies = [r['latency'] for r in successes]
            print_stats('Latency (total)', latencies)

            if args.stream:
                ttfts = [r['ttft'] for r in successes if r['ttft'] is not None]

                if ttfts:
                    print_stats('TTFT           ', ttfts)

            token_counts = [r['tokens'] for r in successes if r['tokens']]

            if token_counts:
                avg_tokens = statistics.mean(token_counts)
                avg_latency = statistics.mean(latencies)
                tps = avg_tokens / avg_latency if avg_latency > 0 else 0
                print(
                    f'  Avg tokens/response: {avg_tokens:.1f}   ' + 
                    f'Throughput: {tps:.1f} tok/s (aggregate)'
                )

    print(separator)
    print('Done.')

    # Write CSV
    if args.output:

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    'timestamp',
                    'concurrency',
                    'latency_s',
                    'ttft_s',
                    'tokens',
                    'error'
                ]
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f'Results saved to {out_path}')


def parse_args() -> argparse.Namespace:
    '''Parse and return command-line arguments.'''

    parser = argparse.ArgumentParser(
        description='Measure llama.cpp server latency vs. concurrency.'
    )

    parser.add_argument(
        '--url',
        default=os.environ.get('LLAMA_BASE_URL', DEFAULT_URL),
        help=f'Server base URL (default: {DEFAULT_URL}, or $LLAMA_BASE_URL)',
    )

    parser.add_argument(
        '--api-key',
        default=os.environ.get('LLAMA_API_KEY'),
        dest='api_key',
        help='API key (default: $LLAMA_API_KEY)',
    )

    parser.add_argument(
        '--levels',
        nargs='+',
        type=int,
        default=DEFAULT_LEVELS,
        metavar='N',
        help=f'Concurrency levels to test (default: {DEFAULT_LEVELS})',
    )

    parser.add_argument(
        '--requests',
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        metavar='N',
        help=f'Repetitions per concurrency level, for averaging (default: {DEFAULT_REQUESTS_PER_LEVEL})',
    )

    parser.add_argument(
        '--prompt',
        default=DEFAULT_PROMPT,
        help='Prompt to send to the model',
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest='max_tokens',
        help=f'Max completion tokens per request (default: {DEFAULT_MAX_TOKENS})',
    )

    _default_output = (
        Path(__file__).resolve().parent
        / 'results'
        / f'{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
    )

    parser.add_argument(
        '--output',
        default=str(_default_output),
        metavar='FILE',
        help=f'Path to write raw results as CSV (default: tests/results/YYYYmmdd_HHMM.csv)',
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming responses (enables TTFT measurement)',
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    asyncio.run(main(args))
