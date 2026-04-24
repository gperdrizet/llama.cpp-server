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
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from helper_funcs.requests import run_level
from helper_funcs.stats import print_stats


try:
    from dotenv import load_dotenv

except ImportError:
    print('ERROR: python-dotenv is required.  Install it with:  pip install python-dotenv')
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
# Main
# ---------------------------------------------------------------------------

async def main(cfg: argparse.Namespace) -> None:
    '''Run the load test across all configured concurrency levels and print results.'''

    base_url = cfg.url.rstrip('/')
    api_key = cfg.api_key  # may be None if server has no --api-key

    # All raw rows accumulated here for CSV output
    csv_rows: list[dict] = []
    run_ts = datetime.now().isoformat(timespec='seconds')

    print(f'Target : {base_url}')
    print(f'Prompt : {cfg.prompt!r}')
    print(f'Tokens : up to {cfg.max_tokens}')
    print(f'Stream : {cfg.stream}')
    print(f'Levels : {cfg.levels}')
    print(f'Repetitions per level: {cfg.requests}')
    print()

    separator = '─' * 72

    for concurrency in cfg.levels:

        print(separator)
        print(
            f'Concurrency = {concurrency}  ({concurrency} simultaneous ' +
            f'requests x {cfg.requests} repetitions)')

        t_wall_start = time.perf_counter()

        results = await run_level(
            concurrency=concurrency,
            n_reps=cfg.requests,
            base_url=base_url,
            api_key=api_key,
            prompt=cfg.prompt,
            max_tokens=cfg.max_tokens,
            stream=cfg.stream,
        )

        wall_time = time.perf_counter() - t_wall_start
        errors = [r for r in results if r['error']]
        successes = [r for r in results if not r['error']]
        total = concurrency * cfg.requests

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

            if cfg.stream:
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
    if cfg.output:

        out_path = Path(cfg.output)
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
        help=f'Concurrency levels to test (default: {DEFAULT_LEVELS})'
    )

    parser.add_argument(
        '--requests',
        type=int,
        default=DEFAULT_REQUESTS_PER_LEVEL,
        metavar='N',
        help=(
            'Repetitions per concurrency level, for averaging ' +
            f'(default: {DEFAULT_REQUESTS_PER_LEVEL})'
        )
    )

    parser.add_argument(
        '--prompt',
        default=DEFAULT_PROMPT,
        help='Prompt to send to the model'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest='max_tokens',
        help=f'Max completion tokens per request (default: {DEFAULT_MAX_TOKENS})'
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
        help='Path to write raw results as CSV (default: tests/results/YYYYmmdd_HHMM.csv)'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming responses (enables TTFT measurement)'
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    asyncio.run(main(args))
