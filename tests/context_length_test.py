#!/usr/bin/env python3
'''
context_length_test.py: llama.cpp server latency vs. input context length

Measures how response latency scales with the length of the input prompt.
For each target context length, sends --concurrency simultaneous requests,
repeated --replicates times, and reports timing statistics.

Usage:
    python tests/context_length_test.py [options]

Environment:
    LLAMA_API_KEY   Bearer token (required if server has --api-key set)
    LLAMA_BASE_URL  Server base URL (default: http://localhost:8502)

Examples:
    python tests/context_length_test.py
    python tests/context_length_test.py --targets 128 512 2048 --replicates 10
    python tests/context_length_test.py --stream --concurrency 8
'''

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import aiohttp

except ImportError:
    print('ERROR: aiohttp is required.  Install it with:  pip install aiohttp')
    sys.exit(1)

try:
    from dotenv import load_dotenv

except ImportError:
    print('ERROR: python-dotenv is required.  Install it with:  pip install python-dotenv')
    sys.exit(1)

from helper_funcs.requests import chat_request
from helper_funcs.stats import print_stats

# Load .env from the repo root (one level above tests/)
_ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_URL = 'http://localhost:8502'
DEFAULT_CONCURRENCY = 4
DEFAULT_REPLICATES = 5
DEFAULT_MAX_TOKENS = 64
DEFAULT_TARGETS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Filler text repeated to build prompts of varying lengths.
_FILLER = (
    'The quick brown fox jumps over the lazy dog. '
    'Pack my box with five dozen liquor jugs. '
    'How vexingly quick daft zebras jump! '
    'The five boxing wizards jump quickly. '
)

# Rough English estimate used as fallback if /tokenize is unavailable.
_CHARS_PER_TOKEN = 4.0

_PREAMBLE = (
    'Read the following passage carefully, '
    'then summarize it in one sentence.\n\nPassage:\n\n'
)
_SUFFIX = '\n\nSummary:'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(target_tokens: int) -> str:
    '''Build a prompt that approximately fills target_tokens input tokens.'''

    overhead_tokens = int(len(_PREAMBLE + _SUFFIX) / _CHARS_PER_TOKEN)
    filler_chars = int((target_tokens - overhead_tokens) * _CHARS_PER_TOKEN)
    filler_chars = max(filler_chars, 1)
    reps = (filler_chars // len(_FILLER)) + 1
    filler = (_FILLER * reps)[:filler_chars]
    return _PREAMBLE + filler + _SUFFIX


async def get_token_count(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: Optional[str],
    text: str,
) -> int:
    '''Return the token count for text using the /tokenize endpoint.'''

    headers = {'Content-Type': 'application/json'}

    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        async with session.post(
            f'{base_url}/tokenize',
            headers=headers,
            json={'content': text},
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return len(data.get('tokens', []))
    except Exception:
        pass
    # Fallback: character-count estimate
    return int(len(text) / _CHARS_PER_TOKEN)


async def run_replicate(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: Optional[str],
    prompt: str,
    concurrency: int,
    max_tokens: int,
    stream: bool,
) -> list[dict]:
    '''Fire `concurrency` simultaneous requests with the given prompt.'''

    tasks = [
        chat_request(session, base_url, api_key, prompt, max_tokens, stream)
        for _ in range(concurrency)
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(cfg: argparse.Namespace) -> None:
    '''Run the context length test across all configured token targets and print results.'''

    base_url = cfg.url.rstrip('/')
    api_key = cfg.api_key

    csv_rows: list[dict] = []
    run_ts = datetime.now().isoformat(timespec='seconds')

    connector = aiohttp.TCPConnector(limit=cfg.concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        print(f'Target     : {base_url}')
        print(f'Slots      : {cfg.slots}')
        print(f'Concurrency: {cfg.concurrency}')
        print(f'Replicates : {cfg.replicates}')
        print(f'Max tokens : {cfg.max_tokens}')
        print(f'Stream     : {cfg.stream}')
        print(f'Targets    : {cfg.targets}')
        print()

        separator = '─' * 72

        for target in cfg.targets:

            prompt = build_prompt(target)
            actual_tokens = await get_token_count(session, base_url, api_key, prompt)

            print(separator)
            print(
                f'Target tokens: {target}  Actual: {actual_tokens}  '
                f'({cfg.concurrency} concurrent x {cfg.replicates} reps)'
            )

            all_results: list[dict] = []

            for _ in range(cfg.replicates):

                results = await run_replicate(
                    session, base_url, api_key,
                    prompt, cfg.concurrency, cfg.max_tokens, cfg.stream,
                )
                all_results.extend(results)

                for r in results:
                    csv_rows.append({
                        'timestamp': run_ts,
                        'slots': cfg.slots,
                        'target_tokens': target,
                        'prompt_tokens': actual_tokens,
                        'concurrency': cfg.concurrency,
                        'latency_s': round(r['latency'], 4) if r['latency'] is not None else 'NaN',
                        'ttft_s': round(r['ttft'], 4) if r['ttft'] is not None else 'NaN',
                        'output_tokens': r['tokens'],
                        'error': r['error'] if r['error'] else 'NaN',
                    })

            errors = [r for r in all_results if r['error']]
            successes = [r for r in all_results if not r['error']]
            total = cfg.concurrency * cfg.replicates

            print(f'  Success: {len(successes)}/{total}')

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

    print(separator)
    print('Done.')

    if cfg.output:

        out_path = Path(cfg.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    'timestamp',
                    'slots',
                    'target_tokens',
                    'prompt_tokens',
                    'concurrency',
                    'latency_s',
                    'ttft_s',
                    'output_tokens',
                    'error'
                ]
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f'Results saved to {out_path}')


def parse_args() -> argparse.Namespace:
    '''Parse and return command-line arguments.'''

    parser = argparse.ArgumentParser(
        description='Measure llama.cpp server latency vs. input context length.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--url',
        default=os.environ.get('LLAMA_BASE_URL', DEFAULT_URL),
        help='Server base URL (or $LLAMA_BASE_URL)',
    )

    parser.add_argument(
        '--api-key',
        default=os.environ.get('LLAMA_API_KEY'),
        dest='api_key',
        help='API key (or $LLAMA_API_KEY)',
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        type=int,
        default=DEFAULT_TARGETS,
        metavar='N',
        help='Target prompt token counts',
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar='N',
        help='Simultaneous requests per replicate',
    )

    parser.add_argument(
        '--replicates',
        type=int,
        default=DEFAULT_REPLICATES,
        metavar='N',
        help='Repetitions per context length (for averaging)',
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest='max_tokens',
        help='Max completion tokens per request',
    )

    parser.add_argument(
        '--slots',
        type=int,
        default=int(os.environ.get('LLAMA_SLOTS', '1')),
        metavar='N',
        help='Server slot count (recorded in CSV, or $LLAMA_SLOTS)',
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming responses (enables TTFT measurement)',
    )

    _default_output = (
        Path(__file__).resolve().parent
        / 'results'
        / f'context_test_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv'
    )
    parser.add_argument(
        '--output',
        default=str(_default_output),
        metavar='FILE',
        help='CSV output path',
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    asyncio.run(main(args))
