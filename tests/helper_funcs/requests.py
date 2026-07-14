'''
requests.py: HTTP request helpers for the llama.cpp load tester.

Provides chat_request() for sending a single chat completion request
and run_level() for firing a batch of concurrent requests.
'''

import sys
import asyncio
import json
import time
from typing import Optional

try:
    import aiohttp

except ImportError:
    print('ERROR: aiohttp is required.  Install it with:  pip install aiohttp')
    sys.exit(1)


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

                return {
                    'error': f'HTTP {resp.status}: {body}',
                    'latency': None, 'ttft': None, 'tokens': 0
                }

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

    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        error_msg = str(exc).strip() or exc.__class__.__name__
        return {'error': error_msg, 'latency': None, 'ttft': None, 'tokens': 0}


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

        for rep_idx in range(n_reps):

            batch_start = time.perf_counter()
            tasks = [
                chat_request(session, base_url, api_key, prompt, max_tokens, stream)
                for _ in range(concurrency)
            ]

            results = await asyncio.gather(*tasks)
            batch_wall_s = time.perf_counter() - batch_start

            success_results = [
                r for r in results
                if not r['error'] and r['latency'] is not None
            ]
            batch_tokens = sum(r['tokens'] for r in success_results if r['tokens'])
            batch_success_count = len(success_results)
            batch_aggregate_tps = (
                batch_tokens / batch_wall_s if batch_wall_s > 0 else 0.0
            )

            for result in results:
                result['replicate_id'] = rep_idx
                result['batch_wall_s'] = batch_wall_s
                result['batch_tokens'] = batch_tokens
                result['batch_success_count'] = batch_success_count
                result['batch_aggregate_tps'] = batch_aggregate_tps

            all_results.extend(results)

    return all_results
