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
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from helper_funcs.requests import run_level

try:
    import yaml
except ImportError:
    yaml = None


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
DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32]
DEFAULT_REQUESTS_PER_LEVEL = 3  # repetitions per concurrency level (for averaging)
DEFAULT_PROMPT = 'In one sentence, explain what a transformer neural network is.'
DEFAULT_MAX_TOKENS = 128
RESULTS_ROOT = Path(__file__).resolve().parent / 'results' / 'load-test'
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / '.env'
DEPLOY_SCRIPT = REPO_ROOT / 'utils' / 'deploy_service.sh'


def print_stats(label: str, values: list[float]) -> None:
    if not values:
        return
    values = sorted(values)
    n = len(values)
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    print(
        f'  {label}: min={values[0]:.3f}s  mean={statistics.mean(values):.3f}s  '
        f'median={statistics.median(values):.3f}s  p95={values[p95_idx]:.3f}s  max={values[-1]:.3f}s'
    )


def _read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding='utf-8').splitlines()


def _set_env_values(path: Path, updates: dict[str, str]) -> None:
    lines = _read_env_lines(path)
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in line:
            out.append(line)
            continue
        key, _, _ = line.partition('=')
        key = key.strip()
        if key in updates:
            out.append(f'{key}={updates[key]}')
            seen.add(key)
        else:
            out.append(line)

    for key, value in updates.items():
        if key not in seen:
            out.append(f'{key}={value}')

    path.write_text('\n'.join(out) + '\n', encoding='utf-8')


def _sanitize_label(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', text).strip('-').lower() or 'case'


def _build_case_output_path(case_label: str, slots: int, run_date: str) -> Path:
    run_dir = f'{run_date}_{_sanitize_label(case_label)}_slots{slots}'
    return RESULTS_ROOT / run_dir / 'load_test.csv'


def _run_deploy(restart: bool) -> None:
    cmd = ['bash', str(DEPLOY_SCRIPT)]
    if restart:
        cmd.append('--restart')
    print(f'Redeploying service: {" ".join(cmd)}')
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _wait_for_server(base_url: str, api_key: str | None, timeout_s: int = 120) -> bool:
    health_url = f'{base_url.rstrip("/")}/health'
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        req = urllib.request.Request(health_url, method='GET')
        if api_key:
            req.add_header('Authorization', f'Bearer {api_key}')
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    return False


def _env_default(*keys: str, fallback: str | None = None) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value != '':
            return value
    return fallback


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
    print(f'Slots  : {cfg.slots}')
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
        errors = [r for r in results if r['error'] or r['latency'] is None]
        successes = [r for r in results if not r['error'] and r['latency'] is not None]
        total = concurrency * cfg.requests

        # Accumulate raw rows for CSV
        for r in results:
            csv_rows.append({
                'timestamp': run_ts,
                'model': cfg.model_label,
                'ctx_size': cfg.ctx_size if cfg.ctx_size is not None else 'unknown',
                'slots': cfg.slots,
                'concurrency': concurrency,
                'replicate_id': r.get('replicate_id', -1),
                'batch_wall_s': round(r.get('batch_wall_s', float('nan')), 4),
                'batch_tokens': r.get('batch_tokens', 0),
                'batch_success_count': r.get('batch_success_count', 0),
                'batch_aggregate_tps': round(r.get('batch_aggregate_tps', float('nan')), 4),
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

                batch_tps = [r.get('batch_aggregate_tps', 0.0) for r in results if r.get('batch_success_count', 0) > 0]
                if batch_tps:
                    print(
                        '  Batch throughput      : ' +
                        f'min={min(batch_tps):.1f}  mean={statistics.mean(batch_tps):.1f}  max={max(batch_tps):.1f} tok/s'
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
                    'model',
                    'ctx_size',
                    'slots',
                    'concurrency',
                    'replicate_id',
                    'batch_wall_s',
                    'batch_tokens',
                    'batch_success_count',
                    'batch_aggregate_tps',
                    'latency_s',
                    'ttft_s',
                    'tokens',
                    'error'
                ]
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f'Results saved to {out_path}')


async def run_suite(args: argparse.Namespace) -> None:
    if yaml is None:
        print('ERROR: PyYAML is required for --suite-config. Install it with: pip install pyyaml')
        sys.exit(1)

    suite_path = Path(args.suite_config)
    if not suite_path.exists():
        print(f'ERROR: suite config not found: {suite_path}')
        sys.exit(1)

    if not ENV_PATH.exists():
        print(f'ERROR: required env file not found: {ENV_PATH}')
        sys.exit(1)

    if not DEPLOY_SCRIPT.exists():
        print(f'ERROR: deploy script not found: {DEPLOY_SCRIPT}')
        sys.exit(1)

    with suite_path.open('r', encoding='utf-8') as f:
        suite_cfg = yaml.safe_load(f) or {}

    global_cfg = suite_cfg.get('global', {})
    cases = suite_cfg.get('cases', [])
    if not cases:
        print('ERROR: suite config has no cases')
        sys.exit(1)

    backup_env_text = ENV_PATH.read_text(encoding='utf-8')
    run_date = datetime.now().strftime('%Y-%m-%d')

    deploy_enabled = bool(global_cfg.get('deploy', True))
    deploy_restart = bool(global_cfg.get('deploy_restart', True))
    restore_deploy = bool(global_cfg.get('restore_deploy', True))
    dry_run = bool(args.dry_run)

    try:
        for case in cases:
            label = str(case.get('label') or case.get('model') or 'case')
            model = str(case.get('model', '')).strip()
            slots = int(case.get('slots', global_cfg.get('slots', _env_default('SLOTS', fallback='1'))))
            ctx_size = int(case.get('ctx_size', global_cfg.get('ctx_size', _env_default('CTX_SIZE', fallback='4096'))))
            gpu_layers = int(case.get('gpu_layers', global_cfg.get('gpu_layers', _env_default('GPU_LAYERS', fallback='-1'))))
            cuda_device = str(case.get('cuda_device', global_cfg.get('cuda_device', _env_default('CUDA_DEVICE', fallback='0'))))
            tensor_split = str(case.get('tensor_split', global_cfg.get('tensor_split', _env_default('TENSOR_SPLIT', fallback=''))))
            prompt_cache_size = int(case.get('prompt_cache_size', global_cfg.get('prompt_cache_size', _env_default('PROMPT_CACHE_SIZE', fallback='0'))))

            base_url = str(case.get('url', global_cfg.get('url', _env_default('BASE_URL', 'LLAMA_BASE_URL', fallback=DEFAULT_URL))))
            api_key = case.get('api_key', global_cfg.get('api_key', _env_default('API_KEY', 'LLAMA_API_KEY', fallback='')))
            levels = case.get('levels', global_cfg.get('levels', DEFAULT_LEVELS))
            requests = int(case.get('requests', global_cfg.get('requests', DEFAULT_REQUESTS_PER_LEVEL)))
            prompt = str(case.get('prompt', global_cfg.get('prompt', DEFAULT_PROMPT)))
            max_tokens = int(case.get('max_tokens', global_cfg.get('max_tokens', DEFAULT_MAX_TOKENS)))
            stream = bool(case.get('stream', global_cfg.get('stream', False)))

            if deploy_enabled:
                updates = {
                    'MODEL': model,
                    'CUDA_DEVICE': cuda_device,
                    'CTX_SIZE': str(ctx_size),
                    'GPU_LAYERS': str(gpu_layers),
                    'SLOTS': str(slots),
                    'PROMPT_CACHE_SIZE': str(prompt_cache_size),
                    'TENSOR_SPLIT': tensor_split,
                }
                if dry_run:
                    print(f'[dry-run] Would update .env keys: {", ".join(sorted(updates.keys()))}')
                    print(f'[dry-run] Would run deploy script: bash {DEPLOY_SCRIPT} {"--restart" if deploy_restart else ""}'.strip())
                else:
                    _set_env_values(ENV_PATH, updates)
                    _run_deploy(restart=deploy_restart)
                    if not _wait_for_server(base_url, str(api_key) if api_key else None):
                        print(f'ERROR: service did not become healthy after deploy for case {label}; skipping case')
                        continue

            output_path = _build_case_output_path(label, slots, run_date)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print('\n' + '=' * 80)
            print(f'Case: {label}')
            print(f'  model={model} ctx_size={ctx_size} slots={slots} gpu_layers={gpu_layers} cuda_device={cuda_device}')
            print(f'  url={base_url} levels={levels} requests={requests} stream={stream}')
            print(f'  output={output_path}')

            case_cfg = argparse.Namespace(
                url=base_url,
                api_key=api_key,
                levels=[int(x) for x in levels],
                requests=requests,
                prompt=prompt,
                max_tokens=max_tokens,
                output=str(output_path),
                slots=slots,
                stream=stream,
                model_label=model,
                ctx_size=ctx_size,
            )
            if dry_run:
                print('[dry-run] Would run load test with above parameters')
            else:
                await main(case_cfg)
    finally:
        ENV_PATH.write_text(backup_env_text, encoding='utf-8')
        if deploy_enabled and restore_deploy and not dry_run:
            try:
                print('\nRestoring original deployed service configuration from .env backup...')
                _run_deploy(restart=True)
            except subprocess.CalledProcessError as exc:
                print(f'WARNING: failed to redeploy original configuration: {exc}')


def parse_args() -> argparse.Namespace:
    '''Parse and return command-line arguments.'''

    parser = argparse.ArgumentParser(
        description='Measure llama.cpp server latency vs. concurrency.'
    )

    parser.add_argument(
        '--suite-config',
        default=None,
        metavar='FILE',
        help='Run a YAML-defined benchmark suite (uses deploy_service.sh between cases)',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='With --suite-config, print planned actions without redeploying or sending requests',
    )

    parser.add_argument(
        '--url',
        default=_env_default('BASE_URL', 'LLAMA_BASE_URL', fallback=DEFAULT_URL),
        help=f'Server base URL (default: {DEFAULT_URL}, or $BASE_URL/$LLAMA_BASE_URL)',
    )

    parser.add_argument(
        '--api-key',
        default=_env_default('API_KEY', 'LLAMA_API_KEY'),
        dest='api_key',
        help='API key (default: $API_KEY/$LLAMA_API_KEY)',
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
        / f'load_test_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.csv'
    )

    parser.add_argument(
        '--output',
        default=str(_default_output),
        metavar='FILE',
        help=(
            'Path to write raw results as CSV (default: ' +
            'tests/results/load_test_YYYY-mm-dd_HH-MM.csv)'
        )
    )

    parser.add_argument(
        '--slots',
        type=int,
        default=int(_env_default('SLOTS', 'LLAMA_SLOTS', fallback='1')),
        metavar='N',
        help='Number of parallel slots the server is configured with (default: $SLOTS/$LLAMA_SLOTS or 1)'
    )

    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use streaming responses (enables TTFT measurement)'
    )

    parser.add_argument(
        '--model-label',
        default='',
        dest='model_label',
        metavar='LABEL',
        help='Model identifier recorded in CSV (default: empty string)',
    )

    parser.add_argument(
        '--ctx-size',
        type=int,
        default=None,
        dest='ctx_size',
        metavar='N',
        help='Server context size in tokens, recorded in CSV',
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    if args.suite_config:
        asyncio.run(run_suite(args))
    else:
        asyncio.run(main(args))
