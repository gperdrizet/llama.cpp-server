#!/usr/bin/env python3
'''
run_llama_bench.py - structured wrapper around llama-bench.

Runs a YAML-defined set of llama-bench test cases, captures CSV output, appends
suite metadata columns, and writes a combined results CSV plus a text log.

This is the canonical benchmark workflow for model / hardware comparisons.
'''

import argparse
import csv
import io
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml

except ImportError:
    print('ERROR: PyYAML is required. Install it with: pip install pyyaml')
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'tests' / 'results' / 'llama_bench'
ENV_FILE = REPO_ROOT / '.env'


def read_env(path: Path) -> dict[str, str]:
    '''Read the .env file and return a dictionary of key-value pairs.'''

    env: dict[str, str] = {}

    if not path.exists():
        print(f'WARNING: .env file not found at {path}')
        return env

    for line in path.read_text().splitlines():

        line = line.strip()

        if not line or line.startswith('#') or '=' not in line:
            continue

        key, _, val = line.partition('=')
        env[key.strip()] = val.strip()

    return env


def as_list(value, default=None):
    '''Return the value as a list. If the value is None, returns empty list or
    the provided default.'''

    if value is None:
        return [] if default is None else default

    if isinstance(value, list):
        return value

    return [value]


def build_bench_bin(global_cfg: dict, env_cfg: dict) -> Path:
    '''Return the path to the llama-bench binary, using the global configuration
    and environment configuration.'''

    if global_cfg.get('bench_bin'):
        return Path(global_cfg['bench_bin'])

    llama_path = env_cfg.get('LLAMA_PATH', '/opt/llama.cpp')

    return Path(llama_path) / 'build' / 'bin' / 'llama-bench'


def build_model_path(case: dict, global_cfg: dict, env_cfg: dict) -> Path:
    '''Return the absolute path to the model file, using the case, global
    configuration, and environment configuration.'''

    model = case['model']
    model_path = Path(model)

    if model_path.is_absolute():
        return model_path

    model_dir = global_cfg.get('model_dir') or env_cfg.get('MODEL_DIR', '/opt/models')

    return Path(model_dir) / model


def workload_label(n_prompt: int, n_gen: int) -> str:
    '''Return a standardized label for a workload based on the number of prompt
    and generated tokens.'''

    return f'pp{n_prompt}_tg{n_gen}'


def normalize_flash_attn(value) -> str:
    '''Normalize the flash attention setting to 'on', 'off', or 'auto'.'''

    if isinstance(value, bool):
        return 'on' if value else 'off'

    text = str(value).strip().lower()

    if text in {'on', 'off', 'auto'}:
        return text

    return 'on'


def build_command(
        bench_bin: Path,
        model_path: Path,
        case: dict,
        workload: dict,
        repetitions: int
) -> list[str]:
    '''Build the command line arguments for running a benchmark case with a specific workload.'''

    cmd = [
        str(bench_bin),
        '-m', str(model_path),
        '-p', str(workload['n_prompt']),
        '-n', str(workload['n_gen']),
        '-r', str(repetitions),
        '-o', 'csv',
        '-ngl', str(case.get('n_gpu_layers', -1)),
        '-sm', str(case.get('split_mode', 'none')),
        '-fa', normalize_flash_attn(case.get('flash_attn', 'on')),
    ]

    if case.get('main_gpu') is not None:
        cmd += ['-mg', str(case['main_gpu'])]
    if case.get('tensor_split'):
        cmd += ['-ts', str(case['tensor_split'])]
    if case.get('threads') is not None:
        cmd += ['-t', str(case['threads'])]
    if case.get('batch_size') is not None:
        cmd += ['-b', str(case['batch_size'])]
    if case.get('ubatch_size') is not None:
        cmd += ['-ub', str(case['ubatch_size'])]
    if case.get('no_warmup'):
        cmd += ['--no-warmup']
    if case.get('embeddings'):
        cmd += ['-embd', '1']

    return cmd


def workloads_for_case(case: dict) -> list[dict]:
    '''Return a list of workloads for the given benchmark case, normalizing the input format.'''

    if case.get('workloads'):
        result = []

        for workload in case['workloads']:
            result.append({
                'label': workload.get('label') or workload_label(
                    workload['n_prompt'], workload['n_gen']
                ),
                'n_prompt': int(workload['n_prompt']),
                'n_gen': int(workload['n_gen']),
            })

        return result

    n_prompts = as_list(case.get('n_prompt'), default=[512])
    n_gens = as_list(case.get('n_gen'), default=[128])
    result = []

    for n_prompt in n_prompts:
        for n_gen in n_gens:
            result.append({
                'label': workload_label(int(n_prompt), int(n_gen)),
                'n_prompt': int(n_prompt),
                'n_gen': int(n_gen),
            })

    return result


def run_case(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    '''Run a single benchmark case with the specified command and environment.'''

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )

    return result


def parse_csv_rows(stdout: str) -> list[dict[str, str]]:
    '''Parse CSV rows from the given standard output string and return a list of dictionaries.'''

    lines = [line for line in stdout.splitlines() if line.strip()]

    if not lines:
        return []

    return list(csv.DictReader(io.StringIO('\n'.join(lines))))


def main() -> None:
    '''Entry point for running the llama-bench benchmark suites based on the provided YAML config.'''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run llama-bench benchmark suites from YAML configs'
    )

    parser.add_argument(
        'config',
        help='Path to YAML benchmark config'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Override default output run name'
    )

    args = parser.parse_args()

    # Load the YAML benchmark config
    config_path = Path(args.config)

    if not config_path.exists():
        print(f'ERROR: config not found: {config_path}')
        sys.exit(1)

    with config_path.open(encoding='utf-8') as f:
        bench_cfg = yaml.safe_load(f)

    # Read environment and global configuration
    env_cfg = read_env(ENV_FILE)
    global_cfg = bench_cfg.get('global', {})
    bench_bin = build_bench_bin(global_cfg, env_cfg)

    # Ensure the llama-bench binary exists before proceeding
    if not bench_bin.exists():
        print(f'ERROR: llama-bench not found: {bench_bin}')
        sys.exit(1)

    # Setup the output directories and file paths
    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{config_path.stem}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f'{run_name}.csv'
    out_log = RESULTS_DIR / f'{run_name}.log'

    # Read the number of repetitions for each benchmark case
    repetitions = int(global_cfg.get('repetitions', 5))

    # Initialize the suite rows and failures lists
    suite_rows: list[dict[str, str]] = []
    failures: list[str] = []

    # Open the log file for writing
    with out_log.open('w', encoding='utf-8') as log:

        # Iterate over each benchmark case in the configuration
        for case in bench_cfg['cases']:

            # Build the model path for the current case
            model_path = build_model_path(case, global_cfg, env_cfg)

            # Skip the case if the model path does not exist
            if not model_path.exists():
                msg = f"SKIP {case['label']}: model not found: {model_path}"
                print(msg)
                log.write(msg + '\n')
                failures.append(msg)
                continue

            # Set up the environment variables for CUDA devices
            visible_devices = str(case.get('cuda_visible_devices', '')).strip()
            case_env = os.environ.copy()
            case_env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

            # Apply the CUDA_VISIBLE_DEVICES setting if specified
            if visible_devices:
                case_env['CUDA_VISIBLE_DEVICES'] = visible_devices

            elif 'CUDA_VISIBLE_DEVICES' in case_env:
                del case_env['CUDA_VISIBLE_DEVICES']

            # Iterate over each workload for the current case
            for workload in workloads_for_case(case):

                # Build the llama-bench command for the current workload
                cmd = build_command(bench_bin, model_path, case, workload, repetitions)
                print(f"{case['label']} / {workload['label']}: {' '.join(cmd)}")
                log.write(f"\n[{case['label']} / {workload['label']}]\n")
                log.write('CMD: ' + ' '.join(cmd) + '\n')
                log.write(f"CUDA_VISIBLE_DEVICES={visible_devices or '<unset>'}\n")

                # Skip execution if this is a dry run
                if args.dry_run:
                    continue

                # Execute the command and capture the result
                result = run_case(cmd, case_env)

                # Check for errors in the command execution and log them
                if result.stderr:
                    log.write('--- stderr ---\n')
                    log.write(result.stderr)

                    if not result.stderr.endswith('\n'):
                        log.write('\n')

                # Check for standard output and log it
                if result.stdout:
                    log.write('--- stdout ---\n')
                    log.write(result.stdout)
                    if not result.stdout.endswith('\n'):
                        log.write('\n')

                # Parse the CSV rows from the command's standard output
                rows = parse_csv_rows(result.stdout)

                # Handle the case where the command failed or produced no CSV rows
                if result.returncode != 0 or not rows:
                    msg = (
                        f"FAIL {case['label']} / {workload['label']}: " + 
                       f"exit={result.returncode} rows={len(rows)}"
                    )

                    print(msg)
                    log.write(msg + '\n')
                    failures.append(msg)

                    continue

                # Process each parsed CSV row and add additional metadata
                for row in rows:
                    row['suite_run'] = run_name
                    row['suite_config'] = config_path.name
                    row['case_label'] = case['label']
                    row['workload_label'] = workload['label']
                    row['cuda_visible_devices'] = visible_devices
                    row['split_mode_cfg'] = str(case.get('split_mode', 'none'))
                    row['tensor_split_cfg'] = str(case.get('tensor_split', ''))
                    row['model_path_cfg'] = str(model_path)
                    suite_rows.append(row)

    # Finish dry run
    if args.dry_run:
        print('Dry run complete.')
        print(f'Would write CSV to {out_csv}')
        print(f'Would write log to {out_log}')
        return

    # Write the collected suite rows to the output CSV if any exist
    if suite_rows:

        fieldnames = [
            'suite_run', 'suite_config', 'case_label', 'workload_label',
            'cuda_visible_devices', 'split_mode_cfg', 'tensor_split_cfg', 'model_path_cfg',
        ] + [key for key in suite_rows[0].keys() if key not in {
            'suite_run', 'suite_config', 'case_label', 'workload_label',
            'cuda_visible_devices', 'split_mode_cfg', 'tensor_split_cfg', 'model_path_cfg',
        }]

        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(suite_rows)

        print(f'Wrote {out_csv}')

    else:
        print('No successful llama-bench rows captured.')

    # Print failures if any occurred
    if failures:
        print('\nFailures:')

        for failure in failures:
            print(f'  - {failure}')

    # Print the location of the log file
    print(f'Log: {out_log}')


if __name__ == '__main__':
    main()
