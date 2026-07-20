# tests directory quick reference

Minimal metadata for what each file and subdirectory is for.

## Top-level

| Path | Purpose |
|---|---|
| `context_fit.py` | Context-size benchmark runner (coarse scan, bisection refinement, verification, CSV/log/summary/plot outputs). |
| `fit-context-size.sh` | Legacy shell wrapper for direct `llama-bench` context probing. |
| `load_test.py` | API load/latency test runner (concurrency levels, repeat batches, optional deploy/restart workflow). |
| `results/` | Benchmark and load-test output artifacts (organized by benchmark). |
| `config/` | Benchmark configuration files, organized by benchmark type. |
| `benchmarks/` | Legacy leftover directory from earlier load-test layout; currently expected to be empty. |
| `helper_funcs/` | Shared helper modules used by test runners. |

## config/

| Path | Purpose |
|---|---|
| `config/context_fit/` | Context-size benchmark configs and model lists (`context_fit.yaml`, fast discovery, single-model refs). |
| `config/load_test/` | Load-test suite YAMLs for API benchmarking scenarios. |
| `config/performance/` | Reserved for model performance benchmark configs/evals. |

## config/load_test/

| Path | Purpose |
|---|---|
| `load-test-GTP-OSS-20B.yaml` | End-to-end load-test suite for GPT-OSS-20B scenarios (1 GPU and 2 GPU slot variants). |
| `load-test-Qwen3.6-27B-Q4_K_M.yaml` | Multi-GPU load-test suite for Qwen3.6-27B-Q4_K_M across context sizes. |

## helper_funcs/

| Path | Purpose |
|---|---|
| `requests.py` | Async HTTP request helpers used by `load_test.py` (single request + concurrent batch execution). |
| `__pycache__/` | Python bytecode cache (auto-generated). |

## Notes

- Context-fit runs write per-run artifacts under `tests/results/context-size/...`.
- Context-fit artifact names are `results.csv`, `run.log`, `summary.json`, and `plot.png`.
- Load-test runs write artifacts under `tests/results/load-test/...`.
- Load-test suites in `tests/config/load_test/*.yaml` are consumed by `tests/load_test.py`.
- Prefer YAML configs over `fit-context-size.sh` for repeatable benchmark runs.

## results/ layout

| Path | Purpose |
|---|---|
| `results/context-size/` | Context-size benchmark outputs grouped by config/profile and run name. |
| `results/load-test/` | API load-test outputs grouped by date/case/slots. |
