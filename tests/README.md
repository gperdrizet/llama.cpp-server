# Benchmark tooling

Utilities for tuning and measuring the llama.cpp inference stack. Run every
command below from the repository root.

| Tool | What it does |
| --- | --- |
| `llama_server_optuna_bench.py` | Finds the best server settings for a model |
| `llama_context_tuner.py` | Finds the largest context window a model can safely use |
| `inference_api_bench.py` | Benchmarks the live API at `promptlyapi.com` |
| `make_bench_prompt.py` | Generates the long test prompts the first two tools use |

The first two launch models locally in Docker. `inference_api_bench.py` talks
to a remote endpoint over HTTPS and needs no GPU or Docker.

> **⚠️ The local tools take the production models offline.** To free VRAM they
> stop the `llama-cuda` container at startup and restart it when they finish —
> or if they crash. While a run is in progress, the pyrite-local models are
> unavailable, including the model this session may be using. Don't run them
> casually. `--no-stop-production` opts out, but only works if production is
> already down.

## Running a tool

Each tool takes a JSON config:

```bash
.venv/bin/python tests/bench/<tool>.py --config tests/config/<tool>/<config>.json
```

Add `--dry-run` to validate the config and see the plan without launching
containers or making network calls. Results go to `tests/results/<tool>/`.

## Finding the best server settings

`llama_server_optuna_bench.py` searches combinations of server flags — flash
attention, split mode, cache dtypes, batch sizes, thread counts, speculative
decoding — using [Optuna](https://optuna.org/). It tries to find settings that
work well *together*, rather than tuning one flag at a time.

Each trial is scored on a fixed reference workload, so a model can't "win" by
stopping early or by being given a shorter prompt. The exact formula is in the
tool's module docstring; the important thing is that all trials are measured
on the same footing.

```bash
# check the config without launching anything
.venv/bin/python tests/bench/llama_server_optuna_bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --dry-run

# run 20 trials
.venv/bin/python tests/bench/llama_server_optuna_bench.py \
    --config tests/config/llama-server-optuna-bench/Qwen3.8-27B.json --n-trials 20
```

Two configs ship: `Qwen3.8-27B.json` and `Qwen3.8-Flash-Next.json` (the latter
is set up for the CPU-offloaded MoE model). To change which flags are searched,
edit the `space` section of a config — the config files are the best examples.

The search is **resumable**: running it again with the same config adds more
trials to the same study. Results are an Optuna SQLite database plus one JSON
file per launch under `tests/results/llama-server-optuna-bench/`, with a
`summary.json` recording the best trial.

## Finding the maximum context size

`llama_context_tuner.py` answers one question: with everything else about the
server fixed, what's the largest `--ctx-size` this model can handle?

It tries progressively larger contexts until allocation fails, then narrows
down the boundary. Once found, it makes one final run at that size to record
the prompt-processing speed curve.

```bash
# validate inputs and print the probe ladder
.venv/bin/python tests/bench/llama_context_tuner.py \
    --config tests/config/llama-context-tuner/Qwen3.8-27B.json --dry-run

# run the search
.venv/bin/python tests/bench/llama_context_tuner.py \
    --config tests/config/llama-context-tuner/Qwen3.8-27B.json
```

The config specifies the model, all server flags, and the model's native
context limit (`native_ctx`). Useful overrides: `--min-ctx`, `--native-ctx`,
`--no-stop-production`. A `.jsonl` results file and a `.summary.json` with the
winning context are written to `tests/results/llama-context-tuner/`.

## Benchmarking the public API

`inference_api_bench.py` lists the models available from an OpenAI-compatible
endpoint, skips embedding models, and runs a short streaming request against
each remaining model to measure prompt-processing and generation speed. It
works with any OpenAI-compatible API; the shipped config points at
`promptlyapi.com`.

Set your API key before running it (never commit keys):

```bash
export PROMPTLY_API_KEY=sk-...

# see which models would be tested, without benchmarking
.venv/bin/python tests/bench/inference_api_bench.py \
    --config tests/config/inference-api-bench/promptly-prod.json --list-models

# benchmark every text model
.venv/bin/python tests/bench/inference_api_bench.py \
    --config tests/config/inference-api-bench/promptly-prod.json
```

The config controls the endpoint, prompt, and output length. Embedding models
are skipped by a name filter (`exclude_regex`) — adjust it if the catalog
changes. Per-request results and per-model averages are written to
`tests/results/inference-api-bench/`.

## Generating test prompts

`make_bench_prompt.py` builds long, realistic prompts from Claude Code
transcripts, so benchmark inputs look like real agentic sessions (code, tool
output, and prose) rather than repeated filler. The shipped prompts are about
32k tokens each.

```bash
.venv/bin/python tests/bench/make_bench_prompt.py \
    --out tests/prompts/qwen3_8_27b_v1.txt --target 32000 --offset 0
```

Different `--offset` values produce different prompts from the same source
transcripts. Keep the target comfortably below the context size you're
testing, since generated tokens need room too.

## Self-tests

These need no GPU, Docker, or network. Run them after changing the tooling:

```bash
.venv/bin/python tests/self_tests/test_llama_server_lib.py
.venv/bin/python tests/self_tests/test_llama_server_optuna_bench.py
.venv/bin/python tests/self_tests/test_llama_context_tuner.py
.venv/bin/python tests/self_tests/test_inference_api_bench.py
```

## Where things live

```
tests/
├── bench/          # the tools and their shared code
├── self_tests/     # host-side self-tests
├── config/         # one JSON config per tool and model
├── prompts/        # generated test prompts
└── results/        # run outputs (not committed)
```
