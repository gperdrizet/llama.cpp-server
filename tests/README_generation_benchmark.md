# Generation Benchmark

Realistic generation-rate benchmark for llama.cpp server under growing conversation load.

## Purpose

Measures token generation rate as:
1. **Context grows** (from 256 to 65536 tokens) - simulates conversation history accumulation
2. **Slot count varies** (1, 2, 4 parallel slots) - tests concurrent request handling
3. **KV cache quantization changes** - compares f16 vs q8_0 KV cache impact

Unlike `context_fit.py` (which tests max-context at fixed deployment), this benchmark simulates real usage:
- Pre-fills context token-by-token, not all at once
- Tests generation latency under loaded GPU
- Runs multiple repetitions to measure variance

## Quick start

```bash
# Test default model list (Q8_0 baseline + IQ4_XS variants)
python tests/generation_benchmark.py

# Test single model with specific cache types
python tests/generation_benchmark.py --model Qwen3.8-27B-UD-IQ4_XS.gguf --cache-k q8_0 --cache-v q8_0

# Custom context sizes and slot counts
python tests/generation_benchmark.py --context-sizes 256 1024 4096 16384 --slot-counts 1 2

# Override output location
python tests/generation_benchmark.py --output /tmp/results.csv
```

## Configuration

Model list: `tests/config/generation_benchmark/models.csv`

| Column | Purpose |
|--------|---------|
| `model` | GGUF filename in `MODEL_DIR` |
| `cache_type_k` | KV cache type for K buffers (`f16`, `q8_0`) |
| `cache_type_v` | KV cache type for V buffers (`f16`, `q8_0`) |
| `description` | Notes on the configuration |

### Example CSV:
```
model,cache_type_k,cache_type_v,description
Qwen3.8-27B-Q8_0.gguf,f16,f16,Baseline (Q8_0 weights + f16 KV)
Qwen3.8-27B-UD-IQ4_XS.gguf,f16,f16,IQ4_XS weights + f16 KV
Qwen3.8-27B-UD-IQ4_XS.gguf,q8_0,q8_0,IQ4_XS weights + q8_0 KV (optimized)
```

## CLI Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--config-csv` | `tests/config/generation_benchmark/models.csv` | Model configuration file |
| `--model` | (none) | Override: test single model only |
| `--cache-k` | `f16` | Override: KV cache type for K (used with `--model`) |
| `--cache-v` | `f16` | Override: KV cache type for V (used with `--model`) |
| `--slot-counts` | `1 2 4` | Parallel slot configurations to test |
| `--context-sizes` | `256 512 1024 2048 4096 8192 16384 32768 65536` | Context sizes to test (tokens) |
| `--repetitions` | `5` | Repetitions per condition |
| `--url` | `http://localhost:8502` | Server URL |
| `--api-key` | (from `.env`) | API key if server has `--api-key` set |
| `--output` | `tests/results/generation-benchmark/generation-benchmark-TIMESTAMP.csv` | Output CSV file |

## Workload

Uses a **realistic conversation corpus** (~500 words of technical discussion) repeated as needed to reach target context sizes. Each test:
1. Pre-fills context with growing amounts of text
2. Sends prompt: `"Briefly describe the key concepts mentioned above."`
3. Collects response latency and token count
4. Repeats 5 times, reports mean ± stddev

## Output CSV

Columns:
- `model`: GGUF filename
- `cache_k`, `cache_v`: KV cache types
- `slot_count`: Number of parallel slots configured
- `context_size`: Tokens of pre-filled context
- `repetition`: Repetition number (1-5)
- `latency`: Full round-trip time (seconds)
- `tokens`: Tokens generated
- `tok_per_sec`: Generation rate (tokens/second)
- `success`: Request succeeded (true/false)
- `error`: Error message if failed
- `timestamp`: ISO 8601 timestamp

## Interpreting Results

**Expected pattern**:
- Higher quantization (Q8_0 → IQ4_XS) = faster generation
- Quantized KV cache (q8_0) = 2-3x faster than f16 KV
- Deeper context = slower generation (GPU sync overhead)
- More slots = slight overhead per slot, but better parallelism for concurrent requests

**Production check**:
Compare `IQ4_XS/q8_0` vs baseline `Q8_0/f16`:
- Should see ~19x throughput improvement at 262K context
- Real conversations (0-32K) will be 2-3x faster due to hybrid attention

## Deployment Integration

Each test automatically:
1. Updates `.env` with new model, cache type, and slot count
2. Runs `utils/deploy_service.sh --restart` to reload systemd service
3. Waits for server health (max 30 retries)
4. Runs generation tests
5. Records results

**Note**: Requires `sudo` access to reload systemd service (or run as `root`).

## Troubleshooting

**Deployment fails with permission denied**:
```bash
sudo chown llama:llama /home/siderealyear/llama.cpp/models/*
```

**Server not healthy**:
```bash
systemctl status llamacpp.service
journalctl -u llamacpp.service -n 50
```

**Models not found**:
- Verify `MODEL_DIR` is readable by `llama` user
- Verify model files exist in `MODEL_DIR`

## Extending the Benchmark

Add new models to `tests/config/generation_benchmark/models.csv`:
```bash
echo "Qwen3.8-27B-UD-Q4_K_M.gguf,q4_k_m,q4_k_m,Q4_K_M weights + quantized KV" >> tests/config/generation_benchmark/models.csv
```

Then run with default config to include all rows.

## See Also

- `tests/context_fit.py` - Max-context benchmarking (coarse → bisection → verification)
- `tests/load_test.py` - Latency benchmarking under concurrent load
- `docs/project-state.md` - Current deployment state and port assignments
