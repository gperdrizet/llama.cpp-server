#!/bin/bash

# Quick A/B comparison of --threads 12 (physical cores only) vs 24 (all
# HyperThreads) for Qwen3.8-Flash-Next decode/prefill throughput on the
# CPU-only E5-2697 v2 box. Mirrors the flags set in
# utils/llama-flash-next.service (flash-attn, quantized KV cache, batch
# sizes) so the numbers are representative of the deployed config.
#
# Caveat: llama-bench only has a single -t/--threads knob, it can't split
# generation threads from batch threads the way llama-server's
# --threads/--threads-batch can. So this tells you whether N threads is
# better or worse overall, not the full prefill-vs-decode picture the
# live service gets from using different values for each.
#
# For clean numbers, stop the live service first so the benchmark isn't
# fighting the production process for the same physical cores:
#   sudo systemctl stop llama-flash-next.service
#
# Usage:
#   bash tests/bench-flash-next-threads.sh [/path/to/model-00001-of-00006.gguf]

MODEL_PATH="${1:-/home/siderealyear/llama.cpp/models/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: model file not found: $MODEL_PATH"
  echo "Usage: $0 [/path/to/model-00001-of-00006.gguf]"
  exit 1
fi

# CPU-only, same as the systemd unit — never let this touch the P100s.
export CUDA_VISIBLE_DEVICES=

/opt/llama.cpp/build/bin/llama-bench \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -b 4096 \
  -ub 4096 \
  -p 256 \
  -n 32 \
  -r 1 \
  -t 12,24 \
  --progress \
  -o md
