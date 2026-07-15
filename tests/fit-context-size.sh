#!/bin/bash

# Checks maximum context size for the specified model

# Check if a model file argument was passed to the script
if [ -z "$1" ]; then
  echo "Error: Please specify the model path."
  echo "Usage: $0 /path/to/model.gguf"
  exit 1
fi

# Store the first argument passed to the script as the model path
MODEL_PATH="$1"

# Force CUDA to match the exact device numbering order shown in nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Isolate the system so llama.cpp only sees physical GPUs 1 and 2 (the P100s)
export CUDA_VISIBLE_DEVICES=1,2

# Execute the benchmark
/opt/llama.cpp/build/bin/llama-bench \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -sm layer \
  -ts 1,1 \
  --fit-target 512 \
  --fit-ctx 2048 \
  -p 512 \
  -n 128 \
  -d 4096,8192,16384,32768,65536