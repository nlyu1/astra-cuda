#!/bin/bash

# Qwen Code Server Setup Script
# This script launches a vLLM server for the Qwen3-30B-A3B model

# Check if model exists
MODEL_PATH="$HOME/models/qwen3-30b-a3b-4bit"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please ensure the model has finished downloading."
    exit 1
fi

# Check if any .gguf files exist
if ! ls "$MODEL_PATH"/*.gguf 1> /dev/null 2>&1; then
    echo "Error: No .gguf files found in $MODEL_PATH"
    echo "Model download may still be in progress."
    exit 1
fi

echo "Starting vLLM server for Qwen3-30B-A3B..."
echo "Model path: $MODEL_PATH"
echo "Server will be available at: http://localhost:8000/v1"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate qwen

# Launch vLLM server
export MODEL="$MODEL_PATH"
vllm serve "$MODEL" \
   --trust-remote-code \
   --tensor-parallel-size 1 \
   --max-model-len 32768 \
   --enable-reasoning \
   --reasoning-parser deepseek_r1 \
   --port 8000 \
   --host 0.0.0.0 