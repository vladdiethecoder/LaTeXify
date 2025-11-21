#!/bin/bash
# Start vLLM server for Qwen2.5-VL-72B-Instruct
# This serves the VLM for figure captioning and LaTeX refinement

set -e

MODEL_ID="Qwen/Qwen2.5-VL-72B-Instruct"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.9}"

echo "Starting vLLM server for $MODEL_ID on port $PORT..."

vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 8192 \
  --trust-remote-code

echo "vLLM server started successfully!"
echo "API endpoint: http://localhost:$PORT/v1"
