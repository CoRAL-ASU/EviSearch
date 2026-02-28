#!/bin/bash
set -euo pipefail

# Qwen3-VL-32B-Instruct vLLM server (local, H200-friendly defaults)
# Override any value via env vars, e.g.:
#   CUDA_VISIBLE_DEVICES=0 PORT=8001 MAX_MODEL_LEN=16384 ./run_vllm_qwen3_vl_32b.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

# Local HF cache (writable location)
HF_HOME="${HF_HOME:-/mnt/data1/nahuja11/.cache/huggingface}"
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

# Pick the GPU(s) to use (set to your H200 index)
export CUDA_VISIBLE_DEVICES=4

# Optional: free the port
if [[ "${1:-}" == "--kill-port" ]]; then
  if command -v fuser &>/dev/null; then
    echo "Freeing port ${PORT}..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
  else
    echo "Install 'fuser' (e.g. psmisc) or kill manually: lsof -i :${PORT}"
    exit 1
  fi
fi

if command -v lsof &>/dev/null && lsof -i ":${PORT}" &>/dev/null; then
  echo "Port ${PORT} is already in use."
  echo "Free it with: ./run_vllm_qwen3_vl_32b.sh --kill-port"
  exit 1
fi
if command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use."
  echo "Free it with: ./run_vllm_qwen3_vl_32b.sh --kill-port"
  exit 1
fi

exec vllm serve "$MODEL" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --dtype "$DTYPE"
