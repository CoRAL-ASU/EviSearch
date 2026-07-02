#!/bin/bash
set -euo pipefail

# OpenAI-compatible vLLM server for local Qwen3.6-27B.
# Override values inline, e.g.:
#   CUDA_VISIBLE_DEVICES=2,3 PORT=8002 ./shell-scripts/run_vllm_qwen36_27b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_BIN="${VLLM_BIN:-/mnt/data1/nahuja11/micromamba/envs/llminfra/bin/vllm}"
MODEL="${MODEL:-${REPO_ROOT}/../../../../shared/shared_hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.6-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"

# Pick the GPU(s) to use unless the caller already set CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${1:-}" == "--kill-port" ]]; then
  if command -v fuser &>/dev/null; then
    echo "Freeing port ${PORT}..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
  else
    echo "Install 'fuser' or kill manually: lsof -i :${PORT}"
    exit 1
  fi
fi

if command -v lsof &>/dev/null && lsof -i ":${PORT}" &>/dev/null; then
  echo "Port ${PORT} is already in use."
  echo "Free it with: ./shell-scripts/run_vllm_qwen36_27b.sh --kill-port"
  exit 1
fi
if command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use."
  echo "Free it with: ./shell-scripts/run_vllm_qwen36_27b.sh --kill-port"
  exit 1
fi

cmd=(
  "$VLLM_BIN" serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TP_SIZE"
  --dtype "$DTYPE"
)

if [[ -n "$LIMIT_MM_PER_PROMPT" ]]; then
  cmd+=(--limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
fi

exec "${cmd[@]}"
