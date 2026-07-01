#!/bin/bash
set -euo pipefail

# Minimal OpenAI-compatible vLLM server for local Qwen3-8B.
# Override values inline, e.g.:
#   CUDA_VISIBLE_DEVICES=0 PORT=8002 ./shell-scripts/run_vllm_qwen3_8b.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_BIN="${VLLM_BIN:-/mnt/data1/nahuja11/micromamba/envs/llminfra/bin/vllm}"
MODEL="${MODEL:-${REPO_ROOT}/../../../../shared/shared_hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"

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
  echo "Free it with: ./shell-scripts/run_vllm_qwen3_8b.sh --kill-port"
  exit 1
fi
if command -v ss &>/dev/null && ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use."
  echo "Free it with: ./shell-scripts/run_vllm_qwen3_8b.sh --kill-port"
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype "$DTYPE"
