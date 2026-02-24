# # #!/bin/bash

# # # Point to the shared Hugging Face cache
# # export HF_HOME="/mnt/shared/shared_hf_home"
# # export TRANSFORMERS_CACHE="/mnt/shared/shared_hf_home"
# # export HF_DATASETS_CACHE="/mnt/shared/shared_hf_home"

# # # Use these 4 GPUs
# # export CUDA_VISIBLE_DEVICES=4,5,6,7

# # vllm serve meta-llama/Llama-3.3-70b-Instruct \
# #   --max-model-len 32768 \
# #   --gpu-memory-utilization 0.35 \
# #   --max-num-seqs 64 \
# #   --port 8001 \
# #   --tensor-parallel-size 4

# #!/bin/bash
# set -euo pipefail

# # Use project-local cache (avoids PermissionError on shared /mnt paths)
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CACHE_DIR="${SCRIPT_DIR}/.cache/huggingface"
# mkdir -p "$CACHE_DIR"
# export HF_HOME="$CACHE_DIR"
# export TRANSFORMERS_CACHE="$HF_HOME"
# export HF_DATASETS_CACHE="$HF_HOME"

# # Select the 4 GPUs you want vLLM to use
# export CUDA_VISIBLE_DEVICES=5
# PORT=8001

# # If port is in use, optionally free it (e.g. ./run_vllm.sh --kill-port)
# if [[ "${1:-}" == "--kill-port" ]]; then
#   if command -v fuser &>/dev/null; then
#     echo "Freeing port ${PORT}..."
#     fuser -k "${PORT}/tcp" 2>/dev/null || true
#     sleep 2
#   else
#     echo "Install 'fuser' (e.g. psmisc) or kill the process manually: lsof -i :${PORT}"
#     exit 1
#   fi
# fi

# if lsof -i ":${PORT}" &>/dev/null || ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
#   echo "Port ${PORT} is already in use. Either:"
#   echo "  1. Use the existing server and run main.py (no need to run this script again)"
#   echo "  2. Free the port and restart: ./run_vllm.sh --kill-port"
#   echo "  3. Find and kill manually: lsof -i :${PORT}   then  kill <PID>"
#   exit 1
# fi

# vllm serve Qwen/Qwen3-8B \
#   --max-model-len 32k \
#   --gpu-memory-utilization 0.90 \
#   --max-num-seqs 64 \
#   --tensor-parallel-size 1 \
#   --port "${PORT}" \
#   --dtype auto

# # # Get the directory where this script is located
# # SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # CHAT_TEMPLATE="${SCRIPT_DIR}/llama3.1_chat_template.jinja"

# # vllm serve meta-llama/Llama-3.1-8B \
# #   --max-model-len 32k \
# #   --gpu-memory-utilization 0.80 \
# #   --max-num-seqs 64 \
# #   --tensor-parallel-size 1 \
# #   --port "${PORT}" \
# #   --dtype auto \
# #   --trust-remote-code \
# #   --chat-template "${CHAT_TEMPLATE}"

# #!/bin/bash

# # Point to the shared Hugging Face cache
# export HF_HOME="/mnt/shared/shared_hf_home"
# export TRANSFORMERS_CACHE="/mnt/shared/shared_hf_home"
# export HF_DATASETS_CACHE="/mnt/shared/shared_hf_home"

# # Use these 4 GPUs
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# vllm serve meta-llama/Llama-3.3-70b-Instruct \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.35 \
#   --max-num-seqs 64 \
#   --port 8001 \
#   --tensor-parallel-size 4

#!/bin/bash
set -euo pipefail

# Point to your local Hugging Face cache (writable location)
export HF_HOME="/mnt/data1/nahuja11/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"

# Select the 4 GPUs you want vLLM to use
export CUDA_VISIBLE_DEVICES=7
# Optional: tune these values to your hardware and model (see notes below)
# MODEL="meta-llama/Llama-3.1-8B"
PORT=8001

vllm serve Qwen/Qwen3-8B \
  --max-model-len 32k \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --tensor-parallel-size 1 \
  --port "${PORT}" \
  --dtype auto

# # Get the directory where this script is located
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CHAT_TEMPLATE="${SCRIPT_DIR}/llama3.1_chat_template.jinja"

# vllm serve meta-llama/Llama-3.1-8B \
#   --max-model-len 32k \
#   --gpu-memory-utilization 0.80 \
#   --max-num-seqs 64 \
#   --tensor-parallel-size 1 \
#   --port "${PORT}" \
#   --dtype auto \
#   --trust-remote-code \
#   --chat-template "${CHAT_TEMPLATE}"