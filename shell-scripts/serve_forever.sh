#!/usr/bin/env bash
# Keep the local model servers running on ONE GPU, and keep holding that GPU.
#
#   shell-scripts/serve_forever.sh [preset]        # preset: local (Qwen3.6-27B, default) or local_mistral
#
# Run it inside tmux. It waits until a GPU in EVISEARCH_SERVE_GPUS (default "4 5 6 7") has at most
# EVISEARCH_SERVE_MAX_USED_MIB in use (default 5000: the three servers need 0.91 of an H200 and the launcher caps a
# GPU at 0.95), starts the chat model, the embedding model and the reranker there with src.inference.serve in the
# foreground, and starts them again whenever they exit (a crash, or Ctrl+C), preferring the same GPU.
#
# To stop for good: touch .cache/serve/STOP, then press Ctrl+C in this window. The servers stop and the loop ends.
set -u
cd "$(dirname "$0")/.."
PRESET="${1:-local}"
case "$PRESET" in
  local) CHAT=qwen36_27b ;;
  local_mistral) CHAT=mistral_small_24b ;;
  *) echo "unknown preset '$PRESET' (local or local_mistral)"; exit 2 ;;
esac
CANDIDATES="${EVISEARCH_SERVE_GPUS:-4 5 6 7}"
MAX_USED="${EVISEARCH_SERVE_MAX_USED_MIB:-5000}"
STOP=.cache/serve/STOP
LOG=.cache/serve/supervisor.log
mkdir -p .cache/serve
rm -f "$STOP"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pick_gpu() {  # the previous GPU first if it still has room, then the candidates in order
  local used
  for gpu in ${LAST_GPU:-} $CANDIDATES; do
    case "$gpu" in 0|1|2|3) continue ;; esac  # never GPUs 0-3
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -le "$MAX_USED" ]; then echo "$gpu"; return 0; fi
  done
  return 1
}

LAST_GPU=""
waiting=0
while [ ! -f "$STOP" ]; do
  if ! GPU=$(pick_gpu); then
    [ $((waiting % 30)) -eq 0 ] && say "waiting for a GPU in [$CANDIDATES] with <= $MAX_USED MiB in use"
    waiting=$((waiting + 1)); sleep 10; continue
  fi
  waiting=0
  LAST_GPU="$GPU"
  say "starting $CHAT + qwen3_embed_8b + qwen3_rerank_8b on GPU $GPU (preset $PRESET)"
  EVISEARCH_PRESET="$PRESET" EVISEARCH_GPUS="$CHAT=$GPU;qwen3_embed_8b=$GPU;qwen3_rerank_8b=$GPU" \
    venv/bin/python -m src.inference.serve
  code=$?
  say "launcher exited with code $code"
  [ -f "$STOP" ] && break
  sleep 5
done
say "stopped (found $STOP)"
