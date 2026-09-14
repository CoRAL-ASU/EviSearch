#!/bin/bash
# Run the agent pipeline over the benchmark trials.
#
# Usage (from any directory):
#   shell-scripts/run_benchmarks.sh pdf_query|search|reconcile|full [--resume] [extra CLI args...]
#
#   full       runs pdf_query -> search -> reconcile for each trial before moving to the next
#   --resume   keep existing results and only run missing columns (default: start fresh)
#   extra args are passed to every CLI, e.g. --max-batches 1 or --groups "Trial,Control Arm"
#
# Trials come from dataset/available_trial_manual_benchmarks.txt (override with TRIALS_FILE=...).
# Models come from src/config/config.py, e.g.:  EVISEARCH_PRESET=cloud shell-scripts/run_benchmarks.sh full
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
TRIALS_FILE="${TRIALS_FILE:-dataset/available_trial_manual_benchmarks.txt}"

STAGE="${1:-}"
[[ $# -gt 0 ]] && shift
RESUME_FLAG="--no-resume"
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_FLAG=""
  shift
fi
EXTRA=("$@")

case "$STAGE" in
  pdf_query|search|reconcile|full) ;;
  *) echo "Usage: $0 pdf_query|search|reconcile|full [--resume] [extra CLI args...]" >&2; exit 2 ;;
esac
[[ -f "$TRIALS_FILE" ]] || { echo "Trials file not found: $TRIALS_FILE" >&2; exit 1; }

run_stage() {
  local stage="$1" doc_id="$2" script
  case "$stage" in
    pdf_query) script=experiment-scripts/run_pdf_query_agent.py ;;
    search) script=experiment-scripts/run_search_agent.py ;;
    reconcile) script=experiment-scripts/run_reconciliation_agent.py ;;
  esac
  "$PYTHON" "$script" "$doc_id" $RESUME_FLAG "${EXTRA[@]}"
}

"$PYTHON" -m src.config

while IFS= read -r line || [[ -n "$line" ]]; do
  doc_id="$(printf '%s' "${line//$'\r'/}" | sed -e 's/^[[:space:]"]*//' -e 's/[[:space:]"]*$//' -e 's/ - done$//' -e 's/\.[pP][dD][fF]$//')"
  [[ "$doc_id" =~ ^NCT ]] || continue
  echo ""
  echo "=== $doc_id ==="
  if [[ "$STAGE" == "full" ]]; then
    { run_stage pdf_query "$doc_id" && run_stage search "$doc_id" && run_stage reconcile "$doc_id"; } || echo "FAILED: $doc_id"
  else
    run_stage "$STAGE" "$doc_id" || echo "FAILED: $doc_id"
  fi
done < "$TRIALS_FILE"
echo "Done."
