#!/bin/bash
# Run reconciliation_agent for all trials in available_trials_manual_benchmarks.txt
#
# Prerequisite: BOTH must exist for each doc:
#   new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_results.json
#   new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json
#
# Output (web-compatible):
#   new_pipeline_outputs/results/<doc_id>/reconciliation_agent/reconciled_results.json
#   Format: {"doc_id": "...", "columns": {col: {"value", "reasoning", "source", "attribution", "tried"}}}
#   Web uses this for extract page, attribution, highlights.
#
# Usage: ./run_reconciliation_agent_benchmarks.sh [--resume]
#   Default: run all trials fresh (--no-resume)
#   --resume: load checkpoints, skip already-reconciled columns

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRIALS_FILE="${SCRIPT_DIR}/available_trials_manual_benchmarks.txt"
# Default: run fresh. Pass --resume to load checkpoints.
RESUME_FLAG="--no-resume"
if [[ "$1" == "--resume" ]]; then
  RESUME_FLAG=""
fi

if [[ ! -f "$TRIALS_FILE" ]]; then
  echo "Error: $TRIALS_FILE not found"
  exit 1
fi

echo "Running reconciliation_agent for trials in $TRIALS_FILE"
echo ""

while IFS= read -r line || [[ -n "$line" ]]; do
  # Normalize doc_id: strip CRLF, surrounding quotes, " - done", .pdf; trim
  line="${line//$'\r'/}"
  line="$(echo "$line" | xargs)"
  line="${line#\"}"
  line="${line%\"}"
  line="${line% - done}"
  line="${line%.pdf}"
  line="${line%.PDF}"
  line="$(echo "$line" | xargs)"

  if [[ -z "$line" ]] || [[ ! "$line" =~ ^NCT ]]; then
    continue
  fi

  echo "=== Running reconciliation_agent: $line ==="
  python experiment-scripts/run_reconciliation_agent.py "$line" $RESUME_FLAG
  echo ""
done < "$TRIALS_FILE"

echo "Done."
