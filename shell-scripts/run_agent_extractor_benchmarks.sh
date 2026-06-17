#!/bin/bash
# Run agent_extractor for all trials in available_trials_manual_benchmarks.txt
#
# Output format (pipeline-compatible):
#   new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_results.json
#   Format: {"doc_id": "...", "columns": {col: {"value", "reasoning", "found", "tried", "attribution"}}}
#   This matches what run_search_agent and run_reconciliation_agent expect.
#
# Usage: ./run_agent_extractor_benchmarks.sh [--resume]
#   Default: run all trials fresh (no-resume, no skip-if-done)
#   --resume: load checkpoints and skip if already done

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRIALS_FILE="${SCRIPT_DIR}/available_trials_manual_benchmarks.txt"
# Default: run fresh (--no-resume). Pass --resume to load checkpoints and skip if done.
RESUME_FLAGS="--no-resume"
if [[ "$1" == "--resume" ]]; then
  RESUME_FLAGS="--skip-if-done"
fi

if [[ ! -f "$TRIALS_FILE" ]]; then
  echo "Error: $TRIALS_FILE not found"
  exit 1
fi

echo "Running agent_extractor for trials in $TRIALS_FILE"
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

  echo "=== Running agent_extractor: $line ==="
  python experiment-scripts/agent_extractor.py "$line" $RESUME_FLAGS
  echo ""
done < "$TRIALS_FILE"

echo "Done."
