#!/bin/bash
# Run search_agent for all trials in available_trials_manual_benchmarks.txt
#
# Prerequisite: parsed_markdown.md must exist for each doc at:
#   new_pipeline_outputs/results/<doc_id>/chunking/parsed_markdown.md
#   OR experiment-scripts/baselines_landing_ai_new_results/<doc_id>/parsed_markdown.md
#
# Output (web-compatible):
#   new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json
#   Format: {"doc_id": "...", "columns": {col: {"value", "reasoning", "found", "tried", "attribution"}}}
#
# Usage: ./run_search_agent_benchmarks.sh [--resume]
#   Default: run all trials fresh (--no-resume)
#   --resume: load checkpoints, skip already-done columns

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

echo "Running search_agent for trials in $TRIALS_FILE"
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

  echo "=== Running search_agent: $line ==="
  python experiment-scripts/run_search_agent.py "$line" $RESUME_FLAG
  echo ""
done < "$TRIALS_FILE"

echo "Done."
