#!/bin/bash
# Run full pipeline for each doc sequentially: agent_extractor → search_agent → reconciliation_agent.
# Complete all 3 agents for one doc before moving to the next.
#
# Prerequisites:
#   - PDF in dataset/ for agent_extractor
#   - parsed_markdown.md for search_agent (chunking/Landing AI output)
#
# Output: new_pipeline_outputs/results/<doc_id>/{agent_extractor,search_agent,reconciliation_agent}/
#
# Usage: ./run_full_pipeline_benchmarks.sh [--resume]
#   Default: run all docs fresh (--no-resume)
#   --resume: load checkpoints, skip completed docs/steps

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Hardcoded doc_ids (no file reading, no xargs - avoids quote/CRLF issues)
DOC_IDS=(
  # "NCT00104715_Gravis_GETUG_EU'15"
  "NCT00268476_Attard_STAMPEDE_Lancet'23" 
  "NCT00268476_James_STAMPEDE_IJC'22"
  "NCT01809691_Aggarwal_SWOG1216_JCO'22"
  "NCT00309985_Kriayako_CHAARTED_JCO'18" 
  "NCT02799602_Hussain_ARASENS_JCO'23" 
  "NCT02799602_Smith_ARASENS_NEJM'22" 
  "NCT02446405_Sweeney_ENZAMET_Lancet Onc'23" 
  "NCT00309985_Sweeney_CHAARTED_NEJM'15" 
  "NCT01957436_Fizazi_PEACE1_Lancet'22" 
)

RESUME_FLAG="--no-resume"
if [[ "$1" == "--resume" ]]; then
  RESUME_FLAG=""
fi

AGENT_RESUME="--no-resume"
if [[ "$1" == "--resume" ]]; then
  AGENT_RESUME="--skip-if-done"
fi

echo "Running full pipeline (agent → search → reconcile) for ${#DOC_IDS[@]} docs"
echo "Mode: $([[ -z "$RESUME_FLAG" ]] && echo "resume" || echo "fresh")"
echo ""

for doc_id in "${DOC_IDS[@]}"; do
  echo "=========================================="
  echo "=== DOC: $doc_id ==="
  echo "=========================================="

  echo ""
  echo "--- 1/3 agent_extractor ---"
  python experiment-scripts/agent_extractor.py "$doc_id" $AGENT_RESUME || { echo "agent_extractor failed for $doc_id"; exit 1; }

  echo ""
  echo "--- 2/3 search_agent ---"
  python experiment-scripts/run_search_agent.py "$doc_id" $RESUME_FLAG || { echo "search_agent failed for $doc_id"; exit 1; }

  echo ""
  echo "--- 3/3 reconciliation_agent ---"
  python experiment-scripts/run_reconciliation_agent.py "$doc_id" $RESUME_FLAG || { echo "reconciliation_agent failed for $doc_id"; exit 1; }

  echo ""
  echo "=== Done: $doc_id ==="
  echo ""
done

echo "All docs processed."
