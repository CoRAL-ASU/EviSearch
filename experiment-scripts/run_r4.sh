#!/usr/bin/env bash
# R4: the knowledge notes, then the two-phase arbiter, as a ladder so each change can be attributed.
#
#   rung A   notes + arbiter v4 (the R3 arbiter)   -> r4-notes-r1, r4-notes-r2      full pipeline
#   rung B   notes + arbiter v5 (two-phase)        -> r4-notes-v5-r1, r4-notes-v5-r2
#            Agent A and Agent B are identical to rung A (same prompts, same knowledge), so they are copied
#            rather than re-run and only the reconciliation stage differs. That is what makes the comparison
#            an arbiter comparison and not a second sample of the agents.
#   baseline notes + markdown baseline             -> r4-notes-b1
#            B1 gets the same knowledge as the agents, or EviSearch - B1 stops measuring the architecture.
#
# Code lives in this worktree; results, parses, schemas and the gold table stay in the shared tree.

set -euo pipefail

WT="/mnt/data1/nahuja11_home/EviSearch/.claude/worktrees/kb-notes-arbiter-v2"
SHARED="/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs"
PY="/mnt/data1/nahuja11_home/EviSearch/venv/bin/python"
SCHEMA="mhspc-trials-20260919020503"
VERSION=4
DOCS="all"
PARALLEL=2

export EVISEARCH_RESULTS_ROOT="$SHARED/results"
export EVISEARCH_SCHEMAS_DIR="$SHARED/schemas"
export EVISEARCH_CHUNK_EMBEDDINGS_DIR="$SHARED/chunk_embeddings"
export EVISEARCH_DATASET_DIR="/mnt/data1/nahuja11_home/EviSearch/dataset"
export EVISEARCH_KNOWLEDGE_DIR="$WT/new_pipeline_outputs/knowledge"
export EVISEARCH_PRESET="${EVISEARCH_PRESET:-local}"
export EVISEARCH_ROLE_RERANKER="${EVISEARCH_ROLE_RERANKER:-none}"

cd "$WT"
LOGS="$WT/new_pipeline_outputs/r4_logs"
mkdir -p "$LOGS"

schema_run() {  # schema_run <run-name> <system> <log>
  $PY experiment-scripts/run_schema.py --schema "$SCHEMA" --version "$VERSION" --system "$2" \
      --docs "$DOCS" --kb notes --run "$1" --parallel "$PARALLEL" >"$LOGS/$3.log" 2>&1
}

echo "[r4] rung A: notes + arbiter v4, two runs in parallel"
schema_run r4-notes-r1 E a_r1 &
schema_run r4-notes-r2 E a_r2 &
wait
echo "[r4] rung A done"

echo "[r4] copying Agent A and Agent B into the v5 runs"
$PY experiment-scripts/copy_agent_stages.py r4-notes-r1 r4-notes-v5-r1 >"$LOGS/copy_r1.log" 2>&1
$PY experiment-scripts/copy_agent_stages.py r4-notes-r2 r4-notes-v5-r2 >"$LOGS/copy_r2.log" 2>&1

echo "[r4] rung B: same agents, arbiter v5"
reconcile_only() {  # reconcile_only <run-name> <log>
  EVISEARCH_RUN="$1" EVISEARCH_ARBITER=v5 EVISEARCH_KB=notes \
    $PY experiment-scripts/reconcile_run.py --parallel "$PARALLEL" >"$LOGS/$2.log" 2>&1
}
reconcile_only r4-notes-v5-r1 b_r1 &
reconcile_only r4-notes-v5-r2 b_r2 &
wait
echo "[r4] rung B done"

echo "[r4] baseline: markdown baseline with the same knowledge"
schema_run r4-notes-b1 B1 b1
echo "[r4] all done"
