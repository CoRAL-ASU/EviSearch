#!/usr/bin/env bash
# Reconcile an existing run's agent outputs again with a different arbiter configuration.
#
#   run_variant.sh <source-run> <target-run> [env assignments...]
#
# The agents are copied, never re-run, so the only difference between the source and the target is the arbiter. That is
# what makes the comparison an arbiter comparison instead of a second sample of the agents.
#
#   run_variant.sh r4-notes-r1 r4-contested-r1 EVISEARCH_ARBITER=v5 EVISEARCH_OWN_READING=contested
#
# Batch concurrency defaults to 3 here: measured at 10.0 min against 19.4 min serial for the v5 arbiter on one paper,
# and the reader's document cache is locked, so batches of a stage no longer race over it.

set -euo pipefail

SRC="${1:?source run}"
DST="${2:?target run}"
shift 2

WT="/mnt/data1/nahuja11_home/EviSearch/.claude/worktrees/kb-notes-arbiter-v2"
SHARED="/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs"
PY="/mnt/data1/nahuja11_home/EviSearch/venv/bin/python"
SCHEMA="mhspc-trials-20260919020503"

export EVISEARCH_RESULTS_ROOT="$SHARED/results"
export EVISEARCH_SCHEMAS_DIR="$SHARED/schemas"
export EVISEARCH_CHUNK_EMBEDDINGS_DIR="$SHARED/chunk_embeddings"
export EVISEARCH_DATASET_DIR="/mnt/data1/nahuja11_home/EviSearch/dataset"
export EVISEARCH_KNOWLEDGE_DIR="$WT/new_pipeline_outputs/knowledge"
export EVISEARCH_DEFINITIONS_CSV="$SHARED/schemas/$SCHEMA/versions/v4.csv"
export EVISEARCH_PRESET="${EVISEARCH_PRESET:-local}"
export EVISEARCH_ROLE_RERANKER="${EVISEARCH_ROLE_RERANKER:-none}"
export EVISEARCH_KB="${EVISEARCH_KB:-notes}"
export EVISEARCH_STAGE_CONCURRENCY="${EVISEARCH_STAGE_CONCURRENCY:-3}"
if [ "$EVISEARCH_KB" = "notes" ]; then
  export EVISEARCH_KB_NOTES_SNAPSHOT="${EVISEARCH_KB_NOTES_SNAPSHOT:-$WT/new_pipeline_outputs/knowledge/note_snapshots/853a2edb7ecd.json}"
fi
export EVISEARCH_RUN="$DST"
for assignment in "$@"; do export "$assignment"; done

cd "$WT"
LOGS="$WT/new_pipeline_outputs/r4_logs"
mkdir -p "$LOGS"

echo "[variant] $SRC -> $DST  arbiter=${EVISEARCH_ARBITER:-v4} own_reading=${EVISEARCH_OWN_READING:-all} kb=$EVISEARCH_KB concurrency=$EVISEARCH_STAGE_CONCURRENCY"
$PY experiment-scripts/copy_agent_stages.py "$SRC" "$DST"
$PY experiment-scripts/reconcile_run.py --parallel "${VARIANT_PARALLEL:-2}" 2>&1 | tee "$LOGS/$DST.log"
echo "[variant] $DST done"
