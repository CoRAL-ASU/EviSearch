#!/usr/bin/env bash
# The control the knowledge claim needs: schema v4 with NO learned knowledge (the fixed rules text), every other
# setting identical to r4-notes - same schema version, same arbiter (v4), same parallelism, launched the same way.
#
#   v4-kboff-r1, v4-kboff-r2  ->  compare with r4-notes-r1/r2 : notes vs no learned knowledge, schema held at v4
#                              ->  compare with v0draft       : three review rounds, knowledge held at fixed rules
#
# Before this, notes had only ever been compared with flat learned rules, and the one clean knowledge on/off pair
# (at v3) went the wrong way: -0.73 for the system, about -1.0 for each agent.

set -euo pipefail

WT="/mnt/data1/nahuja11_home/EviSearch/.claude/worktrees/kb-notes-arbiter-v2"
SHARED="/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs"
PY="/mnt/data1/nahuja11_home/EviSearch/venv/bin/python"
SCHEMA="mhspc-trials-20260919020503"

export EVISEARCH_RESULTS_ROOT="$SHARED/results"
export EVISEARCH_SCHEMAS_DIR="$SHARED/schemas"
export EVISEARCH_CHUNK_EMBEDDINGS_DIR="$SHARED/chunk_embeddings"
export EVISEARCH_DATASET_DIR="/mnt/data1/nahuja11_home/EviSearch/dataset"
export EVISEARCH_KNOWLEDGE_DIR="$WT/new_pipeline_outputs/knowledge"
export EVISEARCH_PRESET="${EVISEARCH_PRESET:-local}"
export EVISEARCH_ROLE_RERANKER="${EVISEARCH_ROLE_RERANKER:-none}"
export EVISEARCH_ARBITER=v4          # the arbiter every rung of the ladder used

cd "$WT"
LOGS="$WT/new_pipeline_outputs/r4_logs"
mkdir -p "$LOGS"
run() { $PY experiment-scripts/run_schema.py --schema "$SCHEMA" --version 4 --system E --docs all \
          --kb off --run "$1" --parallel 2 >"$LOGS/$1.log" 2>&1; }

echo "[kboff] two runs in parallel, as r4-notes was"
run schema-$SCHEMA-v4-kboff    &
run schema-$SCHEMA-v4-kboff-r2 &
wait
echo "[kboff] done"
