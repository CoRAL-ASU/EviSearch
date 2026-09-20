#!/usr/bin/env bash
# The configuration R4 selected, as one command.
#
#   run_frozen.sh <run-name> [docs]
#
# Knowledge notes, delivered whole and role-gated; arbiter v5 reading the columns the two agents disagree on (and the
# ones both left empty); the absence guard on, so it cannot ship "Not reported" while an agent's stated value has never
# been checked. Measured 92.31 and 92.73 over two runs - the best mean of the round - against the v4 arbiter's 91.97
# and 92.48 on identical agent outputs. Cells the agents disagree on are flagged for review: 20.7 per paper holding
# 38% of the errors at 25% precision. See notes/R4_ARCHITECTURE.md.
#
# The global defaults are deliberately NOT changed by this file. EVISEARCH_ARBITER still defaults to v4 and
# EVISEARCH_OWN_READING to `all`, so the web app on :8111 and every earlier run keep behaving as they did and stay
# reproducible. Flipping the defaults is a decision for the owner once the evidence has been read, not a side effect
# of an overnight experiment.

set -euo pipefail

RUN="${1:?run name}"
DOCS="${2:-all}"

WT="/mnt/data1/nahuja11_home/EviSearch/.claude/worktrees/kb-notes-arbiter-v2"
SHARED="/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs"
PY="/mnt/data1/nahuja11_home/EviSearch/venv/bin/python"
SCHEMA="mhspc-trials-20260919020503"
VERSION=4

export EVISEARCH_RESULTS_ROOT="$SHARED/results"
export EVISEARCH_SCHEMAS_DIR="$SHARED/schemas"
export EVISEARCH_CHUNK_EMBEDDINGS_DIR="$SHARED/chunk_embeddings"
export EVISEARCH_DATASET_DIR="/mnt/data1/nahuja11_home/EviSearch/dataset"
export EVISEARCH_KNOWLEDGE_DIR="$WT/new_pipeline_outputs/knowledge"
export EVISEARCH_PRESET="${EVISEARCH_PRESET:-local}"
export EVISEARCH_ROLE_RERANKER="${EVISEARCH_ROLE_RERANKER:-none}"

# the frozen choices
export EVISEARCH_ARBITER=v5
export EVISEARCH_OWN_READING=contested
export EVISEARCH_STAGE_CONCURRENCY="${EVISEARCH_STAGE_CONCURRENCY:-3}"

cd "$WT"
echo "[frozen] notes + arbiter v5 (both_silent) + absence guard -> run $RUN"
exec $PY experiment-scripts/run_schema.py --schema "$SCHEMA" --version "$VERSION" --system E \
     --docs "$DOCS" --kb notes --run "$RUN" --parallel "${PARALLEL:-2}"
