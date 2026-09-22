#!/usr/bin/env bash
# Deploy the web demo to Fly.io (https://evisearch.fly.dev) with the outputs this checkout serves locally.
#
#   shell-scripts/deploy_fly.sh             # set the API keys as Fly secrets, build remotely, deploy
#
# The image carries new_pipeline_outputs (results, embeddings, schemas, notes, reviews, run headers; see
# .dockerignore) and the dataset's PDFs. SEED_VERSION names that data: on its first boot the deployed app moves the
# volume's previous data to /data/_previous/ and serves this checkout's (src/config/runtime_paths.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.fly/bin:$PATH"
APP="${FLY_APP:-evisearch}"

data_hash="$(cd new_pipeline_outputs && find results chunk_embeddings feedback schemas knowledge benchmark_runs -type f \
    ! -name '*.log' ! -path 'knowledge/retired/*' -printf '%p %s %T@\n' 2>/dev/null | sort | sha1sum | cut -c1-12)"
echo "$(git rev-parse --short HEAD)-${data_hash}" > new_pipeline_outputs/SEED_VERSION
echo "[deploy] data version $(cat new_pipeline_outputs/SEED_VERSION)"

# API keys from .env, staged so they take effect with this deploy (values are never printed)
grep -E '^(OPEN_ROUTER_API_KEY|VISION_AGENT_API_KEY)=.+' .env | fly secrets import --stage -a "$APP" >/dev/null
echo "[deploy] secrets staged: $(grep -cE '^(OPEN_ROUTER_API_KEY|VISION_AGENT_API_KEY)=.+' .env) key(s)"

fly deploy --remote-only --ha=false -a "$APP"
