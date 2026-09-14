#!/bin/bash
# Start the EviSearch web interface from the repository root.
#
#   shell-scripts/start_web_interface.sh              # http://127.0.0.1:8007
#   EVISEARCH_PRESET=cloud shell-scripts/start_web_interface.sh
#
# Local presets need the vLLM servers running: python -m src.inference.serve --detach
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    for candidate in venv .venv; do
        if [[ -f "$candidate/bin/activate" ]]; then
            # shellcheck disable=SC1090
            source "$candidate/bin/activate"
            break
        fi
    done
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No virtual environment found. Create one first:"
    echo "  python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Show the selected models and check credentials / local servers (warn only; the UI reports errors per request).
python -m src.config --check || echo "⚠️  Some selected models are not ready; see the checks above."

echo ""
echo "Starting web interface at http://127.0.0.1:${PORT:-8007} (Ctrl+C to stop)"
exec python web/main_app.py
