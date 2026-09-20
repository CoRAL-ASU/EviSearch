"""Is a run actually finished? Counts documents whose reconciled table has every column, not directories that exist.

A stage directory appears when the stage starts, so directory presence is not completion.

  python experiment-analysis/run_complete.py <run> [<run> ...]      exit 0 when every named run is complete
"""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
EXPECTED_COLUMNS = 133
EXPECTED_DOCS = 10

ok = True
for run in sys.argv[1:]:
    done = partial = 0
    for doc in sorted(ROOT.iterdir()):
        path = doc / "runs" / run / "reconciliation_agent" / "reconciled_results.json"
        if not path.exists():
            continue
        try:
            n = len(json.loads(path.read_text()).get("columns") or {})
        except (json.JSONDecodeError, OSError):
            partial += 1  # being written right now
            continue
        if n >= EXPECTED_COLUMNS:
            done += 1
        else:
            partial += 1
    complete = done >= EXPECTED_DOCS
    ok = ok and complete
    print(f"{run}: {done}/{EXPECTED_DOCS} documents complete"
          + (f", {partial} partial" if partial else "")
          + ("  [COMPLETE]" if complete else ""))
sys.exit(0 if ok else 1)
