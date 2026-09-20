"""Cells where the arbiter shipped a value that was neither A's nor B's: what it was, and which tool found it."""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1] if len(sys.argv) > 1 else "schema-mhspc-trials-20260919020503-v4-r2"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4

sq = lambda v: re.sub(r"\s+", " ", str(v or "")).strip().lower().rstrip(".")
found = []
for doc in sorted(ROOT.iterdir()):
    base = doc / "runs" / RUN
    rec = base / "reconciliation_agent" / "reconciled_results.json"
    a = base / "agent_extractor" / "extraction_results.json"
    b = base / "search_agent" / "extraction_results.json"
    if not (rec.exists() and a.exists()):
        continue
    cols = json.loads(rec.read_text()).get("columns") or {}
    av = json.loads(a.read_text()).get("columns") or {}
    bv = json.loads(b.read_text()).get("columns") or {} if b.exists() else {}
    for name, c in cols.items():
        val = sq(c.get("value"))
        if not val or val.startswith("not reported"):
            continue
        a_val, b_val = sq((av.get(name) or {}).get("value")), sq((bv.get(name) or {}).get("value"))
        if val not in (a_val, b_val):
            found.append((doc.name, name, c, a_val, b_val))

print(f"run {RUN}: {len(found)} cells where the shipped value was neither A's nor B's\n")
for doc, name, c, a_val, b_val in found[:N]:
    print(f"=== {doc[:30]} | {name[:52]}")
    print(f"  A said      : {a_val[:70]!r}")
    print(f"  B said      : {b_val[:70]!r}")
    print(f"  ARBITER said: {str(c.get('value'))[:70]!r}   (verified={c.get('verified')})")
    print(f"  its reasoning: {re.sub(chr(10), ' ', str(c.get('reasoning') or ''))[:320]}")
    src = c.get("source") or {}
    print(f"  source: page {src.get('page')} ({src.get('modality')})")
    print(f"  checks run on this column: {[(k.get('value'), k.get('verdict')) for k in (c.get('checks') or [])][:4]}")
    print()
