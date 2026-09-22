"""Per-stage wall-clock per document, from the timing block each stage saves in its metadata."""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUNS = sys.argv[1:] or ["schema-mhspc-trials-20260919020503-v4", "schema-mhspc-trials-20260919020503-v4-r2"]
STAGES = {"agent_extractor": "Agent A", "search_agent": "Agent B", "reconciliation_agent": "arbiter"}

for run in RUNS:
    print(f"\n=== {run}")
    per_stage = {}
    rows = []
    for doc in sorted(ROOT.iterdir()):
        d = doc / "runs" / run
        if not d.is_dir():
            continue
        times, pages = {}, None
        for stage, label in STAGES.items():
            meta = d / stage / "extraction_metadata.json"
            if not meta.exists():
                continue
            m = json.loads(meta.read_text())
            t = m.get("timing") or {}
            secs = t.get("seconds") or t.get("elapsed_s") or t.get("wall_seconds") or t.get("duration_s")
            if secs is None:
                for v in t.values():
                    if isinstance(v, (int, float)) and v > 1:
                        secs = v
                        break
            if secs:
                times[label] = float(secs)
                per_stage.setdefault(label, []).append(float(secs))
        if times:
            rows.append((doc.name, times, sum(times.values())))
    if not rows:
        print("  no timing metadata found")
        continue
    print(f"  {'document':36} {'Agent A':>9} {'Agent B':>9} {'arbiter':>9} {'total':>9}")
    for name, times, total in rows:
        cells = "".join(f"{times.get(l, 0)/60:9.1f}" for l in ("Agent A", "Agent B", "arbiter"))
        print(f"  {name[:36]:36}{cells}{total/60:9.1f}")
    print(f"  {'-'*36}{'-'*36}")
    tot = [t for _, _, t in rows]
    for label, xs in per_stage.items():
        print(f"  mean {label:12} {sum(xs)/len(xs)/60:6.1f} min   (median {sorted(xs)[len(xs)//2]/60:.1f}, max {max(xs)/60:.1f})")
    print(f"  mean END TO END   {sum(tot)/len(tot)/60:6.1f} min per document over {len(tot)} documents")
    print(f"  sum of all stages {sum(tot)/3600:6.2f} h of model time for the run")
