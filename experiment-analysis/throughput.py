"""Aggregate decode throughput of a finished run: does the server scale with concurrency, or is it already full?"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUNS = sys.argv[1:] or ["schema-mhspc-trials-20260919020503-v4", "schema-mhspc-trials-20260919020503-v4-r2"]
STAGES = ("agent_extractor", "search_agent", "reconciliation_agent")

t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
spans, out_tokens, model_secs = [], 0, 0.0

for run in RUNS:
    for doc in sorted(ROOT.iterdir()):
        for stage in STAGES:
            meta = doc / "runs" / run / stage / "extraction_metadata.json"
            if not meta.exists():
                continue
            tm = (json.loads(meta.read_text()).get("timing") or {})
            if tm.get("started_at") and tm.get("finished_at"):
                spans.append((t(tm["started_at"]), t(tm["finished_at"])))
            out_tokens += tm.get("output_tokens", 0)
            model_secs += float(tm.get("model_seconds") or 0)

if not spans:
    print("no timing data")
    raise SystemExit(1)

wall_start, wall_end = min(s for s, _ in spans), max(e for _, e in spans)
wall = (wall_end - wall_start).total_seconds()
busy = sum((e - s).total_seconds() for s, e in spans)

# mean concurrency: how many stage-spans overlap on average across the wall clock
edges = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
live, last, weighted = 0, None, 0.0
for moment, delta in edges:
    if last is not None and live > 0:
        weighted += live * (moment - last).total_seconds()
    live += delta
    last = moment

print(f"runs: {', '.join(RUNS)}")
print(f"  wall clock            {wall/3600:6.2f} h  ({wall_start:%H:%M} -> {wall_end:%H:%M})")
print(f"  summed stage time     {busy/3600:6.2f} h over {len(spans)} stage runs")
print(f"  mean concurrent stages{weighted/wall:7.2f}")
print(f"  output tokens         {out_tokens:,}")
print(f"  AGGREGATE throughput  {out_tokens/wall:6.1f} output tok/s across the whole run")
print(f"  per-stream throughput {out_tokens/max(busy,1):6.1f} output tok/s inside one stage")
print(f"\n  reading: if per-stream throughput holds as concurrency rises, the server has headroom and batch")
print(f"  concurrency buys real time. If aggregate is flat against concurrency, the card is the ceiling.")
