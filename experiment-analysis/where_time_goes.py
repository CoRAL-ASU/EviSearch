"""Per-stage token and call accounting: what the pipeline actually spends its time on."""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1] if len(sys.argv) > 1 else "schema-mhspc-trials-20260919020503-v4"
STAGES = {"agent_extractor": "Agent A", "search_agent": "Agent B", "reconciliation_agent": "arbiter"}

tot = {}
for doc in sorted(ROOT.iterdir()):
    for stage, label in STAGES.items():
        meta = doc / "runs" / RUN / stage / "extraction_metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        u, t = m.get("usage") or {}, m.get("timing") or {}
        secs = next((v for k, v in t.items() if isinstance(v, (int, float)) and v > 1), 0)
        d = tot.setdefault(label, {"docs": 0, "secs": 0.0, "calls": 0, "inp": 0, "cached": 0, "out": 0, "imgs": 0})
        d["docs"] += 1
        d["secs"] += float(secs)
        d["calls"] += u.get("api_calls", 0)
        d["inp"] += u.get("input_tokens", 0)
        d["cached"] += u.get("cached_input_tokens", 0)
        d["out"] += u.get("output_tokens", 0)
        d["imgs"] += u.get("input_images", 0)

print(f"run {RUN}\n")
hdr = f"{'stage':10} {'min/doc':>8} {'calls':>7} {'s/call':>7} {'in tok/doc':>11} {'cached%':>8} {'out tok/doc':>12} {'imgs':>6} {'out tok/s':>10}"
print(hdr)
print("-" * len(hdr))
for label, d in tot.items():
    n = d["docs"]
    cached_pct = 100 * d["cached"] / d["inp"] if d["inp"] else 0
    print(f"{label:10} {d['secs']/n/60:8.1f} {d['calls']/n:7.1f} {d['secs']/max(d['calls'],1):7.1f} "
          f"{d['inp']/n:11,.0f} {cached_pct:7.1f}% {d['out']/n:12,.0f} {d['imgs']/n:6.1f} {d['out']/max(d['secs'],1):10.1f}")

print("\nreading of the numbers:")
for label, d in tot.items():
    n = d["docs"]
    print(f"  {label}: {d['inp']/n:,.0f} input tokens per document over {d['calls']/n:.1f} calls "
          f"= {d['inp']/max(d['calls'],1):,.0f} per call; output {d['out']/max(d['calls'],1):,.0f} per call")
