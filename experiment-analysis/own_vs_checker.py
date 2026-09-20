"""On cells the arbiter blanked although an agent was right: what had its OWN phase-1 reading found?

If phase 1 independently found the same value the agent did, and the cell was still blanked, then one checker pass
overruled two independent readings.

  python experiment-analysis/own_vs_checker.py <run>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1]
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: N(v).lower().rstrip(".").replace("%", "").replace(" ", "")
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def cells(stage):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={RUN}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred)) for r in rows}


A, B, E = cells("agent_extractor"), cells("search_agent"), cells("reconciliation_agent")
keys = sorted(set(A) & set(B) & set(E))
blanked = [k for k in keys if NR(E[k][1]) and E[k][0] < max(A[k][0], B[k][0]) and not (NR(A[k][1]) and NR(B[k][1]))]

kinds = Counter()
recoverable = []
for k in blanked:
    path = ROOT / k[0] / "runs" / RUN / "reconciliation_agent" / "reconciled_results.json"
    entry = json.loads(path.read_text())["columns"].get(k[1]) or {}
    own = entry.get("own_finding") or {}
    right = A[k][1] if A[k][0] >= B[k][0] else B[k][1]
    if own.get("skipped"):
        kinds["phase 1 skipped it (agents agreed)"] += 1
    elif own.get("unread"):
        kinds["phase 1 never answered it"] += 1
    elif NR(own.get("value")):
        kinds["phase 1 also found nothing"] += 1
    elif SQ(own.get("value")) == SQ(right):
        kinds["PHASE 1 FOUND THE SAME VALUE and it was blanked anyway"] += 1
        recoverable.append((k, own.get("value"), right))
    else:
        kinds[f"phase 1 found something else"] += 1

print(f"{RUN}: {len(blanked)} cells blanked although an agent had the right value\n")
for kind, n in kinds.most_common():
    print(f"  {n:4}  {kind}")
if recoverable:
    print(f"\n  recoverable by trusting phase 1 over a checker rejection ({len(recoverable)}):")
    for (k, own_value, right) in recoverable[:8]:
        print(f"    {k[0][:20]:20} | {k[1][:40]:40} own={own_value[:24]!r} agent={right[:24]!r}")
