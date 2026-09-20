"""The cells an arbiter blanked although an agent had the right value: did the agents agree, and what did the checker say?

If the agents independently produced the same value and the arbiter still shipped "Not reported", the evidence that
the value is wrong is one checker pass, against two independent extractions that agree.

  python experiment-analysis/absence_losses.py <run>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1]
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: N(v).lower().rstrip(".").replace("%", "")
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def cells(stage):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={RUN}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred)) for r in rows}


A, B, E = cells("agent_extractor"), cells("search_agent"), cells("reconciliation_agent")
keys = sorted(set(A) & set(B) & set(E))
blanked = [k for k in keys
           if NR(E[k][1]) and E[k][0] < max(A[k][0], B[k][0])
           and not (NR(A[k][1]) and NR(B[k][1]))]

print(f"{RUN}: {len(blanked)} cells blanked although an agent had a better answer\n")
agreed = [k for k in blanked if SQ(A[k][1]) == SQ(B[k][1]) and not NR(A[k][1])]
print(f"  BOTH agents produced the same value and it was blanked : {len(agreed)}")
print(f"  only one agent had it                                  : {len(blanked) - len(agreed)}")
print(f"  points recoverable if unanimous values were never blanked: "
      f"{sum(max(A[k][0], B[k][0]) - E[k][0] for k in agreed):.2f} "
      f"({100 * sum(max(A[k][0], B[k][0]) - E[k][0] for k in agreed) / len(keys):.2f}% of the table)")

reasons = Counter()
for k in blanked:
    path = ROOT / k[0] / "runs" / RUN / "reconciliation_agent" / "reconciled_results.json"
    entry = (json.loads(path.read_text())["columns"].get(k[1]) or {})
    checks = entry.get("checks") or []
    verdicts = {c.get("verdict") for c in checks}
    if not checks:
        reasons["no check was ever run on this column"] += 1
    elif "not_supported" in verdicts:
        reasons["the checker rejected the value"] += 1
    else:
        reasons[f"checked but blanked anyway ({'/'.join(sorted(v for v in verdicts if v))})"] += 1
print("\n  why the value did not survive:")
for reason, n in reasons.most_common():
    print(f"    {n:4}  {reason}")

print("\n  the unanimous ones:")
for k in agreed[:8]:
    print(f"    {k[0][:22]:22} | {k[1][:44]:44} both agents said {A[k][1][:26]!r}")
