"""Why a run's arbiter scores below max(A,B): the losing cells, grouped by what happened.

  python experiment-analysis/losses.py <run> [limit]
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
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def cells(stage):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={RUN}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred), N(r.cell.gold)) for r in rows}


A, B, E = cells("agent_extractor"), cells("search_agent"), cells("reconciliation_agent")
keys = sorted(set(A) & set(B) & set(E))
losing = [k for k in keys if E[k][0] < max(A[k][0], B[k][0])]
winning = [k for k in keys if E[k][0] > max(A[k][0], B[k][0])]
print(f"{RUN}: {len(losing)} cells below max(A,B), {len(winning)} above, on {len(keys)} cells")
print(f"  points given away: {sum(max(A[k][0],B[k][0]) - E[k][0] for k in losing):.2f}"
      f"   points won back: {sum(E[k][0] - max(A[k][0],B[k][0]) for k in winning):.2f}")

kinds = Counter()
for k in losing:
    best = "A" if A[k][0] > B[k][0] else "B"
    right = A[k] if best == "A" else B[k]
    if NR(E[k][1]) and not NR(right[1]):
        kinds["shipped an absence over a correct value"] += 1
    elif NR(right[1]) and not NR(E[k][1]):
        kinds["shipped a value where the correct answer was absence"] += 1
    else:
        kinds[f"took the worse of two stated values (better was {best})"] += 1
print("\nwhat went wrong:")
for kind, n in kinds.most_common():
    print(f"  {n:4}  {kind}")

fams = Counter(k[1].split("|")[0].strip() for k in losing)
print(f"\ncolumn families losing most: {dict(fams.most_common(6))}")

print(f"\nfirst {LIMIT} losing cells:")
for k in losing[:LIMIT]:
    print(f"  {k[0][:24]:24} | {k[1][:42]:42}")
    print(f"     gold {k and E[k][2][:54]!r}")
    print(f"     A={A[k][1][:34]!r}({A[k][0]})  B={B[k][1][:34]!r}({B[k][0]})  ARB={E[k][1][:34]!r}({E[k][0]})")

if winning:
    print(f"\ncells the arbiter got right that BOTH agents missed ({len(winning)}):")
    for k in winning[:6]:
        print(f"  {k[0][:24]:24} | {k[1][:40]:40} gold={E[k][2][:32]!r}")
        print(f"     A={A[k][1][:28]!r}  B={B[k][1][:28]!r}  ARB={E[k][1][:36]!r}({E[k][0]})")
