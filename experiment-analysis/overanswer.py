"""Cells where the arbiter gave a value and gold is empty: is the paper silent, or does gold decline to record it?

If the value is printed in the paper and gold is blank, the arbiter read correctly and the disagreement is about what
the column should hold - a definition question, not an extraction failure.

  python experiment-analysis/overanswer.py <run>
"""
import sys
from collections import Counter

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

RUN = sys.argv[1]
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def cells(stage):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={RUN}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred), N(r.cell.gold)) for r in rows}


A, B, E = cells("agent_extractor"), cells("search_agent"), cells("reconciliation_agent")
keys = sorted(set(A) & set(B) & set(E))
over = [k for k in keys if not NR(E[k][1]) and NR(E[k][2]) and E[k][0] < max(A[k][0], B[k][0])]

print(f"{RUN}: {len(over)} cells where the arbiter answered and gold is empty (and an agent's abstention scored better)\n")
print(f"  by column family: {dict(Counter(k[1].split('|')[0].strip() for k in over).most_common())}")
backed = [k for k in over if not NR(A[k][1]) or not NR(B[k][1])]
print(f"  an agent produced the same kind of value too : {len(backed)} of {len(over)}")
print("     (so the paper does carry a number; the disagreement is whether this column should hold it)\n")
for k in over:
    who = "A" if not NR(A[k][1]) else ("B" if not NR(B[k][1]) else "neither agent")
    print(f"  {k[0][:22]:22} | {k[1][:40]:40}")
    print(f"     arbiter {E[k][1][:40]!r}   found also by: {who}")
