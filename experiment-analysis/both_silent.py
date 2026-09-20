"""On cells where BOTH agents abstained, what does answering them buy or cost?

This is the narrowest use of the arbiter's own reading: it can only add a value where there was none, so it cannot
take the worse of two stated values. The question is whether the values it adds are right more often than wrong.

  python experiment-analysis/both_silent.py <run> [<run> ...]
"""
import sys

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def cells(run, stage):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={run}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred), N(r.cell.gold)) for r in rows}


for run in sys.argv[1:]:
    A, B, E = cells(run, "agent_extractor"), cells(run, "search_agent"), cells(run, "reconciliation_agent")
    keys = sorted(set(A) & set(B) & set(E))
    silent = [k for k in keys if NR(A[k][1]) and NR(B[k][1])]
    answered = [k for k in silent if not NR(E[k][1])]
    kept = [k for k in silent if NR(E[k][1])]
    gain = sum(E[k][0] - max(A[k][0], B[k][0]) for k in answered)
    print(f"\n{run}: {len(silent)} cells where both agents abstained ({100*len(silent)/len(keys):.0f}% of the table)")
    print(f"  the arbiter answered {len(answered)} of them, left {len(kept)} empty")
    if answered:
        won = [k for k in answered if E[k][0] > max(A[k][0], B[k][0])]
        lost = [k for k in answered if E[k][0] < max(A[k][0], B[k][0])]
        print(f"    of those it answered: right {len(won)}, wrong {len(lost)}, no change {len(answered)-len(won)-len(lost)}")
        print(f"    NET from answering them: {gain:+.2f} cell-points ({100*gain/len(keys):+.2f}% of the table)")
    if kept:
        missed = [k for k in kept if not NR(k and E[k][2])]
        print(f"    of those it left empty, gold actually had a value on {len(missed)} "
              f"(worth {len(missed)*100/len(keys):.2f}% if they could be found)")
