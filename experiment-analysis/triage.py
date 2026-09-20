"""Which cells should a human open? Competing triage policies, scored by what they would catch per cell read.

Each policy selects a subset of the 133 columns per paper. For each: how many cells the reviewer opens, how many of
the run's wrong cells are inside that subset (recall), how many opened cells are actually wrong (precision), and the
accuracy the table reaches if every wrong cell opened is corrected.

  python experiment-analysis/triage.py <run>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1]
PAPERS = 10
N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: N(v).lower().rstrip(".").replace("%", "")
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def stage(name):
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={RUN}/{name}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred)) for r in rows}


A, B, E = stage("agent_extractor"), stage("search_agent"), stage("reconciliation_agent")
keys = sorted(set(A) & set(B) & set(E))
flags, verified = {}, {}
for doc in docs:
    path = ROOT / doc / "runs" / RUN / "reconciliation_agent" / "reconciled_results.json"
    if not path.exists():
        continue
    for name, entry in (json.loads(path.read_text()).get("columns") or {}).items():
        flags[(doc, name)] = bool(entry.get("needs_review"))
        verified[(doc, name)] = bool(entry.get("verified"))

total = sum(E[k][0] for k in keys)
wrong = [k for k in keys if E[k][0] < 1.0]

disagree = lambda k: SQ(A[k][1]) != SQ(B[k][1])
both_empty = lambda k: NR(A[k][1]) and NR(B[k][1])

# disagree and both_empty are disjoint: if both agents abstain their values match, so those cells are not in the
# disagree set. The union row is therefore exactly the two parts added, and each part's marginal value is its own row.
POLICIES = [
    ("arbiter's needs_review flag", lambda k: flags.get(k, False)),
    ("A: the two agents disagree", disagree),
    ("B: both agents left it empty", both_empty),
    ("A + B (disjoint union)", lambda k: disagree(k) or both_empty(k)),
    ("the shipped value is not verified", lambda k: not verified.get(k, False)),
    ("every column (133 per paper)", lambda k: True),
]

print(f"{RUN}: {len(keys)} cells, {len(wrong)} of them wrong ({len(wrong)/PAPERS:.1f} per paper), "
      f"table at {100*total/len(keys):.2f}%\n")
print(f"{'policy':34} {'open':>6} {'/paper':>7} {'caught':>7} {'recall':>7} {'prec':>6} {'ceiling':>8} {'per 10 read':>12}")
print("-" * 96)
for label, pick in POLICIES:
    opened = [k for k in keys if pick(k)]
    caught = [k for k in opened if E[k][0] < 1.0]
    gain = sum(1.0 - E[k][0] for k in caught)
    ceiling = 100 * (total + gain) / len(keys)
    per10 = 10 * gain / len(opened) if opened else 0
    print(f"{label:34} {len(opened):6} {len(opened)/PAPERS:7.1f} {len(caught):7} "
          f"{100*len(caught)/len(wrong):6.1f}% {100*len(caught)/len(opened) if opened else 0:5.1f}% "
          f"{ceiling:7.2f}% {per10:11.2f} pts")
print("\n'ceiling' assumes every wrong cell opened is corrected and no right cell is broken.")
print("'per 10 read' is accuracy points gained per ten cells the reviewer opens - the efficiency of the policy.")
