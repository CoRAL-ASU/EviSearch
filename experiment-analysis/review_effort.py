"""What would a human reviewer actually have to look at, and what would it buy?

For each run: how many cells the arbiter flags for review, whether those flags land on the cells that are actually
wrong (precision / recall), and the accuracy a reviewer would reach under three policies:

  flagged-only   fix every flagged cell that is wrong, leave the rest
  all-133        review every column of every paper
  perfect-triage the ceiling for any flagging scheme: fix every wrong cell, review nothing else

  python experiment-analysis/review_effort.py <run> [<run> ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
PAPERS = 10
labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)

print(f"{'run':26} {'cells':>6} {'flagged':>8} {'/paper':>7} {'wrong':>6} {'prec':>6} {'recall':>7} "
      f"{'now':>7} {'flagged-only':>13} {'all-133':>8}")
print("-" * 104)

for run in sys.argv[1:]:
    rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={run}/reconciliation_agent"), docs, gold=gold)), labels)
    scored = {(r.cell.doc, r.cell.column): r for r in rows}

    flagged = set()
    for doc in docs:
        path = ROOT / doc / "runs" / run / "reconciliation_agent" / "reconciled_results.json"
        if not path.exists():
            continue
        for name, entry in (json.loads(path.read_text()).get("columns") or {}).items():
            if entry.get("needs_review") and (doc, name) in scored:
                flagged.add((doc, name))

    n = len(scored)
    wrong = {k for k, r in scored.items() if r.score < 1.0}
    hit = flagged & wrong
    precision = 100 * len(hit) / len(flagged) if flagged else 0.0
    recall = 100 * len(hit) / len(wrong) if wrong else 0.0

    now = 100 * sum(r.score for r in scored.values()) / n
    # a reviewer who opens only the flagged cells can fix at most the wrong ones among them
    gained = sum(1.0 - scored[k].score for k in hit)
    flagged_only = 100 * (sum(r.score for r in scored.values()) + gained) / n
    print(f"{run:26} {n:6} {len(flagged):8} {len(flagged)/PAPERS:7.1f} {len(wrong):6} "
          f"{precision:5.1f}% {recall:6.1f}% {now:6.2f}% {flagged_only:12.2f}% {100.0:7.1f}%")

print()
print("`flagged-only` assumes the reviewer corrects every wrong cell they open and never breaks a right one -")
print("an upper bound on that policy. `all-133` is 100% by construction, at the cost of reading every cell.")
