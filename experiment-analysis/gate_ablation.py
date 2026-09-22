"""Appendix B: what the verifier's negative readings did to the table under each admission policy.

  python experiment-analysis/gate_ablation.py <run> [<run> ...]      # two runs of one policy, pooled

A negative reading is a verifier verdict of not_supported or partial. After one, the value read either shipped anyway
or was dropped in favour of another. A dropped value is scored by the rubric labels (keyed on document, column and
value, so a value any stage of the pooled runs proposed has a label), which says whether the drop removed a correct or
an incorrect value; the cell's final score says whether it corrected the cell or made it incorrect.

  dropped            values dropped after a negative reading, % of cells
  of which correct   % of the dropped values that the rubric scores fully correct
  corrected          cells where a partial or not_supported reading dropped an incorrect value and the cell ended correct
  made incorrect     cells where such a reading dropped a correct value and the cell ended incorrect

With no arguments, both policies: agreement-gated (AGREEMENT_GATE_RUNS) and provenance-gated (the system runs).
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402
from paper_runs import AGREEMENT_GATE_RUNS, SYSTEM_RUNS, root_of  # noqa: E402

N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: N(v).lower().rstrip(".").replace("%", "")


def ablation(runs):
    labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
    docs = cs.select_docs("all", gold)
    score_of, final = {}, {}
    for run in runs:
        for stage in ("agent_extractor", "search_agent", "reconciliation_agent"):
            rows, _ = cs.score(list(cs.cells(cs.System.parse(f"X={run}/{stage}"), docs, root_of(run), gold=gold)), labels)
            for r in rows:
                score_of.setdefault((r.cell.doc, r.cell.column, SQ(r.cell.pred)), r.score)
                if stage == "reconciliation_agent":
                    final[(run, r.cell.doc, r.cell.column)] = r.score
    n = len(final)
    dropped = collections.Counter()
    fixed = broke = 0
    for run in runs:
        for doc in docs:
            path = root_of(run) / doc / "runs" / run / "reconciliation_agent" / "reconciled_results.json"
            for name, entry in (json.loads(path.read_text()).get("columns") or {}).items():
                shipped, fs = N(entry.get("value")), final.get((run, doc, name))
                for c in entry.get("checks") or []:
                    verdict, cv = c.get("verdict"), N(c.get("value"))
                    if verdict not in ("not_supported", "partial") or SQ(cv) == SQ(shipped):
                        continue
                    s = score_of.get((doc, name, SQ(cv)))
                    if s is None:
                        continue
                    dropped["correct" if s >= 1.0 else "incorrect"] += 1
                    if fs is not None:
                        fixed += s < 1.0 and fs >= 1.0
                        broke += s >= 1.0 and fs < 1.0
    total = dropped["correct"] + dropped["incorrect"]
    return {"cells": n, "dropped": total, "dropped_correct": dropped["correct"], "dropped_incorrect": dropped["incorrect"],
            "fixed": fixed, "broke": broke, "dropped_pct": 100 * total / n,
            "dropped_correct_share": 100 * dropped["correct"] / max(1, total),
            "fixed_pct": 100 * fixed / n, "broke_pct": 100 * broke / n}


def report(runs):
    r = ablation(runs)
    print(f"{' '.join(runs)}: {r['cells']} cells pooled")
    print(f"  dropped after a negative reading  {r['dropped_pct']:.1f}% of cells   "
          f"({r['dropped_correct_share']:.0f}% of them correct)   [{r['dropped']}: {r['dropped_correct']} correct, {r['dropped_incorrect']} incorrect]")
    print(f"  cells corrected by the reading    {r['fixed_pct']:.2f}%   [{r['fixed']}]")
    print(f"  cells made incorrect              {r['broke_pct']:.2f}%   [{r['broke']}]")


if __name__ == "__main__":
    for runs in ([sys.argv[1:]] if sys.argv[1:] else [list(AGREEMENT_GATE_RUNS), list(SYSTEM_RUNS)]):
        report(runs)
