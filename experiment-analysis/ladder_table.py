#!/usr/bin/env python3
"""The authoritative comparison table for any set of runs.

Every system is scored on ONE common cell set (the intersection of cells all named systems have a judgement for), so
the numbers are comparable. The gold correction overlay is applied unless --raw is passed, and the eleven cells it
changes are re-judged in memory because the label store is keyed on (doc, column, prediction) and does not invalidate
when gold changes.

  python experiment-analysis/ladder_table.py r4-notes-r1 r4-notes-r2 [--raw] [--baseline qwen_final_b1]
"""
from __future__ import annotations

import re
import sys

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda s: re.sub(r"[\s,.]+$", "", N(s).replace("–", "-").replace("—", "-")).lower()
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

CORRECTIONS = [
    ("NCT01809691_Aggarwal", "COE_RCT_IND_OVERALL_RJ", "Yes (randomised phase 3)"),
    ("NCT02446405_Sweeney", "Median Age (years) | Control", "69 (64-75)"),
    ("NCT02446405_Sweeney", "Median Age (years) | Treatment", "69 (63-74)"),
    ("NCT02799602_Hussain", "COE_RCT_IND_OVERALL_RJ", "Yes (randomised, double-blind, placebo-controlled phase 3)"),
    ("NCT02799602_Smith", "Add-on Treatment", "Darolutamide"),
    ("NCT02799602_Smith", "Region - N (%) | South America | Control", "Included in Rest of the world"),
    ("NCT02799602_Smith", "Region - N (%) | South America | Treatment", "Included in Rest of the world"),
    ("NCT01957436_Fizazi", "PS - N (%) | 0 | Control", "412 (70%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 0 | Treatment", "412 (71%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 1-2 | Control", "177 (30%)"),
    ("NCT01957436_Fizazi", "PS - N (%) | 1-2 | Treatment", "171 (29%)"),
]
STAGES = [("agent_extractor", "Agent A"), ("search_agent", "Agent B"), ("reconciliation_agent", "ARBITER")]


def build(overlay: bool):
    gold, labels = cs.load_gold(), dict(cs.load_labels(cs.labels_path()))
    touched = set()
    if overlay:
        for prefix, col, new in CORRECTIONS:
            hits = [d for d in gold if str(d).startswith(prefix)]
            if not hits:
                continue
            entry = gold[hits[0]]
            holder = entry["columns"] if isinstance(entry, dict) and "columns" in entry else entry
            cur = holder.get(col)
            if isinstance(cur, dict):
                cur["value"] = new
            else:
                holder[col] = new
            touched.add((hits[0], col))
    docs = cs.select_docs("all", gold)

    def score(spec):
        cells = list(cs.cells(cs.System.parse(f"X={spec}"), docs, gold=gold))
        for c in cells:
            if (c.doc, c.column) not in touched:
                continue
            g, p = N(c.gold), N(c.pred)
            if NR(p):
                v = (0.0, 0.0) if g else (1.0, 1.0)
            elif SQ(p) == SQ(g) or (re.match(r"^\d+(\.\d+)?$", p) and SQ(g).startswith(SQ(p))):
                v = (1.0, 1.0)
            else:
                v = (1.0, 1.0) if SQ(p).split("(")[0] == SQ(g).split("(")[0] else (0.0, 0.0)
            labels[c.id] = {"correctness": v[0], "completeness": v[1]}
        rows, unscored = cs.score(cells, labels)
        return {(r.cell.doc, r.cell.column): r.score for r in rows}, len(unscored)

    return score


def main() -> int:
    argv = sys.argv[1:]
    overlay = "--raw" not in argv
    baseline = "qwen_final_b1"
    args = []
    skip = False
    for i, a in enumerate(argv):  # --baseline takes a value, which is not itself a run to table
        if skip:
            skip = False
            continue
        if a == "--baseline":
            baseline = argv[i + 1] if i + 1 < len(argv) else baseline
            skip = True
        elif not a.startswith("--"):
            args.append(a)
    if not args:
        print(__doc__)
        return 2
    score = build(overlay)
    print(f"gold overlay: {'ON (11 page-verified corrections)' if overlay else 'OFF (raw gold)'}")

    b1, b1_un = score(f"{baseline}/markdown_baseline")
    for run in args:
        series, unjudged = {}, {}
        for stage, label in STAGES:
            series[label], unjudged[label] = score(f"{run}/{stage}")
        keys = sorted(set.intersection(*(set(s) for s in series.values()), set(b1)))
        if not keys:
            print(f"\n### {run}: no comparable cells yet")
            continue
        pct = lambda d: 100 * sum(d[k] for k in keys) / len(keys)
        a, b, e = pct(series["Agent A"]), pct(series["Agent B"]), pct(series["ARBITER"])
        best = max(a, b)
        print(f"\n### {run}   common cell set: {len(keys)}")
        print(f"  B1 baseline   {pct(b1):6.2f}   (unjudged {b1_un})")
        print(f"  Agent A       {a:6.2f}   (unjudged {unjudged['Agent A']})")
        print(f"  Agent B       {b:6.2f}   (unjudged {unjudged['Agent B']})")
        print(f"  ARBITER       {e:6.2f}   (unjudged {unjudged['ARBITER']})")
        print(f"  max(A,B)      {best:6.2f}")
        print(f"  BAR arbiter >= max(A,B): {'PASS by %.2f' % (e - best) if e >= best else 'FAIL by %.2f' % (best - e)}")
        print(f"  arbiter - B1  {e - pct(b1):+.2f}")
        A, B, E = series["Agent A"], series["Agent B"], series["ARBITER"]
        win = sum(1 for k in keys if E[k] > max(A[k], B[k]))
        loss = sum(1 for k in keys if E[k] < max(A[k], B[k]))
        print(f"  per cell vs max(A,B): better {win}, worse {loss}, equal {len(keys) - win - loss}")
        right = lambda d, k: d[k] == 1.0
        only_a = [k for k in keys if right(A, k) and not right(B, k)]
        only_b = [k for k in keys if not right(A, k) and right(B, k)]
        for label, group in (("only Agent A right", only_a), ("only Agent B right", only_b)):
            if group:
                kept = sum(1 for k in group if E[k] == 1.0)
                print(f"    {label:20} {len(group):4} cells -> arbiter fully right on {kept} ({100*kept/len(group):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
