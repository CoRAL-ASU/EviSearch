"""Every number in the EACL 2027 paper's Table 1 and Figure 2, from the scored runs.

  python experiment-analysis/paper_numbers.py            # prints the tables and emits LaTeX macros + pgfplots data

Table 1   extraction accuracy and traceability: the single-pass baseline, each extraction agent alone, the oracle
          selection over the two agents, and EviSearch. Means over two independent runs.
Table 2   extraction-agent accuracy as the schema and the curation knowledge base are aligned: auto-drafted schema,
          schema after two and after three review rounds (all with the fixed extraction guidelines), and the final
          schema with the knowledge base. Reconciled accuracy for the first two rows, whose reconciliation is fixed.
Figure 2  accuracy after human review against the fraction of cells reviewed, for review ordered by the disagreement
          signal and for review in random order. A reviewer is assumed to correct every cell they open (an upper
          bound: no user study). Random review of a fraction f lifts accuracy by f times the error mass in
          expectation, since errors are then uniformly distributed over the cells opened.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402
from paper_runs import BASELINE_RUNS, LADDER, SYSTEM_RUNS, root_of  # noqa: E402

N = lambda v: " ".join(str(v if v is not None else "").split()).strip()
SQ = lambda v: N(v).lower().rstrip(".").replace("%", "")
NR = lambda v: (not N(v)) or N(v).lower().startswith(("not reported", "not found", "n/a", "not applicable"))

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
docs = cs.select_docs("all", gold)


def scored(run, stage):
    rows, missing = cs.score(list(cs.cells(cs.System.parse(f"X={run}/{stage}"), docs, root_of(run), gold=gold)), labels)
    if missing:
        raise SystemExit(f"{run}/{stage}: {len(missing)} cells unjudged - judge them before reporting")
    return {(r.cell.doc, r.cell.column): (r.score, N(r.cell.pred)) for r in rows}


def acc(cells):
    return 100 * sum(s for s, _ in cells.values()) / len(cells)


def mean(xs):
    return sum(xs) / len(xs)


out = {}
# ---- Table 1 -----------------------------------------------------------------------------------------------------
base = [acc(scored(r, "markdown_baseline")) for r in BASELINE_RUNS]
A = {r: scored(r, "agent_extractor") for r in SYSTEM_RUNS}
B = {r: scored(r, "search_agent") for r in SYSTEM_RUNS}
E = {r: scored(r, "reconciliation_agent") for r in SYSTEM_RUNS}
oracle = [100 * sum(max(A[r][k][0], B[r][k][0]) for k in A[r]) / len(A[r]) for r in SYSTEM_RUNS]

cover, corro = [], []
flags = {}
for r in SYSTEM_RUNS:
    values = cited = agreed = 0
    flags[r] = set()
    for doc in docs:
        cols = json.loads((root_of(r) / doc / "runs" / r / "reconciliation_agent" / "reconciled_results.json").read_text())["columns"]
        for col, e in cols.items():
            if e.get("needs_review"):
                flags[r].add((doc, col))
            v = N(e.get("value")).lower()
            if NR(v) or v in {"no", "none"}:
                continue
            values += 1
            if e.get("attribution"):
                cited += 1
                agreed += bool(e.get("verified"))
    cover.append(100 * cited / values)
    corro.append(100 * agreed / cited)

rows = [
    ("baseline", "Single-pass extraction", base),
    ("agentA", "PDF Query Agent (A)", [acc(A[r]) for r in SYSTEM_RUNS]),
    ("agentB", "Search Agent (B)", [acc(B[r]) for r in SYSTEM_RUNS]),
    ("oracle", "Oracle selection max(A,B)", oracle),
    ("system", "EviSearch", [acc(E[r]) for r in SYSTEM_RUNS]),
]
print("TABLE 1")
for key, label, vals in rows:
    out[f"acc_{key}"] = mean(vals)
    print(f"  {label:30} " + "  ".join(f"{v:6.2f}" for v in vals) + f"   mean {mean(vals):6.2f}")
out["cover"], out["corro"] = mean(cover), mean(corro)
print(f"  attribution coverage {cover} -> {mean(cover):.1f}   corroboration {[round(c, 2) for c in corro]} -> {mean(corro):.1f}")

# ---- Figure 2 ----------------------------------------------------------------------------------------------------
# tiers reviewed in order; inside a tier the order is random, so accuracy rises linearly across it
curve_runs = []
for r in SYSTEM_RUNS:
    keys = list(E[r])
    wrong = {k: 1.0 - E[r][k][0] for k in keys}
    silent = {k for k in keys if NR(A[r][k][1]) and NR(B[r][k][1])} - flags[r]
    tiers = [flags[r], silent, set(keys) - flags[r] - silent]
    points, x, y = [(0.0, acc(E[r]))], 0.0, acc(E[r])
    for tier in tiers:
        x += 100 * len(tier) / len(keys)
        y += 100 * sum(wrong[k] for k in tier) / len(keys)
        points.append((x, y))
    curve_runs.append(points)
triage = [(mean([p[i][0] for p in curve_runs]), mean([p[i][1] for p in curve_runs])) for i in range(4)]
auto_sys, auto_b = out["acc_system"], out["acc_agentB"]
print("\nFIGURE 2 (mean of runs)")
for x, y in triage:
    rand = auto_sys + x / 100 * (100 - auto_sys)
    randb = auto_b + x / 100 * (100 - auto_b)
    print(f"  reviewed {x:5.1f}%   triage {y:6.2f}   random(EviSearch) {rand:6.2f}   random(Agent B alone) {randb:6.2f}")
q, qacc = triage[1]
out["queue_frac"], out["queue_acc"] = q, qacc
out["queue_random_sys"] = auto_sys + q / 100 * (100 - auto_sys)
out["queue_random_b"] = auto_b + q / 100 * (100 - auto_b)
out["tier2_frac"], out["tier2_acc"] = triage[2]
out["queue_cells"] = mean([len(flags[r]) for r in SYSTEM_RUNS]) / len(docs)

# ---- Table 2 -----------------------------------------------------------------------------------------------------
ladder = {}
print("\nTABLE 2")
for key, runs in LADDER:
    a = mean([acc(scored(r, "agent_extractor")) for r in runs])
    b = mean([acc(scored(r, "search_agent")) for r in runs])
    ladder[key] = (a, b)
    line = f"  {key:8} Agent A {a:6.2f}   Agent B {b:6.2f}"
    if key in ("draft", "rev"):
        e = mean([acc(scored(r, "reconciliation_agent")) for r in runs]); ladder[key + "Sys"] = e
        line += f"   reconciled {e:6.2f}"
    print(line)

# ---- emit --------------------------------------------------------------------------------------------------------
macro = {
    "accBaseline": out["acc_baseline"], "accAgentA": out["acc_agentA"], "accAgentB": out["acc_agentB"],
    "accOracleAB": out["acc_oracle"], "accSystem": out["acc_system"],
    "citedNow": out["cover"], "corroborated": out["corro"],
    "queueFrac": out["queue_frac"], "queueAcc": out["queue_acc"], "queueCells": out["queue_cells"],
    "queueRandomSys": out["queue_random_sys"], "queueRandomB": out["queue_random_b"],
    "tierTwoFrac": out["tier2_frac"], "tierTwoAcc": out["tier2_acc"],
    "draftA": ladder["draft"][0], "draftB": ladder["draft"][1], "revA": ladder["rev"][0], "revB": ladder["rev"][1],
    "revfourA": ladder["revfour"][0], "revfourB": ladder["revfour"][1], "kbA": ladder["kb"][0], "kbB": ladder["kb"][1],
    "draftSys": ladder["draftSys"], "revSys": ladder["revSys"],
}
print("\nMACROS")
for k, v in macro.items():
    print(f"\\renewcommand{{\\{k}}}{{{v:.1f}}}" if k in ("citedNow", "corroborated", "queueFrac", "queueCells", "tierTwoFrac")
          else f"\\renewcommand{{\\{k}}}{{{v:.2f}}}")
print("\nPGFPLOTS triage curve:", " ".join(f"({x:.1f},{y:.2f})" for x, y in triage))
