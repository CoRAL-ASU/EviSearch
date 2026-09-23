"""Every number in the EACL 2027 paper's Tables 1 and 2 and Figure 2, from the scored runs.

  python experiment-analysis/paper_numbers.py            # prints the tables and emits LaTeX macros + pgfplots data

Table 1   extraction accuracy and traceability: the single-pass baseline, each extraction agent alone, the oracle
          selection over the two agents, and EviSearch. Means over two independent runs. Cited: the share of stated
          values carrying a page and a quotation. Verified: the share a second reading of the cited page reproduced;
          only EviSearch reads a value a second time, so it is zero for every other row by construction.
Table 2   extraction-agent accuracy as the schema and the curation knowledge base are aligned: auto-drafted schema,
          schema after two and after three review rounds (all with the fixed extraction guidelines), and the final
          schema with the knowledge base; the Reconciliation Agent's accuracy for every row.
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


def absent(value):
    v = N(value).lower()
    return NR(v) or v in {"no", "none"}


def stage_columns(run, stage):
    name = "reconciled_results.json" if stage == "reconciliation_agent" else "extraction_results.json"
    out = {}
    for doc in docs:
        cols = json.loads((root_of(run) / doc / "runs" / run / stage / name).read_text())["columns"]
        out.update({(doc, col): entry for col, entry in cols.items()})
    return out


def cites(entry):
    """A stated value carrying a page and a quotation."""
    return any(isinstance(a, dict) and a.get("page") and (a.get("evidence") or a.get("verbatim_quote"))
               for a in (entry or {}).get("attribution") or [])


def cited_share(entries):
    stated = [e for e in entries if not absent((e or {}).get("value"))]
    return 100 * sum(cites(e) for e in stated) / len(stated) if stated else 0.0


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

# citations: each agent's own, the baseline's (it produces none), and the oracle's (the chosen agent's; ties go to B)
citations = {"baseline": [cited_share(stage_columns(r, "markdown_baseline").values()) for r in BASELINE_RUNS],
         "agentA": [], "agentB": [], "oracle": []}
for r in SYSTEM_RUNS:
    ca, cb = stage_columns(r, "agent_extractor"), stage_columns(r, "search_agent")
    citations["agentA"].append(cited_share(ca.values()))
    citations["agentB"].append(cited_share(cb.values()))
    citations["oracle"].append(cited_share([ca.get(k) if A[r][k][0] > B[r][k][0] else cb.get(k) for k in A[r]]))

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
out["runs"] = {key: vals for key, _, vals in rows}
for key, vals in citations.items():
    out[f"cited_{key}"] = mean(vals)
print("  cited: " + "  ".join(f"{k} {mean(v):.1f}" for k, v in citations.items()))
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
out["queue_acc_runs"] = [p[1][1] for p in curve_runs]
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
    e = mean([acc(scored(r, "reconciliation_agent")) for r in runs])
    ladder[key + "Sys"] = e
    print(f"  {key:8} Agent A {a:6.2f}   Agent B {b:6.2f}   EviSearch {e:6.2f}")

# ---- emit --------------------------------------------------------------------------------------------------------
def gain(to: float, frm: float) -> float:
    """A difference the text reports, taken between the values as printed so a reader can recompute it."""
    return round(to, 2) - round(frm, 2)


macro = {
    "accBaseline": out["acc_baseline"], "accAgentA": out["acc_agentA"], "accAgentB": out["acc_agentB"],
    "accOracleAB": out["acc_oracle"], "accSystem": out["acc_system"],
    "citedNow": out["cover"], "corroborated": out["corro"],
    "queueFrac": out["queue_frac"], "queueAcc": out["queue_acc"],
    "queueRandomSys": out["queue_random_sys"], "queueRandomB": out["queue_random_b"],
    "tierTwoFrac": out["tier2_frac"], "tierTwoAcc": out["tier2_acc"],
    "draftA": ladder["draft"][0], "draftB": ladder["draft"][1], "revA": ladder["rev"][0], "revB": ladder["rev"][1],
    "revfourA": ladder["revfour"][0], "revfourB": ladder["revfour"][1], "kbA": ladder["kb"][0], "kbB": ladder["kb"][1],
    "draftSys": ladder["draftSys"], "revSys": ladder["revSys"], "revfourSys": ladder["revfourSys"], "kbSys": ladder["kbSys"],
    "accBaselineRunOne": out["runs"]["baseline"][0], "accBaselineRunTwo": out["runs"]["baseline"][1],
    "accAgentARunOne": out["runs"]["agentA"][0], "accAgentARunTwo": out["runs"]["agentA"][1],
    "accAgentBRunOne": out["runs"]["agentB"][0], "accAgentBRunTwo": out["runs"]["agentB"][1],
    "accOracleRunOne": out["runs"]["oracle"][0], "accOracleRunTwo": out["runs"]["oracle"][1],
    "accSystemRunOne": out["runs"]["system"][0], "accSystemRunTwo": out["runs"]["system"][1],
    "queueAccRunOne": out["queue_acc_runs"][0], "queueAccRunTwo": out["queue_acc_runs"][1],
    "gainOverBaseline": gain(out["acc_system"], out["acc_baseline"]),
    "citedBaseline": out["cited_baseline"], "citedAgentA": out["cited_agentA"], "citedAgentB": out["cited_agentB"],
    "citedOracle": out["cited_oracle"],
    "kbGainA": gain(ladder["kb"][0], ladder["revfour"][0]), "kbGainB": gain(ladder["kb"][1], ladder["revfour"][1]),
    "kbGainSys": gain(ladder["kbSys"], ladder["revfourSys"]),
    "totalGainA": gain(ladder["kb"][0], ladder["draft"][0]), "totalGainB": gain(ladder["kb"][1], ladder["draft"][1]),
    "totalGainSys": gain(ladder["kbSys"], ladder["draftSys"]),
}
ONE_DECIMAL = ("citedNow", "corroborated", "queueFrac", "tierTwoFrac", "citedBaseline", "citedAgentA",
               "citedAgentB", "citedOracle")
print("\nMACROS")
for k, v in macro.items():
    print(f"\\renewcommand{{\\{k}}}{{{v:.1f}}}" if k in ONE_DECIMAL else f"\\renewcommand{{\\{k}}}{{{v:.2f}}}")
print("\nPGFPLOTS triage curve:", " ".join(f"({x:.1f},{y:.2f})" for x, y in triage))
