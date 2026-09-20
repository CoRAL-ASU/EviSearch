"""Early R4 signal: the knowledge notes against R3, on the documents R4 has finished, same arbiter (v4).

Only documents whose reconciliation stage has every column are used, and only cells judged on both sides are scored.
Unjudged predictions are reported, never guessed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
R3 = "schema-mhspc-trials-20260919020503-v4"
R4 = sys.argv[1] if len(sys.argv) > 1 else "r4-notes-r1"
EXPECTED = 133

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
all_docs = cs.select_docs("all", gold)


def complete(run, doc):
    p = ROOT / doc / "runs" / run / "reconciliation_agent" / "reconciled_results.json"
    if not p.exists():
        return False
    cols = json.loads(p.read_text()).get("columns") or {}
    return len(cols) >= EXPECTED


docs = [d for d in all_docs if complete(R4, d) and complete(R3, d)]
print(f"comparing {R4} against {R3}")
print(f"documents finished on both sides: {len(docs)} of {len(all_docs)}")
if not docs:
    raise SystemExit(0)
for d in docs:
    print(f"  {d}")


def score(run, stage="reconciliation_agent"):
    rows, un = cs.score(list(cs.cells(cs.System.parse(f"X={run}/{stage}"), docs, gold=gold)), labels)
    return {(r.cell.doc, r.cell.column): r.score for r in rows}, un


for stage, label in (("agent_extractor", "Agent A"), ("search_agent", "Agent B"), ("reconciliation_agent", "ARBITER")):
    s3, u3 = score(R3, stage)
    s4, u4 = score(R4, stage)
    keys = sorted(set(s3) & set(s4))
    if not keys:
        print(f"\n{label}: no comparable cells")
        continue
    p3 = 100 * sum(s3[k] for k in keys) / len(keys)
    p4 = 100 * sum(s4[k] for k in keys) / len(keys)
    better = [k for k in keys if s4[k] > s3[k]]
    worse = [k for k in keys if s4[k] < s3[k]]
    print(f"\n{label:8} on {len(keys)} comparable cells:  R3 {p3:.2f}   R4-notes {p4:.2f}   delta {p4-p3:+.2f}")
    print(f"         better on {len(better)}, worse on {len(worse)}, same on {len(keys)-len(better)-len(worse)}"
          f"   | unjudged: R3 {len(u3)}, R4 {len(u4)}")
    cells4 = {(c.doc, c.column): c for c in cs.cells(cs.System.parse(f"X={R4}/{stage}"), docs, gold=gold)}
    cells3 = {(c.doc, c.column): c for c in cs.cells(cs.System.parse(f"X={R3}/{stage}"), docs, gold=gold)}
    for tag, group in (("BETTER", better), ("WORSE", worse)):
        for k in group[:5]:
            print(f"    [{tag}] {k[1][:44]:44} gold={str(cells4[k].gold)[:24]!r}")
            print(f"             R3={str(cells3[k].pred)[:30]!r} ({s3[k]})  R4={str(cells4[k].pred)[:30]!r} ({s4[k]})")
