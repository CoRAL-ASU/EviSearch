"""Score the v5 smoke run against v4 on the same document and the same agent outputs.

Both arbiters saw identical Agent A and Agent B results (v5's run was seeded by copying them), so the difference is
the arbiter. Cells whose prediction has no cached judgement are reported, never guessed.
"""
import sys

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

DOC = "NCT00104715_Gravis_GETUG_EU'15"
V4 = "schema-mhspc-trials-20260919020503-v4"
V5 = "r4-smoke-v5"

labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()


def cells(run, stage="reconciliation_agent"):
    return {(c.doc, c.column): c for c in cs.cells(cs.System.parse(f"X={run}/{stage}"), [DOC], gold=gold)}


v4, v5 = cells(V4), cells(V5)
a, b = cells(V4, "agent_extractor"), cells(V4, "search_agent")
keys = sorted(set(v4) & set(v5))
print(f"{DOC}: {len(keys)} columns\n")

rows4, un4 = cs.score([v4[k] for k in keys], labels)
rows5, un5 = cs.score([v5[k] for k in keys], labels)
s4 = {(r.cell.doc, r.cell.column): r.score for r in rows4}
s5 = {(r.cell.doc, r.cell.column): r.score for r in rows5}
scored = sorted(set(s4) & set(s5))
print(f"scored by cached judgements on both sides: {len(scored)} of {len(keys)}")
print(f"unjudged predictions -> v4 {len(un4)}, v5 {len(un5)} (new strings need a fresh judgement)\n")
if scored:
    p = lambda d: 100 * sum(d[k] for k in scored) / len(scored)
    print(f"  arbiter v4 {p(s4):.2f}   arbiter v5 {p(s5):.2f}   delta {p(s5) - p(s4):+.2f}  (on the {len(scored)} comparable cells)")
    changed = [k for k in scored if s4[k] != s5[k]]
    better = [k for k in changed if s5[k] > s4[k]]
    worse = [k for k in changed if s5[k] < s4[k]]
    print(f"  v5 better on {len(better)}, worse on {len(worse)}, same on {len(scored) - len(changed)}")
    for tag, group in (("V5 BETTER", better), ("V5 WORSE", worse)):
        for k in group[:6]:
            print(f"    [{tag}] {k[1][:46]:46} gold={str(v5[k].gold)[:26]!r}")
            print(f"              v4={str(v4[k].pred)[:34]!r} ({s4[k]})   v5={str(v5[k].pred)[:34]!r} ({s5[k]})")

print(f"\nunjudged v5 predictions (first 10 of {len(un5)}):")
for c in un5[:10]:
    own = ""
    print(f"  {c.column[:44]:44} gold={str(c.gold)[:24]!r}  v5={str(c.pred)[:30]!r}  v4={str(v4[(c.doc, c.column)].pred)[:24]!r}")
