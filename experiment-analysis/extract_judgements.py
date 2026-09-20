"""Pull a scoring workflow's judgements out of its task output and check they cover the batch exactly.

The workflow's return value is in the task output file, not the per-agent journal (the journal holds one record per
agent). Nothing is written unless the batch lines up: a duplicate id, an id nobody asked for, or a score outside the
allowed set means the run is refused rather than partly merged.

  python experiment-analysis/extract_judgements.py <task-output.json> <unjudged.json> <out.json>
"""
import json
import sys
from collections import Counter
from pathlib import Path

task_output, source_path, out_path = (Path(p) for p in sys.argv[1:4])

payload = json.loads(task_output.read_text())
final = payload.get("result") if isinstance(payload.get("result"), dict) else payload
if isinstance(final, dict) and "judgements" not in final:
    final = None
for note in payload.get("logs") or []:
    print(f"workflow log: {note}")
source = json.loads(source_path.read_text())
wanted = {it["id"]: it for it in source["items"]}

if final is None:
    print("no workflow return value with judgements in the journal")
    raise SystemExit(1)

rows = final["judgements"]
ids = [r["id"] for r in rows]
dupes = [i for i, n in Counter(ids).items() if n > 1]
unknown = sorted(set(ids) - set(wanted))
missing = sorted(set(wanted) - set(ids))
bad = [r for r in rows if r.get("correctness") not in (0.0, 0.5, 1.0) or r.get("completeness") not in (0.0, 0.5, 1.0)]

print(f"batch {final.get('batch')}: {len(rows)} judgements for {len(wanted)} items")
print(f"  duplicate ids : {len(dupes)}")
print(f"  unknown ids   : {len(unknown)}" + (f" -> {unknown[:4]}" if unknown else ""))
print(f"  missing items : {len(missing)}" + (f" -> {missing[:4]}" if missing else ""))
print(f"  out-of-range  : {len(bad)}")
print(f"  adjudicated   : {sum(1 for r in rows if '[adjudicated]' in str(r.get('reason','')))}")
print(f"  score mix     : {dict(Counter((r['correctness'], r['completeness']) for r in rows))}")

if dupes or unknown or bad:
    print("\nREFUSING to write: the batch does not line up with what was asked for")
    raise SystemExit(1)

enriched = []
for r in rows:
    item = wanted[r["id"]]
    enriched.append({**r, "doc": item["doc"], "column": item["column"], "category": item["category"],
                     "gold": item["gold"], "pred": item["pred"]})
out_path.write_text(json.dumps({"judgements": enriched, "source": list(wanted.values())}, ensure_ascii=False, indent=1))
print(f"\nwrote {len(enriched)} judgements -> {out_path}")
if missing:
    print(f"NOTE {len(missing)} items were not judged and stay unscored rather than guessed")
