#!/usr/bin/env python3
"""Collect the cells of one or more runs that have no cached judgement, for double-blind scoring.

Each item carries only the column, its definition, the gold value and the predicted value - never which system or run
produced it, and never the other systems' answers. Identical predictions across systems collapse to one item, because
the label store is keyed on (doc, column, normalised prediction).

  python experiment-analysis/collect_unjudged.py out.json <run>/<stage> [<run>/<stage> ...]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out_path, specs = Path(sys.argv[1]), sys.argv[2:]
    labels, gold = cs.load_labels(cs.labels_path()), cs.load_gold()
    docs = cs.select_docs("all", gold)
    columns = cs.load_columns()

    items: dict[str, dict] = {}
    per_spec = {}
    for spec in specs:
        cells = list(cs.cells(cs.System.parse(f"X={spec}"), docs, gold=gold))
        scored, unscored = cs.score(cells, labels)
        per_spec[spec] = {"cells": len(cells), "scored": len(scored), "unjudged": len(unscored)}
        for c in unscored:
            col = columns.get(c.column)
            items.setdefault(c.id, {
                "id": c.id,
                "doc": c.doc,
                "column": c.column,
                "category": c.category,
                "definition": (col.definition if col else "") or c.definition,
                "gold": str(c.gold or ""),
                "pred": str(c.pred or ""),
            })

    ordered = list(items.values())
    random.Random(0).shuffle(ordered)  # shuffled so a judge cannot infer a system from position
    out_path.write_text(json.dumps({"items": ordered, "by_spec": per_spec}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(ordered)} distinct (cell, prediction) pairs need a judgement -> {out_path}")
    for spec, n in per_spec.items():
        print(f"  {spec:62} {n['unjudged']:4} unjudged of {n['cells']} cells")
    by_doc: dict[str, int] = {}
    for it in ordered:
        by_doc[it["doc"]] = by_doc.get(it["doc"], 0) + 1
    print("  per document:", dict(sorted(by_doc.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
