#!/usr/bin/env python3
"""Score extraction outputs against the gold table with Claude as the judge (evaluator_v2 rubric).

  score_outputs.py rubric                                  # write scoring/RUBRIC.md (evaluator_v2 prompts, verbatim)
  score_outputs.py queue  --system B1=b1_qwen/markdown_baseline --system E0=e0_qwen/reconciliation_agent --docs all
  score_outputs.py ingest scoring/results/<batch>.json    # after the scorer fills a batch
  score_outputs.py report --system ... --docs all [--json out.json]
  score_outputs.py recheck-queue --fraction 0.15 --seed 1 # independent second pass on a sample
  score_outputs.py recheck-ingest scoring/recheck_results/<batch>.json
  score_outputs.py agreement

Systems are name=run/stage (stage: agent_extractor, search_agent, reconciliation_agent, markdown_baseline).
Queued batches never show system names. See src/evaluation/claude_scoring.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import claude_scoring as cs

OUTPUT_FORMAT = """
## How to return scores

You are scoring one queued batch. Every item is one column of one paper: its definition, the gold value (GT) and a
predicted value (Pred). Apply the rules above for the batch's category to each item on its own. You do not know
which system produced a prediction, and you must not look anything up: judge only GT vs Pred under the rules.

## Conventions (fixed after the pilot; they resolve ambiguities in the rules above, for every system alike)

- C1. Tolerance, by kind of number. We are not looking for exact decimals, but counts are different populations when
  they differ:
  counts (N) must match exactly (654 vs 655: no match; 1305 vs 1306: no match);
  percentages match within ±1 percentage point or ±2% relative, whichever is more lenient (4.3 vs 4.2: match;
  23 vs 22.3: match; 19.3 vs 18.7: match; 63.5 vs 67.5: no match);
  other values (medians, durations, rates in the paper's units) match within ±0.1 or at a different rounding
  (35 vs 35.1: match; 41 vs 41.9: no match).
- C2. For completeness, a required GT number counts as present only if Pred has it within tolerance; a wrong number is
  not "present". A count that matches with a percentage outside tolerance is 0.5 / 0.5 when the Definition requires
  both (see C7 for "count and/or percentage"). A GT total counts as present only if Pred states it: per-arm numbers
  whose sum equals it do not make it present.
- C3. "Not reached", "NR (not reached)" and "not estimable" are substantive values, not empty-equivalent: GT empty and
  Pred "Not reached" is 0/0; GT "Not reached" and Pred "Not reached" is 1/1. A bare "NR" in a median-survival or
  time-to-event column means "not reached" (so GT "NR" vs Pred "Not reported" is a miss, 0/0).
- C4. Structured text: a category label implied by the components Pred names counts as present (e.g. GT "Triplet:
  ARPI + ADT + docetaxel" vs Pred "darolutamide plus ADT and docetaxel" is complete).
- C5. Exact match: parenthetical context in GT is optional ("2 arms (A vs B)" vs "2" matches); "Surname et al" formats
  match when the definition allows them.
- C6. In the yes/no columns ("Quality of Life reported", "Reporting by prognostic groups - Y/N | ..."), "No"/"N" and an
  empty value mean the same thing (not reported): GT empty vs Pred "No" is 1/1; GT "No" vs Pred "Not reported" is 1/1;
  "Yes" vs "No" or vs empty is 0/0.
- C7. When the Definition says "count and/or percentage", either the count or the percentage alone is complete (GT
  "143 (36.4%)" vs Pred "143" is 1/1). If Pred gives both and only one is within tolerance, correctness is 0.5 and
  completeness 1.0 (GT "397 (100%)" vs Pred "390 (98.2%)" is 0.5 / 1.0). When it says "count and percentage" or
  "N (%)" without "or", both are required.
  A Definition that says "title or identifier" is satisfied by the identifier alone.

Write a JSON file with exactly this shape, one entry per item id in the batch:

    {"batch": "<batch name>",
     "results": [{"id": "<item id>", "correctness": 0.0 | 0.5 | 1.0, "completeness": 0.0 | 0.5 | 1.0,
                  "reason": "<one short sentence>"}]}
"""


def _systems(specs):
    return [cs.System.parse(spec) for spec in specs or []]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-dir", type=Path, default=cs.SCORING_DIR)
    parser.add_argument("--results-root", type=Path, default=cs.RESULTS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("rubric")
    for name in ("queue", "report"):
        p = sub.add_parser(name)
        p.add_argument("--system", action="append", required=True)
        p.add_argument("--docs", default="all", help="all | dev | heldout | comma-separated doc ids")
        if name == "queue":
            p.add_argument("--batch-size", type=int, default=40)
        else:
            p.add_argument("--json", type=Path)
    for name in ("ingest", "recheck-ingest"):
        p = sub.add_parser(name)
        p.add_argument("results", type=Path, nargs="+")
        p.add_argument("--scorer", default="claude-opus-5")
    p = sub.add_parser("recheck-queue")
    p.add_argument("--fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1)
    sub.add_parser("agreement")
    args = parser.parse_args(argv)

    d = args.scoring_dir
    store, queue_dir = cs.labels_path(d), d / "queue"
    recheck_store, recheck_dir = d / "labels_recheck.jsonl", d / "recheck_queue"
    d.mkdir(parents=True, exist_ok=True)

    if args.command == "rubric":
        parts = ["# Scoring rubric\n\nThe evaluator_v2 prompts, verbatim. Use the section matching the batch category.\n"]
        for category in cs.CATEGORIES:
            parts.append(f"\n## Category: {category}\n\n```text\n{cs.rubric_text(category)}\n```\n")
        parts.append(OUTPUT_FORMAT)
        (d / "RUBRIC.md").write_text("".join(parts))
        print(f"Wrote {d / 'RUBRIC.md'}")
        return 0

    if args.command in ("queue", "report"):
        gold = cs.load_gold()
        docs = cs.select_docs(args.docs, gold)
        labels = cs.load_labels(store)
        if args.command == "queue":
            all_cells = [c for s in _systems(args.system) for c in cs.cells(s, docs, args.results_root)]
            written = cs.build_queue(all_cells, labels, queue_dir, args.batch_size)
            mechanical = sum(c.mechanical for c in all_cells)
            print(f"{len(all_cells)} cells: {mechanical} empty/empty (1/1 by rule), "
                  f"{sum(c.id in labels for c in all_cells)} already scored, {sum(len(json.loads(p.read_text())['items']) for p in written)} "
                  f"new pairs in {len(written)} batches under {queue_dir}")
            return 0
        reports = [cs.report(s, docs, labels, args.results_root) for s in _systems(args.system)]
        for r in reports:
            o = r["overall"]
            flag = "" if r["complete"] else f"  ({r['unscored']} cells not scored yet)"
            print(f"{r['system']:<10} acc {o['accuracy']}  corr {o['correctness']}  comp {o['completeness']}  "
                  f"gold-filled {r['gold_filled']['accuracy']}  dev {r['dev']['accuracy']}  held-out {r['heldout']['accuracy']}{flag}")
            if r["review"]:
                rv = r["review"]
                print(f"{'':<10} agreed cells {rv['agreed']['accuracy']} (n={rv['agreed']['n']})  flag rate {rv['flag_rate']}  "
                      f"flag precision {rv['flag_precision']}  flag recall {rv['flag_recall']}  after review {rv['accuracy_after_simulated_review']}")
        if args.json:
            args.json.write_text(json.dumps(reports, indent=1))
            print(f"Wrote {args.json}")
        return 0

    if args.command in ("ingest", "recheck-ingest"):
        target, qdir = (store, queue_dir) if args.command == "ingest" else (recheck_store, recheck_dir)
        total = 0
        for path in args.results:
            total += cs.ingest(path, qdir, target, args.scorer)
        print(f"Ingested {total} scores into {target}")
        return 0

    if args.command == "recheck-queue":
        written = cs.build_recheck(cs.load_labels(store), args.fraction, args.seed, recheck_dir)
        print(f"Queued {len(written)} re-check batches under {recheck_dir}")
        return 0

    if args.command == "agreement":
        print(json.dumps(cs.agreement(cs.load_labels(store), cs.load_labels(recheck_store)), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
