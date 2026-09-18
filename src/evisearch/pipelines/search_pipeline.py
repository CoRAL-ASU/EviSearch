#!/usr/bin/env python3
"""
Run Arm B (search_agent) over every column batch for one document.

Usage:
  python experiment-scripts/run_search_agent.py "NCT02799602_Hussain_ARASENS_JCO'23"
  python experiment-scripts/run_search_agent.py "<doc_id>" --groups "Add-on Treatment,Adverse Events - N (%)"
  python experiment-scripts/run_search_agent.py "<doc_id>" --max-batches 1 --model gemini-2.5-flash
  python experiment-scripts/run_search_agent.py "<doc_id>" --dry-run

Outputs (under runs/<run>/ when --run or EVISEARCH_RUN is set):
  new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json
  new_pipeline_outputs/results/<doc_id>/search_agent/verification_logs/batch_N_conversation.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evisearch.columns import EVIDENCE_FORMAT, count_found
from src.evisearch.pipelines import results_store
from src.evisearch.services.extraction_rules import rules_setting
from src.evisearch.pipelines.batching import (
    add_usage,
    build_batches,
    definitions_map,
    done_columns,
    empty_usage,
    load_groups,
    parse_group_names,
    stage_timing,
    unknown_groups,
)


def run_settings(model_key: str) -> Dict[str, Any]:
    """Settings that must match for saved Arm B results to be resumed."""
    return {"model": model_key, "evidence_format": EVIDENCE_FORMAT, **rules_setting()}


def run_search_agent_pipeline(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    max_batches: Optional[int] = None,
    model: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run all batches; emits phase_start / search_columns_written / search_batch_done / phase_done events.
    Raises results_store.ResumeError when resuming results made with another model."""
    from src.evisearch.services.search import run_search_agent
    from src.inference.factory import model_key_for

    emit = on_event or (lambda event: None)
    started = time.time()
    settings = run_settings(model_key_for("search_agent", model))
    if resume:
        results_store.check_resume(doc_id, "search", settings)
    groups = load_groups()
    definitions = definitions_map(groups)
    existing = results_store.load_columns(doc_id, "search") if resume else {}
    batches = build_batches(groups, group_names, done=done_columns(existing))
    if max_batches is not None:
        batches = batches[: max(max_batches, 0)]
    columns: Dict[str, Any] = dict(existing)
    if not batches:
        emit({"type": "phase_done", "phase": "search_agent", "skipped": True, "filled": count_found(columns), "total": len(columns)})
        return {"columns": columns, "filled": count_found(columns), "total": len(columns), "usage": empty_usage()}

    usage = empty_usage()
    logs = results_store.logs_dir(doc_id, "search")
    emit({"type": "phase_start", "phase": "search_agent", "batches": len(batches), "total": sum(len(b) for b in batches)})
    for index, batch in enumerate(batches):
        results, batch_usage = run_search_agent(doc_id, batch, definitions, log_path=logs / f"batch_{index}.txt", model=model)
        columns.update(results)
        add_usage(usage, batch_usage)
        results_store.save_columns(doc_id, "search", columns)
        results_store.save_metadata(doc_id, "search", {
            "method": "search_agent", **settings, "run": results_store.current_run(), "usage": usage,
            "timing": stage_timing(started, usage, len(existing)),
        })
        emit({"type": "search_columns_written", "columns": [{"column": name, "value": r["value"]} for name, r in results.items()]})
        emit({"type": "search_batch_done", "batch": index + 1, "total_batches": len(batches), "filled": count_found(columns), "total": len(columns)})
    emit({"type": "phase_done", "phase": "search_agent", "filled": count_found(columns), "total": len(columns)})
    return {"columns": columns, "filled": count_found(columns), "total": len(columns), "usage": usage}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arm B (search agent) for one document")
    parser.add_argument("doc_id", help="Document id")
    parser.add_argument("--groups", help="Comma-separated definition groups (default: all)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing search_agent results")
    parser.add_argument("--max-batches", type=int, help="Run only the first N batches")
    parser.add_argument("--model", help="Catalog model key overriding the search_agent role for this run")
    parser.add_argument("--run", help="Run name: results go to results/<doc_id>/runs/<run>/ (default: EVISEARCH_RUN)")
    parser.add_argument("--dry-run", action="store_true", help="Print batches without calling the model")
    args = parser.parse_args(argv)

    from src.inference.factory import model_key_for

    if args.run is not None:
        results_store.use_run(args.run)
    group_names = parse_group_names(args.groups)
    groups = load_groups()
    unknown = unknown_groups(groups, group_names)
    if unknown:
        print(f"[search_agent] unknown group(s) {unknown}; groups are the Label values in the definitions CSV", file=sys.stderr)
        return 2
    if not args.no_resume:
        try:
            results_store.check_resume(args.doc_id, "search", run_settings(model_key_for("search_agent", args.model)))
        except results_store.ResumeError as exc:
            print(f"[search_agent] {exc}", file=sys.stderr)
            return 2
    existing = {} if args.no_resume else results_store.load_columns(args.doc_id, "search")
    batches = build_batches(groups, group_names, done=done_columns(existing))
    if args.max_batches is not None:
        batches = batches[: max(args.max_batches, 0)]
    print(f"[search_agent] doc_id={args.doc_id} model={model_key_for('search_agent', args.model)} run={results_store.current_run() or '-'} batches={len(batches)}")
    if args.dry_run:
        for index, batch in enumerate(batches):
            print(f"  batch {index}: {[c['column_name'] for c in batch]}")
        return 0
    if not batches:
        print("[search_agent] nothing to extract (all columns done; use --no-resume to redo)")
        return 0

    def report(event: Dict[str, Any]) -> None:
        if event["type"] == "search_batch_done":
            print(f"[search_agent] batch {event['batch']}/{event['total_batches']} done ({event['filled']}/{event['total']} filled)")

    result = run_search_agent_pipeline(
        args.doc_id, group_names, resume=not args.no_resume, max_batches=args.max_batches, model=args.model, on_event=report
    )
    print(f"[search_agent] {result['filled']}/{result['total']} columns with values; usage={result['usage']}")
    print(f"[search_agent] wrote {results_store.results_path(args.doc_id, 'search')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
