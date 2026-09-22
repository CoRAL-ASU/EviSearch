#!/usr/bin/env python3
"""
Reconcile Arm A (agent_extractor) vs Arm B (search_agent) for one document.

Usage:
  python experiment-scripts/run_reconciliation_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
  python experiment-scripts/run_reconciliation_agent.py "<doc_id>" --groups "Add-on Treatment,Control Arm"
  python experiment-scripts/run_reconciliation_agent.py "<doc_id>" --dry-run

Reads both arms' results from, and writes to, the same run (runs/<run>/ when --run or EVISEARCH_RUN is set).

Outputs:
  new_pipeline_outputs/results/<doc_id>/reconciliation_agent/reconciled_results.json
  new_pipeline_outputs/results/<doc_id>/reconciliation_agent/verification_logs/batch_N_conversation.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import BATCH_MAX_COLUMNS, PAGE_IMAGE_SCALE, SELECTION
from src.evisearch.pipelines import results_store
from src.evisearch.pipelines.batch_runner import numbered, run_batches
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


SOURCE_METHODS = ("agent", "search")
"""The saved results this stage reads before it can run: Arm A (agent_extractor) and Arm B (search_agent).

It is declared here, next to the loads below, so a runner can ask what this stage depends on instead of assuming a pair.
Arm A and Arm B declare nothing: they read the document (parsed markdown, page images, embeddings), not each other.
"""


def arbiter_module():
    """The Reconciliation Agent: an independent reading of the contested columns, then adjudication."""
    from src.evisearch.services import reconciliation_v5

    return reconciliation_v5


def run_settings(model_key: str) -> Dict[str, Any]:
    """Settings that must match for saved reconciliation results to be resumed."""
    from src.evisearch.services.extraction_rules import rules_setting

    images = SELECTION.option("reconciliation_page_images") == "auto" and SELECTION.catalog.models[model_key].capabilities.images
    return {
        "model": model_key,
        "page_image_scale": PAGE_IMAGE_SCALE if images else None,
        "reconciler": arbiter_module().RECONCILER_VERSION,
        **rules_setting(),
    }


def run_reconciliation_pipeline(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    max_batches: Optional[int] = None,
    model: Optional[str] = None,
    max_per_batch: int = BATCH_MAX_COLUMNS,
) -> Dict[str, Any]:
    """Returns {"columns": {...}, "error": str | None, "usage": {...}}."""
    run_reconciliation_agent = arbiter_module().run_reconciliation_agent
    from src.inference.factory import model_key_for

    started = time.time()
    settings = run_settings(model_key_for("reconciliation", model))
    if resume:
        try:
            results_store.check_resume(doc_id, "reconciliation", settings)
        except results_store.ResumeError as exc:
            return {"columns": {}, "error": str(exc), "usage": empty_usage()}
    for method in SOURCE_METHODS:
        if not results_store.results_path(doc_id, method).exists():
            return {"columns": {}, "error": f"{results_store.METHOD_DIRS[method]} results not found: {results_store.results_path(doc_id, method)}", "usage": empty_usage()}
    source_a = results_store.load_columns(doc_id, "agent")
    source_b = results_store.load_columns(doc_id, "search")

    groups = load_groups()
    definitions = definitions_map(groups)
    existing = results_store.load_columns(doc_id, "reconciliation") if resume else {}
    batches = build_batches(groups, group_names, done=done_columns(existing), max_per_batch=max_per_batch)
    if max_batches is not None:
        batches = batches[: max(max_batches, 0)]
    columns: Dict[str, Any] = dict(existing)
    usage = empty_usage()
    if not batches:
        _locate_evidence(doc_id)
        return {"columns": columns, "error": None, "usage": usage}

    logs = results_store.logs_dir(doc_id, "reconciliation")
    first_log = results_store.next_log_number(logs, start=0)  # a resumed run keeps the logs of earlier batches
    def work(index, batch):
        return run_reconciliation_agent(
            doc_id, batch, definitions, source_a, source_b, log_path=logs / f"batch_{first_log + index}.txt", model=model
        )

    def accumulate(index, batch, payload):
        results, batch_usage = payload
        columns.update({name: {**r, "tried": True} for name, r in results.items()})
        add_usage(usage, batch_usage)
        results_store.save_columns(doc_id, "reconciliation", columns)
        results_store.save_metadata(doc_id, "reconciliation", {
            "method": "reconciliation_agent", **settings, "run": results_store.current_run(), "usage": usage,
            "timing": stage_timing(started, usage, len(existing)),
        })

    run_batches(numbered(batches), work, accumulate)
    _locate_evidence(doc_id)
    return {"columns": columns, "error": None, "usage": usage}


def _locate_evidence(doc_id: str) -> None:
    """Write where each admitted value sits on its pages (services/evidence_locator.py), next to the reconciled
    results, so the viewer can box the value itself. Locating evidence reads only the PDF and the stored outputs,
    and a failure here must never fail the extraction it follows."""
    try:
        from src.evisearch.services import evidence_locator

        evidence_locator.build_stage(doc_id, results_store.method_dir(doc_id, "reconciliation"))
    except Exception as exc:  # noqa: BLE001 - the extraction is done; only the highlight file is missing
        print(f"[reconciliation] {doc_id}: evidence locations not written ({exc})", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile agent_extractor (A) vs search_agent (B) for one document")
    parser.add_argument("doc_id", help="Document id")
    parser.add_argument("--groups", help="Comma-separated definition groups (default: all)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing reconciliation results")
    parser.add_argument("--batch-size", type=int, default=BATCH_MAX_COLUMNS, help="Columns per batch")
    parser.add_argument("--max-batches", type=int, help="Run only the first N batches")
    parser.add_argument("--model", help="Catalog model key overriding the reconciliation role for this run")
    parser.add_argument("--run", help="Run name: arms are read from and results written to results/<doc_id>/runs/<run>/ (default: EVISEARCH_RUN)")
    parser.add_argument("--dry-run", action="store_true", help="Print batches without calling the model")
    args = parser.parse_args(argv)

    from src.inference.factory import model_key_for

    if args.run is not None:
        results_store.use_run(args.run)
    for method in SOURCE_METHODS:
        if not results_store.results_path(args.doc_id, method).exists():
            print(f"[reconciliation] missing {results_store.results_path(args.doc_id, method)}; run that arm first", file=sys.stderr)
            return 1

    group_names = parse_group_names(args.groups)
    groups = load_groups()
    unknown = unknown_groups(groups, group_names)
    if unknown:
        print(f"[reconciliation] unknown group(s) {unknown}; groups are the Label values in the definitions CSV", file=sys.stderr)
        return 2
    if not args.no_resume:
        try:
            results_store.check_resume(args.doc_id, "reconciliation", run_settings(model_key_for("reconciliation", args.model)))
        except results_store.ResumeError as exc:
            print(f"[reconciliation] {exc}", file=sys.stderr)
            return 2
    existing = {} if args.no_resume else results_store.load_columns(args.doc_id, "reconciliation")
    batches = build_batches(groups, group_names, done=done_columns(existing), max_per_batch=args.batch_size)
    if args.max_batches is not None:
        batches = batches[: max(args.max_batches, 0)]
    print(f"[reconciliation] doc_id={args.doc_id} model={model_key_for('reconciliation', args.model)} run={results_store.current_run() or '-'} batches={len(batches)}")
    if args.dry_run:
        for index, batch in enumerate(batches):
            print(f"  batch {index}: {[c['column_name'] for c in batch]}")
        return 0
    if not batches:
        print("[reconciliation] nothing to reconcile (all columns done; use --no-resume to redo)")
        return 0

    result = run_reconciliation_pipeline(
        args.doc_id, group_names, resume=not args.no_resume, max_batches=args.max_batches, model=args.model, max_per_batch=args.batch_size
    )
    if result["error"]:
        print(f"[reconciliation] {result['error']}", file=sys.stderr)
        return 1
    print(f"[reconciliation] {len(result['columns'])} columns reconciled; usage={result['usage']}")
    print(f"[reconciliation] wrote {results_store.results_path(args.doc_id, 'reconciliation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
