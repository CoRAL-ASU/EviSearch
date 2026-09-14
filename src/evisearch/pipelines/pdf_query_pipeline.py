#!/usr/bin/env python3
"""
Run Arm A (pdf_query) over every column batch for one document.

Usage:
  python experiment-scripts/run_pdf_query_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
  python experiment-scripts/run_pdf_query_agent.py "<doc_id>" --groups "Trial,Control Arm" --max-batches 1
  python experiment-scripts/run_pdf_query_agent.py "<doc_id>" --input pdf --model gemini-2.5-flash
  python experiment-scripts/run_pdf_query_agent.py "<doc_id>" --dry-run

Outputs:
  new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_results.json
  new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_metadata.json
  new_pipeline_outputs/results/<doc_id>/agent_extractor/raw_llm_responses/batch_NNN.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import SELECTION
from src.evisearch.columns import count_found
from src.evisearch.pipelines import results_store
from src.evisearch.pipelines.batching import add_usage, build_batches, done_columns, empty_usage, load_groups, parse_group_names


def run_pdf_query_pipeline(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    max_batches: Optional[int] = None,
    model: Optional[str] = None,
    input_mode: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    from src.evisearch.services.pdf_query import run_pdf_query
    from src.inference.factory import model_key_for

    emit = on_event or (lambda event: None)
    existing = results_store.load_columns(doc_id, "agent") if resume else {}
    batches = build_batches(load_groups(), group_names, done=done_columns(existing))
    if max_batches is not None:
        batches = batches[: max(max_batches, 0)]
    columns: Dict[str, Any] = dict(existing)
    usage = empty_usage()
    input_mode = input_mode or SELECTION.option("pdf_query_input")
    metadata = {"method": "pdf_query", "model": model_key_for("pdf_query", model), "input_mode": input_mode}

    emit({"type": "phase_start", "phase": "agent_extractor", "batches": len(batches), "total": sum(len(b) for b in batches)})
    raw_dir = results_store.logs_dir(doc_id, "agent") if batches else None
    for index, batch in enumerate(batches, 1):
        results, batch_usage = run_pdf_query(
            doc_id, batch, input_mode=input_mode, model=model, raw_response_path=raw_dir / f"batch_{index:03d}.json"
        )
        columns.update(results)
        add_usage(usage, batch_usage)
        results_store.save_columns(doc_id, "agent", columns)
        results_store.save_metadata(doc_id, "agent", {**metadata, "usage": usage})
        emit({
            "type": "columns_written",
            "batch": index,
            "total_batches": len(batches),
            "columns": [{"column": name, "value": r["value"]} for name, r in results.items()],
        })
    emit({"type": "phase_done", "phase": "agent_extractor", "filled": count_found(columns), "total": len(columns)})
    return {"columns": columns, "filled": count_found(columns), "total": len(columns), "usage": usage}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Arm A (pdf_query) for one document")
    parser.add_argument("doc_id", help="Document id (usually the PDF stem)")
    parser.add_argument("--groups", help="Comma-separated definition groups (default: all)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing agent_extractor results")
    parser.add_argument("--max-batches", type=int, help="Run only the first N batches")
    parser.add_argument("--model", help="Catalog model key overriding the pdf_query role for this run")
    parser.add_argument("--input", choices=SELECTION.catalog.options["pdf_query_input"], help="Document input (default from config.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print batches without calling the model")
    args = parser.parse_args(argv)

    from src.inference.factory import model_key_for

    model_key = model_key_for("pdf_query", args.model)
    input_mode = args.input or SELECTION.option("pdf_query_input")
    if input_mode == "pdf" and not SELECTION.catalog.models[model_key].capabilities.pdf:
        readers = [k for k in SELECTION.catalog.models_for_role("pdf_query") if SELECTION.catalog.models[k].capabilities.pdf]
        print(f"[pdf_query] model '{model_key}' cannot read PDFs; use --input markdown or --model one of: {', '.join(readers)}", file=sys.stderr)
        return 2

    group_names = parse_group_names(args.groups)
    existing = {} if args.no_resume else results_store.load_columns(args.doc_id, "agent")
    batches = build_batches(load_groups(), group_names, done=done_columns(existing))
    if args.max_batches is not None:
        batches = batches[: max(args.max_batches, 0)]
    print(f"[pdf_query] doc_id={args.doc_id} model={model_key_for('pdf_query', args.model)} "
          f"input={args.input or SELECTION.option('pdf_query_input')} batches={len(batches)}")
    if args.dry_run:
        for index, batch in enumerate(batches, 1):
            print(f"  batch {index}: {[c['column_name'] for c in batch]}")
        return 0
    if not batches:
        print("[pdf_query] nothing to extract (all columns done; use --no-resume to redo)")
        return 0

    def report(event: Dict[str, Any]) -> None:
        if event["type"] == "columns_written":
            print(f"[pdf_query] batch {event['batch']}/{event['total_batches']}: {len(event['columns'])} columns")

    result = run_pdf_query_pipeline(
        args.doc_id, group_names, resume=not args.no_resume, max_batches=args.max_batches,
        model=args.model, input_mode=args.input, on_event=report,
    )
    print(f"[pdf_query] {result['filled']}/{result['total']} columns with values; usage={result['usage']}")
    print(f"[pdf_query] wrote {results_store.results_path(args.doc_id, 'agent')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
