#!/usr/bin/env python3
"""Run the local full-markdown PDF query agent for one document.

Writes:
  new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_results.json
  new_pipeline_outputs/results/<doc_id>/agent_extractor/extraction_metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))

from src.config.runtime_paths import RESULTS_ROOT
from src.evisearch.pipelines.unified_extraction import build_batches, load_definitions
from src.evisearch.services.markdown_pdf_query import (
    resolve_parsed_markdown_path,
    run_markdown_pdf_query_agent,
)


def _load_existing_columns(doc_id: str) -> Dict[str, Any]:
    path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        columns = data.get("columns", {})
        return columns if isinstance(columns, dict) else {}
    except Exception:
        return {}


def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}


def _add_usage(total: Dict[str, int], batch_usage: Dict[str, Any]) -> None:
    for key in total:
        total[key] += int(batch_usage.get(key, 0) or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local markdown PDF-query agent for one document.")
    parser.add_argument("doc_id", help="Document id, usually the PDF stem.")
    parser.add_argument("--groups-only", nargs="+", help="Limit extraction to specific definition groups.")
    parser.add_argument("--max-per-batch", type=int, default=15, help="Maximum columns per LLM call.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens per LLM call. Keep this below the remaining vLLM context window.",
    )
    parser.add_argument("--provider", default=os.getenv("PDF_QUERY_PROVIDER") or "local")
    parser.add_argument("--model", default=os.getenv("PDF_QUERY_MODEL") or None)
    parser.add_argument("--resume", action="store_true", help="Skip columns already present in agent_extractor output.")
    parser.add_argument("--dry-run", action="store_true", help="Print batches and input paths without calling the model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["PDF_QUERY_MAX_TOKENS"] = str(args.max_tokens)

    markdown_path = resolve_parsed_markdown_path(args.doc_id)
    if not markdown_path.exists():
        print(f"ERROR: parsed markdown not found: {markdown_path}", file=sys.stderr)
        return 1

    existing = _load_existing_columns(args.doc_id) if args.resume else {}
    groups = load_definitions()
    batches = build_batches(
        groups,
        group_names=args.groups_only,
        resume_agent=existing if args.resume else None,
        resume_search=existing if args.resume else None,
        max_per_batch=args.max_per_batch,
    )

    total_columns = sum(len(batch) for batch in batches)
    print(f"doc_id: {args.doc_id}")
    print(f"markdown: {markdown_path}")
    print(f"provider: {args.provider}")
    print(f"model: {args.model or '(provider default)'}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"batches: {len(batches)}")
    print(f"columns_to_run: {total_columns}")

    if args.dry_run:
        for idx, batch in enumerate(batches, start=1):
            names = [col.get("column_name", "") for col in batch]
            print(f"batch {idx}/{len(batches)}: {len(batch)} columns: {names}")
        return 0

    out_dir = RESULTS_ROOT / args.doc_id / "agent_extractor"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_llm_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    columns: Dict[str, Any] = dict(existing) if args.resume else {}
    usage = _empty_usage()
    errors: List[Dict[str, Any]] = []

    for idx, batch in enumerate(batches, start=1):
        names = [col.get("column_name", "") for col in batch]
        print(f"batch {idx}/{len(batches)}: {len(batch)} columns", flush=True)
        try:
            result, batch_usage = run_markdown_pdf_query_agent(
                args.doc_id,
                batch,
                provider_name=args.provider,
                model=args.model,
                raw_response_path=raw_dir / f"batch_{idx:03d}.json",
            )
        except Exception as exc:
            print(f"batch {idx} ERROR: {exc}", flush=True)
            errors.append({"batch": idx, "columns": names, "error": str(exc)})
            result = {
                name: {
                    "value": f"Error: {exc}",
                    "reasoning": "",
                    "found": False,
                    "attribution": [],
                    "tried": True,
                }
                for name in names
            }
            batch_usage = {}

        columns.update(result)
        _add_usage(usage, batch_usage)

    extraction_path = out_dir / "extraction_results.json"
    extraction_path.write_text(
        json.dumps({"doc_id": args.doc_id, "columns": columns}, indent=2),
        encoding="utf-8",
    )

    metadata_path = out_dir / "extraction_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "doc_id": args.doc_id,
                "method": "markdown_pdf_query",
                "provider": args.provider,
                "model": args.model,
                "markdown_path": str(markdown_path),
                "usage": usage,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {extraction_path}")
    print(f"wrote {metadata_path}")
    print(f"columns: {len(columns)} errors: {len(errors)} usage: {usage}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
