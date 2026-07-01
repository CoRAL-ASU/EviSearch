#!/usr/bin/env python3
"""Run the search agent with a local/OpenAI-compatible LLM backend.

Writes:
  new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json
  new_pipeline_outputs/results/<doc_id>/search_agent/extraction_metadata.json
  new_pipeline_outputs/results/<doc_id>/search_agent/verification_logs/batch_N_conversation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

from src.config.runtime_paths import RESULTS_ROOT
from src.evisearch.pipelines.search_pipeline import build_extraction_batches, load_definitions
from src.evisearch.services.search import run_search_agent


def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}


def _add_usage(total: Dict[str, int], batch_usage: Dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "api_calls"):
        total[key] += int(batch_usage.get(key, 0) or 0)
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]


def _load_existing_columns(doc_id: str) -> Dict[str, Any]:
    path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        columns = data.get("columns", {})
        return columns if isinstance(columns, dict) else {}
    except Exception:
        return {}


def _definitions_map(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _group, cols in groups.items():
        for col in cols:
            out[col["Column Name"]] = col.get("Definition", "")
    return out


def _normalize_attr(attr: Any, found: bool) -> list:
    if not found or not isinstance(attr, list):
        return []
    out = []
    for item in attr:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page < 1:
            continue
        mod = str(item.get("modality") or item.get("source_type") or "text").lower()
        if mod not in ("text", "table", "figure"):
            mod = "text"
        out.append({"page": page, "modality": mod})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local search agent for one document.")
    parser.add_argument("doc_id", help="Document id, usually the PDF stem.")
    parser.add_argument("--groups-only", nargs="+", help="Limit extraction to specific definition groups.")
    parser.add_argument("--max-batches", type=int, default=None, help="Run only the first N batches for smoke testing.")
    parser.add_argument("--provider", default=os.getenv("SEARCH_AGENT_PROVIDER") or "local")
    parser.add_argument("--model", default=os.getenv("SEARCH_AGENT_MODEL") or os.getenv("LOCAL_OPENAI_MODEL") or None)
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("SEARCH_AGENT_MAX_TOKENS", "4096") or "4096"))
    parser.add_argument("--resume", action="store_true", help="Resume from existing search_agent output instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true", help="Print batches without calling the model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["SEARCH_AGENT_PROVIDER"] = args.provider
    os.environ["SEARCH_AGENT_MAX_TOKENS"] = str(args.max_tokens)
    if args.model:
        os.environ["SEARCH_AGENT_MODEL"] = args.model

    groups = load_definitions()
    definitions_map = _definitions_map(groups)
    existing = _load_existing_columns(args.doc_id) if args.resume else {}
    batches = build_extraction_batches(groups, group_names=args.groups_only, resume_from=existing if args.resume else None)
    if args.max_batches is not None:
        batches = batches[: max(args.max_batches, 0)]

    total_columns = sum(len(batch) for batch in batches)
    out_dir = RESULTS_ROOT / args.doc_id / "search_agent"
    extraction_path = out_dir / "extraction_results.json"
    metadata_path = out_dir / "extraction_metadata.json"
    logs_dir = out_dir / "verification_logs"

    print(f"doc_id: {args.doc_id}")
    print(f"provider: {args.provider}")
    print(f"model: {args.model or '(provider default)'}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"resume: {args.resume}")
    print(f"output: {extraction_path}")
    print(f"batches: {len(batches)}")
    print(f"columns_to_run: {total_columns}")

    if args.dry_run:
        for idx, batch in enumerate(batches, start=1):
            names = [col.get("column_name", "") for col in batch]
            print(f"batch {idx}/{len(batches)}: {len(batch)} columns: {names}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    db: Dict[str, Any] = {}
    if existing:
        for col_name, value in existing.items():
            if isinstance(value, dict):
                found = bool(value.get("found", True))
                db[col_name] = {**value, "attribution": _normalize_attr(value.get("attribution", []), found)}
            else:
                db[col_name] = value

    usage = _empty_usage()
    errors: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"batch {batch_idx}/{len(batches)}: {len(batch)} columns", flush=True)
        log_path = logs_dir / f"batch_{batch_idx - 1}.txt"
        try:
            results, batch_usage = run_search_agent(
                args.doc_id,
                batch,
                definitions_map,
                log_path=log_path,
                provider_name=args.provider,
                model=args.model,
            )
        except Exception as exc:
            names = [col.get("column_name", "") for col in batch]
            print(f"batch {batch_idx} ERROR: {exc}", flush=True)
            errors.append({"batch": batch_idx, "columns": names, "error": str(exc)})
            results = {
                name: {"value": f"Error: {exc}", "reasoning": "", "found": False, "attribution": []}
                for name in names
            }
            batch_usage = {}

        _add_usage(usage, batch_usage)
        for col_spec in batch:
            col_name = col_spec.get("column_name", "")
            result = results.get(col_name, {"value": "Not reported", "reasoning": "", "found": False, "attribution": []})
            db[col_name] = {
                "value": result.get("value", "Not reported"),
                "reasoning": result.get("reasoning", ""),
                "found": bool(result.get("found", False)),
                "attribution": _normalize_attr(result.get("attribution", []), bool(result.get("found", False))),
                "tried": True,
            }

        extraction_path.write_text(json.dumps({"doc_id": args.doc_id, "columns": db}, indent=2), encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "doc_id": args.doc_id,
                    "method": "search_agent",
                    "provider": args.provider,
                    "model": args.model,
                    "usage": usage,
                    "errors": errors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    filled = sum(1 for value in db.values() if isinstance(value, dict) and value.get("found"))
    print(f"wrote {extraction_path}")
    print(f"wrote {metadata_path}")
    print(f"columns: {len(db)} found: {filled} errors: {len(errors)} usage: {usage}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
