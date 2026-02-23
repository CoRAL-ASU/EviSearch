#!/usr/bin/env python3
"""
plan_and_extract_columns.py

Plan + extract for one or more columns in a single run.
Uses Landing AI chunks. Run only columns marked with 1 in a JSON file.

Usage:
  # From JSON file (1=run, 0=skip):
  python plan_and_extract_columns.py --pdf-name "NCT00268476_Attard_STAMPEDE_Lancet'23" --columns-file columns_to_run.json

  # Or specify columns directly:
  python plan_and_extract_columns.py --pdf-name "..." --column-name "Median_Age_(years)"
  python plan_and_extract_columns.py --pdf-name "..." --column-names "Median_Age_(years)" "Total_Participants_-_N"

Columns file format (JSON object, 1=run, 0=skip):
  {"Median_Age_(years)": 1, "Total_Participants_-_N": 1, "ORR_-_N_(%)": 0}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.planning.plan_generator import Column, ColumnGroup, PlanGenerator
from src.LLMProvider.provider import LLMProvider
from src.table_definitions.definitions import load_definitions

from extract_with_landing_ai import load_column_definitions, extract_column_multi_candidate, find_chunks_for_column_tiered

# Config: if True, plan all columns in one LLM call; if False, one call per group
GROUP_QUERIES = True

# Keyword retrieval defaults (when --use-keywords)
DEFAULT_KEYWORD_TOP_K = 5
DEFAULT_KEYWORD_TOKEN_LIMIT = 150


def _groups_containing_columns(
    definitions: Dict[str, List[Dict[str, str]]],
    column_names: Set[str],
) -> Dict[str, List[Dict[str, str]]]:
    """Return only groups that contain at least one requested column."""
    return {
        g: cols
        for g, cols in definitions.items()
        if {c["Column Name"] for c in cols} & column_names
    }


def plan_and_extract(
    pdf_path: Path,
    column_names: List[str],
    *,
    results_root: Path,
    do_retry: bool = True,
    group_queries: bool = True,
    use_keywords: bool = False,
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
    keyword_token_limit: int = DEFAULT_KEYWORD_TOKEN_LIMIT,
) -> Dict[str, Any]:
    """Plan + extract for the given columns."""
    pdf_path = Path(pdf_path)
    col_set = set(column_names)
    if not col_set:
        return {"success": False, "error": "No columns", "results": {}}

    from landing_ai_chunks import load_landing_ai_chunks

    base_dir = results_root / pdf_path.stem
    cache_dir = base_dir / "chunking"
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_landing_ai_chunks(pdf_path, cache_dir=cache_dir, use_cache=True)

    definitions = load_definitions()
    filtered_defs = _groups_containing_columns(definitions, col_set)
    if not filtered_defs:
        return {"success": False, "error": f"No definitions for: {list(col_set)}", "results": {}}

    definitions_map = load_column_definitions()
    provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
    out_subdir = "plan_extract_columns_with_keywords" if use_keywords else "plan_extract_columns"
    out_dir = base_dir / "planning" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plans: Dict[str, Dict[str, Any]] = {}
    pdf_handle = provider.upload_pdf(pdf_path)

    try:
        if group_queries:
            # Single LLM call: one group with all requested columns
            seen: Set[str] = set()
            all_cols = []
            for g, cols in filtered_defs.items():
                for c in cols:
                    name = c["Column Name"]
                    if name in col_set and name not in seen:
                        seen.add(name)
                        all_cols.append(Column(name=name, definition=c["Definition"]))
            merged_group = ColumnGroup(name="selected_columns", columns=all_cols)
            merged_defs = {"selected_columns": [{"Column Name": c.name, "Definition": c.definition} for c in all_cols]}
            planner = PlanGenerator(provider, merged_defs, name_policy="override")
            plan_data, _ = planner.generate_plan_for_group(
                group=merged_group,
                pdf_handle=pdf_handle,
                chunks=chunks,
                output_dir=out_dir,
            )
            plans["selected_columns"] = plan_data
        else:
            # One LLM call per group
            planner = PlanGenerator(provider, filtered_defs, name_policy="override")
            for group in planner.groups:
                plan_data, _ = planner.generate_plan_for_group(
                    group=group,
                    pdf_handle=pdf_handle,
                    chunks=chunks,
                    output_dir=out_dir,
                )
                plans[plan_data["group_name"]] = plan_data
    finally:
        provider.cleanup_pdf(pdf_handle)

    results: Dict[str, Dict[str, Any]] = {}
    for group_name, plan_data in plans.items():
        for col in plan_data.get("columns", []):
            col_name = col.get("column_name")
            if col_name not in col_set:
                continue
            defn = definitions_map.get(col_name, "")
            # Retrieval: tiered only, or tiered + keyword supplement
            retrieval_meta: Dict[str, Any] = {}
            if use_keywords:
                try:
                    from keyword_retrieval import find_chunks_for_column_with_keywords
                    relevant_chunks, retrieval_source, retrieval_meta = find_chunks_for_column_with_keywords(
                        col, chunks, defn,
                        keyword_top_k=keyword_top_k,
                        keyword_token_limit=keyword_token_limit,
                    )
                except ImportError:
                    relevant_chunks, retrieval_source = find_chunks_for_column_tiered(col, chunks)
                    retrieval_meta = {"keyword_error": "keyword_retrieval not available"}
            else:
                relevant_chunks, retrieval_source = find_chunks_for_column_tiered(col, chunks)
            r = extract_column_multi_candidate(
                col,
                group_name,
                defn,
                chunks,
                logs_dir=logs_dir,
                relevant_chunks_override=relevant_chunks,
                retrieval_source=retrieval_source,
            )
            row: Dict[str, Any] = {
                "value": r.get("value"),
                "primary_value": r.get("primary_value"),
                "candidates": r.get("candidates", []),
                "found": r.get("found"),
                "extraction_plan": col.get("extraction_plan"),
                "page": col.get("page"),
                "source_type": col.get("source_type"),
                "sources": col.get("sources"),
                "retrieval_source": retrieval_source,
            }
            if retrieval_meta:
                row["retrieval_meta"] = retrieval_meta
            results[col_name] = row

    out_file = out_dir / "extraction_results.json"
    out_file.write_text(
        json.dumps({"results": results, "column_names": column_names}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {"success": True, "results": results, "output_file": str(out_file)}


def load_columns_from_file(path: Path) -> List[str]:
    """Load column names where value is 1 (run). JSON: {\"column_name\": 1|0, ...}"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [k for k, v in data.items() if v == 1 or v == "1" or v is True]
    if isinstance(data, list):
        return [
            (item["column_name"] if isinstance(item, dict) else item)
            for item in data
            if (item.get("run") if isinstance(item, dict) else item) in (1, "1", True)
        ]
    return []


def main() -> int:
    p = argparse.ArgumentParser(description="Plan + extract for specific columns (Landing AI chunks)")
    p.add_argument("--pdf-name", required=True, help="PDF stem (e.g. NCT00268476_Attard_STAMPEDE_Lancet'23)")
    p.add_argument("--columns-file", help="JSON file: {\"column_name\": 1|0, ...} - run only 1s")
    p.add_argument("--column-name", help="Single column to extract")
    p.add_argument("--column-names", nargs="+", help="Columns to extract")
    p.add_argument("--results-root", default="new_pipeline_outputs/results")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--no-retry", action="store_true")
    p.add_argument("--group-queries", action="store_true", default=GROUP_QUERIES, help="Plan all columns in one LLM call (default: %(default)s)")
    p.add_argument("--no-group-queries", action="store_false", dest="group_queries", help="One planning call per group")
    p.add_argument("--use-keywords", action="store_true", help="Add keyword (BM25) retrieval to supplement planner chunks")
    p.add_argument("--keyword-top-k", type=int, default=DEFAULT_KEYWORD_TOP_K, help="Max keyword chunks per column (default: %(default)s)")
    p.add_argument("--keyword-token-limit", type=int, default=DEFAULT_KEYWORD_TOKEN_LIMIT, help="Token budget for keyword supplement (default: %(default)s)")
    args = p.parse_args()

    col_names: List[str] = []
    if args.columns_file:
        cf = Path(args.columns_file)
        if not cf.is_absolute():
            cf = PROJECT_ROOT / cf
        if not cf.exists():
            print(f"Columns file not found: {cf}")
            return 1
        col_names = load_columns_from_file(cf)
        if not col_names:
            print("No columns with run=1 in file")
            return 1
    elif args.column_name:
        col_names = [args.column_name]
    elif args.column_names:
        col_names = args.column_names
    else:
        print("Provide --columns-file, --column-name, or --column-names")
        return 1

    results_root = PROJECT_ROOT / args.results_root
    pdf_path = PROJECT_ROOT / args.dataset_dir / f"{args.pdf_name}.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 1

    kw_str = " +keywords" if args.use_keywords else ""
    print(f"Plan + extract for {len(col_names)} column(s) (group_queries={args.group_queries}{kw_str}): {col_names}")
    result = plan_and_extract(
        pdf_path,
        col_names,
        results_root=results_root,
        do_retry=not args.no_retry,
        group_queries=args.group_queries,
        use_keywords=args.use_keywords,
        keyword_top_k=args.keyword_top_k,
        keyword_token_limit=args.keyword_token_limit,
    )

    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        return 1

    for name, r in result["results"].items():
        src = r.get("retrieval_source", "")
        src_str = f" [retrieval: {src}]" if src else ""
        cands = r.get("candidates", [])
        n = len(cands)
        conf = cands[0].get("confidence", "") if cands else ""
        print(f"  {name}: {r.get('value')} ({n} candidate(s), confidence={conf}){src_str}")
    print(f"\nSaved to {result['output_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
