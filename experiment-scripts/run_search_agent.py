#!/usr/bin/env python3
"""
Run the search agent (OpenAI retriever + extraction) over columns in groups of 15.

Uses same batching as agent_extractor: group by definition group, ≤15 per batch.
Search agent uses semantic retriever (not keyword) to find chunks, then extracts values.

Usage:
  python experiment-scripts/run_search_agent.py "NCT02799602_Hussain_ARASENS_JCO'23"
  python experiment-scripts/run_search_agent.py "NCT02799602_Hussain_ARASENS_JCO'23" --groups "Add-on Treatment,Adverse Events - N (%)"
  python experiment-scripts/run_search_agent.py "NCT02799602_Hussain_ARASENS_JCO'23" --dry-run

Outputs:
  new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json
  new_pipeline_outputs/results/<doc_id>/search_agent/verification_logs/batch_N_conversation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_paths import RESULTS_ROOT

MAX_COLUMNS_BATCH = 15


def load_definitions() -> Dict[str, List[Dict]]:
    from src.table_definitions.definitions import load_definitions as _load
    return _load()


def build_extraction_batches(
    groups: Dict[str, List[Dict]],
    group_names: Optional[List[str]] = None,
    resume_from: Optional[Dict[str, Any]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Build batches of columns for extraction. Same logic as agent_extractor get_status.
    - group_names: if provided, only these groups
    - resume_from: {column_name: {...}} already extracted; skip those
    """
    if group_names:
        groups = {k: v for k, v in groups.items() if k in group_names}
    filled = set()
    if resume_from:
        for col_name, val in resume_from.items():
            if val is not None and (not isinstance(val, dict) or val.get("tried", True)):
                filled.add(col_name)

    remaining_with_counts: List[tuple] = []
    for group_name, cols in groups.items():
        col_specs = [
            {"column_name": c["Column Name"], "definition": c.get("Definition", "")}
            for c in cols
            if c["Column Name"] not in filled
        ]
        if col_specs:
            remaining_with_counts.append((group_name, len(col_specs), col_specs))

    batches: List[List[Dict[str, Any]]] = []
    over_limit = [(g, n, c) for g, n, c in remaining_with_counts if n > MAX_COLUMNS_BATCH]
    under_limit = [(g, n, c) for g, n, c in remaining_with_counts if n <= MAX_COLUMNS_BATCH]

    for gname, n, col_specs in over_limit:
        for i in range(0, len(col_specs), MAX_COLUMNS_BATCH):
            batches.append(col_specs[i : i + MAX_COLUMNS_BATCH])

    under_limit.sort(key=lambda x: x[1])
    col_sum = 0
    current: List[Dict[str, Any]] = []
    for gname, n, col_specs in under_limit:
        if col_sum + n <= MAX_COLUMNS_BATCH:
            current.extend(col_specs)
            col_sum += n
        else:
            if current:
                batches.append(current)
            current = list(col_specs)
            col_sum = n
    if current:
        batches.append(current)
    return batches


def run_search_agent_pipeline(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    no_resume: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run full search agent pipeline (all batches). Emits events via on_event for streaming.
    Returns {"columns": {...}, "filled": N, "total": M}.
    """
    def emit(ev: Dict[str, Any]) -> None:
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    groups = load_definitions()
    definitions_map = {}
    for g, cols in groups.items():
        for c in cols:
            definitions_map[c["Column Name"]] = c.get("Definition", "")

    out_dir = RESULTS_ROOT / doc_id / "search_agent"
    extraction_path = out_dir / "extraction_results.json"
    resume_from = None
    if resume and not no_resume and extraction_path.exists():
        try:
            data = json.loads(extraction_path.read_text(encoding="utf-8"))
            resume_from = data.get("columns", {})
        except Exception:
            pass

    batches = build_extraction_batches(groups, group_names=group_names, resume_from=resume_from)
    total_cols = sum(len(b) for b in batches)
    if total_cols == 0:
        if resume_from:
            columns_data = []
            for col_name, r in (resume_from.items() if isinstance(resume_from, dict) else []):
                val = r.get("value", "Not reported") if isinstance(r, dict) else str(r)
                columns_data.append({"column": col_name, "value": val})
            if columns_data:
                emit({"type": "search_columns_written", "columns": columns_data})
        emit({"type": "phase_done", "phase": "search_agent", "skipped": True, "filled": len(resume_from or {}), "total": 0})
        return {"columns": resume_from or {}, "filled": 0, "total": 0}

    emit({"type": "phase_start", "phase": "search_agent", "batches": len(batches), "total": total_cols})

    logs_dir = out_dir / "verification_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_attr(attr: list, found: bool) -> list:
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

    db: Dict[str, Any] = {}
    if resume_from:
        for k, v in resume_from.items():
            if isinstance(v, dict):
                db[k] = {**v, "attribution": _normalize_attr(v.get("attribution", []), bool(v.get("found", True)))}
            else:
                db[k] = v

    filled_so_far = sum(1 for v in db.values() if isinstance(v, dict) and v.get("found"))
    total_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    for batch_idx, batch in enumerate(batches):
        log_path = logs_dir / f"batch_{batch_idx}.txt"
        from web.search_agent import run_search_agent
        results, batch_usage = run_search_agent(doc_id, batch, definitions_map, log_path=log_path)
        total_usage["input_tokens"] += batch_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += batch_usage.get("output_tokens", 0)
        total_usage["api_calls"] += batch_usage.get("api_calls", 0)
        columns_data = []
        for col_spec in batch:
            col_name = col_spec.get("column_name", "")
            r = results.get(col_name, {"value": "Not reported"})
            val = r.get("value", "Not reported")
            db[col_name] = {
                "value": val,
                "reasoning": r.get("reasoning", ""),
                "found": r.get("found", False),
                "attribution": r.get("attribution", []),
                "tried": True,
            }
            columns_data.append({"column": col_name, "value": val})
        filled_so_far = sum(1 for v in db.values() if isinstance(v, dict) and v.get("found"))
        emit({"type": "search_columns_written", "columns": columns_data})
        emit({
            "type": "search_batch_done",
            "batch": batch_idx + 1,
            "total_batches": len(batches),
            "filled": filled_so_far,
            "total": len(db),
        })
        out_dir.mkdir(parents=True, exist_ok=True)
        extraction_path.write_text(json.dumps({"doc_id": doc_id, "columns": db}, indent=2), encoding="utf-8")
        metadata_path = out_dir / "extraction_metadata.json"
        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
        metadata_path.write_text(json.dumps({"doc_id": doc_id, "usage": total_usage}, indent=2), encoding="utf-8")

    emit({"type": "phase_done", "phase": "search_agent", "filled": filled_so_far, "total": len(db)})
    return {"columns": db, "filled": filled_so_far, "total": len(db)}


def main():
    parser = argparse.ArgumentParser(description="Run search agent (OpenAI retriever) for extraction")
    parser.add_argument("doc_id", help="Document ID")
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print batches, don't run")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous extraction")
    args = parser.parse_args()

    doc_id = args.doc_id
    group_names = None
    if args.groups:
        group_names = [g.strip() for g in args.groups.split(",") if g.strip()]

    groups = load_definitions()
    definitions_map = {}
    for g, cols in groups.items():
        for c in cols:
            definitions_map[c["Column Name"]] = c.get("Definition", "")

    # Resume from previous if exists
    out_dir = RESULTS_ROOT / doc_id / "search_agent"
    extraction_path = out_dir / "extraction_results.json"
    resume_from = None
    if not args.no_resume and extraction_path.exists():
        try:
            data = json.loads(extraction_path.read_text(encoding="utf-8"))
            resume_from = data.get("columns", {})
            print(f"[run_search_agent] resuming: {len([k for k, v in resume_from.items() if v])} columns already filled")
        except Exception:
            pass

    batches = build_extraction_batches(groups, group_names=group_names, resume_from=resume_from)
    print(f"[run_search_agent] doc_id={doc_id} group_names={group_names}")
    print(f"[run_search_agent] {len(batches)} batch(es)")

    if args.dry_run:
        for i, b in enumerate(batches):
            print(f"  batch {i}: {[c['column_name'] for c in b]}")
        return 0

    if not batches:
        print("[run_search_agent] No columns to extract (all done or no groups)")
        return 0

    logs_dir = out_dir / "verification_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_attr(attr: list, found: bool) -> list:
        """Ensure attribution is [{page, modality}]."""
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

    db: Dict[str, Any] = {}
    if resume_from:
        for k, v in resume_from.items():
            if isinstance(v, dict):
                found = bool(v.get("found", True))
                db[k] = {
                    **v,
                    "attribution": _normalize_attr(v.get("attribution", []), found),
                }
            else:
                db[k] = v
    total_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    for batch_idx, batch in enumerate(batches):
        print(f"[run_search_agent] batch {batch_idx + 1}/{len(batches)}: {len(batch)} columns")
        log_path = logs_dir / f"batch_{batch_idx}.txt"
        from web.search_agent import run_search_agent
        results, batch_usage = run_search_agent(doc_id, batch, definitions_map, log_path=log_path)
        total_usage["input_tokens"] += batch_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += batch_usage.get("output_tokens", 0)
        total_usage["api_calls"] += batch_usage.get("api_calls", 0)
        for col_name, r in results.items():
            db[col_name] = {
                "value": r.get("value", "Not reported"),
                "reasoning": r.get("reasoning", ""),
                "found": r.get("found", False),
                "attribution": r.get("attribution", []),
                "tried": True,
            }
        # Save after each batch
        out_dir.mkdir(parents=True, exist_ok=True)
        extraction_data = {"doc_id": doc_id, "columns": db}
        extraction_path.write_text(json.dumps(extraction_data, indent=2), encoding="utf-8")
        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
        metadata_path = out_dir / "extraction_metadata.json"
        metadata_path.write_text(json.dumps({"doc_id": doc_id, "usage": total_usage}, indent=2), encoding="utf-8")

    print(f"[run_search_agent] done. Saved to {extraction_path}")
    filled = sum(1 for v in db.values() if isinstance(v, dict) and v.get("found"))
    print(f"[run_search_agent] {filled}/{len(db)} columns with values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
