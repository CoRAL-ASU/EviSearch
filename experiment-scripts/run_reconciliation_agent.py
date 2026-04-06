#!/usr/bin/env python3
"""
Run the reconciliation agent to reconcile Agent Extractor (A) vs Search Agent (B).

Usage:
  python experiment-scripts/run_reconciliation_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
  python experiment-scripts/run_reconciliation_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23" --groups "Add-on Treatment,Control Arm"
  python experiment-scripts/run_reconciliation_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23" --dry-run

Outputs:
  new_pipeline_outputs/results/<doc_id>/reconciliation_agent/reconciled_results.json
  new_pipeline_outputs/results/<doc_id>/reconciliation_agent/verification_logs/batch_N_conversation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.runtime_paths import RESULTS_ROOT


def load_definitions() -> Dict[str, List[Dict]]:
    from src.table_definitions.definitions import load_definitions as _load
    return _load()


def build_reconciliation_batches(
    groups: Dict[str, List[Dict]],
    group_names: Optional[List[str]] = None,
    resume_from: Optional[Dict[str, Any]] = None,
    max_per_batch: int = 15,
) -> List[List[Dict[str, Any]]]:
    """Build batches of columns for reconciliation. Skip columns already in resume_from."""
    if group_names:
        groups = {k: v for k, v in groups.items() if k in group_names}
    filled = set()
    if resume_from:
        for col_name in resume_from.keys():
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
    over_limit = [(g, n, c) for g, n, c in remaining_with_counts if n > max_per_batch]
    under_limit = [(g, n, c) for g, n, c in remaining_with_counts if n <= max_per_batch]

    for gname, n, col_specs in over_limit:
        for i in range(0, len(col_specs), max_per_batch):
            batches.append(col_specs[i : i + max_per_batch])

    under_limit.sort(key=lambda x: x[1])
    col_sum = 0
    current: List[Dict[str, Any]] = []
    for gname, n, col_specs in under_limit:
        if col_sum + n <= max_per_batch:
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


def run_reconciliation_pipeline(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    no_resume: bool = False,
    max_per_batch: int = 15,
) -> Dict[str, Any]:
    """
    Run full reconciliation pipeline. Returns {"columns": {...}, "error": str or None}.
    """
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if not agent_path.exists():
        return {"columns": {}, "error": f"Agent results not found: {agent_path}"}
    if not search_path.exists():
        return {"columns": {}, "error": f"Search agent results not found: {search_path}"}

    agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
    search_data = json.loads(search_path.read_text(encoding="utf-8"))
    source_a = agent_data.get("columns", {}) or {}
    source_b = search_data.get("columns", {}) or {}

    groups = load_definitions()
    definitions_map = {}
    for g, cols in groups.items():
        for c in cols:
            definitions_map[c["Column Name"]] = c.get("Definition", "")

    out_dir = RESULTS_ROOT / doc_id / "reconciliation_agent"
    reconciled_path = out_dir / "reconciled_results.json"
    resume_from = None
    if resume and not no_resume and reconciled_path.exists():
        try:
            data = json.loads(reconciled_path.read_text(encoding="utf-8"))
            resume_from = data.get("columns", {})
        except Exception:
            pass

    batches = build_reconciliation_batches(
        groups, group_names=group_names, resume_from=resume_from, max_per_batch=max_per_batch
    )
    if not batches:
        return {"columns": resume_from or {}, "error": None}

    logs_dir = out_dir / "verification_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    db: Dict[str, Any] = dict(resume_from) if resume_from else {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    for batch_idx, batch in enumerate(batches):
        from web.reconciliation_agent import run_reconciliation_agent
        results, batch_usage = run_reconciliation_agent(
            doc_id=doc_id,
            batch_columns=batch,
            definitions_map=definitions_map,
            source_a_data=source_a,
            source_b_data=source_b,
            log_path=logs_dir / f"batch_{batch_idx}.txt",
        )
        total_usage["input_tokens"] += batch_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += batch_usage.get("output_tokens", 0)
        total_usage["api_calls"] += batch_usage.get("api_calls", 0)
        for col_name, r in results.items():
            db[col_name] = {**r, "tried": True}
        out_dir.mkdir(parents=True, exist_ok=True)
        reconciled_path.write_text(json.dumps({"doc_id": doc_id, "columns": db}, indent=2), encoding="utf-8")
        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
        metadata_path = out_dir / "extraction_metadata.json"
        metadata_path.write_text(json.dumps({"doc_id": doc_id, "usage": total_usage}, indent=2), encoding="utf-8")

    return {"columns": db, "error": None}


def main():
    parser = argparse.ArgumentParser(description="Run reconciliation agent (Agent vs Search Agent)")
    parser.add_argument("doc_id", help="Document ID")
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print batches, don't run")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous results")
    parser.add_argument("--batch-size", type=int, default=15, help="Columns per batch (default: 15)")
    args = parser.parse_args()

    doc_id = args.doc_id
    group_names = None
    if args.groups:
        group_names = [g.strip() for g in args.groups.split(",") if g.strip()]

    # Load Agent (A) and Search Agent (B) results
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if not agent_path.exists():
        print(f"[run_reconciliation_agent] Agent results not found: {agent_path}", file=sys.stderr)
        sys.exit(1)
    if not search_path.exists():
        print(f"[run_reconciliation_agent] Search agent results not found: {search_path}", file=sys.stderr)
        sys.exit(1)

    agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
    search_data = json.loads(search_path.read_text(encoding="utf-8"))
    source_a = agent_data.get("columns", {}) or {}
    source_b = search_data.get("columns", {}) or {}

    groups = load_definitions()
    definitions_map = {}
    for g, cols in groups.items():
        for c in cols:
            definitions_map[c["Column Name"]] = c.get("Definition", "")

    # Resume from previous if exists
    out_dir = RESULTS_ROOT / doc_id / "reconciliation_agent"
    reconciled_path = out_dir / "reconciled_results.json"
    resume_from = None
    if not args.no_resume and reconciled_path.exists():
        try:
            data = json.loads(reconciled_path.read_text(encoding="utf-8"))
            resume_from = data.get("columns", {})
            print(f"[run_reconciliation_agent] resuming: {len(resume_from)} columns already reconciled", file=sys.stderr)
        except Exception:
            pass

    batches = build_reconciliation_batches(
        groups, group_names=group_names, resume_from=resume_from, max_per_batch=args.batch_size
    )
    print(f"[run_reconciliation_agent] doc_id={doc_id} group_names={group_names}")
    print(f"[run_reconciliation_agent] {len(batches)} batch(es)")

    if args.dry_run:
        for i, b in enumerate(batches):
            print(f"  batch {i}: {[c['column_name'] for c in b]}")
        return 0

    if not batches:
        print("[run_reconciliation_agent] No columns to reconcile (all done or no groups)")
        return 0

    logs_dir = out_dir / "verification_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    db: Dict[str, Any] = dict(resume_from) if resume_from else {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    for batch_idx, batch in enumerate(batches):
        print(f"[run_reconciliation_agent] batch {batch_idx + 1}/{len(batches)}: {len(batch)} columns", file=sys.stderr)
        log_path = logs_dir / f"batch_{batch_idx}.txt"
        from web.reconciliation_agent import run_reconciliation_agent
        results, batch_usage = run_reconciliation_agent(
            doc_id=doc_id,
            batch_columns=batch,
            definitions_map=definitions_map,
            source_a_data=source_a,
            source_b_data=source_b,
            log_path=log_path,
        )
        total_usage["input_tokens"] += batch_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += batch_usage.get("output_tokens", 0)
        total_usage["api_calls"] += batch_usage.get("api_calls", 0)
        for col_name, r in results.items():
            db[col_name] = {
                **r,
                "tried": True,
            }
        out_dir.mkdir(parents=True, exist_ok=True)
        extraction_data = {"doc_id": doc_id, "columns": db}
        reconciled_path.write_text(json.dumps(extraction_data, indent=2), encoding="utf-8")
        total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
        metadata_path = out_dir / "extraction_metadata.json"
        metadata_path.write_text(json.dumps({"doc_id": doc_id, "usage": total_usage}, indent=2), encoding="utf-8")

    print(f"[run_reconciliation_agent] done. Saved to {reconciled_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
