#!/usr/bin/env python3
"""
Unified extraction: runs Agent Extractor and Search Agent in parallel per batch,
sequential across batches. Emits batch_complete with both candidate_a and candidate_b.

Usage (from web):
  run_unified_extraction(doc_id, group_names, resume, on_event)
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))


def load_definitions() -> Dict[str, List[Dict]]:
    from src.table_definitions.definitions import load_definitions as _load
    return _load()


def build_batches(
    groups: Dict[str, List[Dict]],
    group_names: Optional[List[str]] = None,
    resume_agent: Optional[Dict[str, Any]] = None,
    resume_search: Optional[Dict[str, Any]] = None,
    max_per_batch: int = 15,
) -> List[List[Dict[str, Any]]]:
    """Build shared batches. Skip columns already in resume_agent and resume_search."""
    if group_names:
        groups = {k: v for k, v in groups.items() if k in group_names}
    filled = set()
    agent_has = set((resume_agent or {}).keys()) if resume_agent else set()
    search_has = set((resume_search or {}).keys()) if resume_search else set()
    for col_name in (agent_has & search_has):
        a, s = resume_agent.get(col_name), resume_search.get(col_name)
        if a is not None and (not isinstance(a, dict) or a.get("tried", True)) and s is not None and (not isinstance(s, dict) or s.get("tried", True)):
            filled.add(col_name)

    remaining: List[tuple] = []
    for group_name, cols in groups.items():
        col_specs = [
            {"column_name": c["Column Name"], "definition": c.get("Definition", "")}
            for c in cols
            if c["Column Name"] not in filled
        ]
        if col_specs:
            remaining.append((group_name, len(col_specs), col_specs))

    batches: List[List[Dict[str, Any]]] = []
    over_limit = [(g, n, c) for g, n, c in remaining if n > max_per_batch]
    under_limit = [(g, n, c) for g, n, c in remaining if n <= max_per_batch]

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


def run_unified_extraction(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    no_resume: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Agent + Search in parallel per batch. Emit batch_complete with both A and B.
    Skips batches where both agent and search already have results.
    """
    def emit(ev: Dict[str, Any]) -> None:
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    from agent_extractor import (
        resolve_pdf_path,
        extract_batch,
        load_previous_extraction,
        LLMProvider,
        _attribution_to_page_modality,
    )
    from web.search_agent import run_search_agent

    # Resolve PDF
    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        emit({"type": "error", "error": f"PDF not found for {doc_id}"})
        return {"error": f"PDF not found for {doc_id}"}

    groups = load_definitions()
    definitions_map = {c["Column Name"]: c.get("Definition", "") for g, cols in groups.items() for c in cols}

    resume_agent = load_previous_extraction(doc_id) if resume and not no_resume else None
    resume_search = None
    search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if resume and not no_resume and search_path.exists():
        try:
            data = json.loads(search_path.read_text(encoding="utf-8"))
            resume_search = data.get("columns", {})
        except Exception:
            pass

    batches = build_batches(
        groups,
        group_names=group_names,
        resume_agent=resume_agent,
        resume_search=resume_search,
    )
    all_column_names = []
    for b in batches:
        all_column_names.extend(c.get("column_name", "") for c in b)
    total = len(all_column_names)

    if total == 0:
        emit({"type": "extraction_start", "total": 0, "column_names": [], "batches": []})
        emit({"type": "done", "filled": 0, "total": 0})
        return {"agent": resume_agent or {}, "search": resume_search or {}}

    batch_column_names = [[c.get("column_name", "") for c in b] for b in batches]
    emit({
        "type": "extraction_start",
        "total": total,
        "column_names": all_column_names,
        "batches": batch_column_names,
    })
    first_batch_size = len(batch_column_names[0]) if batch_column_names else 0
    emit({"type": "stream_message", "text": f"Loaded {first_batch_size} queries — ", "show_columns": 0})
    emit({"type": "stream_message", "text": "Running 2 methods in parallel per batch."})

    provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
    pdf_handle = provider.upload_pdf(pdf_path)

    agent_dir = RESULTS_ROOT / doc_id / "agent_extractor"
    search_dir = RESULTS_ROOT / doc_id / "search_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    agent_db: Dict[str, Any] = dict(resume_agent) if resume_agent else {}
    search_db: Dict[str, Any] = dict(resume_search) if resume_search else {}

    agent_path = agent_dir / "extraction_results.json"
    search_path_out = search_dir / "extraction_results.json"
    logs_dir = search_dir / "verification_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(batches):
        agent_result: Dict[str, Dict[str, Any]] = {}
        search_result: Dict[str, Dict[str, Any]] = {}

        def run_agent_batch():
            nonlocal agent_result
            try:
                agent_result = extract_batch(doc_id, batch, pdf_handle, provider)
            except Exception as e:
                agent_result = {c.get("column_name", ""): {"value": f"Error: {e}", "reasoning": "", "found": False, "attribution": [], "tried": True} for c in batch}

        def run_search_batch():
            nonlocal search_result
            try:
                log_path = logs_dir / f"batch_{batch_idx}.txt"
                search_result, _ = run_search_agent(doc_id, batch, definitions_map, log_path=log_path)
            except Exception as e:
                search_result = {
                    c.get("column_name", ""): {"value": f"Error: {e}", "found": False, "attribution": []}
                    for c in batch
                }

        t_a = threading.Thread(target=run_agent_batch)
        t_s = threading.Thread(target=run_search_batch)
        t_a.start()
        t_s.start()
        t_a.join()
        t_s.join()

        for col_spec in batch:
            col_name = col_spec.get("column_name", "")
            if not col_name:
                continue
            a = agent_result.get(col_name)
            r = search_result.get(col_name, {})
            val_a = a.get("value", "Not reported") if isinstance(a, dict) else str(a or "Not reported")
            val_b = r.get("value", "Not reported") if isinstance(r, dict) else str(r)

            agent_db[col_name] = {
                "value": val_a,
                "reasoning": a.get("reasoning", "") if isinstance(a, dict) else "",
                "found": a.get("found", True) if isinstance(a, dict) else bool(val_a),
                "tried": True,
                "attribution": a.get("attribution", []) if isinstance(a, dict) else [],
            }
            search_db[col_name] = {
                "value": val_b,
                "reasoning": r.get("reasoning", "") if isinstance(r, dict) else "",
                "found": r.get("found", False) if isinstance(r, dict) else False,
                "attribution": r.get("attribution", []) if isinstance(r, dict) else [],
                "tried": True,
            }

        columns_for_event = []
        for col_spec in batch:
            col_name = col_spec.get("column_name", "")
            if not col_name:
                continue
            a = agent_db.get(col_name, {})
            s = search_db.get(col_name, {})
            columns_for_event.append({
                "column": col_name,
                "candidate_a": a.get("value", "Not reported") if isinstance(a, dict) else "Not reported",
                "candidate_b": s.get("value", "Not reported") if isinstance(s, dict) else "Not reported",
            })

        agent_out = {}
        for k, v in agent_db.items():
            if isinstance(v, dict):
                agent_out[k] = {
                    "value": v.get("value"),
                    "reasoning": v.get("reasoning", ""),
                    "found": v.get("found", True),
                    "tried": v.get("tried", True),
                    "attribution": _attribution_to_page_modality(v.get("attribution", [])),
                }
            else:
                agent_out[k] = {"value": v, "reasoning": "", "found": v is not None, "tried": False, "attribution": []}
        search_out = {k: (v if isinstance(v, dict) else {"value": v, "reasoning": "", "found": False, "attribution": [], "tried": True}) for k, v in search_db.items()}
        agent_path.write_text(json.dumps({"doc_id": doc_id, "columns": agent_out, "turns": batch_idx + 1}, indent=2), encoding="utf-8")
        search_path_out.write_text(json.dumps({"doc_id": doc_id, "columns": search_out}, indent=2), encoding="utf-8")

        emit({
            "type": "batch_complete",
            "batch": batch_idx + 1,
            "total_batches": len(batches),
            "columns": columns_for_event,
        })

    filled = sum(1 for v in agent_db.values() if isinstance(v, dict) and v.get("value") and str(v.get("value", "")).lower() not in ("not reported", ""))
    emit({"type": "done", "turns": len(batches), "filled": filled, "total": total})
    return {"agent": agent_db, "search": search_db}
