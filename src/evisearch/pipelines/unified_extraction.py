"""
Unified extraction for the web app: Arm A (pdf_query) and Arm B (search_agent) run in parallel per batch,
batches run sequentially. Emits extraction_start / stream_message / batch_complete / done events.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from src.config.config import SELECTION
from src.evisearch.columns import count_found
from src.evisearch.pipelines import results_store
from src.evisearch.pipelines.batching import build_batches, definitions_map, done_columns, load_groups
from src.evisearch.services.highlight import resolve_pdf_path
from src.retrieval.embedding_retriever import parsed_markdown_path


def run_unified_extraction(
    doc_id: str,
    group_names: Optional[List[str]] = None,
    resume: bool = True,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    from src.evisearch.services.pdf_query import run_pdf_query
    from src.evisearch.services.search import run_search_agent

    def emit(event: Dict[str, Any]) -> None:
        if on_event:
            try:
                on_event(event)
            except Exception:
                pass

    if not parsed_markdown_path(doc_id).exists():
        error = f"Parsed markdown not found for {doc_id}; prepare the document (LandingAI parse) first."
        emit({"type": "error", "error": error})
        return {"error": error}
    if SELECTION.option("pdf_query_input") == "pdf" and not resolve_pdf_path(doc_id):
        error = f"PDF not found for {doc_id}"
        emit({"type": "error", "error": error})
        return {"error": error}

    groups = load_groups()
    definitions = definitions_map(groups)
    agent_db = results_store.load_columns(doc_id, "agent") if resume else {}
    search_db = results_store.load_columns(doc_id, "search") if resume else {}
    batches = build_batches(groups, group_names, done=done_columns(agent_db) & done_columns(search_db))
    names = [c["column_name"] for batch in batches for c in batch]

    emit({"type": "extraction_start", "total": len(names), "column_names": names, "batches": [[c["column_name"] for c in b] for b in batches]})
    if not batches:
        emit({"type": "done", "filled": count_found(agent_db), "total": 0})
        return {"agent": agent_db, "search": search_db}
    emit({"type": "stream_message", "text": f"Loaded {len(batches[0])} queries — ", "show_columns": 0})
    emit({"type": "stream_message", "text": "Running 2 methods in parallel per batch."})

    search_logs = results_store.logs_dir(doc_id, "search")
    agent_logs = results_store.logs_dir(doc_id, "agent")
    for index, batch in enumerate(batches):
        outputs: Dict[str, Dict[str, Any]] = {}

        def run_agent() -> None:
            outputs["agent"], _ = run_pdf_query(doc_id, batch, raw_response_path=agent_logs / f"unified_batch_{index + 1:03d}.json")

        def run_search() -> None:
            outputs["search"], _ = run_search_agent(doc_id, batch, definitions, log_path=search_logs / f"batch_{index}.txt")

        threads = [threading.Thread(target=run_agent), threading.Thread(target=run_search)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        agent_db.update(outputs.get("agent", {}))
        search_db.update(outputs.get("search", {}))
        results_store.save_columns(doc_id, "agent", agent_db, turns=index + 1)
        results_store.save_columns(doc_id, "search", search_db)
        emit({
            "type": "batch_complete",
            "batch": index + 1,
            "total_batches": len(batches),
            "columns": [
                {
                    "column": c["column_name"],
                    "candidate_a": agent_db.get(c["column_name"], {}).get("value", "Not reported"),
                    "candidate_b": search_db.get(c["column_name"], {}).get("value", "Not reported"),
                }
                for c in batch
            ],
        })

    emit({"type": "done", "turns": len(batches), "filled": count_found(agent_db), "total": len(names)})
    return {"agent": agent_db, "search": search_db}
