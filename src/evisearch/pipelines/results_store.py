"""Where each method's per-document results live: RESULTS_ROOT/<doc_id>/<method dir>/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config.runtime_paths import RESULTS_ROOT

METHOD_DIRS = {
    "agent": "agent_extractor",
    "search": "search_agent",
    "reconciliation": "reconciliation_agent",
}
RESULT_FILES = {
    "agent": "extraction_results.json",
    "search": "extraction_results.json",
    "reconciliation": "reconciled_results.json",
}
LOG_DIRS = {
    "agent": "raw_llm_responses",
    "search": "verification_logs",
    "reconciliation": "verification_logs",
}


def method_dir(doc_id: str, method: str) -> Path:
    return RESULTS_ROOT / doc_id / METHOD_DIRS[method]


def results_path(doc_id: str, method: str) -> Path:
    return method_dir(doc_id, method) / RESULT_FILES[method]


def logs_dir(doc_id: str, method: str) -> Path:
    path = method_dir(doc_id, method) / LOG_DIRS[method]
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_columns(doc_id: str, method: str) -> Dict[str, Any]:
    path = results_path(doc_id, method)
    if not path.exists():
        return {}
    try:
        columns = json.loads(path.read_text(encoding="utf-8")).get("columns", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return columns if isinstance(columns, dict) else {}


def save_columns(doc_id: str, method: str, columns: Dict[str, Any], **extra: Any) -> Path:
    path = results_path(doc_id, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, "columns": columns, **extra}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_metadata(doc_id: str, method: str, payload: Dict[str, Any]) -> Path:
    path = method_dir(doc_id, method) / "extraction_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, **payload}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
