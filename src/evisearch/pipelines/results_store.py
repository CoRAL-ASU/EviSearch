"""
Where each method's per-document results live: RESULTS_ROOT/<doc_id>/<method dir>/, or
RESULTS_ROOT/<doc_id>/runs/<run>/<method dir>/ for a named run (EVISEARCH_RUN, or --run on the CLIs), so runs with
different models or inputs never overwrite or resume each other.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

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
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ResumeError(RuntimeError):
    """Saved results were made with different settings than the run that would resume them."""


def use_run(name: Optional[str]) -> None:
    """Select the run whose results are read and written; empty selects the shared layout."""
    global _run
    name = (name or "").strip()
    if name and not RUN_NAME_RE.match(name):
        raise ValueError(f"run name {name!r}: use letters, digits, '.', '_' and '-'")
    _run = name


def current_run() -> str:
    return _run


_run = ""
use_run(os.getenv("EVISEARCH_RUN"))


def method_dir(doc_id: str, method: str) -> Path:
    base = RESULTS_ROOT / doc_id / "runs" / _run if _run else RESULTS_ROOT / doc_id
    return base / METHOD_DIRS[method]


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


def load_metadata(doc_id: str, method: str) -> Dict[str, Any]:
    path = method_dir(doc_id, method) / "extraction_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def check_resume(doc_id: str, method: str, settings: Dict[str, Any]) -> None:
    """Raise ResumeError when saved columns were made with other settings (model, input, image scale)."""
    if not load_columns(doc_id, method):
        return
    saved = load_metadata(doc_id, method)
    changed = [f"{key}: saved {saved.get(key)!r}, now {value!r}" for key, value in settings.items() if saved.get(key) != value]
    if changed:
        raise ResumeError(
            f"{method_dir(doc_id, method)} holds results made with different settings ({'; '.join(changed)}). "
            "Use --no-resume to replace them, or --run <name> to keep both."
        )


def save_metadata(doc_id: str, method: str, payload: Dict[str, Any]) -> Path:
    path = method_dir(doc_id, method) / "extraction_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"doc_id": doc_id, **payload}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
