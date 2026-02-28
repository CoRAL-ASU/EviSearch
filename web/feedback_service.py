"""
feedback_service.py

Record user feedback for Chat (QA) and Verify flows.
Stores to JSONL file for analysis and model tuning.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "new_pipeline_outputs" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


def record_feedback(payload: Dict[str, Any]) -> bool:
    """
    Append a feedback entry to feedback.jsonl.
    Returns True on success, False on error.
    """
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        entry = dict(payload)
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        # Truncate comment to 500 chars
        if "comment" in entry and entry["comment"]:
            entry["comment"] = str(entry["comment"])[:500]
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_feedback(doc_id: str | None = None, source: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Load feedback entries, optionally filtered by doc_id and source.
    Returns most recent first, up to limit.
    """
    if not FEEDBACK_FILE.exists():
        return []
    try:
        entries = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if doc_id and entry.get("doc_id") != doc_id:
                        continue
                    if source and entry.get("source") != source:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        entries.reverse()
        return entries[:limit]
    except Exception:
        return []
