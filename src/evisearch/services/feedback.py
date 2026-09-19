"""
feedback_service.py

The append-only feedback log (FEEDBACK_DIR/feedback.jsonl): chat and attribution feedback, schema review actions, cell
reviews and knowledge-base decisions. Every entry gets an `event_id` so other records (conventions, reviews) can point
to it; entries written before ids existed get a stable id derived from their content (see `event_id_of`).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.config.runtime_paths import FEEDBACK_DIR

FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

_CACHE: Dict[str, Tuple[Tuple[int, int], List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def event_id_of(entry: Dict[str, Any]) -> str:
    """The entry's id; for entries written before ids existed, a stable hash of their content."""
    if entry.get("event_id"):
        return str(entry["event_id"])
    raw = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return "h" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:15]


def append_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append one entry (with a new event_id and a server timestamp) and return it. Raises on I/O errors."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    entry = dict(payload)
    entry["timestamp"] = _timestamp()
    entry.setdefault("event_id", uuid.uuid4().hex[:16])
    # Truncate comment to 500 chars
    if "comment" in entry and entry["comment"]:
        entry["comment"] = str(entry["comment"])[:500]
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # one whole line per writer, also across processes (CLI + web app)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return entry


def record_feedback(payload: Dict[str, Any]) -> bool:
    """
    Append a feedback entry to feedback.jsonl.
    Returns True on success, False on error.
    """
    try:
        append_event(payload)
        return True
    except Exception:
        return False


def all_events() -> List[Dict[str, Any]]:
    """Every entry, oldest first, each with an `event_id`. Cached until the file changes; callers must not mutate."""
    path = FEEDBACK_FILE
    if not path.exists():
        return []
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        cached = _CACHE.get(str(path))
        if cached and cached[0] == key:
            return cached[1]
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entry.setdefault("event_id", event_id_of(entry))
                entries.append(entry)
    with _CACHE_LOCK:
        _CACHE[str(path)] = (key, entries)
    return entries


def load_feedback(doc_id: str | None = None, source: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Load feedback entries, optionally filtered by doc_id and source.
    Returns most recent first, up to limit.
    """
    try:
        entries = [e for e in all_events()
                   if (not doc_id or e.get("doc_id") == doc_id) and (not source or e.get("source") == source)]
        entries = list(reversed(entries))
        return entries[:limit]
    except Exception:
        return []
