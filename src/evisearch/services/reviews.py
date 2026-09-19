"""Human reviews of extracted cells, kept per run.

The feedback log is the store: a review is a `cell_correct` event {doc_id, run, column, before, after, machine_value,
state, reason, note, by} and an undo is a `cell_undo` event naming the review it cancels. A cell's current review is its
last review in that run that was not undone, so a correction made on one run never shows on another. (The legacy
per-paper file RESULTS_ROOT/<doc>/human-edited/ is still written by the old endpoint for older tools, but views of a run
never read it.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.evisearch.services import feedback
from src.evisearch.services.runs import is_not_reported

ACCEPT_REASON = "confirmed correct"
REASONS = (  # why a value is wrong; the closed list shared by the Review page and the rule proposer
    "missed value on the page",
    "value not on the page",
    "wrong statistic",
    "wrong population or subgroup",
    "wrong arm",
    "missing convention",
    "unit or format",
    "should be Not reported",
    "definition or gold issue",
)
STATES = ("accepted", "corrected", "not_reported")


def _norm(value: Any) -> str:
    return " ".join(str(value if value is not None else "").lower().split())


def state_of(value: Any, machine_value: Any) -> str:
    if _norm(value) == _norm(machine_value):
        return "accepted"
    if is_not_reported(value):
        return "not_reported"
    return "corrected"


def _events(doc_id: str, run: Optional[str]) -> List[Dict[str, Any]]:
    return [e for e in feedback.all_events()
            if e.get("event") in ("cell_correct", "cell_undo") and e.get("doc_id") == doc_id and (e.get("run") or None) == (run or None)]


def cell_reviews(doc_id: str, run: Optional[str], machine: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """{column: current review + history} for one paper of one run. Columns whose reviews were all undone have
    `value: None` and keep their history. With `machine` ({column: the run's value}) each current review also gets its
    `state` against the run's own value (accepted / corrected / not_reported)."""
    events = _events(doc_id, run)
    undone = {str(e.get("target")) for e in events if e.get("event") == "cell_undo"}
    out: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if e.get("event") != "cell_correct" or not e.get("column"):
            continue
        item = {"event_id": e["event_id"], "value": str(e.get("after") if e.get("after") is not None else ""),
                "before": e.get("before"), "machine_value": e.get("machine_value"), "reason": str(e.get("reason") or ""),
                "note": str(e.get("note") or ""), "by": str(e.get("by") or ""), "at": e.get("timestamp"),
                "undone": e["event_id"] in undone}
        slot = out.setdefault(str(e["column"]), {"history": []})
        slot["history"].append(item)
    for column, slot in out.items():
        live = [h for h in slot["history"] if not h["undone"]]
        current = live[-1] if live else None
        slot.update({k: (current or {}).get(k) for k in ("event_id", "value", "reason", "note", "by", "at")})
        if machine is not None and current is not None:
            slot["state"] = state_of(current["value"], machine.get(column, ""))
    return out


def record(doc_id: str, run: Optional[str], column: str, value: str, *, machine_value: str, reason: str = "",
           note: str = "", by: str = "", schema_id: Optional[str] = None) -> Dict[str, Any]:
    """Store one review decision and return the stored event (with its event_id and state)."""
    if not column:
        raise ValueError("column required")
    value = "" if value is None else str(value)
    state = state_of(value, machine_value)
    if state == "accepted":
        reason = reason or ACCEPT_REASON
    elif not reason:
        raise ValueError("a correction needs a reason")
    return feedback.append_event({
        "source": "correction", "event": "cell_correct", "doc_id": doc_id, "column": column, "run": run,
        "schema_id": schema_id, "by": by, "before": machine_value, "after": value, "machine_value": machine_value,
        "state": state, "reason": reason, "note": note,
    })


def undo(doc_id: str, run: Optional[str], event_id: str, by: str = "") -> Dict[str, Any]:
    """Cancel one review (the cell falls back to the previous review, or to the machine value)."""
    target = next((e for e in _events(doc_id, run) if e.get("event") == "cell_correct" and e["event_id"] == event_id), None)
    if target is None:
        raise KeyError(f"no review {event_id!r} for this paper and run")
    return feedback.append_event({"source": "correction", "event": "cell_undo", "doc_id": doc_id, "run": run,
                                  "column": target.get("column"), "target": event_id, "by": by,
                                  "schema_id": target.get("schema_id")})


def counts(doc_id: str, run: Optional[str], machine: Dict[str, str]) -> Dict[str, int]:
    """{reviewed, corrected} for one paper of a run; corrected = the reviewer's value differs from the run's."""
    current = [r for r in cell_reviews(doc_id, run, machine).values() if r.get("value") is not None]
    return {"reviewed": len(current), "corrected": sum(1 for r in current if r.get("state") != "accepted")}
