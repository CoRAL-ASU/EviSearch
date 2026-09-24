"""Schemas on disk: a draft that a reviewer edits, then locked versions that extractions run under.

SCHEMAS_DIR/<id>/schema.json is the current state (a Frictionless Table Schema; EviSearch data per field under
"x-evisearch"). Locking writes versions/v<N>.json and versions/v<N>.csv (the definitions CSV format the pipelines read)
and bumps the version. Every review action is appended to the field's history and recorded as a feedback event.
Extractions under a locked version use the run name schema-<id>-v<N>.
"""
from __future__ import annotations

import csv
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import runtime_paths
from src.evisearch.services.feedback import record_feedback

REVIEW_ACTIONS = ("accept", "edit", "answer", "revise")
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def schema_dir(schema_id: str) -> Path:
    return runtime_paths.SCHEMAS_DIR / schema_id


def run_name(schema_id: str, version: int) -> str:
    return f"schema-{schema_id}-v{version}"


def new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "schema"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{slug}-{stamp}"


def record_event(event: str, schema_id: Optional[str] = None, **payload: Any) -> None:
    payload.pop("source", None)  # never overwrite the event's own source
    record_feedback({**payload, "source": "schema", "event": event, "schema_id": schema_id})


def create(name: str, fields: List[Dict[str, Any]], *, source: Dict[str, Any], by: str = "", schema_id: Optional[str] = None) -> Dict[str, Any]:
    schema = {
        "id": schema_id or new_id(name),
        "name": name,
        "version": 1,  # the version the next lock produces
        "locked_versions": [],
        "status": "draft",
        "source": source,
        "created_at": _now(),
        "created_by": by,
        "fields": fields,
    }
    save(schema)
    record_event("schema_draft", schema["id"], by=by, fields=len(fields), origin=source)  # "source" is the event's own field
    return schema


def typed_field(name: str, definition: str, by: str = "") -> Dict[str, Any]:
    """A column its owner typed in: the name and the definition they wrote, so it is reviewed already, with no example
    row behind it."""
    from src.evisearch.schema.facets import parse_header

    facets = parse_header(name)
    return {
        "name": name,
        "title": name,
        "description": definition,
        "type": "string",
        "x-evisearch": {
            "group": facets["family"],
            "facets": {k: facets.get(k, "") for k in ("characteristic", "statistic", "unit", "subgroup", "arm", "category", "cryptic")},
            "answer_format": "",
            "eval_category": "structured_text",
            "nr_policy": "",
            "example": {"doc": None, "value": "", "grounding": None},
            "reading": "",
            "questions": [],
            "confidence": "",
            "conventions": [],
            "review": {"state": "accepted", "by": by or None, "at": _now()},
            "history": [],
        },
    }


def create_typed(name: str, columns: List[Dict[str, Any]], *, description: str = "", by: str = "") -> Dict[str, Any]:
    """A table typed in by hand: its name, and a name and a definition for every column; rows left entirely blank are
    ignored. Raises ValueError saying what is missing. The table starts as a draft, to be locked before extraction."""
    name = " ".join(str(name or "").split())
    if not name:
        raise ValueError("give the table a name")
    rows = [(" ".join(str((c or {}).get("name") or "").split()), str((c or {}).get("definition") or "").strip())
            for c in columns or [] if isinstance(c, dict)]
    rows = [row for row in rows if row[0] or row[1]]
    if not rows:
        raise ValueError("add at least one column with a name and a definition")
    problems, seen = [], {}
    for i, (column, definition) in enumerate(rows, 1):
        if not column:
            problems.append(f"column {i} has no name")
        if not definition:
            problems.append(f"column {i}{f' ({column})' if column else ''} has no definition")
        if column and column.lower() in seen:
            problems.append(f"column {i} ({column}) has the same name as column {seen[column.lower()]}")
        seen.setdefault(column.lower(), i)
    if problems:
        raise ValueError("; ".join(problems))
    return create(name, [typed_field(column, definition, by) for column, definition in rows],
                  source={"kind": "typed", "description": " ".join(str(description or "").split())}, by=by)


def save(schema: Dict[str, Any]) -> Path:
    folder = schema_dir(schema["id"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "schema.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load(schema_id: str, version: Optional[int] = None) -> Dict[str, Any]:
    path = schema_dir(schema_id) / ("schema.json" if version is None else f"versions/v{version}.json")
    if not path.exists():
        raise FileNotFoundError(f"schema {schema_id!r}{'' if version is None else f' v{version}'} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def list_schemas() -> List[Dict[str, Any]]:
    out = []
    root = runtime_paths.SCHEMAS_DIR
    for path in sorted(root.glob("*/schema.json")) if root.exists() else []:
        s = json.loads(path.read_text(encoding="utf-8"))
        states = [f["x-evisearch"]["review"]["state"] for f in s["fields"]]
        out.append({
            "id": s["id"], "name": s["name"], "status": s["status"], "version": s["version"],
            "locked_versions": s.get("locked_versions", []), "fields": len(s["fields"]),
            "reviewed": sum(st != "proposed" for st in states), "created_at": s.get("created_at"),
            "open_questions": sum(1 for f in s["fields"] for q in f["x-evisearch"].get("questions", []) if not q.get("answer")),
        })
    return out


def _same_text(a: str, b: str) -> bool:
    """Whether two definitions say the same thing as stored: equal once every run of whitespace is one space. A line
    break or a double space is not a change worth showing a reviewer; anything else is."""
    return " ".join((a or "").split()) == " ".join((b or "").split())


def get_field(schema: Dict[str, Any], column: str) -> Dict[str, Any]:
    for field in schema["fields"]:
        if field["name"] == column:
            return field
    raise KeyError(f"column {column!r} is not in schema {schema['id']!r}")


def review_field(
    schema_id: str, column: str, action: str, *, by: str = "", definition: Optional[str] = None,
    question_id: Optional[str] = None, answer: Optional[str] = None, reason: str = "", note: str = "",
    claimed_revised: Optional[bool] = None,
) -> Dict[str, Any]:
    """A reviewer action on one field: accept it, edit its definition, or answer one of its questions.
    ("revise" is the agent rewriting a definition after answers; it is recorded the same way.)
    `claimed_revised` is what the revise agent said it did (its own `revised` flag); the text decides what is recorded,
    and a disagreement between the two is kept on the entry as `claimed`."""
    if action not in REVIEW_ACTIONS:
        raise ValueError(f"action must be one of {REVIEW_ACTIONS}")
    with _LOCK:
        schema = load(schema_id)
        if schema["status"] == "locked" and action != "revise":
            schema["status"] = "draft"  # editing a locked schema starts the next version
        field = get_field(schema, column)
        x = field["x-evisearch"]
        before = field["description"]
        entry: Dict[str, Any] = {"at": _now(), "by": by, "action": action, "reason": reason, "note": note}
        if action in ("edit", "revise"):
            if not definition or not definition.strip():
                raise ValueError("an edit needs the new definition")
            field["description"] = definition.strip()
            # Two signals for "did the definition change": what the writer says (`claimed_revised`, the revise agent's
            # own flag) and what the text shows, compared with whitespace normalised. The text decides, because it
            # cannot be wrong; the claim is kept when the two disagree, which is how a reviewer sees that the agent
            # rewrote wording it said it would leave alone. Recording every revision as a change made 22 of the first
            # schema's 63 revisions look like edits and overwrote the review state of fields the owner had accepted.
            if _same_text(field["description"], before):
                entry["unchanged"] = True  # keep the entry: its action and time stop pending_feedback revising again
                if claimed_revised:
                    entry["claimed"] = "revised"  # the agent said it changed the definition, but the text is the same
            else:
                entry.update(before=before, after=field["description"])
                if claimed_revised is False:
                    entry["claimed"] = "unchanged"  # the agent said it changed nothing, yet the definition differs
                x["review"] = {"state": "edited" if action == "edit" else "revised", "by": by, "at": entry["at"]}
        elif action == "answer":
            q = next((q for q in x.get("questions", []) if q["id"] == question_id), None)
            if q is None:
                raise KeyError(f"question {question_id!r} not found on {column!r}")
            q["answer"], q["answered_by"], q["answered_at"] = answer, by, entry["at"]
            entry.update(question=q["question"], answer=answer)
            if x["review"]["state"] == "proposed":
                x["review"] = {"state": "answered", "by": by, "at": entry["at"]}
        else:
            x["review"] = {"state": "accepted", "by": by, "at": entry["at"]}
        x.setdefault("history", []).append(entry)
        save(schema)
    payload: Dict[str, Any] = {"column": column, "by": by, "reason": reason, "note": note, "before": entry.get("before"),
                               "after": entry.get("after"), "question": entry.get("question"), "answer": answer}
    if entry.get("unchanged"):
        payload["unchanged"] = True  # so a page never renders it as a change
    record_event(f"definition_{action}", schema_id, **payload)
    return field


def export_csv(schema: Dict[str, Any], path: Path) -> Path:
    """The definitions CSV format the pipelines read (Label = column group)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Label", "Column Name", "Definition", "eval_category"])
        writer.writeheader()
        for field in schema["fields"]:
            x = field["x-evisearch"]
            writer.writerow({"Label": x.get("group") or field["name"], "Column Name": field["name"],
                             "Definition": field["description"], "eval_category": x.get("eval_category", "structured_text")})
    return path


def lock(schema_id: str, by: str = "") -> Dict[str, Any]:
    """Snapshot the current state as the next version; returns {version, json, csv, run}."""
    with _LOCK:
        schema = load(schema_id)
        version = schema["version"]
        schema["status"], schema["locked_at"], schema["locked_by"] = "locked", _now(), by
        schema["locked_versions"] = sorted(set(schema.get("locked_versions", [])) | {version})
        folder = schema_dir(schema_id) / "versions"
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = dict(schema, version=version)
        (folder / f"v{version}.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        csv_path = export_csv(snapshot, folder / f"v{version}.csv")
        schema["version"] = version + 1
        save(schema)
    states = [f["x-evisearch"]["review"]["state"] for f in snapshot["fields"]]
    record_event("schema_lock", schema_id, by=by, version=version, fields=len(states),
                 accepted=states.count("accepted"), edited=states.count("edited"), revised=states.count("revised"))
    return {"version": version, "json": str(folder / f"v{version}.json"), "csv": str(csv_path), "run": run_name(schema_id, version)}
