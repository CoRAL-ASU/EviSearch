"""Conventions knowledge base: how to read papers for a table, learned from human feedback and reused everywhere.

A convention is a structured record, not free text:
  trigger   {scope: global | table | family | column, table, family, columns, facets, condition}
  action    {type (closed list ACTION_TYPES), params}
  instruction  the sentence the extraction prompts receive (rendered once, then kept)
  examples, source {kind: seed | schema_review | extraction_review, event, by, at}, status, support, supersedes.

Storage: KNOWLEDGE_DIR/conventions.jsonl is an append-only log (create, approve, reject, retire, merge, update); the
current state is rebuilt from it. Seeded from the extraction rules (one global convention per rule bullet), so a KB
holding only the seed renders exactly the rules text the prompts used before.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import runtime_paths

ACTION_TYPES = {
    "statistic_rule": "which statistic goes in the column and what is never used for it (e.g. a rate must be stated, never computed)",
    "include_variants": "report every named variant of an endpoint together, each with its label",
    "label_vocabulary": "use the table owner's label for a category (e.g. 'Triplet therapy' for ADT + docetaxel + an AR inhibitor)",
    "synonym": "treat the paper's term as the column's term (same population, arm, subgroup or event under another name)",
    "population_scope": "which population, cohort or subgroup table answers the column (and which does not)",
    "design_implied": "a value fixed by the trial design (e.g. every patient of an arm receives the protocol treatment)",
    "counting_rule": "how to count or add (arms of a platform trial, grade thresholds, events vs patients)",
    "answer_format": "the form of the cell value (count with percent, value with timepoint, list per arm or trial)",
    "unit_rule": "units and conversions",
    "not_reported_policy": "when the cell is 'Not reported' (and when it is not)",
    "classification": "how to classify the paper or trial (follow-up vs original, add-on vs backbone, randomised)",
}
SCOPES = ("column", "family", "table", "global")  # most specific first
STATUSES = ("proposed", "approved", "rejected", "retired")
RULES_HEADER = "\n\nCOLUMN AND TRIAL CONVENTIONS (apply to every column):\n"
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def kb_dir() -> Path:
    return runtime_paths.KNOWLEDGE_DIR


def log_path() -> Path:
    return kb_dir() / "conventions.jsonl"


def _append(op: str, cid: str, by: str = "", **payload: Any) -> Dict[str, Any]:
    entry = {"at": _now(), "op": op, "id": cid, "by": by, **payload}
    kb_dir().mkdir(parents=True, exist_ok=True)
    with log_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_all() -> Dict[str, Dict[str, Any]]:
    """Current state of every convention, rebuilt from the log."""
    state: Dict[str, Dict[str, Any]] = {}
    if not log_path().exists():
        return state
    for line in log_path().read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        cid, op = e["id"], e["op"]
        if op == "create":
            state[cid] = dict(e["record"], history=[{"at": e["at"], "op": op, "by": e["by"]}])
            continue
        rec = state.get(cid)
        if rec is None:
            continue
        rec["history"].append({k: e[k] for k in ("at", "op", "by") if k in e} | ({"note": e["note"]} if e.get("note") else {}))
        if op in ("approve", "reject", "retire"):
            rec["status"] = {"approve": "approved", "reject": "rejected", "retire": "retired"}[op]
        elif op == "merge":  # another proposal was a duplicate of this one
            rec["support"] = rec.get("support", 1) + 1
            rec.setdefault("examples", []).extend(e.get("examples", []))
        elif op == "update":
            rec.update(e.get("changes", {}))
    return state


def active() -> List[Dict[str, Any]]:
    return [r for r in load_all().values() if r["status"] == "approved"]


def next_id(state: Optional[Dict[str, Any]] = None) -> str:
    state = load_all() if state is None else state
    return f"cv-{len(state) + 1:04d}"


def create(record: Dict[str, Any], by: str = "", status: str = "proposed") -> Dict[str, Any]:
    """Store a new convention (normally after the gate); returns it with its id."""
    if record.get("action", {}).get("type") not in ACTION_TYPES:
        raise ValueError(f"action type must be one of {sorted(ACTION_TYPES)}")
    if record.get("trigger", {}).get("scope") not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if not str(record.get("instruction", "")).strip():
        raise ValueError("a convention needs an instruction")
    with _LOCK:
        state = load_all()
        cid = record.get("id") or next_id(state)
        rec = {"examples": [], "support": 1, "supersedes": [], "version": 1, **record, "id": cid, "status": status}
        rec.setdefault("source", {}).setdefault("at", _now())
        _append("create", cid, by, record=rec)
    return rec


def decide(cid: str, op: str, by: str = "", note: str = "") -> Dict[str, Any]:
    if op not in ("approve", "reject", "retire"):
        raise ValueError("op must be approve, reject or retire")
    with _LOCK:
        if cid not in load_all():
            raise KeyError(cid)
        _append(op, cid, by, note=note)
    return load_all()[cid]


def merge_into(cid: str, examples: Iterable[Dict[str, Any]], by: str = "", note: str = "") -> Dict[str, Any]:
    with _LOCK:
        if cid not in load_all():
            raise KeyError(cid)
        _append("merge", cid, by, examples=list(examples), note=note)
    return load_all()[cid]


def seed_from_rules(version: str = "v5", by: str = "seed") -> List[Dict[str, Any]]:
    """One approved global convention per rule bullet (idempotent: an existing seed is not duplicated)."""
    from src.evisearch.services.extraction_rules import RULES

    text = RULES[version]
    if not text.startswith(RULES_HEADER):
        raise ValueError(f"rules {version} do not start with the expected header")
    bullets = ["- " + b for b in text[len(RULES_HEADER):].lstrip("- ").split("\n- ")]
    existing = {r["instruction"] for r in load_all().values() if r.get("source", {}).get("kind") == "seed"}
    out = []
    for i, bullet in enumerate(bullets, 1):
        if bullet in existing:
            continue
        out.append(create({
            "trigger": {"scope": "global", "facets": {}, "condition": ""},
            "action": {"type": "statistic_rule", "params": {"seed_rule": i}},
            "instruction": bullet,
            "source": {"kind": "seed", "rules": version, "bullet": i},
        }, by=by, status="approved"))
    return out


def render(conventions: Optional[List[Dict[str, Any]]] = None) -> str:
    """The prompt text: seed rules exactly as before, then learned conventions, general before specific."""
    conventions = active() if conventions is None else conventions
    seeds = sorted([c for c in conventions if c.get("source", {}).get("kind") == "seed"], key=lambda c: c["source"].get("bullet", 0))
    learned = [c for c in conventions if c.get("source", {}).get("kind") != "seed"]
    learned.sort(key=lambda c: (-SCOPES.index(c["trigger"]["scope"]), c["id"]))
    lines = [c["instruction"] for c in seeds]
    for c in learned:
        t = c["trigger"]
        where = {"global": "", "table": "", "family": f" [{t.get('family') or ', '.join(t.get('columns', []))} columns]",
                 "column": f" [{', '.join(t.get('columns', []))}]"}[t["scope"]]
        text = c["instruction"].strip()
        lines.append(f"-{where} {text[2:] if text.startswith('- ') else text}")
    if not lines:
        return ""
    body = RULES_HEADER + "\n".join(lines)
    if learned:
        body += "\n- When a convention for specific columns differs from a general one, the specific one applies to those columns."
    return body


def fingerprint(conventions: Optional[List[Dict[str, Any]]] = None) -> str:
    """Short hash of the active conventions, recorded in run settings so results of different KBs are not mixed."""
    conventions = active() if conventions is None else conventions
    blob = json.dumps(sorted((c["id"], c["instruction"]) for c in conventions), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
