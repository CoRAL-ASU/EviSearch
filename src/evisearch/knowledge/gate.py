"""Integrity gate for the conventions knowledge base: no duplicates, no silent overlaps, no active conflicts.

For a proposed convention:
1. overlapping(): conventions whose triggers can apply to the same column (decided by code: scope and facets);
2. relation(): each overlapping one is a duplicate, narrower, broader, conflict or independent of the proposal —
   exact duplicates (same trigger, action and parameters) are decided by code, the rest by the model;
3. check(): the verdict the reviewer sees. A duplicate is merged into the existing convention (support + 1); a conflict
   blocks approval until the reviewer retires one, narrows a scope, or rejects the proposal; narrower/broader are kept
   (the more specific applies where both fire; a broader one may replace narrower ones).
impact(): the columns of a schema the trigger applies to, for the reviewer to judge (and for a regression replay).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from src.evisearch.knowledge import conventions as kb
from src.inference import InferenceError, Message

RELATIONS = ("duplicate", "narrower", "broader", "conflict", "independent")
FACET_KEYS = ("characteristic", "statistic", "arm", "subgroup", "category")

RELATION_PROMPT = """You maintain a knowledge base of conventions that tell an extraction system how to fill a
clinical-trial table. Compare a PROPOSED convention with an EXISTING one whose trigger can apply to some of the same
columns, and classify their relation:
- duplicate: they say the same thing for the same columns (wording may differ);
- narrower: the proposal applies to a subset of the existing one's columns or cases and agrees with it;
- broader: the proposal covers the existing one's columns or cases and more, and agrees with it;
- conflict: on some column or case they would make the extractor write different values;
- independent: they concern different things even though their columns overlap.
Return JSON {"relation": ..., "reason": one sentence naming the column or case that decides it}."""


def _columns(trigger: Dict[str, Any]) -> List[str]:
    return list(trigger.get("columns") or [])


def matches_column(trigger: Dict[str, Any], column: str, facets: Dict[str, Any], table: Optional[str] = None) -> bool:
    scope = trigger.get("scope")
    if scope == "global":
        return True
    if scope == "table":
        return not trigger.get("table") or trigger.get("table") == table
    if scope == "column":
        return column in _columns(trigger)
    # family: the header family, plus any facet constraints
    family = trigger.get("family")
    if family and not column.startswith(family) and column not in _columns(trigger):
        return False
    for key in FACET_KEYS:
        want = (trigger.get("facets") or {}).get(key)
        if want and str(facets.get(key, "")).lower() != str(want).lower():
            return False
    return True


def triggers_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if "global" in (a.get("scope"), b.get("scope")):
        return True
    if "table" in (a.get("scope"), b.get("scope")):
        return not (a.get("table") and b.get("table") and a.get("table") != b.get("table"))
    cols_a, cols_b = set(_columns(a)), set(_columns(b))
    if cols_a & cols_b:
        return True
    fam_a, fam_b = a.get("family"), b.get("family")
    if fam_a and fam_b and fam_a != fam_b and not (cols_a or cols_b):
        return False
    if (fam_a and any(c.startswith(fam_a) for c in cols_b)) or (fam_b and any(c.startswith(fam_b) for c in cols_a)):
        return True
    if fam_a and fam_b and fam_a == fam_b:
        fa, fb = a.get("facets") or {}, b.get("facets") or {}
        return all(not (fa.get(k) and fb.get(k) and str(fa[k]).lower() != str(fb[k]).lower()) for k in FACET_KEYS)
    return False


def overlapping(proposal: Dict[str, Any], existing: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    existing = [c for c in (existing if existing is not None else kb.load_all().values()) if c["status"] in ("approved", "proposed")]
    return [c for c in existing if c.get("id") != proposal.get("id") and triggers_overlap(proposal["trigger"], c["trigger"])]


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    key = lambda c: json.dumps([c["trigger"].get("scope"), sorted(_columns(c["trigger"])), c["trigger"].get("family"),  # noqa: E731
                                c["action"], " ".join(c["instruction"].lower().split())], sort_keys=True)
    return key(a) == key(b)


def relation(chat: Any, proposal: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, str]:
    if _same(proposal, other):
        return {"relation": "duplicate", "reason": "same trigger, action and instruction"}
    show = lambda c: json.dumps({k: c.get(k) for k in ("trigger", "action", "instruction")}, ensure_ascii=False)  # noqa: E731
    schema = {"type": "object", "properties": {"relation": {"type": "string", "enum": list(RELATIONS)}, "reason": {"type": "string"}},
              "required": ["relation", "reason"]}
    try:
        result = chat.chat([Message.system(RELATION_PROMPT), Message.user(f"PROPOSED: {show(proposal)}\nEXISTING ({other['id']}): {show(other)}")],
                           response_schema=schema, max_tokens=400)
        out = result.json() or {}
    except (InferenceError, ValueError) as exc:
        return {"relation": "conflict", "reason": f"could not be classified ({exc}); treated as a conflict for the reviewer"}
    rel = out.get("relation") if out.get("relation") in RELATIONS else "conflict"
    return {"relation": rel, "reason": str(out.get("reason", ""))}


def check(chat: Any, proposal: Dict[str, Any], existing: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """{verdict: new | duplicate | blocked, relations: [{id, relation, reason, instruction}], duplicate_of}."""
    relations = []
    for other in overlapping(proposal, existing):
        rel = relation(chat, proposal, other)
        relations.append({"id": other["id"], "instruction": other["instruction"], "scope": other["trigger"]["scope"],
                          "status": other["status"], **rel})
    dup = next((r for r in relations if r["relation"] == "duplicate"), None)
    conflict = [r for r in relations if r["relation"] == "conflict"]
    verdict = "duplicate" if dup else ("blocked" if conflict else "new")
    return {"verdict": verdict, "relations": relations, "duplicate_of": dup["id"] if dup else None,
            "conflicts": [r["id"] for r in conflict]}


def impact(trigger: Dict[str, Any], fields: Sequence[Dict[str, Any]], table: Optional[str] = None) -> List[str]:
    """Columns of a schema (field records with x-evisearch facets) that the trigger applies to."""
    return [f["name"] for f in fields if matches_column(trigger, f["name"], f.get("x-evisearch", {}).get("facets", {}), table)]
