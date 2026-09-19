"""Rule proposer: turn a reviewer's feedback (a correction with its reason, or an answer during schema review) into a
structured convention for the knowledge base. The reviewer confirms or edits it before the gate and approval.

The proposal must generalise: it names the columns or header family it applies to and a condition, never this paper's
values. Paper-specific facts stay cell corrections.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.evisearch.knowledge.conventions import ACTION_TYPES, SCOPES
from src.inference import InferenceError, Message

PROPOSER_PROMPT = """You turn a human reviewer's feedback on a clinical-trial extraction table into a reusable
CONVENTION: a rule that tells the extraction system how to fill some columns for ANY paper of this table.

Rules:
- Generalise: say which columns it applies to (scope "column" with exact column names, scope "family" with the header
  family, e.g. "Median PFS (mo)", or scope "table"/"global" for every column) and under which condition. Never put this
  paper's numbers, names or pages into the instruction.
- If the feedback only fixes a fact of this one paper (a misread number, a wrong page) it is NOT a convention: return
  is_convention=false.
- Pick exactly one action type: {types}
- instruction: one or two sentences in the imperative, starting with "- ", e.g. "- When the paper reports progression-free
  survival only as named variants (for example biochemical and radiographic PFS), give each variant with its label."
Return JSON."""


def _schema() -> Dict[str, Any]:
    return {"type": "object", "properties": {
        "is_convention": {"type": "boolean"},
        "why_not": {"type": "string"},
        "scope": {"type": "string", "enum": list(SCOPES)},
        "family": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "condition": {"type": "string"},
        "action_type": {"type": "string", "enum": list(ACTION_TYPES)},
        "instruction": {"type": "string"},
    }, "required": ["is_convention", "why_not", "scope", "family", "columns", "condition", "action_type", "instruction"]}


def propose(chat: Any, *, column: str, definition: str, feedback: str, before: str = "", after: str = "", reason: str = "",
            columns: Sequence[str] = (), paper: str = "", source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{is_convention, why_not, record} where record is ready for the gate (status set on create)."""
    types = "; ".join(f"{k} ({v})" for k, v in ACTION_TYPES.items())
    user = "\n".join([
        f"Column: {column}", f"Column definition: {definition}",
        f"Value before feedback: {before or '(none)'}", f"Value after feedback: {after or '(none)'}",
        f"Reviewer's reason: {reason or '(none)'}", f"Reviewer's note: {feedback}",
        f"Other columns of the table (for scope): {', '.join(columns[:160])}",
    ])
    try:
        result = chat.chat([Message.system(PROPOSER_PROMPT.format(types=types)), Message.user(user)], response_schema=_schema(), max_tokens=800)
        out = result.json() or {}
    except (InferenceError, ValueError) as exc:
        return {"is_convention": False, "why_not": f"proposer failed: {exc}", "record": None}
    if not out.get("is_convention"):
        return {"is_convention": False, "why_not": out.get("why_not", ""), "record": None}
    instruction = str(out.get("instruction", "")).strip()
    record = {
        "trigger": {"scope": out.get("scope") if out.get("scope") in SCOPES else "column", "family": out.get("family", ""),
                    "columns": [c for c in out.get("columns", []) if c in columns or c == column] or [column],
                    "facets": {}, "condition": out.get("condition", "")},
        "action": {"type": out.get("action_type") if out.get("action_type") in ACTION_TYPES else "answer_format", "params": {}},
        "instruction": instruction if instruction.startswith("- ") else "- " + instruction,
        "examples": [{"doc": paper, "column": column, "before": before, "after": after}],
        "source": {**(source or {}), "feedback": feedback, "reason": reason},
    }
    return {"is_convention": True, "why_not": "", "record": record}
