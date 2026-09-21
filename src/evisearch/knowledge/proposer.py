"""Turn a reviewer's correction into an edit to the knowledge notes.

The reviewer fixes a cell and says why. Two outcomes are possible and only one of them is knowledge:

* the system misread this page - a wrong number, the wrong row, the wrong page. Worth exactly one cell, and the
  proposal returns is_knowledge=false;
* the column's meaning or the reading convention was ambiguous and the system resolved it differently than the
  reviewer would. That is worth something on every future paper, and belongs in a note.

The proposal is an edit to a NAMED note, not a free-standing rule, and the notes that already govern the column are
sent with the request. That is what replaced the old integrity gate: a one-line convention arrived with no context, so
duplicates and contradictions had to be found by comparing sentence pairs, and the knowledge base accumulated rules
that each looked reasonable alone. A note is a document about one topic, so the way not to contradict it is to read it
and edit it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.evisearch.knowledge import notes as notes_kb
from src.inference import InferenceError, Message

PROPOSER_PROMPT = """You maintain the knowledge notes that tell a clinical-trial extraction system how to read papers
for one table. A reviewer has corrected a cell. Decide whether that correction is knowledge, and if it is, write the
edit.

You are shown the notes that already govern this column. Read them first. If one of them covers this topic, your edit
belongs in THAT note, under the heading it uses - extend or correct the existing text rather than adding a second rule
beside it. Propose a new note only when no existing note is about this topic at all.

What is knowledge:
- what a column means, which statistic or endpoint or population it asks for, what counts as its term;
- how to find and assemble a value: which table, which row, whether a derivation is permitted and how;
- what to do when the paper is silent, or reports the quantity only under a different name.

What is NOT knowledge: this paper's numbers, names, pages or arms. A misread page is a cell fix, not a note. If the
correction is only a fact about this one paper, return is_knowledge=false and nothing else.

Write the edit as one or two imperative bullets starting with "- ", in the voice of the surrounding note. Name the
condition it applies under. Never name this paper, its drugs or its values.

Return JSON: {"is_knowledge": bool, "note": "<id of the note to edit, or a new short kebab-case id>",
"heading": "<the heading in that note the text belongs under>", "text": "- ...", "why": "<one sentence: what was
ambiguous, and what the edit settles>", "new_note": bool, "role": "definitions" | "extraction"}"""


def _notes_block(governing: Sequence[notes_kb.Note]) -> str:
    if not governing:
        return "(no note currently governs this column)"
    out = []
    for note in governing:
        where = note.family or (", ".join(note.columns) if note.columns else note.scope)
        out.append(f"### note id: {note.id}   (role: {note.role}, scope: {note.scope}, applies to: {where})\n{note.body}")
    return "\n\n".join(out)


def propose(chat: Any, *, column: str, definition: str, feedback: str, before: str = "", after: str = "",
            reason: str = "", governing: Optional[Sequence[notes_kb.Note]] = None) -> Dict[str, Any]:
    """Ask the model for a note edit. Returns the parsed proposal; is_knowledge=false means "cell fix only"."""
    if governing is None:
        governing = notes_kb.governing([column])
    request = (
        f"COLUMN: {column}\nDEFINITION: {definition}\n"
        + (f"THE SYSTEM WROTE: {before!r}\n" if before else "")
        + (f"THE REVIEWER CORRECTED IT TO: {after!r}\n" if after else "")
        + (f"THE REVIEWER'S REASON: {reason}\n" if reason else "")
        + f"THE REVIEWER'S FEEDBACK: {feedback}\n\n"
        f"NOTES THAT ALREADY GOVERN THIS COLUMN:\n{_notes_block(governing)}"
    )
    schema = {
        "type": "object",
        "properties": {
            "is_knowledge": {"type": "boolean"},
            "note": {"type": "string"},
            "heading": {"type": "string"},
            "text": {"type": "string"},
            "why": {"type": "string"},
            "new_note": {"type": "boolean"},
            "role": {"type": "string", "enum": ["definitions", "extraction"]},
        },
        "required": ["is_knowledge"],
    }
    try:
        result = chat.chat([Message.system(PROPOSER_PROMPT), Message.user(request)],
                           response_schema=schema, max_tokens=800).json() or {}
    except (InferenceError, ValueError) as exc:
        return {"is_knowledge": False, "why": f"proposer failed: {exc}"}
    if not result.get("is_knowledge"):
        return {"is_knowledge": False, "why": str(result.get("why") or "a fact about this paper, not a convention")}
    return {
        "is_knowledge": True,
        "note": str(result.get("note") or "").strip(),
        "heading": str(result.get("heading") or "").strip(),
        "text": str(result.get("text") or "").strip(),
        "why": str(result.get("why") or "").strip(),
        "new_note": bool(result.get("new_note")),
        "role": result.get("role") if result.get("role") in ("definitions", "extraction") else "extraction",
        "governing": [n.id for n in governing],
    }
