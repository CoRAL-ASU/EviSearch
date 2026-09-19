"""Schema agent: drafts a definition for every column of a table from its header, what the header asks for (facets),
and one example value found in its paper; and revises definitions after a reviewer's answers and edits.

Drafting runs one structured call per group of at most BATCH columns (columns of the same header family stay together).
The agent must ask the reviewer, rather than guess, when the example value is not printed in the paper (often a label
the table owner uses), when the header has cryptic tokens, or when the example combines several things.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.config import MAX_TOKENS
from src.evisearch.schema.facets import parse_header
from src.evisearch.schema.grounding import ground
from src.inference import InferenceError, Message
from src.retrieval import embedding_retriever as retriever

BATCH = 12
EVAL_CATEGORIES = ("exact_match", "numeric_tolerance", "structured_text")
CONFIDENCE = ("high", "medium", "low")

DRAFT_PROMPT = """You write the column definitions of a clinical-trial evidence table. An extraction system reads each
definition to fill the column for any paper, so a definition must say exactly what goes in the cell. A human reviews
your definitions before they are used.

For each column you get its header, what the header seems to ask for (parsed), and one example value that the table's
owner entered for one paper, with where that value is printed in the paper (or that it is not printed there).

Write for each column:
- definition: one question, then what to include, then the absence rule, e.g. "What is the median overall survival, in
  months, of patients in the treatment arm of the high-volume subgroup? If reported, include the value as stated. Use
  'Not reported' if missing." Name the statistic, the characteristic or endpoint, the population or subgroup, the arm,
  the timepoint and the unit. Read the example value and its page text to learn what the owner means (which statistic,
  which population, the answer format such as "count (percent)", the vocabulary of labels), but write a definition that
  works for any paper: never put this paper's numbers or names into it.
- answer_format: the form of a cell value, e.g. "count (percent)", "median months", "Yes/No", "free text list".
- eval_category: exact_match for identifiers and short categorical answers (NCT, year, phase, yes/no, a fixed label),
  numeric_tolerance for numeric values, structured_text for descriptions and lists.
- not_reported_policy: when the cell is "Not reported" (e.g. "when the paper does not state it for this arm; never
  computed from other numbers").
- reading: one sentence on how the example value was read from the paper (page and what it is), or why it is not there.
- questions: 0-3 questions for the reviewer, each with 2-4 short options. ASK instead of guessing when:
  - the example value is not printed in the paper (it may be a label or convention of the table owner, or computed);
  - the header has cryptic tokens or abbreviations you cannot expand from the paper;
  - the example combines several things (both arms, several trials, endpoint variants such as bPFS and rPFS);
  - the header could mean more than one statistic, population or arm.
- confidence: high, medium or low.

Return JSON with one entry per column, using the column names exactly as given."""

REVISE_PROMPT = """You revise column definitions of a clinical-trial evidence table after a human reviewer's feedback.
For each column you get the current definition, the example value, the reviewer's answers to your questions and the
reviewer's notes. Rewrite the definition so it follows the feedback exactly, in the same style (question; what to
include; absence rule), still general for any paper. Keep everything the feedback does not change. Return JSON with one
entry per column: column, definition, revised, change (one sentence: what changed and why).
Set revised to false when the feedback needs no change to the definition — for example when an answer only confirms what
the definition already says — and then return the current definition word for word. Set it to true only when you changed
the definition. Never rewrite wording the feedback does not ask you to change."""


def _schema(names: Sequence[str], revise: bool = False) -> Dict[str, Any]:
    if revise:
        props = {"column": {"type": "string", "enum": list(names)}, "definition": {"type": "string"},
                 "revised": {"type": "boolean"}, "change": {"type": "string"}}
        required = ["column", "definition", "revised", "change"]
    else:
        props = {
            "column": {"type": "string", "enum": list(names)},
            "definition": {"type": "string"},
            "answer_format": {"type": "string"},
            "eval_category": {"type": "string", "enum": list(EVAL_CATEGORIES)},
            "not_reported_policy": {"type": "string"},
            "reading": {"type": "string"},
            "questions": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
                "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
            }, "required": ["question", "options"]}},
            "confidence": {"type": "string", "enum": list(CONFIDENCE)},
        }
        required = list(props)
    return {"type": "object", "properties": {"columns": {"type": "array", "minItems": len(names), "maxItems": len(names),
            "items": {"type": "object", "properties": props, "required": required}}}, "required": ["columns"]}


def batches(headers: Sequence[str], size: int = BATCH) -> List[List[str]]:
    """Consecutive columns of the same header family stay together; families are packed into batches of <= size."""
    families: List[List[str]] = []
    for header in headers:
        family = parse_header(header)["family"]
        if families and parse_header(families[-1][0])["family"] == family:
            families[-1].append(header)
        else:
            families.append([header])
    out: List[List[str]] = []
    for fam in families:
        for i in range(0, len(fam), size):
            part = fam[i:i + size]
            if out and len(out[-1]) + len(part) <= size:
                out[-1].extend(part)
            else:
                out.append(list(part))
    return out


def _column_block(i: int, header: str, facets: Dict[str, Any], value: str, grounding: Dict[str, Any], doc_id: str) -> str:
    parsed = ", ".join(f"{k}={facets[k]!r}" for k in ("characteristic", "statistic", "unit", "subgroup", "arm", "category") if facets.get(k))
    lines = [f"---\nColumn {i}: {header}", f"Parsed: {parsed or '(nothing recognised)'}"]
    if facets.get("cryptic"):
        lines.append(f"Cryptic tokens: {', '.join(facets['cryptic'])}")
    if not value:
        lines.append(f"Example value ({doc_id}): empty — the owner left this cell blank for this paper")
    else:
        where = {"found": f"printed on page(s) {grounding['pages']}", "not_in_paper": "NOT printed in the paper",
                 "short": "a short answer (not searched)", "empty": ""}[grounding["status"]]
        lines.append(f'Example value ({doc_id}): "{value}" — {where}')
        for snip in grounding.get("snippets", [])[:2]:
            lines.append(f'  page {snip["page"]}: "{snip["text"]}"')
    return "\n".join(lines)


def _ask(chat: Any, system: str, user: str, names: Sequence[str], revise: bool = False) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    result = chat.chat([Message.system(system), Message.user(user)], response_schema=_schema(names, revise), max_tokens=MAX_TOKENS["pdf_query"])
    log = {"finish_reason": result.finish_reason, "usage": result.usage.to_dict(), "response_text": result.text}
    try:
        parsed = result.json() or {}
    except ValueError as exc:
        log["error"] = str(exc)
        return {}, log
    return {str(c.get("column")): c for c in parsed.get("columns", []) if isinstance(c, dict) and c.get("column") in names}, log


def draft_fields(
    chat: Any, doc_id: str, headers: Sequence[str], example: Dict[str, str], *, conventions: str = "", description: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Field records (Table Schema fields with x-evisearch) for every header, in order, plus call logs."""
    total = retriever.get_total_pages(doc_id)
    pages = retriever.get_page_content(doc_id, list(range(1, total + 1)))
    info = {h: (parse_header(h), example.get(h, ""), ground(doc_id, example.get(h, ""), pages)) for h in headers}
    system = DRAFT_PROMPT + (f"\n\nConventions the table's owner has already given (apply them):\n{conventions}" if conventions else "")
    drafted: Dict[str, Dict[str, Any]] = {}
    logs: List[Dict[str, Any]] = []
    for batch in batches(headers):
        blocks = [_column_block(i, h, *info[h], doc_id) for i, h in enumerate(batch, 1)]
        intro = (f"Table: {description}\n\n" if description else "") + f"Columns ({len(batch)}):\n"
        got, log = _ask(chat, system, intro + "\n".join(blocks), batch)
        missing = [h for h in batch if h not in got]
        if missing:  # one follow-up for the columns a reply left out
            more, follow = _ask(chat, system, intro + "\n".join(_column_block(i, h, *info[h], doc_id) for i, h in enumerate(missing, 1)), missing)
            got.update(more)
            log["follow_up"] = follow
        log["columns"] = batch
        logs.append(log)
        drafted.update(got)
    fields = []
    for h in headers:
        facets, value, grounding = info[h]
        d = drafted.get(h, {})
        questions = [{"id": f"q{i}", "question": q.get("question", ""), "options": q.get("options", []), "answer": None}
                     for i, q in enumerate(d.get("questions", []) or [], 1)]
        if not d:
            questions.append({"id": "q_missing", "question": "The schema agent returned no definition for this column. What should it contain?",
                              "options": ["Write it myself"], "answer": None})
        fields.append({
            "name": h,
            "title": h,
            "description": d.get("definition", ""),
            "type": "string",
            "x-evisearch": {
                "group": facets["family"],
                "facets": {k: facets[k] for k in ("characteristic", "statistic", "unit", "subgroup", "arm", "category", "cryptic")},
                "answer_format": d.get("answer_format", ""),
                "eval_category": d.get("eval_category") if d.get("eval_category") in EVAL_CATEGORIES else "structured_text",
                "nr_policy": d.get("not_reported_policy", ""),
                "example": {"doc": doc_id, "value": value, "grounding": grounding},
                "reading": d.get("reading", ""),
                "questions": questions,
                "confidence": d.get("confidence", "low"),
                "conventions": [],
                "review": {"state": "proposed", "by": None, "at": None},
                "history": [],
            },
        })
    return fields, logs


def pending_feedback(field: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """The reviewer's answers and notes that the current definition does not reflect yet: those given after the last
    edit or revision. An edit is the reviewer's own wording (its note explains it), so it is never rewritten."""
    x = field["x-evisearch"]
    history = x.get("history", [])
    last = max((h["at"] for h in history if h.get("action") in ("edit", "revise")), default="")
    answers = [f'- {q["question"]} → {q["answer"]}' for q in x.get("questions", [])
               if q.get("answer") and (q.get("answered_at") or "") > last]
    notes = [f'- {h["note"]}' for h in history if h.get("note") and h.get("action") not in ("edit", "revise") and h["at"] > last]
    return answers, notes


def revise_fields(chat: Any, fields: Sequence[Dict[str, Any]], *, conventions: str = "") -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Rewrite the definitions of fields with pending answers or notes; returns ({column: {definition, change}}, logs)."""
    todo = [f for f in fields if any(pending_feedback(f))]
    system = REVISE_PROMPT + (f"\n\nConventions the table's owner has given (apply them):\n{conventions}" if conventions else "")
    out: Dict[str, Dict[str, Any]] = {}
    logs: List[Dict[str, Any]] = []
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        blocks = []
        for j, f in enumerate(part, 1):
            x = f["x-evisearch"]
            answers, notes = pending_feedback(f)
            blocks.append("\n".join([f"---\nColumn {j}: {f['name']}", f"Current definition: {f['description']}",
                                     f'Example value: "{x["example"]["value"]}"', "Answers:", *(answers or ["- none"]),
                                     "Notes:", *(notes or ["- none"])]))
        got, log = _ask(chat, system, "\n".join(blocks), [f["name"] for f in part], revise=True)
        log["columns"] = [f["name"] for f in part]
        logs.append(log)
        out.update(got)
    return out, logs


def definitions_text(fields: Sequence[Dict[str, Any]]) -> str:
    return json.dumps({f["name"]: f["description"] for f in fields}, ensure_ascii=False, indent=1)


def draft_schema_from_sheet(chat: Any, sheet_path: str, doc_id: str, name: str, *, row_hint: Optional[str] = None,
                            by: str = "", conventions: str = "", description: str = "") -> Dict[str, Any]:
    """Ingest a spreadsheet, draft every column from the example row of doc_id, and store the draft schema."""
    from src.evisearch.schema import store
    from src.evisearch.schema.ingest import DOC_COLUMN, read_sheet

    sheet = read_sheet(sheet_path)
    row = sheet.row_for(row_hint or doc_id) or (sheet.rows[0] if sheet.rows else {})
    headers = [h for h in sheet.headers if h != DOC_COLUMN]
    if not headers:
        raise InferenceError(f"{sheet_path}: no columns")
    fields, logs = draft_fields(chat, doc_id, headers, row, conventions=conventions, description=description)
    schema = store.create(name, fields, source={"sheet": str(sheet_path), "example_doc": doc_id, "description": description}, by=by)
    log_path = store.schema_dir(schema["id"]) / "draft_calls.json"
    log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=1), encoding="utf-8")
    return schema
