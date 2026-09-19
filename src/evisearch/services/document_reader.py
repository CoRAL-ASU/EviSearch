"""
Document reader: answers the reconciler's questions from the whole paper, in a separate model call.

The reconciler never holds the paper in its own context. Its ask_document tool sends questions here; this call gets
the full document exactly as Agent A reads it (every page's parsed text followed by its image, with the same token
budget fallback) and returns, per question, the answer with the page(s) and the text that shows it. The document comes
first in the request, so repeated questions about one paper reuse the cached prefix on vLLM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.config.config import MAX_TOKENS, SELECTION
from src.evisearch.columns import MODALITIES
from src.evisearch.services import pdf_query
from src.evisearch.services.extraction_rules import shared_rules
from src.inference import InferenceError, Message, TextPart, Usage
from src.inference.base import ChatModel

QUESTIONS_PER_CALL = 8
QUESTION_RESERVE_CHARS = 6000  # room kept for the questions when sizing the document, so every call sees the same document

SYSTEM_PROMPT = """You answer questions about one clinical trial paper, using only the document provided.

For each question return:
- answer: the value or fact as the paper states it, with the population, arm and timepoint it refers to. When the
  paper gives several candidates (overall and subgroup values, two timepoints, two trials), list them with what each
  refers to. Answer "Not reported" when the paper does not state it.
- pages: the 1-based page(s) that show the answer (empty when not reported).
- evidence: the text from that page, copied as printed: the sentence, or for a table the row label, column header
  and cell, or for a figure its label and the number read from it.
- modality: "table", "figure" or "text"."""

_documents: Dict[Tuple[str, str, str], pdf_query.DocumentInput] = {}


def input_mode(chat: ChatModel) -> str:
    mode = SELECTION.option("pdf_query_input")
    return mode if mode == "markdown" or chat.capabilities.images else "markdown"


def document_for(chat: ChatModel, doc_id: str) -> pdf_query.DocumentInput:
    """The document parts for this model, built once per process (page rendering is the slow part)."""
    mode = input_mode(chat)
    key = (doc_id, chat.key, mode)
    if key not in _documents:
        budget = pdf_query.document_token_budget(
            chat.spec.context_tokens, SYSTEM_PROMPT + pdf_query.IMAGE_RULES + "x" * QUESTION_RESERVE_CHARS, MAX_TOKENS["reader"]
        )
        _documents.clear()  # one paper at a time is enough; page images are several MB
        _documents[key] = pdf_query.build_document_input(doc_id, mode, budget, chat.spec.image_tokens)
    return _documents[key]


def response_schema(ids: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "answer": {"type": "string"},
                        "pages": {"type": "array", "items": {"type": "integer"}},
                        "evidence": {"type": "string"},
                        "modality": {"type": "string", "enum": list(MODALITIES)},
                    },
                    "required": ["id", "answer", "pages", "evidence", "modality"],
                },
            }
        },
        "required": ["answers"],
    }


def answer_questions(
    chat: ChatModel, doc_id: str, questions: List[Dict[str, str]], definitions: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], Usage, List[Dict[str, Any]]]:
    """questions: [{question, column?}] (the first QUESTIONS_PER_CALL are answered). Returns (answers in question
    order, usage, call logs). Never raises for model errors: failed answers carry the error. A failed call with several
    questions is retried as two halves: structured output can loop inside a string until max_tokens, and at
    temperature 0 an identical retry loops again, so the retry changes the prompt."""
    questions = questions[:QUESTIONS_PER_CALL]
    answers, usage, call = _answer_once(chat, doc_id, questions, definitions)
    if "error" not in call or len(questions) < 2 or "document" not in call:
        return answers, usage, [call]
    call["recovered_by_split"] = True
    half = (len(questions) + 1) // 2
    answers, calls = [], [call]
    for part in (questions[:half], questions[half:]):
        part_answers, part_usage, part_calls = answer_questions(chat, doc_id, part, definitions)
        answers += part_answers
        usage.add(part_usage)
        calls += part_calls
    return answers, usage, calls


def _answer_once(
    chat: ChatModel, doc_id: str, questions: List[Dict[str, str]], definitions: Optional[Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], Usage, Dict[str, Any]]:
    ids = [f"q{i}" for i in range(1, len(questions) + 1)]
    usage = Usage()
    call: Dict[str, Any] = {"questions": len(questions)}
    try:
        document = document_for(chat, doc_id)
    except (FileNotFoundError, ValueError) as exc:
        call["error"] = str(exc)
        return [{**q, "answer": "", "pages": [], "evidence": "", "modality": "text", "error": str(exc)} for q in questions], usage, call
    call["document"] = {key: document.info.get(key) for key in ("pages", "image_pages", "fallback", "estimated_tokens")}

    lines = ["QUESTIONS:"]
    for question_id, question in zip(ids, questions):
        column = question.get("column") or ""
        definition = (definitions or {}).get(column, "")
        about = f"\nColumn: {column}" + (f"\nDefinition: {definition}" if definition else "") if column else ""
        lines.append(f"\n---\nid: {question_id}{about}\nQuestion: {question.get('question', '')}")
    lines.append('\nReturn JSON: {"answers": [{"id": ..., "answer": ..., "pages": [...], "evidence": ..., "modality": ...}]}')
    columns = [q["column"] for q in questions if q.get("column")] or None
    system = SYSTEM_PROMPT + (pdf_query.IMAGE_RULES if document.info["image_pages"] else "") + shared_rules(columns=columns)
    messages = [Message.system(system), Message.user(*document.parts, TextPart("\n".join(lines)))]
    schema = response_schema(ids) if chat.capabilities.json_schema else None
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["reader"])
        usage.add(result.usage)
        call.update(result.call_record(), finish_reason=result.finish_reason)
        if str(result.finish_reason).lower() == "length":
            raise ValueError("reply cut off at max_tokens")
        parsed = result.json()
    except (InferenceError, ValueError) as exc:
        call["error"] = str(exc)
        return [{**q, "answer": "", "pages": [], "evidence": "", "modality": "text", "error": str(exc)} for q in questions], usage, call

    by_id = {str(item.get("id")): item for item in (parsed or {}).get("answers", []) if isinstance(item, dict)}
    answers = []
    for question_id, question in zip(ids, questions):
        item = by_id.get(question_id, {})
        pages = [int(p) for p in item.get("pages") or [] if isinstance(p, (int, float)) and int(p) >= 1]
        modality = str(item.get("modality") or "text").lower()
        answers.append({
            **question,
            "answer": str(item.get("answer") or "").strip(),
            "pages": pages,
            "evidence": str(item.get("evidence") or "").strip(),
            "modality": modality if modality in MODALITIES else "text",
            **({} if item else {"error": "reader returned no answer for this question"}),
        })
    return answers, usage, call
