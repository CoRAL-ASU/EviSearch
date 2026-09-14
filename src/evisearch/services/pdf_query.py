"""
Arm A (pdf_query): extract a batch of columns from the whole document in one structured call.

The model reads either the parsed markdown (any chat model) or the PDF itself (models that accept PDFs),
chosen by OPTIONS["pdf_query_input"] in src/config/config.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.catalog import ConfigError
from src.config.config import MAX_TOKENS, PDF_QUERY_MAX_MARKDOWN_CHARS, SELECTION
from src.evisearch.columns import column_names, extraction_items_schema, fill_missing, parse_column_entries
from src.evisearch.knowledge.preferences import load_extraction_preferences
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services.highlight import resolve_pdf_path
from src.inference import InferenceError, Message, PdfPart, TextPart, Usage, get_chat
from src.retrieval.embedding_retriever import parsed_markdown_path

SYSTEM_PROMPT = """You extract clinical trial data from a research paper.

Use ONLY the document provided. Return a value for every requested column.

Rules:
- Use "Not reported" and found=false when the document does not report the value.
- For N (%) columns include both the count and the percentage when reported.
- Check table and figure captions for scope: use overall/all-patient values for overall columns, and the matching
  subgroup for subgroup columns. When only subgroups are reported and the column asks for the whole population,
  combine the subgroups that make up the whole population.
- Attribution lists the 1-based page number(s) the value came from, with modality "table", "figure" or "text"."""

MARKDOWN_PAGES_NOTE = (
    "Pages in the markdown are separated by <!-- PAGE BREAK -->: page 1 is the text before the first marker, "
    "page 2 the text after it, and so on."
)


def extraction_schema(names: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": {"columns": extraction_items_schema(names)}, "required": ["columns"]}


def build_columns_prompt(batch_columns: List[Dict[str, Any]], preferences: str) -> str:
    blocks = [
        f"---\nColumn {i}: {col.get('column_name', '')}\nDefinition: {col.get('definition', '')}"
        for i, col in enumerate(batch_columns, 1)
    ]
    preference_block = f"\nHUMAN EXTRACTION PREFERENCES:\n{preferences}\n" if preferences else ""
    return (
        f"{preference_block}\nCOLUMNS TO EXTRACT:\n" + "\n".join(blocks) + "\n\n"
        'Return JSON: {"columns": [{"column": <exact column name>, "value": ..., "reasoning": ..., '
        '"found": true|false, "attribution": [{"page": N, "modality": "text"|"table"|"figure"}]}]}'
    )


def _document_parts(doc_id: str, input_mode: str) -> List[Any]:
    if input_mode == "pdf":
        pdf_path = resolve_pdf_path(doc_id)
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found for {doc_id}")
        return [TextPart("DOCUMENT: the attached PDF."), PdfPart(pdf_path.read_bytes(), filename=pdf_path.name)]

    path = parsed_markdown_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(f"Parsed markdown not found: {path}. Prepare the document (LandingAI parse) first.")
    markdown = path.read_text(encoding="utf-8")
    if PDF_QUERY_MAX_MARKDOWN_CHARS and len(markdown) > PDF_QUERY_MAX_MARKDOWN_CHARS:
        markdown = markdown[:PDF_QUERY_MAX_MARKDOWN_CHARS] + "\n\n[... markdown truncated ...]"
    # Document first: requests for the same paper share a prefix, which vLLM prefix caching reuses.
    return [TextPart(f"DOCUMENT MARKDOWN ({MARKDOWN_PAGES_NOTE}):\n\n{markdown}\n\nEND DOCUMENT MARKDOWN.")]


def run_pdf_query(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    *,
    input_mode: Optional[str] = None,
    model: Optional[str] = None,
    preferences: Optional[str] = None,
    raw_response_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Returns ({column: {value, reasoning, found, attribution, tried}}, usage). Never raises for model errors:
    affected columns come back as "Not reported" with the error as reasoning."""
    names = column_names(batch_columns)
    usage = Usage()
    input_mode = input_mode or SELECTION.option("pdf_query_input")

    try:
        chat = get_chat("pdf_query", model)
        if input_mode == "pdf" and not chat.capabilities.pdf:
            raise ConfigError(f"model '{chat.key}' cannot read PDFs; use pdf_query_input=markdown")
        document = _document_parts(doc_id, input_mode)
    except (ConfigError, InferenceError, FileNotFoundError) as exc:
        return fill_missing({}, names, f"pdf_query not run: {exc}"), usage.to_dict()

    prefs = load_extraction_preferences() if preferences is None else preferences
    columns_prompt = build_columns_prompt(batch_columns, prefs)
    messages = [Message.system(SYSTEM_PROMPT), Message.user(*document, columns_prompt)]
    schema = extraction_schema(names) if chat.capabilities.json_schema else None

    log: Dict[str, Any] = {"model": chat.key, "input_mode": input_mode, "system": SYSTEM_PROMPT, "prompt": columns_prompt}
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["pdf_query"])
    except InferenceError as exc:
        log["error"] = str(exc)
        if raw_response_path:
            write_json(raw_response_path, log)
        return fill_missing({}, names, f"pdf_query failed: {exc}"), usage.to_dict()

    usage.add(result.usage)
    log.update({"response_text": result.text, "finish_reason": result.finish_reason, "usage": usage.to_dict()})
    if raw_response_path:
        write_json(raw_response_path, log)

    try:
        parsed = result.json()
    except ValueError as exc:
        return fill_missing({}, names, f"JSON parse error: {exc}"), usage.to_dict()
    results = parse_column_entries(parsed, names)
    return fill_missing(results, names, "Not returned by the model"), usage.to_dict()
