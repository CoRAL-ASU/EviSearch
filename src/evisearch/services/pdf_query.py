"""
Arm A (pdf_query): extract a batch of columns from the whole document in one structured call.

The document is the parsed markdown, page by page. With OPTIONS["pdf_query_input"] = "markdown_images" (the default)
each page's text is followed by its rendered image; with "markdown" the text is sent alone. Every provider gets
exactly this input: the PDF file itself is never uploaded, so local and API models read the same evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.catalog import ConfigError, ImageTokens
from src.config.config import MAX_TOKENS, PAGE_IMAGE_SCALE, PDF_QUERY_MAX_PAGE_IMAGES, SELECTION
from src.evisearch.columns import column_names, extraction_items_schema, fill_missing, parse_column_entries
from src.evisearch.knowledge.preferences import load_extraction_preferences
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services.extraction_rules import shared_rules
from src.evisearch.services.highlight import resolve_pdf_path
from src.evisearch.services.page_images import pdf_page_count, render_pages
from src.inference import ImagePart, InferenceError, Message, TextPart, Usage, get_chat
from src.retrieval.embedding_retriever import parsed_markdown_path
from src.retrieval.markdown_preprocessor import PAGE_BREAK

SYSTEM_PROMPT = """You extract clinical trial data from a research paper.

Use ONLY the document provided. Return a value for every requested column.

Rules:
- Use "Not reported" and found=false when the document does not report the value.
- For N (%) columns include both the count and the percentage when reported.
- Check table and figure captions for scope: use overall/all-patient values for overall columns, and the matching
  subgroup for subgroup columns. When only subgroups are reported and the column asks for the whole population,
  combine the subgroups that make up the whole population.
- Attribution lists the 1-based page number(s) the value came from, with modality "table", "figure" or "text", and
  evidence: the text on that page that supports the value, copied as printed (the sentence; for a table, the row
  label, the column header and the cell; for a figure, its label and what you read from it). Every value is checked
  against the page and evidence you give, so cite the page that actually shows it."""

IMAGE_RULES = """
- Pages come with their parsed text and, where included, their rendered image. Use the parsed text for exact wording
  and numbers in text and tables; use the image for figures (Kaplan-Meier curves, forest plots, flow diagrams) and to
  check table layout. When the parsed text and the image disagree about a value, trust the image."""

def system_prompt_text(images: bool = True) -> str:
    """Agent A's system prompt: base rules, the page-image rules when images are sent, then the shared conventions."""
    return SYSTEM_PROMPT + (IMAGE_RULES if images else "") + shared_rules()


ANCHOR_RE = re.compile(r"<a\s+id=['\"][^'\"]*['\"][^>]*>\s*</a>\s*")
FIGURE_RE = re.compile(r"<::(?!\s*logo)", re.IGNORECASE)  # LandingAI figure descriptions; logos are not evidence
CHARS_PER_TEXT_TOKEN = 2  # benchmark papers measure 2.5-3.4 characters per Qwen or Mistral token; 2 keeps estimates above the real count

# Steps tried in order until the document fits the token budget; None means every page went in with its image.
FALLBACK_STEPS = (None, "figure_table_pages", "markdown_only")
FOLLOW_UPS = 1  # extra calls per batch for columns the reply left out or lost (cut off, unreadable JSON)


@dataclass
class DocumentInput:
    parts: List[Any]
    info: Dict[str, Any]  # input_mode, pages, image_pages, page_image_scale, fallback, estimated_tokens, warnings


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
        '"found": true|false, "attribution": [{"page": N, "modality": "text"|"table"|"figure", "evidence": ...}]}]}'
    )


def load_markdown_pages(doc_id: str) -> List[str]:
    """Parsed markdown split into pages, without LandingAI's <a id> anchors (~15-20% of the tokens)."""
    path = parsed_markdown_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(f"Parsed markdown not found: {path}. Prepare the document (LandingAI parse) first.")
    markdown = ANCHOR_RE.sub("", path.read_text(encoding="utf-8"))
    return [page.strip() for page in markdown.split(PAGE_BREAK)]


def figure_or_table_pages(pages: List[str]) -> List[int]:
    return [number for number, text in enumerate(pages, 1) if FIGURE_RE.search(text) or "<table" in text.lower()]


def document_token_budget(context_tokens: Optional[int], prompt_text: str, max_output_tokens: int) -> Optional[int]:
    """Tokens left for the document once the output and the rest of the prompt are reserved (None: no limit known)."""
    if not context_tokens:
        return None
    return context_tokens - max_output_tokens - len(prompt_text) // CHARS_PER_TEXT_TOKEN


def build_document_input(
    doc_id: str, input_mode: str, token_budget: Optional[int] = None, image_tokens: Optional[ImageTokens] = None
) -> DocumentInput:
    """Document parts for one call. In markdown_images mode every page gets its image; when that would exceed
    token_budget, images are limited to pages with figures or tables, then dropped (info["fallback"] says which).
    image_tokens is the model's image token cost (catalog.yaml models.<key>.image_tokens)."""
    pages = load_markdown_pages(doc_id)
    texts = {number: f"=== PAGE {number}: parsed text ===\n{text}" for number, text in enumerate(pages, 1)}
    text_tokens = sum(len(text) for text in texts.values()) // CHARS_PER_TEXT_TOKEN
    info: Dict[str, Any] = {
        "input_mode": input_mode,
        "pages": len(pages),
        "image_pages": [],
        "page_image_scale": None,
        "fallback": None,
        "estimated_tokens": text_tokens,
        "token_budget": token_budget,
        "warnings": [],
    }
    if input_mode == "markdown":
        parts = [TextPart(f"DOCUMENT: {len(pages)} pages of parsed text.")]
        parts += [TextPart(texts[number]) for number in sorted(texts)]
        return DocumentInput(parts + [TextPart("END OF DOCUMENT.")], info)
    if input_mode != "markdown_images":
        raise ConfigError(f"pdf_query_input={input_mode!r}: choose markdown_images or markdown")

    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found for {doc_id}; markdown_images renders the page images from it")
    pdf_pages = pdf_page_count(Path(pdf_path))
    if pdf_pages != len(pages):
        info["warnings"].append(f"page count mismatch: markdown has {len(pages)} pages, PDF has {pdf_pages}")
    images = {image.page: image for image in render_pages(Path(pdf_path), list(range(1, pdf_pages + 1)), PAGE_IMAGE_SCALE)}
    candidates = {
        None: sorted(images),
        "figure_table_pages": [number for number in figure_or_table_pages(pages) if number in images],
        "markdown_only": [],
    }
    for fallback in FALLBACK_STEPS:
        image_pages = candidates[fallback]
        estimate = text_tokens + sum(images[number].estimated_tokens(image_tokens) for number in image_pages)
        fits = token_budget is None or estimate <= token_budget
        if (fits and len(image_pages) <= PDF_QUERY_MAX_PAGE_IMAGES) or fallback == "markdown_only":
            break
    if token_budget is not None and estimate > token_budget:
        info["warnings"].append(f"document needs ~{estimate} tokens but only {token_budget} are available")
    info.update(image_pages=image_pages, fallback=fallback, estimated_tokens=estimate,
                page_image_scale=PAGE_IMAGE_SCALE if image_pages else None)

    if fallback is None:
        header = f"DOCUMENT: {len(pages)} pages, each given as its parsed text followed by its rendered image."
    elif image_pages:
        header = f"DOCUMENT: {len(pages)} pages of parsed text; rendered images follow the pages with figures or tables ({', '.join(map(str, image_pages))})."
    else:
        header = f"DOCUMENT: {len(pages)} pages of parsed text (page images left out: they do not fit the model's context)."
    parts: List[Any] = [TextPart(header)]
    for number in range(1, max(len(pages), pdf_pages) + 1):
        text = texts.get(number, f"=== PAGE {number}: parsed text ===\n(no parsed text for this page)")
        if number in image_pages:
            parts += [TextPart(f"{text}\n=== PAGE {number}: image ==="), ImagePart(images[number].png)]
        else:
            parts.append(TextPart(text))
    return DocumentInput(parts + [TextPart("END OF DOCUMENT.")], info)


def run_pdf_query(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    *,
    input_mode: Optional[str] = None,
    model: Optional[str] = None,
    preferences: Optional[str] = None,
    raw_response_path: Optional[Path] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Returns ({column: {value, reasoning, found, attribution, tried}}, usage). Never raises for model errors:
    affected columns come back as "Not reported" with the error as reasoning. `details`, when given, receives the
    document info (pages, image pages, fallback, estimated tokens)."""
    names = column_names(batch_columns)
    usage = Usage()
    input_mode = input_mode or SELECTION.option("pdf_query_input")
    prefs = load_extraction_preferences() if preferences is None else preferences
    columns_prompt = build_columns_prompt(batch_columns, prefs)

    try:
        chat = get_chat("pdf_query", model)
        if input_mode == "markdown_images" and not chat.capabilities.images:
            raise ConfigError(f"model '{chat.key}' cannot read images; use pdf_query_input=markdown")
        budget = document_token_budget(chat.spec.context_tokens, system_prompt_text() + columns_prompt, MAX_TOKENS["pdf_query"])
        document = build_document_input(doc_id, input_mode, budget, chat.spec.image_tokens)
    except (ConfigError, InferenceError, FileNotFoundError) as exc:
        return fill_missing({}, names, f"pdf_query not run: {exc}"), usage.to_dict()
    if details is not None:
        details.update(document.info)

    system = system_prompt_text(bool(document.info["image_pages"]))
    log: Dict[str, Any] = {"model": chat.key, "input_mode": input_mode, "document": document.info, "system": system, "prompt": columns_prompt}
    results: Dict[str, Dict[str, Any]] = {}
    pending, prompt, max_tokens = list(batch_columns), columns_prompt, MAX_TOKENS["pdf_query"]
    for attempt in range(1 + FOLLOW_UPS):
        pending_names, reason = column_names(pending), "Not returned by the model"
        # Document first: every batch for the same paper shares this prefix, which prompt caching reuses.
        messages = [Message.system(system), Message.user(*document.parts, prompt)]
        schema = extraction_schema(pending_names) if chat.capabilities.json_schema else None
        entry = log if attempt == 0 else {"prompt": prompt, "columns": pending_names, "max_tokens": max_tokens}
        try:
            result = chat.chat(messages, response_schema=schema, max_tokens=max_tokens)
        except InferenceError as exc:
            entry["error"] = str(exc)
            reason = f"pdf_query failed: {exc}"
        else:
            usage.add(result.usage)
            entry.update({"response_text": result.text, "finish_reason": result.finish_reason, "started_at": result.started_at,
                          "duration_s": result.duration_s, "usage": result.usage.to_dict() if attempt else usage.to_dict()})
            try:
                results.update(parse_column_entries(result.json(), pending_names))
            except ValueError as exc:
                reason = f"JSON parse error: {exc}"
        if attempt:
            log["follow_up"] = entry
            log["usage"] = usage.to_dict()
        if raw_response_path:
            write_json(raw_response_path, log)
        pending = [col for col in batch_columns if col.get("column_name") not in results]
        if not pending or "error" in entry:
            break
        # A reply that left columns out, or was cut off at the token limit (seen: a long deliberation on one column),
        # gets one more call for just the missing columns; a cut-off reply gets twice the budget, within the context.
        prompt = build_columns_prompt(pending, prefs)
        if entry.get("finish_reason") == "length":
            room = (chat.spec.context_tokens or 0) - result.usage.input_tokens - 512
            max_tokens = max(max_tokens, min(2 * max_tokens, room)) if chat.spec.context_tokens else 2 * max_tokens
    return fill_missing(results, names, reason), usage.to_dict()
