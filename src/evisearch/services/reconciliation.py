"""
Reconciliation agent: resolve Arm A (agent_extractor) vs Arm B (search_agent) per column.

Tools: get_page (page text, plus page images when the model accepts images) and submit_verification.
The model comes from the "reconciliation" role in src/config/config.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config.catalog import ConfigError
from src.config.config import (
    AGENT_MAX_TOOL_CALLS,
    AGENT_MAX_TURNS,
    MAX_TOKENS,
    PAGE_IMAGE_SCALE,
    RECONCILIATION_MAX_PAGE_IMAGES,
    SELECTION,
)
from src.evisearch.columns import MODALITIES, NOT_REPORTED, column_names
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services.highlight import resolve_pdf_path
from src.evisearch.services.page_images import render_pdf_pages_to_png
from src.inference import ImagePart, InferenceError, TextPart, Tool, ToolOutput, ToolSpec, Usage, get_chat, run_tool_loop
from src.retrieval import embedding_retriever as retriever

VERIFICATIONS = ("A_correct_B_wrong", "B_correct_A_wrong", "both_correct", "both_wrong")

SYSTEM_PROMPT = """You reconcile extractions from two sources (A and B) for clinical trial columns.

WORKFLOW:
1. First pass (no get_page): Submit verification immediately ONLY for:
   - Both "Not reported" / "Not found" -> value="Not reported", verification=both_correct
   - Same value -> both_correct
   - BOTH have values, same page/modality, one is superset of the other (e.g. "67.0 (42-85)" vs "67.0") -> keep more complete, verification=both_correct
2. Second pass (MUST use get_page): For ANY column where:
   - One has a value and the other is "Not reported" / empty -> YOU MUST call get_page to verify. Do NOT submit A_correct_B_wrong or B_correct_A_wrong without first fetching and checking the page.
   - Both have values but they differ (conflicting numbers, different text) -> call get_page to resolve
   - Use page numbers from the source that has the value (or both if both have pages)
   - Inspect text and page images if available, then submit verification
3. If get_page does not clarify: pick best guess and submit. Do not loop indefinitely.

CRITICAL: When one source has a value and the other says "Not reported", you MUST call get_page before submitting. The source with the value may have extracted from the wrong row or hallucinated; verify against the document first.

VERIFICATION: A_correct_B_wrong | B_correct_A_wrong | both_correct | both_wrong

SOURCE: When value is not "Not reported", include source: {page: N, modality: "text"|"table"|"figure"}. When modality is "text", also include verbatim_quote: the exact sentence or phrase from the document that supports the value (copy verbatim from the page content you saw via get_page). This supports attribution.

Prefer get_page for pages from A and B. Do not request pages you already have."""

FOLLOW_UP = "Continue. Submit verification for any columns you can resolve. Use get_page for more pages if needed. Submit all columns when done."


def _extract_source_output(col_data: Any) -> Dict[str, Any]:
    """Normalize an A/B column result to {value, page, modality, reasoning}; page from the first attribution."""
    if not isinstance(col_data, dict):
        return {"value": NOT_REPORTED, "page": None, "modality": "text", "reasoning": ""}
    value = col_data.get("value") or NOT_REPORTED
    reasoning = str(col_data.get("reasoning") or "").strip()[:400]
    attribution = col_data.get("attribution") or []
    page, modality = None, "text"
    if attribution and isinstance(attribution, list) and isinstance(attribution[0], dict):
        page = attribution[0].get("page")
        modality = attribution[0].get("modality") or attribution[0].get("source_type") or "text"
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    if page is not None and page < 1:
        page = None
    modality = str(modality).lower()
    return {"value": str(value), "page": page, "modality": modality if modality in MODALITIES else "text", "reasoning": reasoning}


def normalize_verification(raw: Dict[str, Any]) -> Dict[str, Any]:
    verification = str(raw.get("verification", "both_wrong")).strip()
    if verification not in VERIFICATIONS:
        verification = "both_wrong"
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    try:
        page = int(source.get("page")) if str(source.get("page", "")).strip() else None
    except (TypeError, ValueError):
        page = None
    if page is not None and page < 1:
        page = None
    modality = str(source.get("modality") or source.get("source_type") or "text").lower()
    if modality not in MODALITIES:
        modality = "text"
    quote = str(source.get("verbatim_quote") or "").strip()
    source_obj: Dict[str, Any] = {"page": page, "modality": modality}
    if quote:
        source_obj["verbatim_quote"] = quote
    return {
        "value": str(raw.get("value", NOT_REPORTED)),
        "reasoning": str(raw.get("reasoning", "")),
        "verification": verification,
        "source": source_obj,
        "attribution": [dict(source_obj)] if page or quote else [],
    }


def unresolved(reason: str) -> Dict[str, Any]:
    return {"value": NOT_REPORTED, "reasoning": reason, "verification": "both_wrong", "source": {"page": None, "modality": "text"}, "attribution": []}


def tool_specs(names: List[str]) -> List[ToolSpec]:
    return [
        ToolSpec(
            name="get_page",
            description="Load the full content (and page images when supported) of specific pages. Use when A and B disagree. Pages you already have show 'already provided'.",
            parameters={
                "type": "object",
                "properties": {"page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "1-based page numbers"}},
                "required": ["page_numbers"],
            },
        ),
        ToolSpec(
            name="submit_verification",
            description="Submit reconciled values for one or more columns. Call it several times: easy columns first, disputed columns after get_page.",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names)},
                                "value": {"type": "string"},
                                "reasoning": {"type": "string"},
                                "verification": {"type": "string", "enum": list(VERIFICATIONS)},
                                "source": {
                                    "type": "object",
                                    "properties": {
                                        "page": {"type": "integer"},
                                        "modality": {"type": "string", "enum": list(MODALITIES)},
                                        "verbatim_quote": {"type": "string", "description": "Exact supporting sentence when modality is text"},
                                    },
                                },
                            },
                            "required": ["column", "value", "reasoning", "verification"],
                        },
                    }
                },
                "required": ["results"],
            },
        ),
    ]


class _ReconciliationSession:
    def __init__(self, doc_id: str, names: List[str], total_pages: int, image_scale: Optional[float]):
        self.doc_id = doc_id
        self.names = names
        self.total_pages = total_pages
        self.image_scale = image_scale
        self.pdf_path = resolve_pdf_path(doc_id) if image_scale else None
        self.pages_sent: Set[int] = set()
        self.images_sent = 0
        self.submitted: Dict[str, Dict[str, Any]] = {}

    def get_page(self, args: Dict[str, Any]) -> ToolOutput:
        pages = sorted({int(p) for p in args.get("page_numbers") or [] if isinstance(p, (int, float))})
        content_map = retriever.get_page_content(self.doc_id, pages)
        parts, returned = [], []
        for page in pages:
            if page < 1 or page > self.total_pages:
                parts.append(f"[Page {page}] Page {page} does not exist. Document has {self.total_pages} pages.")
            elif page in self.pages_sent:
                parts.append(f"[Page {page}] already provided; check your context.")
            else:
                parts.append(f"[Page {page}]\n{content_map.get(page, '')}")
                returned.append(page)
        self.pages_sent.update(returned)

        attachments: List[Any] = []
        if self.image_scale and self.pdf_path and self.pdf_path.exists():
            budget = max(RECONCILIATION_MAX_PAGE_IMAGES - self.images_sent, 0)
            for page_number, png in render_pdf_pages_to_png(self.pdf_path, returned[:budget], self.image_scale):
                attachments += [TextPart(f"[Image: Page {page_number}]"), ImagePart(png)]
                self.images_sent += 1

        def forget() -> None:
            self.pages_sent.difference_update(returned)

        return ToolOutput({"formatted_chunks": "\n\n---\n\n".join(parts), "pages_returned": returned}, attachments, on_evict=forget)

    def submit_verification(self, args: Dict[str, Any]) -> ToolOutput:
        raw = args.get("results")
        entries: List[Tuple[str, Any]] = []
        if isinstance(raw, list):
            entries = [(item.get("column"), item) for item in raw if isinstance(item, dict)]
        elif isinstance(raw, dict):
            entries = list(raw.items())
        accepted = []
        for name, item in entries:
            if name not in self.names or name in self.submitted:
                continue
            self.submitted[name] = normalize_verification(item if isinstance(item, dict) else {"value": item, "verification": "both_correct"})
            accepted.append(name)
        remaining = [n for n in self.names if n not in self.submitted]
        return ToolOutput({"submitted": accepted, "remaining": remaining})

    def done(self) -> bool:
        return all(name in self.submitted for name in self.names)


def run_reconciliation_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    source_a_data: Dict[str, Dict[str, Any]],
    source_b_data: Dict[str, Dict[str, Any]],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Reconcile one batch. Returns ({column: {value, reasoning, verification, source, attribution}}, usage)."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("reconciliation", model)
    except (ConfigError, InferenceError) as exc:
        return {name: unresolved(f"reconciliation not run: {exc}") for name in names}, Usage().to_dict()

    use_images = SELECTION.option("reconciliation_page_images") == "auto" and chat.capabilities.images
    total_pages = retriever.get_total_pages(doc_id)
    session = _ReconciliationSession(doc_id, names, total_pages, PAGE_IMAGE_SCALE if use_images else None)

    blocks = []
    for i, col in enumerate(batch_columns, 1):
        name = col.get("column_name", "")
        a, b = _extract_source_output(source_a_data.get(name)), _extract_source_output(source_b_data.get(name))
        blocks.append(
            f"\n---\nColumn {i}: {name}\nDefinition: {definitions_map.get(name, '') or col.get('definition', '')}\n"
            f"  A: value=\"{a['value'][:200]}\" | page={a['page']} | modality={a['modality']} | reasoning=\"{a['reasoning'].replace(chr(34), chr(39))[:300]}\"\n"
            f"  B: value=\"{b['value'][:200]}\" | page={b['page']} | modality={b['modality']} | reasoning=\"{b['reasoning'].replace(chr(34), chr(39))[:300]}\"\n"
        )
    user_prompt = (
        f"Reconcile the following columns. Document has {total_pages} pages. Sources are anonymous (A and B).\n\n"
        f"COLUMNS:\n{''.join(blocks)}\n\nSubmit verification for easy columns first. For disputed columns, use get_page then submit."
    )
    specs = {spec.name: spec for spec in tool_specs(names)}
    loop = run_tool_loop(
        chat,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        tools=[Tool(specs["get_page"], session.get_page), Tool(specs["submit_verification"], session.submit_verification)],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["reconciliation"],
        follow_up=FOLLOW_UP,
        is_done=session.done,
        finish_tool="submit_verification",
    )

    results = dict(session.submitted)
    reason = "Agent did not resolve before limits" + (f": {loop.error}" if loop.error else "")
    for name in names:
        results.setdefault(name, unresolved(reason))

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "stopped_by": loop.stopped_by,
                "error": loop.error,
                "page_images": session.images_sent,
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in loop.transcript if e["role"] == "tool"],
                "conversation": loop.transcript,
                "calls": loop.calls,
                "results": results,
            },
        )
    return results, loop.usage.to_dict()
