"""
reconciliation_agent.py

Reconcile Agent Extractor (A) vs Search Agent (B) per column.
Tools: get_page (text + page images), submit_verification.
Phase 1: No search_chunks - get_page and submit_verification only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config.runtime_paths import DATASET_DIR, RESULTS_ROOT
from src.LLMProvider.google_genai_client import (
    create_vertex_genai_client,
    ensure_genai_modules,
    has_vertex_auth,
    vertex_auth_error_message,
)

MAX_TOOL_CALLS = 15
MAX_TURNS = 25
PIXMAP_RESOLUTION = 8  # High resolution for page images

VALID_MODALITIES = frozenset({"text", "table", "figure"})
VALID_VERIFICATION = frozenset({"A_correct_B_wrong", "B_correct_A_wrong", "both_correct", "both_wrong"})


def resolve_pdf_path(doc_id: str) -> Optional[Path]:
    """Resolve PDF path for doc_id."""
    for base in (RESULTS_ROOT / doc_id, DATASET_DIR):
        if not base.exists():
            continue
        for p in base.glob("**/*.pdf"):
            if p.stem.replace("'", "'") == doc_id.replace("'", "'"):
                return p
        for p in base.glob("*.pdf"):
            if p.stem in doc_id or doc_id in p.stem:
                return p
    return None


def render_pdf_pages_to_png(pdf_path: Path, page_numbers: List[int]) -> List[tuple]:
    """
    Render PDF pages to PNG bytes. Returns [(page_num, bytes), ...] for valid pages.
    """
    import fitz
    doc = fitz.open(pdf_path)
    out = []
    for p in page_numbers:
        if p < 1 or p > len(doc):
            continue
        page = doc[p - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(PIXMAP_RESOLUTION, PIXMAP_RESOLUTION))
        img_bytes = pix.tobytes("png")
        out.append((p, img_bytes))
    doc.close()
    return out


def _ensure_genai():
    return ensure_genai_modules()


def _run_get_page(
    doc_id: str,
    page_numbers: List[int],
    pages_sent: Set[int],
    total_pages: int,
    pdf_path: Optional[Path],
) -> tuple:
    """
    Return content for requested pages: text (from markdown) + page images.
    For pages already sent: "already provided—check your context."
    Returns (response_dict, list of Parts for user message including images).
    """
    from src.retrieval.openai_embedding_retriever import get_page_content

    content_map = get_page_content(doc_id, page_numbers)
    text_parts = []
    pages_returned = []

    for p in sorted(set(page_numbers)):
        if p < 1 or p > total_pages:
            text_parts.append(f"[Page {p}] Page {p} does not exist. Document has {total_pages} pages.")
        elif p in pages_sent:
            text_parts.append(f"[Page {p}] already provided—check your context.")
        else:
            text_parts.append(f"[Page {p}]\n{content_map.get(p, '')}")
            pages_returned.append(p)

    response_dict = {
        "formatted_chunks": "\n\n---\n\n".join(text_parts),
        "pages_returned": pages_returned,
    }

    # Build Parts: function response + images for new pages
    types = _ensure_genai()[1]
    parts_for_turn: List[Any] = [types.Part.from_function_response(name="get_page", response=response_dict)]

    if pdf_path and pdf_path.exists() and pages_returned:
        images = render_pdf_pages_to_png(pdf_path, pages_returned)
        for page_num, img_bytes in images:
            parts_for_turn.append(
                types.Part.from_text(text=f"\n[Image: Page {page_num}]\n")
            )
            parts_for_turn.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            )

    return response_dict, parts_for_turn


def _normalize_source(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract page and modality from attribution or single source."""
    if not item:
        return {"page": None, "modality": "text"}
    page = item.get("page")
    if page is not None and page != "Not applicable":
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    mod = str(item.get("modality") or item.get("source_type") or "text").lower()
    if mod not in VALID_MODALITIES:
        mod = "text"
    return {"page": page if page and page >= 1 else None, "modality": mod}


def _build_tool_declarations():
    return [
        {
            "name": "get_page",
            "description": "Load full content and images of specific pages by page number. Use when A and B disagree to verify the correct value. Pages you already have will show 'already provided'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers to load (1-based, e.g. [1, 2, 3])",
                    },
                },
                "required": ["page_numbers"],
            },
        },
        {
            "name": "submit_verification",
            "description": "Submit reconciled value and verification for one or more columns. Call multiple times: first for easy columns (match, not reported, more complete), then after get_page for disputed columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "object",
                        "description": "Map of column_name to {value, reasoning, verification, source}. verification: A_correct_B_wrong | B_correct_A_wrong | both_correct | both_wrong. source: {page, modality, verbatim_quote?} when value is not 'Not reported'. When modality is 'text', include verbatim_quote: the exact quote/sentence from the document that supports the value (copy from page text seen via get_page).",
                    },
                },
                "required": ["results"],
            },
        },
    ]


def _load_source_data(doc_id: str, source: str) -> Dict[str, Dict[str, Any]]:
    """Load Agent (A) or Search Agent (B) extraction results."""
    if source == "A":
        path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    else:
        path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("columns", {}) or {}
    except Exception:
        return {}


def _extract_source_output(col_data: Any) -> Dict[str, Any]:
    """Normalize column data to {value, page, modality, reasoning}. Page from first attribution."""
    if not isinstance(col_data, dict):
        return {"value": "Not reported", "page": None, "modality": "text", "reasoning": ""}
    val = col_data.get("value") or "Not reported"
    reasoning = str(col_data.get("reasoning") or "").strip()[:400]
    attr = col_data.get("attribution") or []
    page, modality = None, "text"
    if attr and isinstance(attr, list):
        first = attr[0] if isinstance(attr[0], dict) else {}
        page = first.get("page")
        modality = first.get("modality") or first.get("source_type") or "text"
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    if page is None or page < 1:
        page = None
    if str(modality).lower() not in VALID_MODALITIES:
        modality = "text"
    return {"value": str(val), "page": page, "modality": str(modality).lower(), "reasoning": reasoning}


def run_reconciliation_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    source_a_data: Dict[str, Dict[str, Any]],
    source_b_data: Dict[str, Dict[str, Any]],
    log_path: Optional[Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Run reconciliation agent for one batch.
    Returns (results, usage) where results is {column_name: {value, reasoning, verification, source, attribution}}
    and usage is {input_tokens, output_tokens, api_calls, total_tokens}.
    """
    genai, types = _ensure_genai()
    if not has_vertex_auth():
        empty_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}
        reason = vertex_auth_error_message()
        return ({
            c.get("column_name", ""): {
                "value": "Not reported",
                "reasoning": reason,
                "verification": "both_wrong",
                "source": {"page": None, "modality": "text"},
                "attribution": [],
            }
            for c in batch_columns
        }, empty_usage)

    pdf_path = resolve_pdf_path(doc_id)
    from src.retrieval.openai_embedding_retriever import get_total_pages
    total_pages = get_total_pages(doc_id)

    client = create_vertex_genai_client()
    tools = types.Tool(function_declarations=_build_tool_declarations())

    col_blocks = []
    for i, col in enumerate(batch_columns, 1):
        name = col.get("column_name", "")
        defn = definitions_map.get(name, "") or col.get("definition", "")
        a_out = _extract_source_output(source_a_data.get(name))
        b_out = _extract_source_output(source_b_data.get(name))
        a_reason = (a_out.get("reasoning") or "").replace('"', "'")[:300]
        b_reason = (b_out.get("reasoning") or "").replace('"', "'")[:300]
        col_blocks.append(
            f"\n---\nColumn {i}: {name}\nDefinition: {defn}\n"
            f"  A: value=\"{a_out['value'][:200]}\" | page={a_out['page']} | modality={a_out['modality']} | reasoning=\"{a_reason}\"\n"
            f"  B: value=\"{b_out['value'][:200]}\" | page={b_out['page']} | modality={b_out['modality']} | reasoning=\"{b_reason}\"\n"
        )

    system_instruction = """You reconcile extractions from two sources (A and B) for clinical trial columns.

WORKFLOW:
1. First pass (no get_page): Submit verification immediately ONLY for:
   - Both "Not reported" / "Not found" → value="Not reported", verification=both_correct
   - Same value → both_correct
   - BOTH have values, same page/modality, one is superset of the other (e.g. "67.0 (42-85)" vs "67.0") → keep more complete, verification=both_correct
2. Second pass (MUST use get_page): For ANY column where:
   - One has a value and the other is "Not reported" / empty → YOU MUST call get_page to verify. Do NOT submit A_correct_B_wrong or B_correct_A_wrong without first fetching and checking the page.
   - Both have values but they differ (conflicting numbers, different text) → call get_page to resolve
   - Use page numbers from the source that has the value (or both if both have pages)
   - Inspect text and page images, then submit verification
3. If get_page does not clarify: pick best guess and submit. Do not loop indefinitely.

CRITICAL: When one source has a value and the other says "Not reported", you MUST call get_page before submitting. The source with the value may have extracted from the wrong row or hallucinated—verify against the document first.

VERIFICATION: A_correct_B_wrong | B_correct_A_wrong | both_correct | both_wrong

SOURCE: When value is not "Not reported", include source: {page: N, modality: "text"|"table"|"figure"}. When modality is "text", also include verbatim_quote: the exact sentence or phrase from the document that supports the value (copy verbatim from the page content you saw via get_page). This supports attribution.

Prefer get_page for pages from A and B. Do not request pages you already have."""

    config = types.GenerateContentConfig(
        temperature=0.0,
        tools=[tools],
        system_instruction=system_instruction,
    )

    user_prompt = f"""Reconcile the following columns. Document has {total_pages} pages. Sources are anonymous (A and B).

COLUMNS:
{"".join(col_blocks)}

Submit verification for easy columns first. For disputed columns, use get_page then submit."""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
    pages_sent: Set[int] = set()
    tool_call_count = 0
    submitted_results: Dict[str, Dict[str, Any]] = {}
    conversation_log: List[Dict[str, Any]] = [{"turn": 0, "role": "user", "content": user_prompt}]
    tool_calls_sequence: List[Dict[str, Any]] = []
    total_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    for turn in range(MAX_TURNS):
        if tool_call_count >= MAX_TOOL_CALLS:
            break

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        # Accumulate token usage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_usage["input_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
            total_usage["output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0
        total_usage["api_calls"] += 1

        if not response.candidates:
            break
        cand = response.candidates[0]
        if not cand.content:
            break
        parts = cand.content.parts or []
        function_calls = [p for p in parts if hasattr(p, "function_call") and getattr(p, "function_call", None)]
        reasoning_text = (response.text or "").strip()

        if not function_calls:
            conversation_log.append({"turn": len(conversation_log) + 1, "role": "model", "content": reasoning_text})
            break

        if reasoning_text:
            conversation_log.append({"turn": len(conversation_log) + 1, "role": "model", "content": reasoning_text})

        response_parts: List[Any] = []
        for fc in function_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                break
            tool_call_count += 1
            fc_obj = getattr(fc, "function_call", None)
            name = getattr(fc_obj, "name", None) or ""
            args = getattr(fc_obj, "args", None) or {}

            if name == "submit_verification":
                res = args.get("results") or {}
                batch_col_names = {c.get("column_name", "") for c in batch_columns}
                for k, v in (res.items() if isinstance(res, dict) else []):
                    if k not in batch_col_names or k in submitted_results:
                        continue
                    if isinstance(v, dict):
                        verif = str(v.get("verification", "both_wrong")).strip()
                        if verif not in VALID_VERIFICATION:
                            verif = "both_wrong"
                        src = v.get("source") or {}
                        page = src.get("page") if isinstance(src, dict) else None
                        mod = (src.get("modality") or "text") if isinstance(src, dict) else "text"
                        if mod not in VALID_MODALITIES:
                            mod = "text"
                        verbatim = (src.get("verbatim_quote") or "").strip() if isinstance(src, dict) else ""
                        try:
                            page_int = int(page) if page is not None and str(page).strip() else None
                        except (TypeError, ValueError):
                            page_int = None
                        if page_int is not None and page_int < 1:
                            page_int = None
                        source_obj = {"page": page_int, "modality": mod}
                        if verbatim:
                            source_obj["verbatim_quote"] = verbatim
                        attr_item = {"page": page_int, "modality": mod}
                        if verbatim:
                            attr_item["verbatim_quote"] = verbatim
                        submitted_results[str(k)] = {
                            "value": str(v.get("value", "Not reported")),
                            "reasoning": str(v.get("reasoning", "")),
                            "verification": verif,
                            "source": source_obj,
                            "attribution": [attr_item] if page_int or verbatim else [],
                        }
                    else:
                        submitted_results[str(k)] = {
                            "value": str(v),
                            "reasoning": "",
                            "verification": "both_correct",
                            "source": {"page": None, "modality": "text"},
                            "attribution": [],
                        }
                response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"submitted": list(res.keys()) if isinstance(res, dict) else []},
                    )
                )
                tool_calls_sequence.append({"name": name, "args": {"results_keys": list(res.keys()) if isinstance(res, dict) else []}})
                conversation_log.append({
                    "turn": len(conversation_log) + 1,
                    "role": "tool",
                    "name": name,
                    "args": {"results_keys": list(res.keys()) if isinstance(res, dict) else []},
                    "response": {"submitted": list(res.keys()) if isinstance(res, dict) else []},
                })

            elif name == "get_page":
                page_nums = args.get("page_numbers") or []
                page_nums = [int(p) for p in page_nums if isinstance(p, (int, float))]
                resp_dict, parts_for_turn = _run_get_page(
                    doc_id, page_nums, pages_sent, total_pages, pdf_path
                )
                for p in resp_dict.get("pages_returned", []):
                    pages_sent.add(p)
                for part in parts_for_turn:
                    response_parts.append(part)
                tool_calls_sequence.append({"name": name, "args": dict(args)})
                conversation_log.append({"turn": len(conversation_log) + 1, "role": "tool", "name": name, "args": dict(args), "response": resp_dict})
            else:
                response_parts.append(types.Part.from_function_response(name=name, response={"error": "Unknown tool"}))
                tool_calls_sequence.append({"name": name, "args": dict(args)})

        if response_parts:
            contents.append(cand.content)
            continuation = types.Part.from_text(
                text="Continue. Submit verification for any columns you can resolve. Use get_page for more pages if needed. Submit all columns when done."
            )
            contents.append(types.Content(role="user", parts=response_parts + [continuation]))

        if all(c.get("column_name", "") in submitted_results for c in batch_columns):
            break

    # Fill missing columns
    for c in batch_columns:
        name = c.get("column_name", "")
        if name and name not in submitted_results:
            submitted_results[name] = {
                "value": "Not reported",
                "reasoning": "Agent did not resolve before limits",
                "verification": "both_wrong",
                "source": {"page": None, "modality": "text"},
                "attribution": [],
            }

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        conv_path = log_path.with_name(log_path.stem + "_conversation.json")
        conv_path.write_text(
            json.dumps({
                "doc_id": doc_id,
                "tool_calls_sequence": tool_calls_sequence,
                "conversation": conversation_log,
                "results": submitted_results,
            }, indent=2, default=str),
            encoding="utf-8",
        )

    total_usage["total_tokens"] = total_usage["input_tokens"] + total_usage["output_tokens"]
    return submitted_results, total_usage
