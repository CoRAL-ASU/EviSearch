"""
search_agent.py

Search agent that uses OpenAI embedding retriever (parsed_markdown) to find pages,
then extracts values. Tracks pages_sent to avoid re-sending content.
Tools: search_chunks (semantic search), get_chunks_by_page, submit_extraction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from src.config.runtime_paths import RESULTS_ROOT
from src.LLMProvider.google_genai_client import (
    create_vertex_genai_client,
    ensure_genai_modules,
    has_vertex_auth,
    vertex_auth_error_message,
)

MAX_TOOL_CALLS = 15
MAX_TURNS = 25

VALID_MODALITIES = frozenset({"text", "table", "figure"})


def _normalize_attribution(attr: List[Any], found: bool) -> List[Dict[str, Any]]:
    """Convert raw attribution to [{page, modality}]. Default modality='text' if missing."""
    if not found or not isinstance(attr, list):
        return []
    out = []
    for item in attr:
        if not isinstance(item, dict):
            continue
        try:
            page = item.get("page")
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        if page is None or page < 1:
            continue
        mod = str(item.get("modality") or item.get("source_type") or "text").lower()
        if mod not in VALID_MODALITIES:
            mod = "text"
        out.append({"page": page, "modality": mod})
    return out


def _ensure_genai():
    return ensure_genai_modules()


def _run_search_chunks(
    doc_id: str,
    query: str,
    pages_sent: Set[int],
) -> Dict[str, Any]:
    """
    Semantic search. Returns page numbers + full content for pages NOT in pages_sent.
    For pages already sent: "Page N: already provided—check your context."
    """
    from src.retrieval.openai_embedding_retriever import search_chunks as retriever_search

    hits = retriever_search(doc_id, query=query, top_k=5)
    if not hits:
        return {
            "matches": [],
            "formatted_chunks": "No matching chunks found. Try different search terms.",
            "pages_returned": [],
        }

    parts = []
    pages_returned = []
    all_already_sent = True

    for h in hits:
        page = h.get("page", 0)
        text = h.get("text") or ""
        score = h.get("score", 0)

        if page in pages_sent:
            parts.append(f"[Page {page}, score={score:.2f}] already provided—check your context.")
        else:
            all_already_sent = False
            parts.append(f"[Page {page}, score={score:.2f}]\n{text}")
            pages_returned.append(page)

    if all_already_sent:
        formatted = "All retrieved pages have already been provided. Try a different query or submit with what you have."
    else:
        formatted = "\n\n---\n\n".join(parts)

    return {
        "matches": [{"page": h.get("page"), "score": h.get("score")} for h in hits],
        "formatted_chunks": formatted,
        "pages_returned": pages_returned,
    }


def _run_get_chunks_by_page(
    doc_id: str,
    page_numbers: List[int],
    pages_sent: Set[int],
    total_pages: int,
) -> Dict[str, Any]:
    """
    Return content for requested pages. For pages already sent: "already provided."
    For invalid pages: "Page N does not exist. Document has M pages."
    """
    from src.retrieval.openai_embedding_retriever import get_page_content

    content_map = get_page_content(doc_id, page_numbers)
    parts = []
    pages_returned = []

    for p in sorted(set(page_numbers)):
        if p < 1 or p > total_pages:
            parts.append(content_map.get(p, f"Page {p} does not exist. Document has {total_pages} pages."))
        elif p in pages_sent:
            parts.append(f"[Page {p}] already provided—check your context.")
        else:
            parts.append(f"[Page {p}]\n{content_map.get(p, '')}")
            pages_returned.append(p)

    return {
        "formatted_chunks": "\n\n---\n\n".join(parts),
        "pages_returned": pages_returned,
    }


def _build_tool_declarations():
    return [
        {
            "name": "search_chunks",
            "description": "Semantic search over document pages (OpenAI embeddings). Returns top matching pages with full content. Pages you already have will show 'already provided—check your context'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'Median overall survival treatment arm months', 'region demographics N percentage')",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_chunks_by_page",
            "description": "Load full content of specific pages by page number. Use for informational columns (trial name, treatment arm, etc.) that typically appear in the first few pages. Pages you already have will show 'already provided'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers to load (1-based, e.g. [1, 2, 3] for first three pages)",
                    },
                },
                "required": ["page_numbers"],
            },
        },
        {
            "name": "submit_extraction",
            "description": "Submit extracted values when done. Call when you have enough information for all columns (values or 'Not reported'). You may submit again to revise if you find better information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "object",
                        "description": "Map of column_name to {value: string, reasoning: string, found: bool, attribution: [{page: number, modality: 'text'|'table'|'figure'}, ...]}",
                    },
                },
                "required": ["results"],
            },
        },
    ]


def run_search_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    log_path: Optional[Path] = None,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Run search agent for one batch of columns.
    Uses OpenAI retriever for search.
    Returns (results, usage) where results is {column_name: {value, reasoning, found, attribution}}
    and usage is {input_tokens, output_tokens, api_calls, total_tokens}.
    """
    genai, types = _ensure_genai()
    if not has_vertex_auth():
        empty_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}
        reason = vertex_auth_error_message()
        return ({c.get("column_name", ""): {"value": "Not reported", "reasoning": reason, "found": False, "attribution": []} for c in batch_columns}, empty_usage)

    client = create_vertex_genai_client()
    tools = types.Tool(function_declarations=_build_tool_declarations())

    col_blocks = []
    for i, col in enumerate(batch_columns, 1):
        name = col.get("column_name", "")
        defn = definitions_map.get(name, "") or col.get("definition", "")
        col_blocks.append(f"\n---\nColumn {i}: {name}\nDefinition: {defn}")

    from src.retrieval.openai_embedding_retriever import get_total_pages
    total_pages = get_total_pages(doc_id)

    system_instruction = """You extract clinical trial values from document pages.

WORKFLOW (follow this order):
1. Load initial pages: get_chunks_by_page([1, 2]) first.
2. Extract from what you have: Fill as many columns as possible from pages 1–2 before calling search_chunks.
3. Identify gaps: Note which columns are still blank or unclear.
4. Search only for gaps: Call search_chunks only for those specific columns. Do not search for info you may already have.
5. Submit extraction when done.

EXTRACT-FIRST POLICY:
- Do not call search_chunks until you have attempted extraction from the pages you already have.
- Clinical trial papers often have title, authors, endpoints, eligibility, and key design info in the first few pages. Use them first.
- Before each search_chunks call: only use it for columns you cannot find or are unclear in your current content.

DOMAIN POLICY:
- Informational columns (trial name, treatment arm, control arm, phase, design): Start with get_chunks_by_page([1, 2, 3]) where this info usually appears.
- Specific columns (demographics, outcomes, adverse events): Extract from pages 1–3 first; use search_chunks only if still missing.

TABLE SCOPE AND SUBGROUP POLICY:
- Always check table headers for scope: does the table show "All Patients", "Overall", or subgroup-specific columns (e.g. "High Volume", "Low Volume")?
- For columns requesting overall/all population: use the "All Patients" or "Overall" column if present. If not present but subgroups are reported (e.g. High Volume, Low Volume), sum the subgroup values to derive overall (e.g. sum N and recalculate %).
- For columns requesting subgroup-specific data: use the matching subgroup column only.

RULES:
- Do not request pages you already have. We will tell you "already provided—check your context" for pages already sent.
- Submit when you have enough information for all columns (values or "Not reported"). You may submit again to revise if you find better information.
- For N (%) columns include both count and percentage.
- Do NOT include "treatment" or "control" in your search queries as they are generic. Use specific terms (drug names, region names, arm labels, column-specific terms).

Tools:
- search_chunks: semantic search. Only call for columns not found in pages you have. Query = specific column/term you need.
- get_chunks_by_page: load specific pages by number. Use for first pages (trial info) or when you know the page.
- submit_extraction: submit {column: {value, reasoning, found, attribution}} when done.

Attribution: For each column, list sources as [{\"page\": N, \"modality\": \"text\"|\"table\"|\"figure\"}]. Use \"table\" for table content, \"figure\" for figures, \"text\" for prose. If not found: value=\"Not reported\", found=false."""

    config = types.GenerateContentConfig(
        temperature=0.0,
        tools=[tools],
        system_instruction=system_instruction,
    )

    user_prompt = f"""Extract values for these columns. Document has {total_pages} pages.

COLUMNS:
{"".join(col_blocks)}

For informational columns (trial name, arms, etc.), use get_chunks_by_page([1, 2]) first. For specific columns, use search_chunks. Submit when you have enough information."""

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
    pages_sent: Set[int] = set()
    tool_call_count = 0
    submitted_results: Optional[Dict[str, Dict[str, Any]]] = None
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

        # Log model reasoning before tool responses (reasoning step with each call)
        if reasoning_text:
            conversation_log.append({"turn": len(conversation_log) + 1, "role": "model", "content": reasoning_text})

        response_parts = []
        for fc in function_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                break
            tool_call_count += 1
            fc_obj = getattr(fc, "function_call", None)
            name = getattr(fc_obj, "name", None) or ""
            args = getattr(fc_obj, "args", None) or {}

            if name == "submit_extraction":
                res = args.get("results") or {}
                submitted_results = {}
                for k, v in (res.items() if isinstance(res, dict) else []):
                    if isinstance(v, dict):
                        found = bool(v.get("found", True))
                        raw_attr = v.get("attribution") if isinstance(v.get("attribution"), list) else []
                        submitted_results[str(k)] = {
                            "value": str(v.get("value", "Not reported")),
                            "reasoning": str(v.get("reasoning", "")),
                            "found": found,
                            "attribution": _normalize_attribution(raw_attr, found),
                        }
                    else:
                        submitted_results[str(k)] = {"value": str(v), "reasoning": "", "found": True, "attribution": []}
                response_parts.append(types.Part.from_function_response(name=name, response={"submitted": list(submitted_results.keys())}))
                tool_calls_sequence.append({"name": name, "args": {"results_keys": list(submitted_results.keys())}})
                conversation_log.append({"turn": len(conversation_log) + 1, "role": "tool", "name": name, "args": {"results_keys": list(submitted_results.keys())}, "response": {"submitted": list(submitted_results.keys())}})
                break

            if name == "search_chunks":
                query = str(args.get("query", ""))
                result = _run_search_chunks(doc_id, query, pages_sent)
                for p in result.get("pages_returned", []):
                    pages_sent.add(p)
                response_parts.append(types.Part.from_function_response(name=name, response=result))
                tool_calls_sequence.append({"name": name, "args": dict(args)})
                conversation_log.append({"turn": len(conversation_log) + 1, "role": "tool", "name": name, "args": dict(args), "response": dict(result)})
            elif name == "get_chunks_by_page":
                page_nums = args.get("page_numbers") or []
                page_nums = [int(p) for p in page_nums if isinstance(p, (int, float))]
                result = _run_get_chunks_by_page(doc_id, page_nums, pages_sent, total_pages)
                for p in result.get("pages_returned", []):
                    pages_sent.add(p)
                response_parts.append(types.Part.from_function_response(name=name, response=result))
                tool_calls_sequence.append({"name": name, "args": dict(args)})
                conversation_log.append({"turn": len(conversation_log) + 1, "role": "tool", "name": name, "args": dict(args), "response": dict(result)})
            else:
                response_parts.append(types.Part.from_function_response(name=name, response={"error": "Unknown tool"}))
                tool_calls_sequence.append({"name": name, "args": dict(args)})

        if submitted_results is not None:
            break

        if response_parts:
            contents.append(cand.content)
            reason_prompt = types.Part.from_text(
                text="Summarize what you learned. Then: search for more columns, load more pages, or call submit_extraction when you have enough information."
            )
            contents.append(types.Content(role="user", parts=response_parts + [reason_prompt]))

    if submitted_results is None:
        submitted_results = {c.get("column_name", ""): {"value": "Not reported", "reasoning": "Agent did not submit", "found": False, "attribution": []} for c in batch_columns}

    # Fill any missing columns
    for c in batch_columns:
        name = c.get("column_name", "")
        if name and name not in submitted_results:
            submitted_results[name] = {"value": "Not reported", "reasoning": "Not extracted", "found": False, "attribution": []}

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
