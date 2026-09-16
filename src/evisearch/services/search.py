"""
Arm B (search_agent): a tool-using agent that finds evidence page by page and submits column values.

Tools: search_chunks (embedding search, reranked when a reranker is selected), get_chunks_by_page,
submit_extraction. The model comes from the "search_agent" role in src/config/config.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config.catalog import ConfigError
from src.config.config import AGENT_MAX_TOOL_CALLS, AGENT_MAX_TURNS, MAX_TOKENS
from src.evisearch.columns import column_names, extraction_items_schema, fill_missing, parse_column_entries
from src.evisearch.pipelines.results_store import write_json
from src.inference import InferenceError, Tool, ToolOutput, ToolSpec, Usage, get_chat, run_tool_loop
from src.retrieval import embedding_retriever as retriever

SYSTEM_PROMPT = """You extract clinical trial values from document pages.

WORKFLOW (follow this order):
1. Load initial pages: get_chunks_by_page([1, 2]) first.
2. Extract from what you have: Fill as many columns as possible from pages 1-2 before calling search_chunks.
3. Identify gaps: Note which columns are still blank or unclear.
4. Search only for gaps: Call search_chunks only for those specific columns. Do not search for info you may already have.
5. Submit extraction when done.

EXTRACT-FIRST POLICY:
- Do not call search_chunks until you have attempted extraction from the pages you already have.
- Clinical trial papers often have title, authors, endpoints, eligibility, and key design info in the first few pages. Use them first.
- Before each search_chunks call: only use it for columns you cannot find or are unclear in your current content.

DOMAIN POLICY:
- Informational columns (trial name, treatment arm, control arm, phase, design): Start with get_chunks_by_page([1, 2, 3]) where this info usually appears.
- Specific columns (demographics, outcomes, adverse events): Extract from pages 1-3 first; use search_chunks only if still missing.

TABLE SCOPE AND SUBGROUP POLICY:
- Always check table headers for scope: does the table show "All Patients", "Overall", or subgroup-specific columns (e.g. "High Volume", "Low Volume")?
- For columns requesting overall/all population: use the "All Patients" or "Overall" column if present. If not present but subgroups are reported (e.g. High Volume, Low Volume), sum the subgroup values to derive overall (e.g. sum N and recalculate %).
- For columns requesting subgroup-specific data: use the matching subgroup column only.

RULES:
- Do not request pages you already have. We will tell you "already provided; check your context" for pages already sent.
- Submit when you have enough information for all columns (values or "Not reported").
- For N (%) columns include both count and percentage.
- Do NOT include "treatment" or "control" in your search queries as they are generic. Use specific terms (drug names, region names, arm labels, column-specific terms).

Attribution: For each column, list sources as [{"page": N, "modality": "text"|"table"|"figure"}]. Use "table" for table content, "figure" for figures, "text" for prose. If not found: value="Not reported", found=false."""

FOLLOW_UP = "Summarize what you learned. Then: search for more columns, load more pages, or call submit_extraction when you have enough information."


def tool_specs(names: List[str]) -> List[ToolSpec]:
    return [
        ToolSpec(
            name="search_chunks",
            description="Semantic search over document pages. Returns the best matching pages with full content. Pages you already have show 'already provided; check your context'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'median overall survival abiraterone months'"},
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="get_chunks_by_page",
            description="Load the full content of specific pages. Use for informational columns (trial name, arms) that appear in the first pages, or when you know the page.",
            parameters={
                "type": "object",
                "properties": {
                    "page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "1-based page numbers, e.g. [1, 2, 3]"},
                },
                "required": ["page_numbers"],
            },
        ),
        ToolSpec(
            name="submit_extraction",
            description="Submit the extracted values for all columns (values or 'Not reported'). Ends the task.",
            parameters={
                "type": "object",
                "properties": {"results": extraction_items_schema(names)},
                "required": ["results"],
            },
        ),
    ]


class _SearchSession:
    def __init__(self, doc_id: str, names: List[str], total_pages: int):
        self.doc_id = doc_id
        self.names = names
        self.total_pages = total_pages
        self.pages_sent: Set[int] = set()
        self.submitted: Optional[Dict[str, Dict[str, Any]]] = None

    def _forget(self, pages: List[int]):
        return lambda: self.pages_sent.difference_update(pages)

    def search_chunks(self, args: Dict[str, Any]) -> ToolOutput:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutput({"error": "query is required"})
        hits = retriever.search_chunks(self.doc_id, query)
        if not hits:
            return ToolOutput({"matches": [], "formatted_chunks": "No matching chunks found. Try different search terms.", "pages_returned": []})
        parts, returned = [], []
        for hit in hits:
            page = hit["page"]
            if page in self.pages_sent:
                parts.append(f"[Page {page}, score={hit['score']:.2f}] already provided; check your context.")
            else:
                parts.append(f"[Page {page}, score={hit['score']:.2f}]\n{hit['text']}")
                returned.append(page)
        self.pages_sent.update(returned)
        formatted = (
            "\n\n---\n\n".join(parts)
            if returned
            else "All retrieved pages have already been provided. Try a different query or submit with what you have."
        )
        content = {
            "matches": [{"page": h["page"], "score": h["score"]} for h in hits],
            "retrieval": hits[0]["retrieval"],  # "rerank", or "embedding (reranker unavailable: ...)"
            "formatted_chunks": formatted,
            "pages_returned": returned,
        }
        return ToolOutput(content, on_evict=self._forget(returned))

    def get_chunks_by_page(self, args: Dict[str, Any]) -> ToolOutput:
        pages = sorted({int(p) for p in args.get("page_numbers") or [] if isinstance(p, (int, float))})
        content_map = retriever.get_page_content(self.doc_id, pages)
        parts, returned = [], []
        for page in pages:
            if page < 1 or page > self.total_pages:
                parts.append(content_map.get(page, f"Page {page} does not exist. Document has {self.total_pages} pages."))
            elif page in self.pages_sent:
                parts.append(f"[Page {page}] already provided; check your context.")
            else:
                parts.append(f"[Page {page}]\n{content_map.get(page, '')}")
                returned.append(page)
        self.pages_sent.update(returned)
        return ToolOutput({"formatted_chunks": "\n\n---\n\n".join(parts), "pages_returned": returned}, on_evict=self._forget(returned))

    def submit_extraction(self, args: Dict[str, Any]) -> ToolOutput:
        self.submitted = parse_column_entries(args, self.names, list_key="results")
        return ToolOutput({"submitted": sorted(self.submitted)}, stop=True)


def run_search_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Run the search agent for one batch. Returns ({column: result}, usage); never raises for model errors."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("search_agent", model)
    except (ConfigError, InferenceError) as exc:
        return fill_missing({}, names, f"search_agent not run: {exc}"), Usage().to_dict()

    total_pages = retriever.get_total_pages(doc_id)
    session = _SearchSession(doc_id, names, total_pages)
    blocks = "".join(
        f"\n---\nColumn {i}: {col.get('column_name', '')}\nDefinition: {definitions_map.get(col.get('column_name', ''), '') or col.get('definition', '')}"
        for i, col in enumerate(batch_columns, 1)
    )
    user_prompt = (
        f"Extract values for these columns. Document has {total_pages} pages.\n\nCOLUMNS:\n{blocks}\n\n"
        "For informational columns (trial name, arms, etc.), use get_chunks_by_page([1, 2]) first. "
        "For specific columns, use search_chunks. Submit when you have enough information."
    )
    specs = {spec.name: spec for spec in tool_specs(names)}
    loop = run_tool_loop(
        chat,
        system=SYSTEM_PROMPT,
        user=user_prompt,
        tools=[
            Tool(specs["search_chunks"], session.search_chunks),
            Tool(specs["get_chunks_by_page"], session.get_chunks_by_page),
            Tool(specs["submit_extraction"], session.submit_extraction),
        ],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["search_agent"],
        follow_up=FOLLOW_UP,
        finish_tool="submit_extraction",
    )

    if session.submitted is None:
        reason = f"Agent did not submit ({loop.stopped_by}{': ' + loop.error if loop.error else ''})"
        results = fill_missing({}, names, reason)
    else:
        results = fill_missing(dict(session.submitted), names, "Not extracted")

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "stopped_by": loop.stopped_by,
                "error": loop.error,
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in loop.transcript if e["role"] == "tool"],
                "conversation": loop.transcript,
                "results": results,
            },
        )
    return results, loop.usage.to_dict()
