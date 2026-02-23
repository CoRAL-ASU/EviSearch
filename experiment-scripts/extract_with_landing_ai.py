#!/usr/bin/env python3
"""
extract_with_landing_ai.py

Extract column values using Landing AI chunks + LLM (multi-candidate).

Flow (group-wise, default):
1. Load plans and Landing AI chunks
2. For each group: vote-based chunk selection (rank by column votes, ceiling on tokens), one LLM call
3. LLM returns per-column: candidates, primary_value, found

Flow (--no-group-wise): per-column extraction, one LLM call per column.

Output: extraction results with candidates, primary_value, value (=primary_value for compatibility).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.planning.plan_generator import safe_stem
from src.table_definitions.definitions import load_definitions


def load_column_definitions() -> Dict[str, str]:
    grouped = load_definitions()
    out: Dict[str, str] = {}
    for _group, cols in grouped.items():
        for col in cols:
            out[col["Column Name"]] = col["Definition"]
    return out

# Config
PROVIDER = "gemini"
MODEL = "gemini-2.5-flash"
STRUCTURER_BASE_URL = "http://localhost:8001/v1"
STRUCTURER_MODEL = "Qwen/Qwen3-8B"
MAX_WORKERS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 8000
MAX_TOKENS_GROUP = 16000  # Larger output for group extraction (many columns)
TEXT_CHUNK_CHAR_LIMIT = 32000
TABLE_CHUNK_CHAR_LIMIT = 32000
FALLBACK_CHUNK_COUNT = 8
FALLBACK_MAX_CHUNKS = 80  # When plan has no page, include more chunks to cover tables

# Group-wise extraction: vote-based chunk selection
CHUNK_CONTEXT_CEILING_TOKENS = 30000  # Max tokens for chunks in one group call
CHARS_PER_TOKEN = 4  # Heuristic for token count


def _token_count(text: str) -> int:
    """Rough token count: chars / 4."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _chunk_page_as_int(chunk: Dict[str, Any]) -> int:
    """Extract page as int for sorting. Handles int, str like '5-7'."""
    p = chunk.get("page")
    if isinstance(p, int):
        return p
    if isinstance(p, str) and "-" in p:
        try:
            return int(p.split("-", 1)[0].strip())
        except Exception:
            return 0
    try:
        return int(p) if p else 0
    except Exception:
        return 0


def select_chunks_for_group_vote_based(
    group_columns: List[Tuple[str, Dict[str, Any]]],
    chunks: List[Dict[str, Any]],
    ceiling_tokens: int = CHUNK_CONTEXT_CEILING_TOKENS,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Vote-based chunk selection: rank by how many columns reference each chunk,
    add until ceiling. Tie-breaking: keep both unless exceeding ceiling;
    when must choose, pick chunk closest in page to already-selected.
    Returns (selected_chunks, retrieval_source).
    """
    chunk_votes: Dict[int, int] = {}
    chunk_data: Dict[int, Dict[str, Any]] = {}
    retrieval_source = "extraction_plan_pages"

    for group_name, col in group_columns:
        relevant, src = find_chunks_for_column_tiered(col, chunks)
        if src == "planner_sources":
            retrieval_source = "planner_sources"
        for c in relevant:
            cid = c.get("chunk_id")
            if cid is not None:
                chunk_votes[cid] = chunk_votes.get(cid, 0) + 1
                if cid not in chunk_data:
                    chunk_data[cid] = dict(c)

    if not chunk_data:
        return [], retrieval_source

    # Build list of (chunk_id, vote_count, chunk)
    candidates = [
        (cid, chunk_votes.get(cid, 0), chunk_data[cid])
        for cid in chunk_data
    ]
    # Sort by vote count desc, then page asc
    candidates.sort(key=lambda x: (-x[1], _chunk_page_as_int(x[2])))

    selected: List[Dict[str, Any]] = []
    cumulative_tokens = 0
    selected_pages: List[int] = []

    i = 0
    while i < len(candidates):
        vote = candidates[i][1]
        tied = [(cid, v, c) for cid, v, c in candidates[i:] if v == vote]
        i += len(tied)

        tied_tokens = sum(_token_count(str(c.get("content", "") or "") + "\n" + str(c.get("table_content", "") or "")) for _, _, c in tied)

        if cumulative_tokens + tied_tokens <= ceiling_tokens:
            for _, _, c in tied:
                selected.append(c)
                cumulative_tokens += _token_count(str(c.get("content", "") or "") + "\n" + str(c.get("table_content", "") or ""))
                selected_pages.append(_chunk_page_as_int(c))
            continue

        # Must choose: add by proximity to selected pages
        def min_page_dist(chunk: Dict[str, Any]) -> int:
            cp = _chunk_page_as_int(chunk)
            if not selected_pages:
                return 0
            return min(abs(cp - sp) for sp in selected_pages)

        tied_sorted = sorted(tied, key=lambda x: (min_page_dist(x[2]), _chunk_page_as_int(x[2])))

        for _, _, c in tied_sorted:
            c_tokens = _token_count(str(c.get("content", "") or "") + "\n" + str(c.get("table_content", "") or ""))
            if cumulative_tokens + c_tokens > ceiling_tokens:
                break
            selected.append(c)
            cumulative_tokens += c_tokens
            selected_pages.append(_chunk_page_as_int(c))

    return selected, retrieval_source


def _parse_pages_from_plan(extraction_plan: str) -> list[int]:
    """Extract page numbers mentioned in extraction plan (e.g. 'page 5', 'Table 1 (page 5)')."""
    if not extraction_plan:
        return []
    pages: list[int] = []
    # Match "page 5", "page 5)", "(page 5", "pages 5-7", "page 5 and 6"
    for m in re.finditer(r"page[s]?\s+(\d+)(?:\s*[-–]\s*(\d+))?", extraction_plan, re.I):
        start = int(m.group(1))
        pages.append(start)
        if m.group(2):
            end = int(m.group(2))
            pages.extend(range(start + 1, end + 1))
    return sorted(set(p for p in pages if 1 <= p <= 500))


def chunk_page_matches(chunk_page: Any, target_page: int) -> bool:
    if isinstance(chunk_page, int):
        return chunk_page == target_page
    if isinstance(chunk_page, str):
        if "-" in chunk_page:
            try:
                start_s, end_s = chunk_page.split("-", 1)
                return int(start_s) <= target_page <= int(end_s)
            except Exception:
                return False
        try:
            return int(chunk_page) == target_page
        except Exception:
            return False
    return False


def find_chunks_for_column(
    column: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Select chunks relevant to a single column (page + source_type, or pages from plan)."""
    source_type = str(column.get("source_type", "")).lower()
    page = column.get("page", -1)
    extraction_plan = column.get("extraction_plan", "") or ""
    selected: List[Dict[str, Any]] = []
    seen = set()

    # Primary: use page + source_type when available
    if source_type in {"table", "text", "figure"} and isinstance(page, int) and page >= 1:
        for idx, chunk in enumerate(chunks):
            chunk_type = str(chunk.get("type", "text")).lower()
            if chunk_type != source_type:
                continue
            if not chunk_page_matches(chunk.get("page"), page):
                continue
            if idx in seen:
                continue
            selected.append({
                "chunk_id": idx,
                "type": chunk_type,
                "page": chunk.get("page"),
                "content": chunk.get("content", "") or "",
                "table_content": chunk.get("table_content", "") or "",
            })
            seen.add(idx)

    # Fallback when page=-1 / not_applicable: parse extraction_plan for page refs (e.g. "Table 1 (page 5)")
    # Only add chunks whose page is in plan_pages (page + modality together - no blanket "all tables")
    if not selected and fallback:
        plan_pages = _parse_pages_from_plan(extraction_plan)
        for idx, chunk in enumerate(chunks):
            if idx in seen:
                continue
            chunk_page = chunk.get("page")
            chunk_type = str(chunk.get("type", "text")).lower()
            # Get page as int (chunk may have int or range str like "1-4")
            page_val: int | None = None
            if isinstance(chunk_page, int):
                page_val = chunk_page
            elif isinstance(chunk_page, str) and "-" in chunk_page:
                try:
                    page_val = int(chunk_page.split("-", 1)[0].strip())
                except Exception:
                    pass
            if plan_pages and page_val is not None and page_val in plan_pages:
                selected.append({
                    "chunk_id": idx,
                    "type": chunk_type,
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                })
                seen.add(idx)

    # Last resort: include more chunks so we don't miss tables (was first 8, now first N)
    if not selected and fallback:
        for idx, chunk in enumerate(chunks[:FALLBACK_MAX_CHUNKS]):
            selected.append({
                "chunk_id": idx,
                "type": str(chunk.get("type", "text")).lower(),
                "page": chunk.get("page"),
                "content": chunk.get("content", "") or "",
                "table_content": chunk.get("table_content", "") or "",
            })
    return selected


def find_chunks_for_column_tiered(
    column: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Tiered retrieval: use planner sources first; only if empty, use extraction_plan pages.
    Returns (selected_chunks, retrieval_source) where retrieval_source is
    "planner_sources" or "extraction_plan_pages".
    """
    selected: List[Dict[str, Any]] = []
    seen: set[int] = set()
    extraction_plan = column.get("extraction_plan", "") or ""

    # Tier 1: planner sources (list of [page, modality])
    sources = column.get("sources") or []
    if isinstance(sources, list) and sources:
        for item in sources:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            page_val = item[0]
            modality = str(item[1]).lower() if item[1] else "text"
            if modality not in {"table", "text", "figure"}:
                modality = "text"
            target_page = int(page_val) if isinstance(page_val, (int, float)) else None
            if target_page is None or target_page < 1:
                continue
            for idx, chunk in enumerate(chunks):
                if idx in seen:
                    continue
                chunk_type = str(chunk.get("type", "text")).lower()
                if chunk_type != modality:
                    continue
                if not chunk_page_matches(chunk.get("page"), target_page):
                    continue
                selected.append({
                    "chunk_id": idx,
                    "type": chunk_type,
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                })
                seen.add(idx)

    if selected:
        return selected, "planner_sources"

    # Tier 2: extraction_plan pages - only when tier 1 returns empty
    plan_pages = _parse_pages_from_plan(extraction_plan)
    if plan_pages:
        for idx, chunk in enumerate(chunks):
            if idx in seen:
                continue
            chunk_page = chunk.get("page")
            chunk_type = str(chunk.get("type", "text")).lower()
            page_val: int | None = None
            if isinstance(chunk_page, int):
                page_val = chunk_page
            elif isinstance(chunk_page, str) and "-" in chunk_page:
                try:
                    page_val = int(chunk_page.split("-", 1)[0].strip())
                except Exception:
                    pass
            if page_val is not None and page_val in plan_pages:
                selected.append({
                    "chunk_id": idx,
                    "type": chunk_type,
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                })
                seen.add(idx)

    return selected, "extraction_plan_pages"


def _clean_chunk_content(text: str) -> str:
    """Strip Landing AI markup: anchor IDs, table/cell IDs, merge to readable content."""
    if not text:
        return ""
    # Remove <a id='...'></a> anchors (often at start of chunks)
    text = re.sub(r"<a\s+id=['\"][^'\"]*['\"]\s*></a>\s*", "", text)
    # Remove id="4-1", id='4-2' etc from table/td/tr elements
    text = re.sub(r'\s+id=["\'][^"\']*["\']', "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_sort_key(chunk: Dict[str, Any]) -> tuple:
    """Sort key: (page_num, 0). Handles int, str like '5-7', etc."""
    p = chunk.get("page")
    if isinstance(p, int):
        return (p, 0)
    if isinstance(p, str) and "-" in p:
        try:
            return (int(p.split("-", 1)[0].strip()), 0)
        except Exception:
            return (0, 0)
    try:
        return (int(p) if p else 0, 0)
    except Exception:
        return (0, 0)


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks by type: TEXT (chronological), TABLES (chronological), FIGURES (chronological)."""
    if not chunks:
        return "No chunks available."

    by_type: Dict[str, List[Dict[str, Any]]] = {"text": [], "table": [], "figure": []}
    for chunk in chunks:
        ctype = str(chunk.get("type", "text")).lower()
        if ctype not in by_type:
            by_type["text"].append(chunk)  # marginalia etc. -> text
        else:
            by_type[ctype].append(chunk)

    sections: List[str] = []

    # TEXT: all text chunks concatenated together in chronological order (at the top)
    text_group = sorted(by_type["text"], key=_page_sort_key)
    if text_group:
        text_parts: List[str] = []
        for chunk in text_group:
            body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                text_parts.append(body)
        if text_parts:
            sections.append("--- TEXT ---\n" + "\n\n".join(text_parts))

    # TABLES: each table with [Page N] label
    table_group = sorted(by_type["table"], key=_page_sort_key)
    if table_group:
        table_parts: List[str] = []
        for chunk in table_group:
            page = chunk.get("page")
            body = str(chunk.get("table_content", "") or chunk.get("content", ""))[:TABLE_CHUNK_CHAR_LIMIT]
            body = _clean_chunk_content(body)
            if not body:
                body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                table_parts.append(f"[Page {page}]\n{body}")
        if table_parts:
            sections.append("--- TABLES ---\n" + "\n\n".join(table_parts))

    # FIGURES: each figure with [Page N] label
    figure_group = sorted(by_type["figure"], key=_page_sort_key)
    if figure_group:
        figure_parts: List[str] = []
        for chunk in figure_group:
            page = chunk.get("page")
            body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                figure_parts.append(f"[Page {page}]\n{body}")
        if figure_parts:
            sections.append("--- FIGURES ---\n" + "\n\n".join(figure_parts))

    return "\n\n".join(sections) if sections else "No content available."


def build_extraction_prompt(
    column: Dict[str, Any],
    definition: str,
    formatted_chunks: str,
    extraction_plan: str,
) -> str:
    name = column.get("column_name", "")
    return f"""You are extracting a value for a clinical trial data column from the provided document content.

Column: {name}
Definition: {definition}

Extraction plan to follow:
{extraction_plan}

Document content (grouped by type, in chronological page order):
{formatted_chunks}

Task:
1. Extract the value for this column following the plan. Return your best-effort value even if uncertain.
2. If the plan seems wrong, incomplete, or ambiguous, still return the value you found and add a suggestion and/or alternative_plan.
3. Use "not found" or "not reported" only if the value is genuinely absent from the content.

Output a JSON object with:
- value: the extracted value (string)
- found: true if you found a value, false if not reported/absent
- confidence: "high" | "medium" | "low"
- suggestion: optional string with concerns or notes (null if none)
- alternative_plan: optional string describing a different extraction approach to try (null if plan seems fine)
"""


def build_extraction_prompt_multi_candidate(
    column: Dict[str, Any],
    definition: str,
    formatted_chunks: str,
    extraction_plan: str,
) -> str:
    name = column.get("column_name", "")
    return f"""You are extracting values for a clinical trial data column from the provided document content.

Column: {name}
Definition: {definition}

Extraction plan to follow:
{extraction_plan}

Document content (grouped by type, in chronological page order):
{formatted_chunks}

Task:
1. List 1-4 plausible values for this column. For each candidate: value, evidence (short excerpt or location), assumptions (if any), confidence.
2. If only one interpretation is clear, return it as a single candidate.
3. When the source is ambiguous, report all plausible interpretations as separate candidates:
   - **Adverse events**: If the table does not specify whether it reports treatment-related or all-cause AEs (e.g. "Treatment-Emergent AEs" or "Any AE" without causality), report BOTH: one candidate as all-cause (what the table actually reports) and one as treatment-related (if that is what the column asks for), with assumptions noting the ambiguity.
   - **PFS / progression endpoints**: If multiple progression times are reported in different tables (e.g. "Median Time to CRPC" in Table 2 vs "Median Time to Clinical Progression" in Table 3), report BOTH values as separate candidates with their source.
4. Use a candidate with value "Not reported" only when the value is genuinely absent from the content.
5. primary_value: your best guess for automated use (e.g. the most standard or commonly used form).

Output a JSON object with:
- candidates: array of 1-4 objects, each with value, evidence (string or null), assumptions (string or null), confidence ("high"|"medium"|"low")
- primary_value: string (your best guess)
- found: true if at least one value was found in the content, false if not reported/absent

Examples from our dataset:

Median PFS with multiple progression endpoints (CHAARTED):
{{"candidates": [{{"value": "14.9", "evidence": "Table 2, Median Time to CRPC (PSA rise or clinical progression), High volume, ADT Plus Docetaxel", "assumptions": "Time to CRPC used as PFS proxy", "confidence": "medium"}}, {{"value": "27.3", "evidence": "Table 3, Median Time to Clinical Progression, High volume, ADT Plus Docetaxel", "assumptions": "Clinical progression only", "confidence": "medium"}}], "primary_value": "14.9", "found": true}}

Adverse events when table does not specify treatment-related vs all-cause (ARASENS Table 3):
{{"candidates": [{{"value": "458 (70.2%)", "evidence": "Table 3, Any AE worst grade Grade 3-5, Darolutamide arm", "assumptions": "Table reports Treatment-Emergent AEs (all-cause)", "confidence": "medium"}}, {{"value": "458 (70.2%)", "evidence": "Table 3, same row", "assumptions": "Table does not specify causality; if treatment-related, value may be same or different", "confidence": "low"}}], "primary_value": "458 (70.2%)", "found": true}}

Single clear value with optional CI:
{{"candidates": [{{"value": "19.4", "evidence": "Table 2, page 6", "assumptions": null, "confidence": "high"}}, {{"value": "19.4 (95% CI 16.2-22.1)", "evidence": "Footnote b", "assumptions": "Included CI", "confidence": "medium"}}], "primary_value": "19.4", "found": true}}
"""


def build_extraction_prompt_group(
    group_columns: List[Tuple[str, Dict[str, Any]]],
    definitions_map: Dict[str, str],
    formatted_chunks: str,
) -> str:
    """Build prompt for extracting all columns in a group at once."""
    col_specs: List[str] = []
    for i, (group_name, col) in enumerate(group_columns, 1):
        name = col.get("column_name", "")
        defn = definitions_map.get(name, "")
        plan = col.get("extraction_plan", "") or ""
        col_specs.append(f"""
Column {i}: {name}
  Definition: {defn}
  Extraction plan: {plan}
""")

    return f"""You are extracting values for multiple clinical trial data columns from the same document content.
All columns below share the same evidence pool. Extract each column following its plan.

COLUMNS TO EXTRACT:
{"".join(col_specs)}

Document content (grouped by type, in chronological page order):
{formatted_chunks}

Task:
For EACH column listed above, output a JSON object with:
- candidates: array of 1-4 objects, each with value, evidence (string or null), assumptions (string or null), confidence ("high"|"medium"|"low")
- primary_value: string (your best guess)
- found: true if at least one value was found, false if not reported/absent

Return a single JSON object with column names as keys. Each value is an object with candidates, primary_value, found.
Example: {{"Median PFS (mo) | Overall | Treatment": {{"candidates": [{{"value": "19.4", "evidence": "Table 2", "assumptions": null, "confidence": "high"}}], "primary_value": "19.4", "found": true}}, ...}}
"""


# Pydantic schema for extraction response
try:
    from pydantic import BaseModel, Field
    from typing import Optional

    class ExtractionResult(BaseModel):
        value: str = Field(description="The extracted value")
        found: bool = Field(description="True if value was found in chunks")
        confidence: str = Field(description="high, medium, or low")
        suggestion: Optional[str] = Field(default=None, description="Optional concern or note")
        alternative_plan: Optional[str] = Field(default=None, description="Optional alternative extraction approach")

    class ExtractionCandidate(BaseModel):
        value: str = Field(description="A plausible extracted value")
        evidence: Optional[str] = Field(default=None, description="Short excerpt or location supporting this value")
        assumptions: Optional[str] = Field(default=None, description="Assumptions made if any")
        confidence: str = Field(description="high, medium, or low")

    class ExtractionResultMultiCandidate(BaseModel):
        candidates: List[ExtractionCandidate] = Field(
            description="List of 1-4 plausible values with evidence and assumptions"
        )
        primary_value: str = Field(description="Best guess for automated use; use first candidate if only one")
        found: bool = Field(description="True if at least one value was found")
except ImportError:
    ExtractionResult = None
    ExtractionCandidate = None
    ExtractionResultMultiCandidate = None


def _write_llm_log(
    logs_dir: Path,
    group_name: str,
    column_name: str,
    prompt: str,
    output: str,
    prompt_retry: str | None = None,
    output_retry: str | None = None,
    retrieval_source: str | None = None,
) -> None:
    """Write clean model input/output to a txt file."""
    stem = safe_stem(f"{group_name}_{column_name}")
    parts: List[str] = []
    if retrieval_source:
        parts.append(f"RETRIEVAL SOURCE: {retrieval_source}\n\n")
    parts.extend([
        "============================================================\nINPUT\n============================================================\n",
        prompt,
        "\n\n============================================================\nOUTPUT\n============================================================\n",
        output or "",
    ])
    if prompt_retry is not None and output_retry is not None:
        parts.extend([
            "\n\n============================================================\nINPUT (RETRY)\n============================================================\n",
            prompt_retry,
            "\n\n============================================================\nOUTPUT (RETRY)\n============================================================\n",
            output_retry or "",
        ])
    (logs_dir / f"{stem}_llm_call.txt").write_text("".join(parts), encoding="utf-8")


def extract_column(
    column: Dict[str, Any],
    group_name: str,
    definition: str,
    chunks: List[Dict[str, Any]],
    do_retry: bool = True,
    logs_dir: Path | None = None,
    relevant_chunks_override: List[Dict[str, Any]] | None = None,
    retrieval_source: str | None = None,
) -> Dict[str, Any]:
    """Extract value for one column. Optionally retry with alternative_plan.
    When relevant_chunks_override and retrieval_source are provided, use those instead of find_chunks_for_column."""
    plan = column.get("extraction_plan", "")
    if relevant_chunks_override is not None:
        relevant = relevant_chunks_override
    else:
        relevant = find_chunks_for_column(column, chunks)
    formatted = format_chunks(relevant)

    prompt = build_extraction_prompt(column, definition, formatted, plan)

    provider = LLMProvider(provider=PROVIDER, model=MODEL)
    response = provider.generate(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    result: Dict[str, Any] = {
        "column_index": column.get("column_index"),
        "column_name": column.get("column_name"),
        "group_name": group_name,
        "extraction_plan": column.get("extraction_plan", ""),
        "page": column.get("page"),
        "source_type": column.get("source_type"),
        "retrieval_source": retrieval_source,
        "success": response.success,
        "error": response.error,
        "value": None,
        "found": False,
        "confidence": "low",
        "suggestion": None,
        "alternative_plan": None,
        "retry_value": None,
        "values_match": None,
        "prompt_token_count": response.input_tokens,
        "output_token_count": response.output_tokens,
        "raw_response": response.text or "",
    }

    if not response.success:
        if logs_dir:
            _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
        return result

    # Parse structured output
    if ExtractionResult and OutputStructurer:
        structurer = OutputStructurer(
            base_url=STRUCTURER_BASE_URL,
            model=STRUCTURER_MODEL,
            enable_thinking=False,
        )
        structured = structurer.structure(
            text=response.text or "",
            schema=ExtractionResult,
            max_retries=2,
            return_dict=True,
        )
        if structured.success and isinstance(structured.data, dict):
            d = structured.data
            result["value"] = d.get("value") or ""
            result["found"] = bool(d.get("found", False))
            result["confidence"] = str(d.get("confidence", "low")).lower()
            result["suggestion"] = d.get("suggestion")
            result["alternative_plan"] = d.get("alternative_plan")
        else:
            # Structurer failed; fall back to JSON parse from raw response
            try:
                text = (response.text or "").strip()
                if "```" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                d = json.loads(text)
                result["value"] = d.get("value") or ""
                result["found"] = bool(d.get("found", False))
                result["confidence"] = str(d.get("confidence", "low")).lower()
                result["suggestion"] = d.get("suggestion")
                result["alternative_plan"] = d.get("alternative_plan")
            except Exception as e:
                result["error"] = f"Structuring failed: {getattr(structured, 'error', 'unknown')}; JSON fallback: {e}"
                if logs_dir:
                    _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
                return result
    else:
        # Fallback: try to parse JSON from response
        try:
            text = (response.text or "").strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            d = json.loads(text)
            result["value"] = d.get("value") or ""
            result["found"] = bool(d.get("found", False))
            result["confidence"] = str(d.get("confidence", "low")).lower()
            result["suggestion"] = d.get("suggestion")
            result["alternative_plan"] = d.get("alternative_plan")
        except Exception as e:
            result["error"] = f"Parse failed: {e}"
            if logs_dir:
                _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
            return result

    # Option B: Retry with alternative_plan if provided
    prompt_retry: str | None = None
    output_retry: str | None = None
    if do_retry and result.get("alternative_plan"):
        alt_plan = result["alternative_plan"]
        prompt2 = build_extraction_prompt(column, definition, formatted, alt_plan)
        response2 = provider.generate(prompt=prompt2, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        prompt_retry = prompt2
        output_retry = response2.text or ""
        if response2.success and ExtractionResult:
            structurer = OutputStructurer(
                base_url=STRUCTURER_BASE_URL,
                model=STRUCTURER_MODEL,
                enable_thinking=False,
            )
            structured2 = structurer.structure(
                text=response2.text or "",
                schema=ExtractionResult,
                max_retries=2,
                return_dict=True,
            )
            if structured2.success and isinstance(structured2.data, dict):
                retry_val = structured2.data.get("value", "")
                result["retry_value"] = retry_val
                v1 = str(result.get("value", "")).strip()
                v2 = str(retry_val).strip()
                result["values_match"] = v1 == v2

    if logs_dir:
        _write_llm_log(
            logs_dir,
            group_name,
            column.get("column_name", ""),
            prompt,
            response.text or "",
            prompt_retry=prompt_retry,
            output_retry=output_retry,
            retrieval_source=retrieval_source,
        )

    return result


def extract_column_multi_candidate(
    column: Dict[str, Any],
    group_name: str,
    definition: str,
    chunks: List[Dict[str, Any]],
    logs_dir: Path | None = None,
    relevant_chunks_override: List[Dict[str, Any]] | None = None,
    retrieval_source: str | None = None,
) -> Dict[str, Any]:
    """Extract 1-4 plausible values per column with evidence and assumptions."""
    plan = column.get("extraction_plan", "")
    if relevant_chunks_override is not None:
        relevant = relevant_chunks_override
    else:
        relevant = find_chunks_for_column(column, chunks)
    chunk_count = len(relevant)

    if not relevant:
        col_name = column.get("column_name", "")
        print(f"  WARNING: No chunks for column '{col_name}' (group: {group_name}), skipping LLM call")
        return {
            "column_index": column.get("column_index"),
            "column_name": column.get("column_name"),
            "group_name": group_name,
            "extraction_plan": plan,
            "page": column.get("page"),
            "source_type": column.get("source_type"),
            "retrieval_source": retrieval_source or "",
            "chunk_count": 0,
            "success": True,
            "error": None,
            "candidates": [{"value": "Not reported", "evidence": None, "assumptions": "No chunks available", "confidence": "low"}],
            "primary_value": "Not reported",
            "value": "Not reported",
            "found": False,
            "prompt_token_count": 0,
            "output_token_count": 0,
            "raw_response": "",
        }

    formatted = format_chunks(relevant)
    prompt = build_extraction_prompt_multi_candidate(column, definition, formatted, plan)

    provider = LLMProvider(provider=PROVIDER, model=MODEL)
    response = provider.generate(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    result: Dict[str, Any] = {
        "column_index": column.get("column_index"),
        "column_name": column.get("column_name"),
        "group_name": group_name,
        "extraction_plan": column.get("extraction_plan", ""),
        "page": column.get("page"),
        "source_type": column.get("source_type"),
        "retrieval_source": retrieval_source or "",
        "chunk_count": chunk_count,
        "success": response.success,
        "error": response.error,
        "candidates": [],
        "primary_value": None,
        "value": None,
        "found": False,
        "prompt_token_count": response.input_tokens,
        "output_token_count": response.output_tokens,
        "raw_response": response.text or "",
    }

    if not response.success:
        if logs_dir:
            _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
        return result

    if ExtractionResultMultiCandidate and OutputStructurer:
        structurer = OutputStructurer(
            base_url=STRUCTURER_BASE_URL,
            model=STRUCTURER_MODEL,
            enable_thinking=False,
        )
        structured = structurer.structure(
            text=response.text or "",
            schema=ExtractionResultMultiCandidate,
            max_retries=2,
            return_dict=True,
        )
        if structured.success and isinstance(structured.data, dict):
            d = structured.data
            candidates = d.get("candidates") or []
            result["candidates"] = [
                {
                    "value": c.get("value", ""),
                    "evidence": c.get("evidence"),
                    "assumptions": c.get("assumptions"),
                    "confidence": str(c.get("confidence", "low")).lower(),
                }
                for c in candidates
            ]
            result["primary_value"] = d.get("primary_value") or (candidates[0].get("value") if candidates else "")
            result["value"] = result["primary_value"]
            result["found"] = bool(d.get("found", False))
        else:
            try:
                text = (response.text or "").strip()
                if "```" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                d = json.loads(text)
                candidates = d.get("candidates") or []
                result["candidates"] = [
                    {"value": c.get("value", ""), "evidence": c.get("evidence"), "assumptions": c.get("assumptions"), "confidence": str(c.get("confidence", "low")).lower()}
                    for c in candidates
                ]
                result["primary_value"] = d.get("primary_value") or (candidates[0].get("value") if candidates else "")
                result["value"] = result["primary_value"]
                result["found"] = bool(d.get("found", False))
            except Exception as e:
                result["error"] = f"Structuring failed: {getattr(structured, 'error', 'unknown')}; JSON fallback: {e}"
                if logs_dir:
                    _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
                return result
    else:
        try:
            text = (response.text or "").strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            d = json.loads(text)
            candidates = d.get("candidates") or []
            result["candidates"] = [
                {"value": c.get("value", ""), "evidence": c.get("evidence"), "assumptions": c.get("assumptions"), "confidence": str(c.get("confidence", "low")).lower()}
                for c in candidates
            ]
            result["primary_value"] = d.get("primary_value") or (candidates[0].get("value") if candidates else "")
            result["value"] = result["primary_value"]
            result["found"] = bool(d.get("found", False))
        except Exception as e:
            result["error"] = f"Parse failed: {e}"
            if logs_dir:
                _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)
            return result

    if logs_dir:
        _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "", retrieval_source=retrieval_source)

    return result


def extract_group_multi_candidate(
    group_name: str,
    group_columns: List[Tuple[str, Dict[str, Any]]],
    definitions_map: Dict[str, str],
    chunks: List[Dict[str, Any]],
    logs_dir: Path | None = None,
    chunk_ceiling_tokens: int = CHUNK_CONTEXT_CEILING_TOKENS,
) -> List[Dict[str, Any]]:
    """Extract all columns in a group with one LLM call. Vote-based chunk selection."""
    selected_chunks, retrieval_source = select_chunks_for_group_vote_based(
        group_columns, chunks, ceiling_tokens=chunk_ceiling_tokens
    )
    chunk_count = len(selected_chunks)

    if not selected_chunks:
        print(f"  WARNING: No chunks for group '{group_name}', skipping LLM call")
        results: List[Dict[str, Any]] = []
        for _, col in group_columns:
            results.append({
                "column_index": col.get("column_index"),
                "column_name": col.get("column_name"),
                "group_name": group_name,
                "extraction_plan": col.get("extraction_plan", ""),
                "page": col.get("page"),
                "source_type": col.get("source_type"),
                "retrieval_source": retrieval_source,
                "chunk_count": 0,
                "success": True,
                "error": None,
                "candidates": [{"value": "Not reported", "evidence": None, "assumptions": "No chunks available", "confidence": "low"}],
                "primary_value": "Not reported",
                "value": "Not reported",
                "found": False,
                "prompt_token_count": 0,
                "output_token_count": 0,
                "raw_response": "",
            })
        return results

    formatted = format_chunks(selected_chunks)
    prompt = build_extraction_prompt_group(group_columns, definitions_map, formatted)

    provider = LLMProvider(provider=PROVIDER, model=MODEL)
    response = provider.generate(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS_GROUP)

    results = []
    for _, col in group_columns:
        col_name = col.get("column_name", "")
        r: Dict[str, Any] = {
            "column_index": col.get("column_index"),
            "column_name": col_name,
            "group_name": group_name,
            "extraction_plan": col.get("extraction_plan", ""),
            "page": col.get("page"),
            "source_type": col.get("source_type"),
            "retrieval_source": retrieval_source,
            "chunk_count": chunk_count,
            "success": False,
            "error": None,
            "candidates": [],
            "primary_value": None,
            "value": None,
            "found": False,
            "prompt_token_count": 0,
            "output_token_count": 0,
            "raw_response": "",
        }
        results.append(r)

    if not response.success:
        for r in results:
            r["error"] = response.error
        if logs_dir:
            _write_llm_log(logs_dir, group_name, "group", prompt, response.text or "", retrieval_source=retrieval_source)
        return results

    for r in results:
        r["prompt_token_count"] = response.input_tokens
        r["output_token_count"] = response.output_tokens
        r["raw_response"] = response.text or ""

    text = (response.text or "").strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        parsed = json.loads(text)
    except Exception as e:
        for r in results:
            r["error"] = f"Parse failed: {e}"
        if logs_dir:
            _write_llm_log(logs_dir, group_name, "group", prompt, response.text or "", retrieval_source=retrieval_source)
        return results

    if not isinstance(parsed, dict):
        for r in results:
            r["error"] = "Response is not a JSON object"
        return results

    for r in results:
        col_name = r["column_name"]
        col_data = parsed.get(col_name)
        if not isinstance(col_data, dict):
            continue
        candidates = col_data.get("candidates") or []
        r["candidates"] = [
            {
                "value": c.get("value", ""),
                "evidence": c.get("evidence"),
                "assumptions": c.get("assumptions"),
                "confidence": str(c.get("confidence", "low")).lower(),
            }
            for c in candidates
        ]
        r["primary_value"] = col_data.get("primary_value") or (candidates[0].get("value") if candidates else "")
        r["value"] = r["primary_value"]
        r["found"] = bool(col_data.get("found", False))
        r["success"] = True

    if logs_dir:
        _write_llm_log(logs_dir, group_name, "group", prompt, response.text or "", retrieval_source=retrieval_source)

    return results


def load_plans(planning_dir: Path) -> Dict[str, Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    compiled = planning_dir / "plans_all_columns.json"
    if compiled.exists():
        data = json.loads(compiled.read_text(encoding="utf-8"))
        for entry in data.get("plans", []):
            if isinstance(entry, dict) and entry.get("group_name"):
                plans[entry["group_name"]] = entry
    return plans


def flatten_columns(plans_by_group: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """(group_name, column) for each column across all groups."""
    out: List[tuple] = []
    for group_name in sorted(plans_by_group.keys()):
        group = plans_by_group[group_name]
        for col in group.get("columns", []):
            out.append((group_name, col))
    return out


def group_columns_by_group(flat: List[tuple]) -> Dict[str, List[tuple]]:
    """Group (group_name, col) by group_name. Returns {group_name: [(group_name, col), ...]}."""
    out: Dict[str, List[tuple]] = {}
    for group_name, col in flat:
        if group_name not in out:
            out[group_name] = []
        out[group_name].append((group_name, col))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract column values using Landing AI chunks + LLM (multi-candidate).")
    p.add_argument("--pdf-name", required=True, help="PDF stem (e.g. NCT02799602_Hussain_ARASENS_JCO'23)")
    p.add_argument("--results-root", default="new_pipeline_outputs/results")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Limit columns to process (0 = all)")
    p.add_argument("--group-wise", action="store_true", default=True, help="Extract by group (one LLM call per group, default)")
    p.add_argument("--no-group-wise", action="store_false", dest="group_wise", help="Extract per column (one LLM call per column)")
    p.add_argument("--chunk-ceiling", type=int, default=CHUNK_CONTEXT_CEILING_TOKENS, help="Max tokens for chunks per group (default: %(default)s)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    base_dir = results_root / args.pdf_name
    planning_dir = base_dir / "planning"
    output_dir = base_dir / "planning" / "extract_landing_ai"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    if not planning_dir.exists():
        raise FileNotFoundError(f"Planning dir not found: {planning_dir}")

    # Load Landing AI chunks
    from landing_ai_chunks import load_landing_ai_chunks
    pdf_path = PROJECT_ROOT / args.dataset_dir / f"{args.pdf_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    cache_dir = base_dir / "chunking"
    print(f"Loading Landing AI chunks...")
    chunks = load_landing_ai_chunks(pdf_path, cache_dir=cache_dir, use_cache=True)
    print(f"Loaded {len(chunks)} chunks")

    plans_by_group = load_plans(planning_dir)
    if not plans_by_group:
        raise RuntimeError(f"No plans in {planning_dir}")

    definitions_map = load_column_definitions()
    flat = flatten_columns(plans_by_group)
    if args.limit > 0:
        flat = flat[: args.limit]
        print(f"Processing first {len(flat)} columns (--limit={args.limit})")

    mode_str = "group-wise" if args.group_wise else "per-column"
    print(f"Mode: {mode_str}")

    logs_dir: Path | None = None
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total_api_calls = 0

    if args.group_wise:
        groups = group_columns_by_group(flat)
        for group_name in sorted(groups.keys()):
            group_cols = groups[group_name]
            try:
                group_results = extract_group_multi_candidate(
                    group_name,
                    group_cols,
                    definitions_map,
                    chunks,
                    logs_dir=logs_dir,
                    chunk_ceiling_tokens=args.chunk_ceiling,
                )
                results.extend(group_results)
                if group_results and group_results[0].get("chunk_count", 0) > 0:
                    total_api_calls += 1
                for r in group_results:
                    col_name = r.get("column_name", "")
                    cands = r.get("candidates", [])
                    n = len(cands)
                    conf = cands[0].get("confidence", "") if cands else ""
                    src = r.get("retrieval_source", "")
                    src_str = f" [{src}]" if src else ""
                    n_chunks = r.get("chunk_count", "?")
                    print(f"  {group_name} / {col_name}: {r.get('value', 'N/A')} ({n} candidate(s), {conf}, {n_chunks} chunks){src_str}")
            except Exception as e:
                for _, col in group_cols:
                    results.append({
                        "group_name": group_name,
                        "column_name": col.get("column_name", ""),
                        "success": False,
                        "error": str(e),
                    })
                    print(f"  {group_name} / {col.get('column_name', '')}: ERROR {e}")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {}
            for group_name, col in flat:
                defn = definitions_map.get(col.get("column_name", ""), "")
                relevant_chunks, retrieval_source = find_chunks_for_column_tiered(col, chunks)
                fut = ex.submit(
                    extract_column_multi_candidate,
                    col,
                    group_name,
                    defn,
                    chunks,
                    logs_dir=logs_dir,
                    relevant_chunks_override=relevant_chunks,
                    retrieval_source=retrieval_source,
                )
                futures[fut] = (group_name, col.get("column_name"))

            for fut in as_completed(futures):
                group_name, col_name = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    cands = r.get("candidates", [])
                    n = len(cands)
                    conf = cands[0].get("confidence", "") if cands else ""
                    src = r.get("retrieval_source", "")
                    src_str = f" [{src}]" if src else ""
                    n_chunks = r.get("chunk_count", "?")
                    print(f"  {group_name} / {col_name}: {r.get('value', 'N/A')} ({n} candidate(s), {conf}, {n_chunks} chunks){src_str}")
                except Exception as e:
                    results.append({
                        "group_name": group_name,
                        "column_name": col_name,
                        "success": False,
                        "error": str(e),
                    })
                    print(f"  {group_name} / {col_name}: ERROR {e}")
        total_api_calls = sum(1 for r in results if r.get("chunk_count", 0) > 0)

    if args.dry_run:
        print(f"Dry run: {len(results)} columns processed")
        return 0

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pdf_name": args.pdf_name,
        "provider": PROVIDER,
        "model": MODEL,
        "multi_candidate": True,
        "group_wise": args.group_wise,
        "chunk_ceiling_tokens": args.chunk_ceiling,
        "total_columns": len(results),
        "total_api_calls": total_api_calls,
        "columns_with_multiple_candidates": sum(1 for r in results if len(r.get("candidates", [])) > 1),
        "total_candidates": sum(len(r.get("candidates", [])) for r in results),
        "results": results,
    }
    out_file = output_dir / "extraction_results.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_file}")
    print(f"  Total API calls: {summary['total_api_calls']}")
    print(f"  Columns with multiple candidates: {summary['columns_with_multiple_candidates']}")
    print(f"  Total candidates: {summary['total_candidates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
