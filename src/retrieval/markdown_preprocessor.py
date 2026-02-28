"""
Preprocess parsed_markdown.md for retriever embedding.

Takes markdown + optional JSON (landing_ai_parse_output) and produces
page-level chunks for the retriever. Handles:
- Split by <!-- PAGE BREAK -->
- HTML tables → markdown (preserves structure for retrieval)
- Markdown tables kept as-is
- Chunk IDs from <a id='...'></a> extracted for attribution (when JSON not available)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

PAGE_BREAK = "<!-- PAGE BREAK -->"
MAX_CHARS_PER_EMBED = 30000


def _strip_anchor_tags(text: str) -> str:
    """Remove <a id='uuid'></a> tags; keep content. Reduces embedding noise."""
    return re.sub(r"<a\s+id=['\"][^'\"]*['\"][^>]*>.*?</a>", "", text, flags=re.DOTALL)


def _convert_html_tables_to_markdown(text: str) -> str:
    """Replace HTML <table>...</table> blocks with markdown equivalents."""
    from web.table_utils import html_table_to_markdown

    def _replace(match: re.Match) -> str:
        html = match.group(0)
        return html_table_to_markdown(html)

    # Match <table ...>...</table> (non-greedy; assumes no nested tables)
    return re.sub(
        r"<table[^>]*>.*?</table>",
        _replace,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _extract_chunk_ids_from_markdown(text: str) -> List[str]:
    """Extract chunk UUIDs from <a id='uuid'></a> anchors in order of appearance."""
    return re.findall(r"<a\s+id=['\"]([^'\"]+)['\"]", text)


def _is_markdown_table_line(line: str) -> bool:
    """True if line looks like a markdown table row (| ... |)."""
    stripped = line.strip()
    return bool(stripped) and stripped.startswith("|") and stripped.endswith("|")


def build_page_chunks_from_markdown(
    markdown: str,
    json_data: Optional[Dict[str, Any]] = None,
    convert_html_tables: bool = True,
    strip_anchors: bool = True,
) -> List[Tuple[str, int, str]]:
    """
    Build page-level chunks from parsed_markdown for retriever embedding.

    Args:
        markdown: Full content of parsed_markdown.md (Landing AI ADE Parse output).
        json_data: Optional landing_ai_parse_output.json dict with "chunks" array.
                   Used for chunk_id->page mapping when available. Not required for
                   page splitting (that comes from <!-- PAGE BREAK -->).
        convert_html_tables: If True, convert HTML <table> to markdown. Default True.
        strip_anchors: If True, remove <a id='...'></a> tags before embedding. Default True.

    Returns:
        List of (chunk_id, page, text) tuples.
        chunk_id = "page_N", page = 1-based page number, text = content for embedding.
    """
    if not markdown or not markdown.strip():
        return []

    pages_raw = markdown.split(PAGE_BREAK)
    out: List[Tuple[str, int, str]] = []

    for i, page_text in enumerate(pages_raw):
        page_num = i + 1
        chunk_id = f"page_{page_num}"

        text = page_text.strip()
        if not text:
            continue

        if strip_anchors:
            text = _strip_anchor_tags(text)

        if convert_html_tables:
            text = _convert_html_tables_to_markdown(text)

        text = text.strip()
        if not text:
            continue

        # Add page label for consistency with retriever output
        labeled = f"## Page {page_num}\n\n{text}"

        if len(labeled) > MAX_CHARS_PER_EMBED:
            labeled = labeled[:MAX_CHARS_PER_EMBED] + "\n[... truncated]"

        out.append((chunk_id, page_num, labeled))

    return out


def get_chunk_ids_by_page(
    markdown: str,
    json_data: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[str]]:
    """
    Map page number (1-based) to list of chunk IDs on that page.

    Uses json_data when available (chunks have grounding.page); otherwise
    extracts from <a id='...'></a> in markdown and infers page from PAGE BREAK.
    """
    if json_data and json_data.get("chunks"):
        page_to_ids: Dict[int, List[str]] = {}
        for c in json_data["chunks"]:
            cid = c.get("id")
            if not cid:
                continue
            g = c.get("grounding") or {}
            page_0 = int(g.get("page", 0))
            page_1 = page_0 + 1
            page_to_ids.setdefault(page_1, []).append(cid)
        return page_to_ids

    # Fallback: extract from markdown by splitting on PAGE BREAK
    pages_raw = markdown.split(PAGE_BREAK)
    page_to_ids: Dict[int, List[str]] = {}
    for i, page_text in enumerate(pages_raw):
        page_num = i + 1
        ids = _extract_chunk_ids_from_markdown(page_text)
        if ids:
            page_to_ids[page_num] = ids
    return page_to_ids
