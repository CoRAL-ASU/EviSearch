"""
highlight_service.py

Load Landing AI parse output and return bounding boxes for PDF highlighting.
Used by the comparison dashboard to show where extracted values came from.

Matching strategy: prefer exact value match in chunk content; fall back to page+type.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RESULTS = PROJECT_ROOT / "new_pipeline_outputs" / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"


def _normalize_value_for_search(v: str) -> List[str]:
    """Return value and common variants for matching (e.g. brackets vs parens)."""
    v = str(v or "").strip()
    if not v or len(v) < 2:
        return []
    variants = [v]
    swapped = v.replace("[", "(").replace("]", ")")
    if swapped != v:
        variants.append(swapped)
    swapped2 = v.replace("(", "[").replace(")", "]")
    if swapped2 != v and swapped2 not in variants:
        variants.append(swapped2)
    return variants


def _chunk_text(chunk: Dict[str, Any]) -> str:
    """Get searchable text from a Landing AI chunk."""
    markdown = str(chunk.get("markdown") or "")
    # Strip anchor tags for cleaner matching
    markdown = re.sub(r"<a[^>]*>.*?</a>", "", markdown, flags=re.DOTALL)
    return markdown


def _landing_type_to_pipeline(landing_type: str) -> str:
    """Map Landing AI chunk type to pipeline type (text, table, figure)."""
    t = str(landing_type or "").lower()
    if "table" in t or t == "table":
        return "table"
    if "figure" in t or "logo" in t or "scancode" in t:
        return "figure"
    return "text"


def _chunk_page_matches(chunk_page_0: int, target_page_1: int) -> bool:
    """Landing AI uses 0-based page; extraction uses 1-based."""
    return chunk_page_0 + 1 == target_page_1


def load_landing_ai_parse(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Load landing_ai_parse_output.json for a document.
    Returns raw parse data or None if not found.
    """
    path = PIPELINE_RESULTS / doc_id / "chunking" / "landing_ai_parse_output.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_highlights_by_value(
    doc_id: str,
    value: str,
    limit: Optional[int] = 1,
) -> Dict[str, Any]:
    """
    Get highlight boxes for chunks that contain the exact value (or normalized variant).
    Returns { available, highlights: [{ page, box }], match_source: "value" }.
    limit: max highlights to return; None = return all matches.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return {
            "available": False,
            "highlights": [],
            "doc_id": doc_id,
            "match_source": "value",
        }

    chunks = parse_data.get("chunks") or []
    variants = _normalize_value_for_search(value)
    if not variants:
        return {
            "available": False,
            "highlights": [],
            "doc_id": doc_id,
            "match_source": "value",
        }

    highlights: List[Dict[str, Any]] = []
    for chunk in chunks:
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        box = grounding.get("box")
        page_0 = grounding.get("page", 0)
        if not box or not isinstance(box, dict):
            continue
        text = _chunk_text(chunk)
        if not text:
            continue
        for v in variants:
            if v in text:
                b = {
                    "left": float(box.get("left", 0)),
                    "top": float(box.get("top", 0)),
                    "right": float(box.get("right", 1)),
                    "bottom": float(box.get("bottom", 1)),
                }
                area = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
                highlights.append({"page": int(page_0) + 1, "box": b, "_area": area})
                break

    # Prefer smallest box (most specific); apply limit
    if highlights:
        highlights.sort(key=lambda x: x.pop("_area", 1))
        if limit is not None:
            highlights = highlights[:limit]

    return {
        "available": True,
        "highlights": highlights,
        "doc_id": doc_id,
        "match_source": "value",
    }


def get_highlights_by_chunk_ids(
    doc_id: str,
    chunk_ids: List[str],
) -> Dict[str, Any]:
    """
    Get highlight boxes for specific chunk IDs (e.g. from attribution).
    Returns { available, highlights: [{ page, box, chunk_id }] }.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return {"available": False, "highlights": [], "doc_id": doc_id}

    ids_set = {c.strip() for c in chunk_ids if c and str(c).strip()}
    chunks = parse_data.get("chunks") or []
    chunk_by_id = {c.get("id"): c for c in chunks if c.get("id")}

    highlights: List[Dict[str, Any]] = []
    for cid in ids_set:
        chunk = chunk_by_id.get(cid)
        if not chunk:
            continue
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        box = grounding.get("box")
        page_0 = grounding.get("page", 0)
        if not box or not isinstance(box, dict):
            continue
        b = {
            "left": float(box.get("left", 0)),
            "top": float(box.get("top", 0)),
            "right": float(box.get("right", 1)),
            "bottom": float(box.get("bottom", 1)),
        }
        highlights.append({"page": int(page_0) + 1, "box": b, "chunk_id": cid})

    return {
        "available": len(highlights) > 0,
        "highlights": highlights,
        "doc_id": doc_id,
        "match_source": "chunk_ids",
    }


def get_highlights_by_page_type(
    doc_id: str,
    page: int,
    source_type: str,
) -> Dict[str, Any]:
    """
    Get highlight boxes for chunks matching page and source_type.
    Returns { available, highlights: [{ page, box }], page, source_type }.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return {
            "available": False,
            "highlights": [],
            "page": page,
            "source_type": source_type,
            "doc_id": doc_id,
        }

    chunks = parse_data.get("chunks") or []
    target_type = str(source_type or "text").lower()
    if target_type not in ("text", "table", "figure"):
        target_type = "text"

    highlights: List[Dict[str, Any]] = []
    for chunk in chunks:
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        box = grounding.get("box")
        page_0 = grounding.get("page", 0)
        if not box or not isinstance(box, dict):
            continue
        chunk_type = _landing_type_to_pipeline(chunk.get("type", "text"))
        if chunk_type != target_type:
            continue
        if not _chunk_page_matches(int(page_0), page):
            continue
        b = {
            "left": float(box.get("left", 0)),
            "top": float(box.get("top", 0)),
            "right": float(box.get("right", 1)),
            "bottom": float(box.get("bottom", 1)),
        }
        area = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
        highlights.append({"page": int(page_0) + 1, "box": b, "_area": area})

    # Prefer smallest box; limit to 1 chunk for cleanest view
    if highlights:
        highlights.sort(key=lambda x: x.pop("_area", 1))
        highlights = highlights[:1]

    return {
        "available": True,
        "highlights": highlights,
        "page": page,
        "source_type": source_type,
        "doc_id": doc_id,
    }


def get_highlights_for_column(
    doc_id: str,
    column_name: str,
    comparison_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get highlight boxes for a column.
    Strategy: 1) Try exact value match in chunks (most accurate); 2) Fall back to page+type.
    """
    if comparison_data is None:
        try:
            from web.comparison_service import load_comparison_data
            comparison_data = load_comparison_data(doc_id)
        except Exception:
            comparison_data = {}

    comparison_rows = comparison_data.get("comparison", [])
    method_results = comparison_data.get("method_results", {})

    # Resolve value and page/source_type from comparison row
    # Prefer pipeline value (shortest/most precise) for value-based matching
    value = None
    page = None
    source_type = None
    for row in comparison_rows:
        if row.get("column_name") != column_name:
            continue
        methods = row.get("methods") or {}
        for method_name in ("pipeline", "pipeline_plan_extract", "pipeline_keywords", "landing_ai_baseline"):
            col_data = methods.get(method_name)
            if col_data:
                v = col_data.get("value") or col_data.get("primary_value")
                if v and str(v).strip() and str(v).lower() not in ("not found", "not reported", "not applicable", "—", "-"):
                    v_str = str(v).strip()
                    # Prefer shorter value (more likely to match in chunk)
                    if value is None or len(v_str) < len(value):
                        value = v_str
                p = col_data.get("page")
                st = col_data.get("source_type")
                if p is not None and st and page is None:
                    page = p
                    source_type = st
        if page is None:
            for col_data in methods.values():
                if col_data and col_data.get("page") is not None:
                    page = col_data.get("page")
                    source_type = col_data.get("source_type") or "text"
                    break
        break

    if page is None and method_results:
        for method_name in ("pipeline", "pipeline_plan_extract", "landing_ai_baseline"):
            col_data = method_results.get(method_name, {}).get(column_name, {})
            if col_data and col_data.get("page") is not None:
                page = col_data.get("page")
                source_type = col_data.get("source_type") or "text"
                if not value:
                    v = col_data.get("value") or col_data.get("primary_value")
                    if v and str(v).strip() and str(v).lower() not in ("not found", "not reported", "not applicable", "—", "-"):
                        value = str(v).strip()
                break

    # 1. Try value-based matching first (most accurate)
    if value:
        result = get_highlights_by_value(doc_id, value)
        if result.get("highlights"):
            result["column_name"] = column_name
            result["match_source"] = "value"
            return result

    # 2. Fall back to page+type
    if page is not None and source_type:
        result = get_highlights_by_page_type(doc_id, int(page), source_type or "text")
        result["column_name"] = column_name
        result["match_source"] = "page_type"
        return result

    return {
        "available": False,
        "highlights": [],
        "column_name": column_name,
        "doc_id": doc_id,
        "reason": "Could not resolve value or page/source_type for column",
    }


def count_value_matches_for_column(
    doc_id: str,
    column_name: str,
    comparison_data: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Return count of chunks that match the column's value (for value-based search).
    Used to assess impact of highlighting all matches vs. just one.
    """
    if comparison_data is None:
        try:
            from web.comparison_service import load_comparison_data
            comparison_data = load_comparison_data(doc_id)
        except Exception:
            return 0

    comparison_rows = comparison_data.get("comparison", [])
    method_results = comparison_data.get("method_results", {})

    value = None
    for row in comparison_rows:
        if row.get("column_name") != column_name:
            continue
        methods = row.get("methods") or {}
        for method_name in ("pipeline", "pipeline_plan_extract", "pipeline_keywords", "landing_ai_baseline"):
            col_data = methods.get(method_name)
            if col_data:
                v = col_data.get("value") or col_data.get("primary_value")
                if v and str(v).strip() and str(v).lower() not in ("not found", "not reported", "not applicable", "—", "-"):
                    v_str = str(v).strip()
                    if value is None or len(v_str) < len(value):
                        value = v_str
        break

    if not value and method_results:
        for method_name in ("pipeline", "pipeline_plan_extract", "landing_ai_baseline"):
            col_data = method_results.get(method_name, {}).get(column_name, {})
            if col_data:
                v = col_data.get("value") or col_data.get("primary_value")
                if v and str(v).strip() and str(v).lower() not in ("not found", "not reported", "not applicable", "—", "-"):
                    value = str(v).strip()
                    break

    if not value:
        return 0

    result = get_highlights_by_value(doc_id, value, limit=None)
    return len(result.get("highlights", []))


def resolve_pdf_path(doc_id: str) -> Optional[Path]:
    """Resolve PDF file path for a document. Returns Path or None."""
    candidates = [
        DATASET_DIR / f"{doc_id}.pdf",
        PIPELINE_RESULTS / doc_id / f"{doc_id}.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None
