"""
highlight_service.py

Load Landing AI parse output and return bounding boxes for PDF highlighting.
Used by the attribution viewer to show attributed chunks on the PDF.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RESULTS = PROJECT_ROOT / "new_pipeline_outputs" / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"


def _chunk_text(chunk: Dict[str, Any]) -> str:
    """Get searchable text from a Landing AI chunk."""
    markdown = str(chunk.get("markdown") or "")
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
