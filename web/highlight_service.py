"""
highlight_service.py

Load Landing AI parse output and return bounding boxes for PDF highlighting.
Used by the attribution viewer to show attributed chunks on the PDF.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_RESULTS = PROJECT_ROOT / "new_pipeline_outputs" / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"
UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"


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


def get_full_chunk_texts(doc_id: str, chunk_ids: list[str]) -> dict[str, str]:
    """
    Get full text for each chunk_id from Landing AI parse.
    Returns {chunk_id: full_text}. Uses full markdown, not truncated snippet.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return {}
    chunks = parse_data.get("chunks") or []
    chunk_by_id = {c.get("id"): c for c in chunks if c.get("id")}
    out = {}
    for cid in chunk_ids:
        if not cid:
            continue
        chunk = chunk_by_id.get(cid)
        if chunk:
            out[cid] = _chunk_text(chunk)
    return out


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


def get_chunks_by_page_type(
    doc_id: str,
    page: int,
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Get chunks matching page and source_type from Landing AI parse.
    Returns list of {chunk_id, page, source_type, text}.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []

    chunks = parse_data.get("chunks") or []
    target_type = str(source_type or "text").lower()
    if target_type not in ("text", "table", "figure"):
        target_type = "text"

    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        page_0 = grounding.get("page", 0)
        chunk_type = _landing_type_to_pipeline(chunk.get("type", "text"))
        if chunk_type != target_type:
            continue
        if not _chunk_page_matches(int(page_0), page):
            continue
        cid = chunk.get("id")
        if cid:
            out.append({
                "chunk_id": cid,
                "page": int(page_0) + 1,
                "source_type": chunk_type,
                "text": _chunk_text(chunk),
            })
    return out


def _extract_table_number(raw: str) -> Optional[int]:
    """Extract table number from caption, e.g. 'TABLE 1.' or 'Table 2:'. Returns int or None."""
    m = re.search(r"Table\s+(\d+)[.:]?", raw, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_figure_number(raw: str) -> Optional[int]:
    """Extract figure number from caption, e.g. 'Figure 1' or 'Fig. 2A'. Returns int or None."""
    m = re.search(r"(?:Fig\.?|Figure)\s*(\d+)[A-Za-z]?", raw, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _get_chunks_by_page_and_single_number(
    chunks: List[Dict[str, Any]],
    page: int,
    target_num: int,
    target_type: str,
) -> List[Dict[str, Any]]:
    """Helper: get chunks for one (page, type, number)."""
    candidates: List[Tuple[Dict[str, Any], float, Optional[int]]] = []
    for chunk in chunks:
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        page_0 = grounding.get("page", 0)
        chunk_type = _landing_type_to_pipeline(chunk.get("type", "text"))
        if chunk_type != target_type or not _chunk_page_matches(int(page_0), page):
            continue
        raw = _chunk_text(chunk)
        box = grounding.get("box") or {}
        top = float(box.get("top", 0)) if isinstance(box, dict) else 0.0
        num = _extract_table_number(raw) if target_type == "table" else _extract_figure_number(raw)
        candidates.append((chunk, top, num))

    if not candidates:
        return []

    caption_matches = [c for c in candidates if c[2] == target_num]
    if caption_matches:
        chosen = caption_matches
    else:
        chosen = sorted(candidates, key=lambda x: (x[1], x[0].get("id", "")))
        idx = min(target_num - 1, len(chosen) - 1)
        chosen = [chosen[idx]]

    return [
        {"chunk_id": c[0].get("id"), "page": page, "source_type": target_type, "text": _chunk_text(c[0])}
        for c in chosen if c[0].get("id")
    ]


def get_chunks_by_page_and_number(
    doc_id: str,
    page: int,
    table_num: Optional[int] = None,
    figure_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Get Landing AI chunks for a specific table or figure by page and number.
    Use when retrieval points to "Table 2" or "Figure 1" - fetches only that chunk (light memory).

    Algorithm:
    1. Filter chunks by page and source_type (table/figure).
    2. Match by caption number (Table N, Figure N) when present.
    3. Fallback: sort by position (box.top) and take Nth (1-indexed).

    Returns list of {chunk_id, page, source_type, text}.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []

    chunks = parse_data.get("chunks") or []
    out: List[Dict[str, Any]] = []
    if table_num is not None and table_num >= 1:
        out.extend(_get_chunks_by_page_and_single_number(chunks, page, table_num, "table"))
    if figure_num is not None and figure_num >= 1:
        out.extend(_get_chunks_by_page_and_single_number(chunks, page, figure_num, "figure"))
    return out


def _normalize_for_match(text: str) -> str:
    """Whitespace + Unicode normalize for relaxed verbatim match (Options A+B)."""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text).strip())
    return unicodedata.normalize("NFC", t)


def get_chunks_by_page_and_verbatim(
    doc_id: str,
    page: int,
    verbatim_quote: str,
) -> List[Dict[str, Any]]:
    """
    Get text chunks on a page whose markdown contains the verbatim quote (relaxed match).
    Uses Options A+B: whitespace collapse, Unicode NFC. Then substring or 80% word-overlap fallback.
    Returns list of {chunk_id, page, source_type, text}.
    """
    verbatim = (verbatim_quote or "").strip()
    if len(verbatim) < 5:
        return []

    text_chunks = get_chunks_by_page_type(doc_id, page, "text")
    if not text_chunks:
        return []

    norm_verbatim = _normalize_for_match(verbatim)
    verbatim_words = [w for w in norm_verbatim.split() if len(w) > 1]
    min_overlap = max(1, int(0.8 * len(verbatim_words))) if verbatim_words else 1

    candidates: List[Tuple[Dict[str, Any], float]] = []
    for chunk in text_chunks:
        chunk_text = chunk.get("text", "") or ""
        norm_chunk = _normalize_for_match(chunk_text)
        if not norm_chunk:
            continue

        # Substring match (preferred)
        if norm_verbatim in norm_chunk:
            candidates.append((chunk, 1.0))
            continue

        # Word-overlap fallback: verbatim words in chunk, in order
        chunk_words = norm_chunk.split()
        v_idx = 0
        overlap = 0
        for cw in chunk_words:
            if v_idx < len(verbatim_words) and verbatim_words[v_idx].lower() in cw.lower():
                overlap += 1
                v_idx += 1
        if overlap >= min_overlap:
            score = overlap / len(verbatim_words) if verbatim_words else 0.9
            candidates.append((chunk, score))

    if not candidates:
        return []
    # Return best matches (score >= 0.8), sorted by score desc
    best = sorted(candidates, key=lambda x: -x[1])
    return [{"chunk_id": c[0]["chunk_id"], "page": c[0]["page"], "source_type": c[0]["source_type"], "text": c[0]["text"]} for c in best if c[1] >= 0.8]


def get_chunks_by_page(
    doc_id: str,
    page: int,
) -> List[Dict[str, Any]]:
    """
    Get all chunks on a page (any modality).
    Returns list of {chunk_id, page, source_type, text}.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []

    chunks = parse_data.get("chunks") or []
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        grounding = chunk.get("grounding") or {}
        if not isinstance(grounding, dict):
            continue
        page_0 = grounding.get("page", 0)
        if not _chunk_page_matches(int(page_0), page):
            continue
        cid = chunk.get("id")
        if cid:
            chunk_type = _landing_type_to_pipeline(chunk.get("type", "text"))
            out.append({
                "chunk_id": cid,
                "page": int(page_0) + 1,
                "source_type": chunk_type,
                "text": _chunk_text(chunk),
            })
    return out


def resolve_pdf_path(doc_id: str) -> Optional[Path]:
    """Resolve PDF file path for a document. Returns Path or None."""
    # 1. Uploads (upload_* before extraction copies to results)
    if doc_id.startswith("upload_") and UPLOAD_DIR.exists():
        p = UPLOAD_DIR / f"{doc_id}.pdf"
        if p.exists():
            return p
    # 2. Results folder (includes upload_* after extraction)
    p = PIPELINE_RESULTS / doc_id / f"{doc_id}.pdf"
    if p.exists():
        return p
    # 3. Exact match: dataset/<doc_id>.pdf
    p = DATASET_DIR / f"{doc_id}.pdf"
    if p.exists():
        return p
    # 4. Dataset subdirs: doc_id may be "subdir_stem" from dataset/subdir/stem.pdf
    if DATASET_DIR.exists():
        for pdf in DATASET_DIR.glob("**/*.pdf"):
            stem = pdf.stem
            if stem == doc_id or stem in doc_id or doc_id in stem:
                return pdf
    return None
