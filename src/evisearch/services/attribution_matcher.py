"""
attribution_matcher.py

Custom keyword matching for attribution: numeric parts + column-name tokens only.
No free-form words from values (they can mislead).
Implements Phase 0 (attribution: text/table/figure), Phase 1 (numeric match), Phase 2 (planner page/type) per ATTRIBUTION_ALGORITHM_SPEC.md.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Column-name stopwords: short/common words that add noise
COLUMN_STOPWORDS = frozenset({
    "the", "of", "and", "or", "to", "in", "with", "for", "by", "at",
    "standard", "care", "trial", "not", "all", "any", "per", "n",
})


def extract_numeric_parts(value: str) -> List[str]:
    """
    Extract numeric parts only from a value string.
    No words from value — per spec, value words can mislead.

    Keeps: integers ≥3 digits, 2-digit numbers (excl. 00), decimals, percentages.
    """
    value = str(value or "").strip()
    if not value or value.lower() in ("not reported", "not found", "not applicable", "—", "-", "n/a"):
        return []

    parts = []
    # Percentages first (e.g. 38%, (45%)) — normalize to "38%", "45%"
    for m in re.finditer(r"\d+\s*%", value):
        parts.append(m.group(0).replace(" ", ""))
    # Integers ≥3 digits
    for m in re.finditer(r"\b\d{3,}\b", value):
        parts.append(m.group(0))
    # 2-digit numbers (excl 00)
    for m in re.finditer(r"\b([1-9]\d|\d[1-9])\b", value):
        parts.append(m.group(0))
    # Decimals (e.g. 45.7, 76.6)
    for m in re.finditer(r"\b\d+\.\d+\b", value):
        parts.append(m.group(0))

    seen = set()
    unique = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def extract_numeric_parts_from_values(value_strs: List[str]) -> Tuple[List[str], List[str], bool]:
    """
    Extract numeric parts from multiple value strings (final_value + method_values).
    Returns (required_parts, all_parts, has_matchable).
    - required_parts: from first value only — chunk must contain ALL
    - all_parts: merged from all — used for ranking
    """
    if not value_strs:
        return [], [], False

    required = extract_numeric_parts(value_strs[0])
    all_parts = list(required)
    seen = set(all_parts)
    for v in value_strs[1:]:
        for p in extract_numeric_parts(v):
            if p not in seen:
                seen.add(p)
                all_parts.append(p)
    return required, all_parts, len(required) > 0


def extract_column_tokens(column_name: str) -> List[str]:
    """
    Extract tokens from column name for ranking. Split on |, -, space, ().
    Keep: length ≥ 3, not pure numbers, not stopwords.
    """
    if not column_name:
        return []
    text = str(column_name).strip()
    # Split on delimiters
    tokens = re.split(r"[\|\s\-\(\)]+", text)
    out = []
    seen = set()
    for t in tokens:
        t = t.strip(".,;:[]").strip()
        if not t or len(t) < 3:
            continue
        if t.isdigit():
            continue
        low = t.lower()
        if low in COLUMN_STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            out.append(low)
    return out


def chunk_contains_all_parts(parts: List[str], chunk_text: str) -> bool:
    """True if chunk contains ALL required parts."""
    if not parts:
        return False
    chunk_lower = chunk_text.lower()
    chunk_norm = chunk_text.replace(" ", "")
    for p in parts:
        if "%" in p:
            pn = p.replace(" ", "")
            if pn not in chunk_norm and p not in chunk_text:
                return False
        elif p.replace(".", "").isdigit():
            if len(p) <= 2:
                if not re.search(r"(^|[^\d])" + re.escape(p) + r"([^\d]|$)", chunk_text):
                    return False
            elif p not in chunk_text:
                return False
        else:
            if p.lower() not in chunk_lower:
                return False
    return True


def normalize_for_search(text: str) -> str:
    """
    Normalize text for snippet matching: collapse [ ] and ( ) around numbers/percentages,
    unify whitespace, preserve numbers and structure.
    """
    if not text or not isinstance(text, str):
        return ""
    s = str(text).strip()
    # Collapse multiple spaces/newlines
    s = re.sub(r"\s+", " ", s)
    # Normalize [38%] and (38%) to 38% for matching
    s = re.sub(r"[(\[]\s*(\d+)\s*%\s*[)\]]", r"\1%", s)
    s = re.sub(r"[(\[]\s*(\d+(?:\.\d+)?)\s*[)\]]", r"\1", s)
    return s.strip()


def snippet_match_score(snippet: str, chunk_text: str) -> float:
    """
    Score how well snippet matches chunk. Returns 0 if no match.
    - 1.0: exact/normalized substring (snippet in chunk)
    - 0.95: best local alignment >= 90% of snippet length + all numbers in chunk
    - 0.85: best local alignment >= 80% of snippet + all numbers in chunk
    """
    if not snippet or len(snippet.strip()) < 10:
        return 0.0
    snippet_norm = normalize_for_search(snippet)
    chunk_norm = normalize_for_search(chunk_text)
    if not snippet_norm or len(snippet_norm) < 10:
        return 0.0
    # Exact/normalized substring
    if snippet_norm in chunk_norm:
        return 1.0
    # Fuzzy: find longest matching block — snippet-like span in chunk
    matcher = SequenceMatcher(None, snippet_norm, chunk_norm)
    match = matcher.find_longest_match(0, len(snippet_norm), 0, len(chunk_norm))
    if match and match.size > 0:
        overlap_ratio = match.size / len(snippet_norm)
        parts = extract_numeric_parts(snippet)
        if overlap_ratio >= 0.9 and (not parts or chunk_contains_all_parts(parts, chunk_text)):
            return 0.95
        if overlap_ratio >= 0.8 and (not parts or chunk_contains_all_parts(parts, chunk_text)):
            return 0.85
    return 0.0


def _chunk_contains_identifier(chunk_text: str, identifier: str) -> bool:
    """True if chunk text contains identifier (case-insensitive, normalized whitespace)."""
    if not identifier or not chunk_text:
        return False
    id_norm = " ".join(str(identifier).strip().lower().split())
    chunk_norm = " ".join(chunk_text.lower().split())
    return id_norm in chunk_norm


def phase0_text_source(
    valid_chunks: List[Dict],
    page: int,
    snippet: str,
    chunk_text_fn,
    landing_type_to_pipeline_fn,
    top_k: int = 2,
) -> List[Tuple[Dict, float]]:
    """Text source: filter by page+text type, match snippet in chunk."""
    if not snippet or len(snippet.strip()) < 10:
        return []
    scored = []
    for c in valid_chunks:
        if not _chunk_on_page_and_type(c, page, "text", landing_type_to_pipeline_fn):
            continue
        text = chunk_text_fn(c)
        score = snippet_match_score(snippet, text)
        if score >= 0.8:
            scored.append((c, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def phase0_table_source(
    valid_chunks: List[Dict],
    page: int,
    table_number: str,
    caption: str,
    chunk_text_fn,
    landing_type_to_pipeline_fn,
    top_k: int = 2,
) -> List[Tuple[Dict, float]]:
    """Table source: filter by page+table type, prefer chunks containing table_number."""
    if not table_number:
        return []
    candidates = []
    for c in valid_chunks:
        if not _chunk_on_page_and_type(c, page, "table", landing_type_to_pipeline_fn):
            continue
        text = chunk_text_fn(c)
        has_num = _chunk_contains_identifier(text, table_number)
        has_cap = _chunk_contains_identifier(text, caption) if caption else False
        score = 0.95 if has_num else (0.85 if has_cap else 0.8)
        candidates.append((c, score))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def phase0_figure_source(
    valid_chunks: List[Dict],
    page: int,
    figure_number: str,
    caption: str,
    chunk_text_fn,
    landing_type_to_pipeline_fn,
    top_k: int = 2,
) -> List[Tuple[Dict, float]]:
    """Figure source: filter by page+figure type, prefer chunks containing figure_number."""
    if not figure_number:
        return []
    candidates = []
    for c in valid_chunks:
        if not _chunk_on_page_and_type(c, page, "figure", landing_type_to_pipeline_fn):
            continue
        text = chunk_text_fn(c)
        has_num = _chunk_contains_identifier(text, figure_number)
        has_cap = _chunk_contains_identifier(text, caption) if caption else False
        score = 0.95 if has_num else (0.85 if has_cap else 0.8)
        candidates.append((c, score))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def phase0_attribution_match(
    valid_chunks: List[Dict],
    attribution: List[Dict[str, Any]],
    chunk_text_fn,
    landing_type_to_pipeline_fn,
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:
    """
    Structured attribution: for each source (text/table/figure), find matching chunks.
    Merge, dedupe by chunk id, return top_k.
    """
    if not attribution or not isinstance(attribution, list):
        return []
    seen_ids: set = set()
    merged: List[Tuple[Dict, float]] = []
    for src in attribution:
        if not isinstance(src, dict):
            continue
        st = str(src.get("source_type") or "").lower()
        page = src.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        if not page or page < 1:
            continue
        results: List[Tuple[Dict, float]] = []
        if st == "text":
            snippet = (src.get("snippet") or "").strip()
            results = phase0_text_source(
                valid_chunks, page, snippet, chunk_text_fn, landing_type_to_pipeline_fn, top_k=2
            )
        elif st == "table":
            table_number = (src.get("table_number") or "").strip()
            caption = (src.get("caption") or "").strip()
            results = phase0_table_source(
                valid_chunks, page, table_number, caption, chunk_text_fn, landing_type_to_pipeline_fn, top_k=2
            )
        elif st == "figure":
            figure_number = (src.get("figure_number") or "").strip()
            caption = (src.get("caption") or "").strip()
            results = phase0_figure_source(
                valid_chunks, page, figure_number, caption, chunk_text_fn, landing_type_to_pipeline_fn, top_k=2
            )
        for c, score in results:
            cid = c.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged.append((c, score))
    merged.sort(key=lambda x: -x[1])
    return merged[:top_k]


def count_parts_in_chunk(parts: List[str], chunk_text: str) -> int:
    """Count how many of the parts appear in chunk (for ranking)."""
    if not parts:
        return 0
    count = 0
    chunk_norm = chunk_text.replace(" ", "")
    chunk_lower = chunk_text.lower()
    for p in parts:
        if "%" in p:
            if p.replace(" ", "") in chunk_norm or p in chunk_text:
                count += 1
        elif p.replace(".", "").isdigit():
            if len(p) <= 2:
                if re.search(r"(^|[^\d])" + re.escape(p) + r"([^\d]|$)", chunk_text):
                    count += 1
            elif p in chunk_text:
                count += 1
        else:
            if p.lower() in chunk_lower:
                count += 1
    return count


def count_column_tokens_in_chunk(col_tokens: List[str], chunk_text: str) -> int:
    """Count column-name tokens that appear in chunk (for ranking)."""
    if not col_tokens:
        return 0
    chunk_lower = chunk_text.lower()
    count = 0
    for t in col_tokens:
        if t in chunk_lower:
            count += 1
    return count


def _chunk_page_1(chunk: Dict) -> int:
    """1-based page number."""
    g = chunk.get("grounding") or {}
    return int(g.get("page", 0)) + 1 if isinstance(g, dict) else 1


def _chunk_on_page_and_type(
    chunk: Dict,
    page_1: int,
    source_type: str,
    landing_type_to_pipeline_fn,
) -> bool:
    """True if chunk is on page and matches source_type."""
    if page_1 < 1:
        return False
    cp = _chunk_page_1(chunk)
    if cp != page_1:
        return False
    st = str(source_type or "text").lower()
    if st in ("not_applicable", "not applicable", "n/a", "na"):
        return True
    if st not in ("text", "table", "figure"):
        return True
    ct = landing_type_to_pipeline_fn(chunk.get("type", "text"))
    return ct == st


def phase1_numeric_match(
    valid_chunks: List[Dict],
    chunk_text_fn,
    required_parts: List[str],
    all_parts: List[str],
    col_tokens: List[str],
    hint_page: Optional[int],
    hint_source_type: Optional[str],
    landing_type_to_pipeline_fn,
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:
    """
    Phase 1: Chunks containing all required numeric parts, ranked.
    Returns list of (chunk, score) for top_k.
    """
    matching = []
    for c in valid_chunks:
        text = chunk_text_fn(c)
        if not chunk_contains_all_parts(required_parts, text):
            continue
        num_score = count_parts_in_chunk(all_parts, text)
        col_score = count_column_tokens_in_chunk(col_tokens, text)
        page_1 = _chunk_page_1(c)
        on_hint = page_1 == hint_page if hint_page and hint_page >= 1 else False
        st = str(hint_source_type or "").lower()
        type_ok = st in ("text", "table", "figure") and landing_type_to_pipeline_fn(
            c.get("type", "text")
        ) == st if st else False
        location_boost = (2 if (on_hint and type_ok) else 1 if on_hint else 0) * 10
        score = num_score * 100 + col_score * 5 + location_boost
        matching.append((c, score))
    matching.sort(key=lambda x: -x[1])
    return matching[:top_k]


def phase2_planner_location(
    valid_chunks: List[Dict],
    chunk_text_fn,
    page_1: int,
    source_type: str,
    col_tokens: List[str],
    landing_type_to_pipeline_fn,
    top_k: int = 3,
) -> List[Tuple[Dict, float]]:
    """
    Phase 2: Chunks on planner page with matching type, ranked by column tokens.
    Returns list of (chunk, score).
    """
    if page_1 < 1:
        return []
    st = str(source_type or "").lower()
    if st in ("not_applicable", "not applicable", "n/a", "na", ""):
        st = "text"
    if st not in ("text", "table", "figure"):
        st = "text"

    page_chunks = [
        c for c in valid_chunks
        if _chunk_on_page_and_type(c, page_1, st, landing_type_to_pipeline_fn)
    ]
    if not page_chunks:
        return []

    scored = []
    for c in page_chunks:
        text = chunk_text_fn(c)
        col_score = count_column_tokens_in_chunk(col_tokens, text)
        scored.append((c, float(col_score)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def chunks_to_attribution_output(
    chunk_score_pairs: List[Tuple[Dict, float]],
    chunk_text_fn,
    landing_type_to_pipeline_fn,
    score_label: float = 1.0,
) -> List[Dict[str, Any]]:
    """Convert (chunk, score) pairs to attribution output format."""
    out = []
    for c, _ in chunk_score_pairs:
        g = c.get("grounding") or {}
        page_0 = g.get("page", 0) if isinstance(g, dict) else 0
        snippet = chunk_text_fn(c)
        out.append({
            "chunk_id": c["id"],
            "page": int(page_0) + 1,
            "source_type": landing_type_to_pipeline_fn(c.get("type", "text")),
            "snippet": (snippet[:200] + "…") if len(snippet) > 200 else snippet,
            "score": score_label,
        })
    return out
