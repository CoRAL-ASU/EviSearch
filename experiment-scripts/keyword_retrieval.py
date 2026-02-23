#!/usr/bin/env python3
"""
keyword_retrieval.py

Keyword-based retrieval to supplement planner-based chunk selection.
Uses Landing AI chunks. BM25 over chunk content, merge with primary chunks
under a token budget (100-200) for the keyword supplement.

Flow:
1. derive_keywords_for_column(column, definition) -> 3-5 domain keywords
2. find_chunks_by_keywords(chunks, keywords, top_k, exclude_chunk_ids) -> BM25 top-K
3. merge_chunks_with_keyword_supplement(primary, keyword_chunks, token_limit)
4. find_chunks_for_column_with_keywords() orchestrates tiered + keyword
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

# Generic terms to exclude from keyword derivation
GENERIC_TERMS: Set[str] = {
    "median", "n", "%", "years", "mo", "count", "percentage", "number", "value",
    "reported", "include", "arm", "treatment", "control", "missing", "not",
    "patients", "trial", "if", "the", "and", "or", "use", "missing",
    "experimental", "participants", "total", "each", "separately",
}

# Clinical/domain terms to prefer (minimum length 2)
MIN_KEYWORD_LEN = 2
MAX_KEYWORDS = 5
DEFAULT_KEYWORD_TOP_K = 5
DEFAULT_KEYWORD_TOKEN_LIMIT = 150

# Token heuristic: ~4 chars per token
CHARS_PER_TOKEN = 4


def _token_count(text: str) -> int:
    """Rough token count: chars / 4."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _tokenize_simple(text: str) -> List[str]:
    """Simple tokenization: split on non-alphanumeric, lowercase, filter short."""
    if not text:
        return []
    tokens = re.sub(r"[^\w\s-]", " ", text.lower()).split()
    return [t for t in tokens if len(t) >= 2]


def derive_keywords_for_column(column: Dict[str, Any], definition: str) -> List[str]:
    """
    Extract 3-5 domain keywords from column name + definition.
    Excludes generic terms. Prefers clinical/domain terms.
    """
    name = str(column.get("column_name", ""))
    defn = str(definition or "")

    # Parse: split on |, _, -, (, ), keep meaningful parts (max 4 words each)
    parts: List[str] = []
    for s in re.split(r"[|_\-()]+", name + " " + defn):
        s = re.sub(r"[^\w\s-]", "", s).strip().lower()
        if len(s) >= MIN_KEYWORD_LEN:
            word_count = len(s.split())
            if word_count <= 4:  # Short phrases only
                parts.append(s)

    # Extract individual words
    words: Set[str] = set()
    for p in parts:
        for w in _tokenize_simple(p):
            if len(w) >= MIN_KEYWORD_LEN and w not in GENERIC_TERMS:
                words.add(w)

    # Add short phrases from column name (e.g. "adverse events", "grade 3", "high volume")
    for p in parts:
        if p not in GENERIC_TERMS and 2 <= len(p.split()) <= 4:
            words.add(p)
        elif len(p.split()) == 1 and p not in GENERIC_TERMS and len(p) >= 2:
            words.add(p)

    # Order: longer first (prefer phrases), then by length
    ordered = sorted(words, key=lambda x: (-len(x), x))
    return list(ordered)[:MAX_KEYWORDS]


def find_chunks_by_keywords(
    chunks: List[Dict[str, Any]],
    keywords: List[str],
    top_k: int = DEFAULT_KEYWORD_TOP_K,
    exclude_chunk_ids: Set[int] | None = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    BM25 retrieval over chunks. Returns top-K chunks not in exclude_chunk_ids.
    Returns list of (chunk_dict, score) sorted by score descending.
    """
    if not keywords or not chunks:
        return []

    exclude = exclude_chunk_ids or set()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    # Build corpus: each chunk = content + table_content
    corpus_texts: List[str] = []
    chunk_indices: List[int] = []
    for idx, chunk in enumerate(chunks):
        if idx in exclude:
            continue
        content = str(chunk.get("content", "") or "")
        table = str(chunk.get("table_content", "") or "")
        text = (content + "\n" + table).strip()
        if not text:
            continue
        corpus_texts.append(text)
        chunk_indices.append(idx)

    if not corpus_texts:
        return []

    tokenized_corpus = [_tokenize_simple(t) for t in corpus_texts]
    if not any(tokenized_corpus):
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    query = " ".join(keywords)
    tokenized_query = _tokenize_simple(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    scored = [(chunk_indices[i], scores[i]) for i in range(len(chunk_indices)) if scores[i] > 0]
    scored.sort(key=lambda x: -x[1])

    result: List[Tuple[Dict[str, Any], float]] = []
    for idx, score in scored[:top_k]:
        chunk = chunks[idx]
        result.append(({
            "chunk_id": idx,
            "type": str(chunk.get("type", "text")).lower(),
            "page": chunk.get("page"),
            "content": chunk.get("content", "") or "",
            "table_content": chunk.get("table_content", "") or "",
        }, score))
    return result


def merge_chunks_with_keyword_supplement(
    primary_chunks: List[Dict[str, Any]],
    keyword_chunks: List[Tuple[Dict[str, Any], float]],
    token_limit: int = DEFAULT_KEYWORD_TOKEN_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Merge primary chunks + keyword chunks. Keyword supplement capped at token_limit.
    Order: primary first, then keyword by score.
    """
    merged = list(primary_chunks)
    if not keyword_chunks or token_limit <= 0:
        return merged

    primary_ids = {c.get("chunk_id") for c in primary_chunks if c.get("chunk_id") is not None}
    used = 0
    for chunk, _score in keyword_chunks:
        cid = chunk.get("chunk_id")
        if cid is not None and cid in primary_ids:
            continue
        text = str(chunk.get("content", "") or "") + "\n" + str(chunk.get("table_content", "") or "")
        tokens = _token_count(text)
        if used + tokens > token_limit:
            # Truncate to fit
            remaining = token_limit - used
            if remaining <= 0:
                break
            char_limit = remaining * CHARS_PER_TOKEN
            if len(text) > char_limit:
                chunk = dict(chunk)
                content = chunk.get("content", "") or ""
                table = chunk.get("table_content", "") or ""
                if len(content) >= char_limit:
                    chunk["content"] = content[:char_limit] + "..."
                    chunk["table_content"] = ""
                else:
                    chunk["table_content"] = (table or "")[:char_limit - len(content)] + "..."
            merged.append(chunk)
            used = token_limit
            break
        merged.append(chunk)
        used += tokens

    return merged


def find_chunks_for_column_with_keywords(
    column: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    definition: str,
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
    keyword_token_limit: int = DEFAULT_KEYWORD_TOKEN_LIMIT,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Tiered retrieval + keyword supplement. Uses Landing AI chunks.

    Returns (merged_chunks, retrieval_source, retrieval_meta).
    retrieval_meta: {primary_count, keyword_count, keywords, keyword_tokens_added}
    """
    from extract_with_landing_ai import find_chunks_for_column_tiered

    primary_chunks, source = find_chunks_for_column_tiered(column, chunks)
    primary_ids = {c.get("chunk_id") for c in primary_chunks if c.get("chunk_id") is not None}

    meta: Dict[str, Any] = {
        "primary_count": len(primary_chunks),
        "keyword_count": 0,
        "keywords": [],
        "keyword_tokens_added": 0,
    }

    keywords = derive_keywords_for_column(column, definition)
    if not keywords:
        return primary_chunks, source, meta

    meta["keywords"] = keywords

    keyword_scored = find_chunks_by_keywords(
        chunks, keywords, top_k=keyword_top_k, exclude_chunk_ids=primary_ids
    )
    if not keyword_scored:
        return primary_chunks, source, meta

    merged = merge_chunks_with_keyword_supplement(
        primary_chunks, keyword_scored, token_limit=keyword_token_limit
    )

    meta["keyword_count"] = len(merged) - len(primary_chunks)
    meta["keyword_tokens_added"] = sum(
        _token_count(str(c.get("content", "") or "") + "\n" + str(c.get("table_content", "") or ""))
        for c in merged[len(primary_chunks):]
    )

    new_source = f"{source}+keywords" if meta["keyword_count"] > 0 else source
    return merged, new_source, meta
