"""
attribution_service.py

Attribution retrieval: custom matcher (numeric + column tokens) → planner location → semantic fallback.
See web/attribution_matcher.py for Phase 1 & 2; semantic is Phase 3.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from web.highlight_service import _chunk_text, _landing_type_to_pipeline, load_landing_ai_parse
from web.attribution_matcher import (
    extract_numeric_parts_from_values,
    extract_column_tokens,
    phase1_numeric_match,
    phase2_planner_location,
    chunks_to_attribution_output,
)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_ST = True
except ImportError:
    HAS_ST = False
    np = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "new_pipeline_outputs" / "chunk_embeddings"

# Strong retrieval model for document chunks. Options:
# - BAAI/bge-large-en-v1.5: 1024 dim, strong retrieval, English (recommended for clinical docs)
# - BAAI/bge-base-en-v1.5: 768 dim, faster, good retrieval
# - jinaai/jina-embeddings-v3: 1024 dim, 8192 tokens (needs trust_remote_code=True)
# - all-MiniLM-L6-v2: 384 dim, lightweight baseline
MODEL_NAME = os.environ.get("ATTRIBUTION_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")


def _safe_cache_name(doc_id: str) -> str:
    base = re.sub(r"[^\w\-]", "_", doc_id)[:120] or "doc"
    model_tag = re.sub(r"[^\w\-]", "_", MODEL_NAME.split("/")[-1])[:60]
    return f"{base}_{model_tag}"


def _get_model():
    if not HAS_ST:
        return None
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None


def _chunk_text_clean(chunk: Dict) -> str:
    text = _chunk_text(chunk)
    return re.sub(r"<::[^>]*::>", "", text).strip()[:4000]


def get_chunk_embeddings(doc_id: str) -> Tuple[List[Dict], Optional[Any]]:
    """
    Load chunks, return (chunks_list, embeddings_matrix). Uses disk cache per doc.
    embeddings_matrix: (n_chunks, dim) np array. Builds and caches if not present.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return [], None

    chunks = parse_data.get("chunks") or []
    valid = []
    texts = []
    for c in chunks:
        if not c.get("id"):
            continue
        t = _chunk_text_clean(c)
        if not t or len(t) < 10:
            continue
        valid.append(c)
        texts.append(t)

    if not valid:
        return [], None

    cache_name = _safe_cache_name(doc_id)
    cache_path = CACHE_DIR / f"{cache_name}.npz"
    meta_path = CACHE_DIR / f"{cache_name}_meta.json"

    # Try load from cache
    if cache_path.exists() and meta_path.exists():
        try:
            data = np.load(cache_path)
            embs = data["embeddings"]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_ids = meta.get("chunk_ids", [])
            if len(cached_ids) == len(valid) and all(
                valid[i].get("id") == cached_ids[i] for i in range(len(valid))
            ):
                return valid, np.asarray(embs)
        except Exception:
            pass

    # Build and cache
    model = _get_model()
    if not model:
        return valid, None

    embs = model.encode(texts, show_progress_bar=False)
    embs = np.asarray(embs)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embs)
    meta_path.write_text(
        json.dumps({"chunk_ids": [c["id"] for c in valid]}),
        encoding="utf-8",
    )

    return valid, embs


def _chunk_on_page(chunk: Dict, page_1: int) -> bool:
    """True if chunk is on given page (1-based)."""
    g = chunk.get("grounding") or {}
    if not isinstance(g, dict):
        return False
    page_0 = g.get("page", 0)
    return int(page_0) + 1 == page_1


def _chunk_type_matches(chunk: Dict, source_type: str) -> bool:
    """True if chunk type matches pipeline source_type (text/table/figure)."""
    t = _landing_type_to_pipeline(chunk.get("type", "text"))
    st = str(source_type or "text").lower()
    if st not in ("text", "table", "figure"):
        return True
    return t == st


def _parse_pages_from_evidence(evidence: str) -> List[int]:
    """Extract page numbers from evidence text, e.g. 'page 1', 'Table 1 on page 5', 'pages 5-6'."""
    pages = []
    for m in re.finditer(r"page\s+(\d+)", evidence or "", re.I):
        pages.append(int(m.group(1)))
    for m in re.finditer(r"pages?\s+(\d+)\s*[-–]\s*(\d+)", evidence or "", re.I):
        for p in range(int(m.group(1)), int(m.group(2)) + 1):
            pages.append(p)
    return list(dict.fromkeys(pages))  # dedupe preserving order


def retrieve_chunks_for_evidence(
    doc_id: str,
    evidence_query: str,
    top_k: int = 3,
    column_name: Optional[str] = None,
    final_value: Optional[str] = None,
    pipeline_page: Optional[int] = None,
    pipeline_source_type: Optional[str] = None,
    evidence_text: Optional[str] = None,
    method_values: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Strategy: VALUE MATCH FIRST, using pipeline location hints when available.
    - When pipeline has page/type: prefer chunks on that page and type
    - Parse pages from evidence as fallback location hint
    - Fall back to semantic/evidence retrieval only when value has no matchable content.
    Returns list of {chunk_id, page, source_type, snippet, score}.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []

    chunks_raw = parse_data.get("chunks") or []
    chunks = [c for c in chunks_raw if c.get("id")]
    valid = []
    for c in chunks:
        t = _chunk_text_clean(c)
        if t and len(t) >= 10:
            valid.append(c)

    if not valid:
        return []

    # Location hints: pipeline page/type, plus pages parsed from evidence
    hint_pages = set()
    if pipeline_page is not None and pipeline_page >= 1:
        hint_pages.add(int(pipeline_page))
    if evidence_text:
        hint_pages.update(_parse_pages_from_evidence(evidence_text))

    # Collect value strings: final_value + method values (exclude "not found" etc.)
    all_value_strs = [final_value or ""]
    if method_values:
        for v in method_values:
            v = str(v or "").strip()
            if v and v.lower() not in ("not found", "not reported", "not applicable", "n/a"):
                all_value_strs.append(v)

    required_parts, all_parts, has_numeric = extract_numeric_parts_from_values(all_value_strs)
    col_tokens = extract_column_tokens(column_name or "")

    # Resolve page: prefer pipeline_page, fallback to first from evidence
    hint_page = None
    if pipeline_page is not None and pipeline_page >= 1:
        try:
            hint_page = int(pipeline_page)
        except (TypeError, ValueError):
            pass
    if hint_page is None and hint_pages:
        hint_page = min(hint_pages) if hint_pages else None

    # Phase 1: Numeric match — chunks containing ALL required numeric parts
    if has_numeric and required_parts:
        matched = phase1_numeric_match(
            valid,
            _chunk_text_clean,
            required_parts,
            all_parts,
            col_tokens,
            hint_page,
            pipeline_source_type,
            _landing_type_to_pipeline,
            top_k=top_k,
        )
        if matched:
            return chunks_to_attribution_output(
                matched,
                _chunk_text_clean,
                _landing_type_to_pipeline,
                score_label=1.0,
            )

    # Phase 2: Planner location — chunks on page with matching type
    if hint_page and hint_page >= 1 and pipeline_source_type:
        st = str(pipeline_source_type or "").lower()
        if st not in ("not applicable", "not_applicable", "n/a", "na"):
            matched = phase2_planner_location(
                valid,
                _chunk_text_clean,
                hint_page,
                pipeline_source_type,
                col_tokens,
                _landing_type_to_pipeline,
                top_k=top_k,
            )
            if matched:
                return chunks_to_attribution_output(
                    matched,
                    _chunk_text_clean,
                    _landing_type_to_pipeline,
                    score_label=0.9,  # planner-based
                )

    # Phase 3: Semantic retrieval fallback
    if not HAS_ST:
        return []

    chunks_list, embs = get_chunk_embeddings(doc_id)
    if not chunks_list or embs is None:
        return []

    model = _get_model()
    if not model:
        return []

    q = model.encode([evidence_query[:2000]], show_progress_bar=False)[0]
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    scores = np.dot(embs, q_norm)
    fetch_k = min(len(chunks_list), max(top_k * 3, 20))
    order = np.argsort(scores)[::-1][:fetch_k]

    # Re-rank semantic results by pipeline location when we have hints
    if hint_pages:
        scored = []
        for i in order:
            c = chunks_list[i]
            page_1 = int((c.get("grounding") or {}).get("page", 0)) + 1
            on_hint = page_1 in hint_pages
            type_ok = _chunk_type_matches(c, pipeline_source_type or "") if pipeline_source_type else True
            boost = (2 if (on_hint and type_ok) else 1 if on_hint else 0) * 0.2
            scored.append((i, float(scores[i]) + boost))
        scored.sort(key=lambda x: -x[1])
        order = [x[0] for x in scored[:top_k]]
    else:
        order = order[:top_k]

    out = []
    for i in order:
        c = chunks_list[i]
        g = c.get("grounding") or {}
        page_0 = g.get("page", 0) if isinstance(g, dict) else 0
        snippet = _chunk_text_clean(c)
        out.append({
            "chunk_id": c["id"],
            "page": int(page_0) + 1,
            "source_type": _landing_type_to_pipeline(c.get("type", "text")),
            "snippet": (snippet[:200] + "…") if len(snippet) > 200 else snippet,
            "score": float(scores[i]),
        })
    return out


def enrich_reconciled_with_attribution(
    doc_id: str,
    reconciled_columns: List[Dict[str, Any]],
    comparison_rows: Optional[List[Dict]] = None,
    top_k: int = 3,
    use_semantic: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each reconciled column: collate evidence, run semantic retrieval, add attributed_chunks.
    """
    col_to_row = {r.get("column_name"): r for r in (comparison_rows or [])}

    enriched = []
    for col in reconciled_columns:
        col_name = col.get("column_name", "")
        final_value = col.get("final_value", "")

        # Collate evidence from contributing methods
        evidences = []
        row = col_to_row.get(col_name)
        if row and row.get("methods"):
            for m in col.get("contributing_methods") or []:
                meth = row["methods"].get(m)
                if meth:
                    ev = meth.get("evidence") or (meth.get("attribution") or {}).get("evidence", "")
                    if ev:
                        evidences.append(str(ev)[:400])

        query = f"{col_name}: {final_value}. " + " ".join(evidences)[:2000]
        if not query.strip().endswith("."):
            query = query.strip()

        # Pipeline location hints (page, source_type) and evidence for parsing pages
        pipeline_page = col.get("page")
        pipeline_source_type = col.get("source_type")
        if pipeline_page is not None and str(pipeline_page).lower() in ("not applicable", "n/a", "na"):
            pipeline_page = None
        evidence_combined = " ".join(evidences) if evidences else ""

        # Collect method values (Gemini, Landing, Pipeline) for value-match keywords
        method_values = []
        if row and row.get("methods"):
            for m in col.get("contributing_methods") or []:
                meth = row["methods"].get(m)
                if meth:
                    val = meth.get("value") or meth.get("primary_value", "")
                    if val and str(val).strip():
                        method_values.append(str(val).strip())

        chunks_out = retrieve_chunks_for_evidence(
            doc_id,
            query,
            top_k=top_k,
            column_name=col_name,
            final_value=final_value,
            pipeline_page=pipeline_page,
            pipeline_source_type=pipeline_source_type,
            evidence_text=evidence_combined,
            method_values=method_values if method_values else None,
        )

        out = dict(col)
        out["attributed_chunks"] = chunks_out
        out["chunk_ids"] = [c["chunk_id"] for c in chunks_out]
        enriched.append(out)

    return enriched
