"""
Page retrieval over parsed_markdown.md: embeddings for recall, an optional reranker for precision.

The embedding and reranking models come from the selection in src/config/config.py. Page embeddings are
cached per document and per embedding model, so switching models never mixes vectors.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config.config import RERANK_CANDIDATES, RETRIEVAL_TOP_K, SEARCH_PAGE_MAX_CHARS
from src.config.runtime_paths import CHUNK_EMBEDDINGS_DIR, RESULTS_ROOT
from src.inference.factory import embedding_model_id, get_embedder, get_reranker
from src.inference.types import InferenceError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSED_MARKDOWN_BASELINES = PROJECT_ROOT / "experiment-scripts" / "baselines_landing_ai_new_results"
EMBEDDINGS_CACHE = CHUNK_EMBEDDINGS_DIR
MAX_CHARS_PER_EMBED = 30000


def parsed_markdown_path(doc_id: str) -> Path:
    """results/<doc_id>/chunking/parsed_markdown.md, falling back to the LandingAI baseline copy."""
    for base in (RESULTS_ROOT / doc_id / "chunking", PARSED_MARKDOWN_BASELINES / doc_id):
        path = base / "parsed_markdown.md"
        if path.exists():
            return path
    return RESULTS_ROOT / doc_id / "chunking" / "parsed_markdown.md"


def load_page_chunks(doc_id: str) -> List[Tuple[str, int, str]]:
    """[(chunk_id "page_N", page, text)] from parsed markdown."""
    from src.retrieval.markdown_preprocessor import build_page_chunks_from_markdown

    path = parsed_markdown_path(doc_id)
    if not path.exists():
        return []
    return build_page_chunks_from_markdown(path.read_text(encoding="utf-8"))


def _model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", model_id).strip("_")


def _cache_path(doc_id: str, model_id: str) -> Path:
    safe_id = doc_id.replace("/", "_").replace("'", "_")
    return EMBEDDINGS_CACHE / f"{safe_id}_{_model_slug(model_id)}_markdown.npz"


def _markdown_sha(doc_id: str) -> str:
    path = parsed_markdown_path(doc_id)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _load_cache(doc_id: str, model_id: str) -> Optional[Tuple[List[str], np.ndarray]]:
    path = _cache_path(doc_id, model_id)
    sha = _markdown_sha(doc_id)
    if not sha or not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["markdown_sha256"]) != sha:
            return None
        return [str(c) for c in data["chunk_ids"]], data["embeddings"]
    except (OSError, KeyError, ValueError):
        return None


def has_embedding_cache(doc_id: str) -> bool:
    """True when page embeddings for the current parsed markdown and embedding model are cached."""
    return _load_cache(doc_id, embedding_model_id()) is not None


def embed_chunks(doc_id: str, force: bool = False) -> Optional[Tuple[List[str], np.ndarray]]:
    """Embed every page of the parsed markdown (cached). Returns (chunk_ids, embeddings) or None without markdown."""
    page_chunks = load_page_chunks(doc_id)
    if not page_chunks:
        return None
    model_id = embedding_model_id()
    if not force:
        cached = _load_cache(doc_id, model_id)
        if cached is not None:
            return cached

    texts = [text[:MAX_CHARS_PER_EMBED] for _, _, text in page_chunks]
    embeddings = get_embedder().embed(texts, kind="document")
    chunk_ids = [chunk_id for chunk_id, _, _ in page_chunks]
    path = _cache_path(doc_id, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        chunk_ids=np.array(chunk_ids),
        embeddings=embeddings,
        pages=np.array([page for _, page, _ in page_chunks]),
        markdown_sha256=np.array(_markdown_sha(doc_id)),
        model_id=np.array(model_id),
    )
    return chunk_ids, embeddings


def search_chunks(doc_id: str, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Most relevant pages for a query: {chunk_id, page, source_type, text, score, retrieval}."""
    top_k = top_k or RETRIEVAL_TOP_K
    indexed = embed_chunks(doc_id)
    if not indexed:
        return []
    chunk_ids, embeddings = indexed
    page_text = {chunk_id: text for chunk_id, _, text in load_page_chunks(doc_id)}

    query_vector = get_embedder().embed([query], kind="query")[0]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector) + 1e-9
    similarities = embeddings @ query_vector / norms

    reranker = get_reranker()
    candidate_count = max(top_k, RERANK_CANDIDATES) if reranker else top_k
    order = [int(i) for i in np.argsort(-similarities)[:candidate_count]]
    scores = {i: float(similarities[i]) for i in order}
    retrieval = "embedding"

    if reranker and len(order) > 1:
        try:
            ranked = reranker.rerank(query, [page_text.get(chunk_ids[i], "")[:MAX_CHARS_PER_EMBED] for i in order], top_n=top_k)
            order = [order[index] for index, _ in ranked]
            scores = {order[pos]: score for pos, (_, score) in enumerate(ranked)}
            retrieval = "rerank"
        except InferenceError as exc:
            retrieval = f"embedding (reranker unavailable: {exc})"

    hits = []
    for index in order[:top_k]:
        chunk_id = chunk_ids[index]
        text = page_text.get(chunk_id, "")
        if not text:
            continue
        hits.append({
            "chunk_id": chunk_id,
            "page": int(chunk_id.split("_")[1]) if chunk_id.startswith("page_") else 0,
            "source_type": "page",
            "text": text[:SEARCH_PAGE_MAX_CHARS],
            "score": scores.get(index, 0.0),
            "retrieval": retrieval,
        })
    return hits


def get_total_pages(doc_id: str) -> int:
    return len(load_page_chunks(doc_id))


def get_page_content(doc_id: str, page_numbers: List[int]) -> Dict[int, str]:
    """{page: text}; invalid pages map to an explanation."""
    page_to_text = {page: text for _, page, text in load_page_chunks(doc_id)}
    total = len(page_to_text)
    return {
        page: page_to_text.get(page, "") if 1 <= page <= total else f"Page {page} does not exist. Document has {total} pages."
        for page in page_numbers
    }
