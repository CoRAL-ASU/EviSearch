"""
OpenAI text-embedding-3-large semantic retriever.

Uses parsed_markdown.md (split by PAGE BREAK) for embedding. Caches to disk.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config.runtime_paths import CHUNK_EMBEDDINGS_DIR, RESULTS_ROOT

PARSED_MARKDOWN_BASELINES = PROJECT_ROOT / "experiment-scripts" / "baselines_landing_ai_new_results"
EMBEDDINGS_CACHE = CHUNK_EMBEDDINGS_DIR
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
MAX_CHARS_PER_EMBED = 30000


def _get_parsed_markdown_path(doc_id: str) -> Path:
    """
    Path to parsed_markdown.md for doc_id.
    Priority: results/<doc_id>/chunking/parsed_markdown.md, then baselines/<doc_id>/parsed_markdown.md.
    """
    for base in (RESULTS_ROOT / doc_id / "chunking", PARSED_MARKDOWN_BASELINES / doc_id):
        path = base / "parsed_markdown.md"
        if path.exists():
            return path
    # Return results path as default (for writing); caller must check exists
    return RESULTS_ROOT / doc_id / "chunking" / "parsed_markdown.md"


def _load_page_chunks(doc_id: str) -> List[tuple]:
    """Load page chunks from parsed_markdown via preprocessor. Returns [(chunk_id, page, text), ...]."""
    from src.retrieval.markdown_preprocessor import build_page_chunks_from_markdown

    path = _get_parsed_markdown_path(doc_id)
    if not path.exists():
        return []
    markdown = path.read_text(encoding="utf-8")
    return build_page_chunks_from_markdown(markdown)


def _embed_texts(client: Any, texts: List[str]) -> np.ndarray:
    """Embed texts via OpenAI API. Returns (n, dim) array."""
    if not texts:
        return np.array([]).reshape(0, 0)
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Truncate long texts (8192 tokens ~ 32k chars)
        batch = [t[:30000] if len(t) > 30000 else t for t in batch]
        resp = client.embeddings.create(input=batch, model=OPENAI_EMBEDDING_MODEL)
        for d in resp.data:
            out.append(d.embedding)
    return np.array(out, dtype=np.float32)


def _get_cache_path(doc_id: str) -> Path:
    EMBEDDINGS_CACHE.mkdir(parents=True, exist_ok=True)
    safe_id = doc_id.replace("/", "_").replace("'", "_")
    return EMBEDDINGS_CACHE / f"{safe_id}_{OPENAI_EMBEDDING_MODEL.replace('-', '_')}_markdown.npz"


def _parse_mtime(doc_id: str) -> float:
    path = _get_parsed_markdown_path(doc_id)
    return path.stat().st_mtime if path.exists() else 0.0


def embed_chunks(doc_id: str, force: bool = False) -> Optional[tuple]:
    """
    Embed page chunks from parsed_markdown for doc_id. Caches to disk.
    Returns (chunk_ids, embeddings_2d) or None if failed.
    chunk_ids are "page_N".
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY not set")

    page_chunks = _load_page_chunks(doc_id)
    if not page_chunks:
        return None

    cache_path = _get_cache_path(doc_id)
    parse_mtime = _parse_mtime(doc_id)
    if not force and cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_mtime = float(data.get("parse_mtime", 0))
            if cached_mtime >= parse_mtime:
                return list(data["chunk_ids"]), data["embeddings"]
        except Exception:
            pass

    texts = [t for _, _, t in page_chunks]
    embeddings = _embed_texts(client, texts)
    chunk_ids = [sid for sid, _, _ in page_chunks]
    pages = [p for _, p, _ in page_chunks]

    np.savez_compressed(
        cache_path,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        parse_mtime=parse_mtime,
        pages=np.array(pages),
    )
    return chunk_ids, embeddings


def search_chunks(
    doc_id: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search over page chunks using OpenAI text-embedding-3-large.
    Returns list of {chunk_id, page, source_type, text, score}.
    """
    from openai import OpenAI

    result = embed_chunks(doc_id)
    if not result:
        return []
    chunk_ids, embeddings = result

    page_chunks = _load_page_chunks(doc_id)
    id_to_text = {(sid, p): t for sid, p, t in page_chunks}

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    q_emb = _embed_texts(client, [query])
    if q_emb.size == 0:
        return []

    scores = np.dot(embeddings, q_emb[0]) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb[0]) + 1e-9
    )
    top_idx = np.argsort(-scores)[:top_k]

    out = []
    for idx in top_idx:
        cid = chunk_ids[idx]
        page_1 = int(cid.split("_")[1]) if cid.startswith("page_") else 0
        text = id_to_text.get((cid, page_1), "")
        if text:
            out.append({
                "chunk_id": cid,
                "page": page_1,
                "source_type": "page",
                "text": text[:15000],
                "score": float(scores[idx]),
            })
    return out


def get_total_pages(doc_id: str) -> int:
    """Return total page count for doc_id from parsed_markdown."""
    chunks = _load_page_chunks(doc_id)
    return len(chunks) if chunks else 0


def get_page_content(doc_id: str, page_numbers: List[int]) -> Dict[int, str]:
    """
    Return {page: content} for requested pages.
    For invalid pages: returns "Page N does not exist. Document has M pages."
    """
    chunks = _load_page_chunks(doc_id)
    page_to_text = {p: t for _, p, t in chunks}
    total = len(page_to_text)
    out: Dict[int, str] = {}
    for p in page_numbers:
        if p < 1 or p > total:
            out[p] = f"Page {p} does not exist. Document has {total} pages."
        else:
            out[p] = page_to_text.get(p, "")
    return out
