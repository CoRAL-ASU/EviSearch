from __future__ import annotations

import numpy as np
import pytest

import src.retrieval.embedding_retriever as retriever
from src.inference.types import InferenceError

MARKDOWN = "\n<!-- PAGE BREAK -->\n".join(
    [
        "STAMPEDE trial: abiraterone added to ADT",
        "Table 2. Median overall survival 76.6 months",
        "Adverse events grade 3 or higher",
    ]
)


class KeywordEmbedder:
    vocab = ["stampede", "survival", "adverse", "months", "grade"]

    def __init__(self):
        self.calls = []

    def embed(self, texts, kind="document"):
        self.calls.append((kind, list(texts)))
        return np.array([[1.0 if w in t.lower() else 0.0 for w in self.vocab] + [0.01] for t in texts], dtype=np.float32)


@pytest.fixture
def embedder(tmp_path, monkeypatch):
    chunk_dir = tmp_path / "results" / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "parsed_markdown.md").write_text(MARKDOWN, encoding="utf-8")
    fake = KeywordEmbedder()
    monkeypatch.setattr(retriever, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(retriever, "PARSED_MARKDOWN_BASELINES", tmp_path / "baselines")
    monkeypatch.setattr(retriever, "EMBEDDINGS_CACHE", tmp_path / "cache")
    monkeypatch.setattr(retriever, "get_embedder", lambda: fake)
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)
    monkeypatch.setattr(retriever, "embedding_model_id", lambda: "Qwen/Qwen3-Embedding-8B")
    return fake


def test_embeddings_are_cached_per_model_and_invalidated_by_content(embedder, tmp_path, monkeypatch):
    assert not retriever.has_embedding_cache("doc-1")
    chunk_ids, vectors = retriever.embed_chunks("doc-1")
    assert chunk_ids == ["page_1", "page_2", "page_3"]
    assert vectors.shape == (3, 6)
    assert retriever.has_embedding_cache("doc-1")
    assert (tmp_path / "cache" / "doc-1_Qwen_Qwen3_Embedding_8B_markdown.npz").exists()

    retriever.embed_chunks("doc-1")
    assert len(embedder.calls) == 1  # served from cache

    monkeypatch.setattr(retriever, "embedding_model_id", lambda: "text-embedding-3-large")
    assert not retriever.has_embedding_cache("doc-1")  # another model never reuses these vectors

    monkeypatch.setattr(retriever, "embedding_model_id", lambda: "Qwen/Qwen3-Embedding-8B")
    (tmp_path / "results" / "doc-1" / "chunking" / "parsed_markdown.md").write_text(MARKDOWN + " (revised parse)", encoding="utf-8")
    assert not retriever.has_embedding_cache("doc-1")


def test_search_ranks_pages_by_similarity_with_query_embeddings(embedder):
    hits = retriever.search_chunks("doc-1", "overall survival months", top_k=2)
    assert len(hits) == 2
    assert hits[0]["page"] == 2 and hits[0]["score"] > hits[1]["score"]
    assert hits[0]["retrieval"] == "embedding"
    assert embedder.calls[-1][0] == "query"


def test_reranker_reorders_candidates_and_failures_fall_back_to_embeddings(embedder, monkeypatch):
    class AdverseFirst:
        def rerank(self, query, documents, top_n=None):
            index = next(i for i, doc in enumerate(documents) if "Adverse" in doc)
            return [(index, 0.99)]

    monkeypatch.setattr(retriever, "get_reranker", lambda: AdverseFirst())
    hit = retriever.search_chunks("doc-1", "overall survival months", top_k=1)[0]
    assert (hit["page"], hit["score"], hit["retrieval"]) == (3, 0.99, "rerank")

    class Down:
        def rerank(self, *args, **kwargs):
            raise InferenceError("rerank server down")

    monkeypatch.setattr(retriever, "get_reranker", lambda: Down())
    hit = retriever.search_chunks("doc-1", "overall survival months", top_k=1)[0]
    assert hit["page"] == 2
    assert "reranker unavailable" in hit["retrieval"]


def test_page_content_and_page_count(embedder):
    assert retriever.get_total_pages("doc-1") == 3
    content = retriever.get_page_content("doc-1", [2, 9])
    assert "76.6" in content[2]
    assert content[9] == "Page 9 does not exist. Document has 3 pages."
