"""Reranker client for vLLM's Cohere/Jina-compatible /v1/rerank API."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import httpx

from src.config.catalog import ModelSpec
from src.inference.base import Reranker
from src.inference.types import InferenceError


class VLLMReranker(Reranker):
    def __init__(self, key: str, spec: ModelSpec, base_url: str, timeout_s: float = 120.0, client: Optional[httpx.Client] = None):
        super().__init__(key, spec)
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.Client(timeout=timeout_s)

    def rerank(self, query: str, documents: Sequence[str], top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        if not documents:
            return []
        payload = {"model": self.spec.name, "query": query, "documents": list(documents)}
        if top_n:
            payload["top_n"] = top_n
        try:
            response = self.http.post(f"{self.base_url}/v1/rerank", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise InferenceError(f"{self.key} ({self.spec.name}) rerank request failed: {exc}") from exc
        ranked = sorted(
            ((int(item["index"]), float(item["relevance_score"])) for item in data.get("results", [])),
            key=lambda pair: -pair[1],
        )
        return ranked[:top_n] if top_n else ranked
