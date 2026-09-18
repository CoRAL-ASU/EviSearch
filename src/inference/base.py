"""Abstract model interfaces. Adapters implement them; pipeline code depends only on these."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config.catalog import Capabilities, ModelSpec
from src.inference.types import ChatResult, ImagePart, InferenceError, Message, PdfPart, ToolSpec, count_images, utc_timestamp


class ChatModel(ABC):
    def __init__(self, key: str, spec: ModelSpec):
        self.key = key
        self.spec = spec

    @property
    def capabilities(self) -> Capabilities:
        return self.spec.capabilities

    def chat(
        self,
        messages: List[Message],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_choice: str = "auto",
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChatResult:
        """Run one model turn. Raises InferenceError when the call fails or the input is unsupported.

        response_schema is enforced when the model supports structured output; callers still parse
        ChatResult.text (use ChatResult.json()). The result records when the call started, how long it took and how
        many images it sent (ChatResult.started_at / duration_s, usage.model_seconds / input_images).
        """
        self._check_inputs(messages, tools)
        started_at, start = utc_timestamp(), time.perf_counter()
        result = self._chat(messages, list(tools or []), tool_choice, response_schema, temperature, max_tokens)
        result.started_at, result.duration_s = started_at, round(time.perf_counter() - start, 3)
        result.usage.model_seconds = result.duration_s
        result.usage.input_images = count_images(messages)
        return result

    @abstractmethod
    def _chat(
        self,
        messages: List[Message],
        tools: List[ToolSpec],
        tool_choice: str,
        response_schema: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        ...

    def _check_inputs(self, messages: List[Message], tools: Optional[Sequence[ToolSpec]]) -> None:
        caps = self.capabilities
        if tools and not caps.tools:
            raise InferenceError(f"Model '{self.key}' does not support tool calling")
        for message in messages:
            parts = list(message.parts)
            for result in message.tool_results:
                parts.extend(result.attachments)
            for part in parts:
                if isinstance(part, ImagePart) and not caps.images:
                    raise InferenceError(f"Model '{self.key}' does not accept images")
                if isinstance(part, PdfPart) and not caps.pdf:
                    raise InferenceError(f"Model '{self.key}' does not accept PDF input")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={self.key!r}, name={self.spec.name!r})"


class Embedder(ABC):
    def __init__(self, key: str, spec: ModelSpec):
        self.key = key
        self.spec = spec

    @property
    def model_id(self) -> str:
        return self.spec.name

    @abstractmethod
    def embed(self, texts: Sequence[str], kind: str = "document") -> np.ndarray:
        """Embed texts; kind is "query" or "document". Returns a (len(texts), dim) float32 array."""


class Reranker(ABC):
    def __init__(self, key: str, spec: ModelSpec):
        self.key = key
        self.spec = spec

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[str], top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return (document index, relevance score) pairs, most relevant first."""
