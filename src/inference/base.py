"""Abstract model interfaces. Adapters implement them; pipeline code depends only on these."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config.catalog import Capabilities, ModelSpec
from src.inference.types import ChatResult, ImagePart, InferenceError, Message, PdfPart, ToolSpec


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
        ChatResult.text (use ChatResult.json()).
        """
        self._check_inputs(messages, tools)
        return self._chat(messages, list(tools or []), tool_choice, response_schema, temperature, max_tokens)

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
