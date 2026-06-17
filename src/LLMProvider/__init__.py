# src/LLMProvider/__init__.py
"""Unified LLM Provider module with lazy optional-provider imports."""

from .models import SUPPORTED_MODELS, get_model_pricing
from .structurer import OutputStructurer, StructurerResponse

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "PDFHandle",
    "SUPPORTED_MODELS",
    "get_model_pricing",
    "OutputStructurer",
    "StructurerResponse",
]


def __getattr__(name: str):
    if name in {"LLMProvider", "LLMResponse", "PDFHandle"}:
        from .provider import LLMProvider, LLMResponse, PDFHandle

        values = {
            "LLMProvider": LLMProvider,
            "LLMResponse": LLMResponse,
            "PDFHandle": PDFHandle,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
