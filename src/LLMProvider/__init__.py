# src/LLMProvider/__init__.py
"""
Unified LLM Provider module.
Supports: Gemini (Vertex AI), OpenAI, Novita, Groq, DeepInfra
Includes OutputStructurer for converting free-form text to structured JSON.
"""

from .provider import LLMProvider, LLMResponse, PDFHandle
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

