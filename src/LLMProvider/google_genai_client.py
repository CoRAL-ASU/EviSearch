"""
Shared Google Gen AI client helpers for Vertex AI.

Supports two auth modes:
- Local development with `VERTEX_API_KEY`
- Deployed/runtime auth with ADC or an attached service account
"""

from __future__ import annotations

import os
from typing import Any, Tuple

_genai = None
_types = None

DEFAULT_VERTEX_LOCATION = "us-central1"


def ensure_genai_modules() -> Tuple[Any, Any]:
    """Lazy import to avoid requiring google-genai for non-Gemini paths."""
    global _genai, _types
    if _genai is None or _types is None:
        from google import genai as imported_genai
        from google.genai import types as imported_types

        _genai = imported_genai
        _types = imported_types
    return _genai, _types


def get_genai_types() -> Any:
    """Return lazily imported google.genai.types."""
    return ensure_genai_modules()[1]


def get_vertex_project() -> str:
    """Resolve the Google Cloud project used for Vertex AI ADC auth."""
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID")
        or ""
    ).strip()


def get_vertex_location() -> str:
    """Resolve the Vertex AI location."""
    return (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GCP_LOCATION")
        or DEFAULT_VERTEX_LOCATION
    ).strip()


def get_vertex_api_key() -> str:
    """Resolve the local-development Vertex AI API key."""
    return os.getenv("VERTEX_API_KEY", "").strip()


def has_vertex_auth() -> bool:
    """Return whether the current environment can initialize a Vertex client."""
    return bool(get_vertex_api_key() or get_vertex_project())


def get_vertex_http_options(timeout_ms: int | None = None) -> Any:
    """Build Vertex AI SDK HTTP options with the stable v1 API."""
    types = get_genai_types()
    kwargs = {"api_version": "v1"}
    if timeout_ms is not None:
        kwargs["timeout"] = timeout_ms
    return types.HttpOptions(**kwargs)


def vertex_auth_error_message() -> str:
    """Return a consistent auth/configuration error message."""
    project = get_vertex_project() or "<unset>"
    location = get_vertex_location() or "<unset>"
    return (
        "Vertex AI Gemini authentication is not configured. "
        "Set VERTEX_API_KEY for local development, or configure ADC/service-account auth. "
        "For ADC-based auth, set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION. "
        f"Current values: GOOGLE_CLOUD_PROJECT={project}, GOOGLE_CLOUD_LOCATION={location}."
    )


def create_vertex_genai_client(timeout_ms: int | None = None) -> Any:
    """Create a Google Gen AI client configured for Vertex AI."""
    genai, _ = ensure_genai_modules()
    http_options = get_vertex_http_options(timeout_ms)
    api_key = get_vertex_api_key()
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key, http_options=http_options)

    project = get_vertex_project()
    location = get_vertex_location()
    if not project:
        raise ValueError(vertex_auth_error_message())

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=http_options,
    )
