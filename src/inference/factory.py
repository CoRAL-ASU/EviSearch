"""Build chat models, embedders and rerankers for pipeline roles from the selection in config.py."""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import httpx

from src.config.catalog import ConfigError, EndpointSpec, Selection
from src.inference.base import ChatModel, Embedder, Reranker
from src.inference.types import InferenceError

_lock = threading.Lock()
_clients: Dict[str, Any] = {}
_models: Dict[tuple, Any] = {}


def _selection() -> Selection:
    from src.config.config import SELECTION

    return SELECTION


def reset_cache() -> None:
    """Drop cached clients and models (tests, or after changing environment variables)."""
    with _lock:
        _clients.clear()
        _models.clear()


def model_key_for(role: str, model: Optional[str] = None) -> Optional[str]:
    selection = _selection()
    if model is None:
        return selection.model_key(role)
    selection.catalog.check_model_for_role(role, model)
    return model


def endpoint_base_url(endpoint_key: str, selection: Optional[Selection] = None) -> str:
    selection = selection or _selection()
    endpoint = selection.catalog.endpoints[endpoint_key]
    if endpoint.base_url_env and os.getenv(endpoint.base_url_env):
        return os.environ[endpoint.base_url_env].rstrip("/")
    if endpoint.base_url:
        return endpoint.base_url.rstrip("/")
    if endpoint.server:
        server = selection.catalog.servers[endpoint.server]
        host = "127.0.0.1" if server.host in ("0.0.0.0", "", "::") else server.host
        base = f"http://{host}:{server.port}"
        return f"{base}/v1" if endpoint.type == "openai_compatible" else base
    raise ConfigError(f"Endpoint '{endpoint_key}' has no base URL")


def _api_key(endpoint_key: str, endpoint: EndpointSpec) -> str:
    if not endpoint.api_key_env:
        return "EMPTY"  # local vLLM servers do not check keys
    value = os.getenv(endpoint.api_key_env, "").strip()
    if not value:
        raise InferenceError(f"Endpoint '{endpoint_key}' needs {endpoint.api_key_env}; set it in .env")
    return value


def _client(endpoint_key: str) -> Any:
    with _lock:
        if endpoint_key in _clients:
            return _clients[endpoint_key]
    selection = _selection()
    endpoint = selection.catalog.endpoints[endpoint_key]
    if endpoint.type == "gemini":
        from src.inference.gemini import create_vertex_client

        client = create_vertex_client(endpoint)
    elif endpoint.type == "openai_compatible":
        from openai import OpenAI

        client = OpenAI(
            base_url=endpoint_base_url(endpoint_key, selection),
            api_key=_api_key(endpoint_key, endpoint),
            timeout=endpoint.timeout_s,
            max_retries=endpoint.max_retries,
        )
    else:
        client = httpx.Client(timeout=endpoint.timeout_s)
    with _lock:
        return _clients.setdefault(endpoint_key, client)


def openai_client(model_key: str) -> Any:
    """Raw OpenAI SDK client for a model's endpoint, for provider-specific APIs (e.g. file search)."""
    catalog = _selection().catalog
    endpoint_key = catalog.models[model_key].endpoint
    if catalog.endpoints[endpoint_key].type != "openai_compatible":
        raise ConfigError(f"Model '{model_key}' is not served by an OpenAI-compatible endpoint")
    return _client(endpoint_key)


def get_chat(role: str, model: Optional[str] = None) -> ChatModel:
    """Chat model for a role. `model` overrides the selection with a catalog key valid for that role."""
    key = model_key_for(role, model)
    if key is None:
        raise ConfigError(f"No model is selected for role '{role}'")
    cache_key = ("chat", key)
    with _lock:
        if cache_key in _models:
            return _models[cache_key]
    catalog = _selection().catalog
    spec = catalog.models[key]
    endpoint = catalog.endpoints[spec.endpoint]
    if endpoint.type == "gemini":
        from src.inference.gemini import GeminiChat

        chat: ChatModel = GeminiChat(key, spec, _client(spec.endpoint))
    elif endpoint.type == "openai_compatible":
        from src.inference.openai_compat import OpenAICompatChat

        chat = OpenAICompatChat(key, spec, _client(spec.endpoint), local=endpoint.server is not None)
    else:
        raise ConfigError(f"Model '{key}' is not a chat model")
    with _lock:
        return _models.setdefault(cache_key, chat)


def get_embedder(model: Optional[str] = None) -> Embedder:
    key = model_key_for("embedding", model)
    if key is None:
        raise ConfigError("No model is selected for role 'embedding'")
    cache_key = ("embedding", key)
    with _lock:
        if cache_key in _models:
            return _models[cache_key]
    from src.inference.openai_compat import OpenAICompatEmbedder

    spec = _selection().catalog.models[key]
    embedder = OpenAICompatEmbedder(key, spec, _client(spec.endpoint))
    with _lock:
        return _models.setdefault(cache_key, embedder)


def get_reranker(model: Optional[str] = None) -> Optional[Reranker]:
    """The selected reranker, or None when reranking is disabled."""
    key = model_key_for("reranker", model)
    if key is None:
        return None
    cache_key = ("reranker", key)
    with _lock:
        if cache_key in _models:
            return _models[cache_key]
    from src.inference.rerank import VLLMReranker

    catalog = _selection().catalog
    spec = catalog.models[key]
    reranker = VLLMReranker(key, spec, endpoint_base_url(spec.endpoint), client=_client(spec.endpoint))
    with _lock:
        return _models.setdefault(cache_key, reranker)


def embedding_model_id() -> str:
    """Name of the selected embedding model, without creating a client."""
    spec = _selection().model("embedding")
    if spec is None:
        raise ConfigError("No model is selected for role 'embedding'")
    return spec.name


def cost_usd(model_key: Optional[str], input_tokens: int, output_tokens: int) -> float:
    catalog = _selection().catalog
    spec = catalog.models.get(model_key or "")
    if spec is None:
        return 0.0
    return round(
        input_tokens * spec.price_per_1k.input / 1000 + output_tokens * spec.price_per_1k.output / 1000, 6
    )


def check_selection(selection: Optional[Selection] = None) -> List[str]:
    """Problems that would stop the selected models from running: missing credentials, unreachable servers."""
    selection = selection or _selection()
    catalog = selection.catalog
    roles_by_endpoint: Dict[str, List[str]] = {}
    for role, key in selection.roles.items():
        if key:
            roles_by_endpoint.setdefault(catalog.models[key].endpoint, []).append(role)

    problems: List[str] = []
    for endpoint_key, roles in roles_by_endpoint.items():
        endpoint = catalog.endpoints[endpoint_key]
        used_by = f"(used by {', '.join(roles)})"
        if endpoint.type == "gemini":
            from src.inference.gemini import has_vertex_auth, vertex_auth_error_message

            if not has_vertex_auth(endpoint):
                problems.append(f"{endpoint_key} {used_by}: {vertex_auth_error_message(endpoint)}")
            continue
        if endpoint.api_key_env and not os.getenv(endpoint.api_key_env, "").strip():
            problems.append(f"{endpoint_key} {used_by}: {endpoint.api_key_env} is not set")
        if endpoint.server or (endpoint.base_url_env and os.getenv(endpoint.base_url_env)):
            base = endpoint_base_url(endpoint_key, selection)
            root = base[: -len("/v1")] if base.endswith("/v1") else base
            try:
                httpx.get(f"{root}/health", timeout=3.0).raise_for_status()
            except Exception as exc:
                hint = f"start it with: python -m src.inference.serve --only {endpoint.server}" if endpoint.server else ""
                problems.append(f"{endpoint_key} {used_by}: server not reachable at {root} ({type(exc).__name__}). {hint}".strip())
                continue
            expected = sorted({catalog.models[selection.roles[role]].name for role in roles})
            try:
                found = [item.get("id") for item in httpx.get(f"{root}/v1/models", timeout=3.0).json().get("data", [])]
            except Exception:
                continue  # health answered; a server without /v1/models cannot be checked further
            missing = [name for name in expected if name not in found]
            if missing:
                problems.append(f"{endpoint_key} {used_by}: {root} serves {found}, not {missing} (another server on that port?)")
    return problems
