from __future__ import annotations

import pytest

from src.config.catalog import Catalog, ConfigError, env, load_catalog, role_overrides_from_env


@pytest.fixture
def catalog() -> Catalog:
    return load_catalog()


def test_every_preset_resolves(catalog):
    for preset in catalog.presets:
        selection = catalog.resolve(preset, gpu_pool=[0, 1])
        assert selection.preset == preset
        assert selection.option("pdf_query_input") == "markdown_images"


def test_local_preset_needs_local_servers_and_cloud_needs_none(catalog):
    assert catalog.resolve("local", gpu_pool=[0]).servers_needed() == ["qwen36_27b", "qwen3_embed_8b", "qwen3_rerank_8b"]
    assert catalog.resolve("cloud", gpu_pool=[0]).servers_needed() == []


def test_role_override_rejects_wrong_kind_and_names_valid_models(catalog):
    with pytest.raises(ConfigError) as exc:
        catalog.resolve("local", role_overrides={"search_agent": "qwen3-embedding-8b"}, gpu_pool=[0])
    message = str(exc.value)
    assert "search_agent needs a chat model" in message.replace("'", "")
    assert "qwen3.6-27b" in message and "gemini-2.5-flash" in message


def test_image_input_requires_image_capable_model(catalog):
    with pytest.raises(ConfigError, match="cannot read images"):
        catalog.resolve("local", role_overrides={"pdf_query": "qwen3-8b"}, gpu_pool=[0])
    selection = catalog.resolve("local", role_overrides={"pdf_query": "qwen3-8b"}, options={"pdf_query_input": "markdown"}, gpu_pool=[0])
    assert selection.option("pdf_query_input") == "markdown"


def test_unknown_option_value_lists_choices(catalog):
    with pytest.raises(ConfigError, match="markdown_images \\| markdown"):
        catalog.resolve("local", options={"pdf_query_input": "pdf"}, gpu_pool=[0])


def test_optional_reranker_can_be_disabled_but_judge_cannot(catalog):
    selection = catalog.resolve("local", role_overrides={"reranker": None}, gpu_pool=[0])
    assert selection.model("reranker") is None
    assert "qwen3_rerank_8b" not in selection.servers_needed()
    with pytest.raises(ConfigError, match="role 'judge' needs a model"):
        catalog.resolve("local", role_overrides={"judge": None}, gpu_pool=[0])


def test_gpu_assignment_validated_against_pool_and_tensor_parallel(catalog):
    with pytest.raises(ConfigError) as exc:
        catalog.resolve("local", gpus={"qwen36_27b": [9], "qwen3_embed_8b": [0, 1], "nope": "auto"}, gpu_pool=[0, 1])
    message = str(exc.value)
    assert "GPU [9], which is not in GPU_POOL [0, 1]" in message
    assert "needs tensor_parallel=1" in message
    assert "unknown server 'nope'" in message

    selection = catalog.resolve("local", gpus={"qwen36_27b": [1]}, gpu_pool=[0, 1])
    assert selection.gpus["qwen36_27b"] == [1]
    assert selection.gpus["qwen3_embed_8b"] == "auto"


def test_catalog_reference_errors_are_reported():
    raw = {
        "endpoints": {"local": {"type": "openai_compatible", "server": "missing"}},
        "models": {"m": {"kind": "chat", "endpoint": "local", "name": "m"}},
        "roles": {"qa": {"kind": "chat"}},
        "presets": {"p": {"qa": "m"}},
    }
    with pytest.raises(ConfigError, match="unknown server 'missing'"):
        Catalog.model_validate(raw).validate_references()


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("X_POOL", "0, 2,3")
    monkeypatch.setenv("X_GPUS", "qwen36_27b=2,3;qwen3_embed_8b=auto")
    monkeypatch.setenv("EVISEARCH_ROLE_JUDGE", "gemini-2.5-pro")
    monkeypatch.setenv("EVISEARCH_ROLE_RERANKER", "none")

    assert env("X_POOL", [0]) == [0, 2, 3]
    assert env("X_GPUS", {"qwen3_rerank_8b": "auto"}) == {
        "qwen3_rerank_8b": "auto",
        "qwen36_27b": [2, 3],
        "qwen3_embed_8b": "auto",
    }
    assert env("X_MISSING", "local") == "local"
    overrides = role_overrides_from_env()
    assert overrides["judge"] == "gemini-2.5-pro"
    assert overrides["reranker"] is None

    monkeypatch.setenv("X_BAD", "a,b")
    with pytest.raises(ConfigError, match="X_BAD"):
        env("X_BAD", [0])
