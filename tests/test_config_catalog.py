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
    assert catalog.resolve("local", gpu_pool=[0]).servers_needed() == ["qwen36_27b", "qwen3_embed_8b"]  # no reranker
    assert catalog.resolve("cloud", gpu_pool=[0]).servers_needed() == []


def test_every_preset_retrieves_by_embedding_alone(catalog):
    """The configuration the paper reports: no reranker in any preset."""
    assert all(assignment.get("reranker") is None for assignment in catalog.presets.values())


def test_image_token_cost_per_model(catalog):
    letter = (1224, 1584)  # US letter at PAGE_IMAGE_SCALE 2
    assert catalog.models["qwen3.6-27b"].image_tokens.count(*letter) == 39 * 50
    assert catalog.models["qwen3.6-27b"].image_tokens.count(1170, 1566) == 1813  # measured on the server: ~1,815
    # Pixtral: scaled to 1190x1540, 43 x 55 patches of 28 px, plus [IMG_BREAK]/[IMG_END] per row (mistral_common)
    assert catalog.models["mistral-small-3.2-24b"].image_tokens.count(*letter) == 55 * (43 + 1)
    assert catalog.models["mistral-small-3.2-24b"].image_tokens.count(280, 56) == 2 * (10 + 1)
    assert catalog.models["gemini-2.5-flash"].image_tokens.count(*letter) == 39 * 50  # default: 32 px upper bound


def test_serverless_presets_move_only_the_agent_roles_off_the_gpu(catalog):
    for preset, provider_model in (("novita", "novita-vlm"), ("together", "together-vlm")):
        assignment = catalog.presets[preset]
        assert set(assignment) == set(catalog.presets["local"])
        for role, local_model in catalog.presets["local"].items():
            expected = provider_model if local_model == "qwen3.6-27b" else local_model
            assert assignment[role] == expected, (preset, role)
        # retrieval still runs on the local GPU, so page sets and ranking match a `local` run
        assert catalog.resolve(preset, gpu_pool=[0]).servers_needed() == ["qwen3_embed_8b"]
        # and the served id is never hardcoded: it comes from the environment
        assert catalog.models[provider_model].name_env
        assert catalog.models[provider_model].price_per_1k.input == 0.0  # no invented provider prices


def test_name_env_fills_the_served_id_from_the_environment(monkeypatch):
    from src.config import catalog as catalog_module

    monkeypatch.setenv("EVISEARCH_NOVITA_MODEL", "vendor/some-open-model")
    catalog_module._load_catalog_cached.cache_clear()
    try:
        fresh = load_catalog()
        assert fresh.models["novita-vlm"].name == "vendor/some-open-model"
        assert fresh.models["together-vlm"].name == ""  # its own variable is unset
    finally:
        catalog_module._load_catalog_cached.cache_clear()


def _served_model_catalog() -> Catalog:
    return Catalog.model_validate({
        "endpoints": {"host": {"type": "openai_compatible", "base_url": "https://host.test/v1", "api_key_env": "HOST_KEY"}},
        "models": {"served": {"kind": "chat", "endpoint": "host", "name_env": "HOST_MODEL", "capabilities": {"tools": True}}},
        "roles": {"qa": {"kind": "chat", "requires": ["tools"]}},
        "presets": {"host": {"qa": "served"}},
    })


def test_model_without_a_served_id_is_rejected():
    raw = {
        "endpoints": {"host": {"type": "openai_compatible", "base_url": "https://host.test/v1"}},
        "models": {"m": {"kind": "chat", "endpoint": "host"}},
        "roles": {"qa": {"kind": "chat"}},
        "presets": {"p": {"qa": "m"}},
    }
    with pytest.raises(ConfigError, match="needs a served model id"):
        Catalog.model_validate(raw).validate_references()


def test_unset_served_id_is_reported_and_refuses_to_build_a_model(monkeypatch):
    from src.inference import factory

    catalog = _served_model_catalog()  # HOST_MODEL is not set, so the catalog still loads with an empty name
    assert catalog.models["served"].name == ""
    catalog.validate_references()
    selection = catalog.resolve("host")
    monkeypatch.setattr(factory, "_selection", lambda: selection)
    factory.reset_cache()
    try:
        assert any("HOST_MODEL is not set" in problem for problem in factory.check_selection(selection))
        with pytest.raises(ConfigError, match="HOST_MODEL"):
            factory.get_chat("qa")
    finally:
        factory.reset_cache()


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
