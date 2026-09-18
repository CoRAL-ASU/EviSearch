"""Every benchmark paper fits Arm A's markdown_images input on each local model, with every page image and no fallback."""
from __future__ import annotations

import json

import pytest

import src.evisearch.services.pdf_query as pdf_query
from src.config.config import CATALOG, GOLD_TABLE_JSON_PATH, MAX_TOKENS
from src.evisearch.knowledge.preferences import load_extraction_preferences
from src.evisearch.pipelines.batching import build_batches, load_groups

LOCAL_PRESETS = ("local", "local_mistral")  # Qwen3.6-27B and Mistral Small 3.2, both with a 131072-token context


def benchmark_doc_ids():
    if not GOLD_TABLE_JSON_PATH.exists():
        return []
    rows = json.loads(GOLD_TABLE_JSON_PATH.read_text(encoding="utf-8"))["data"]
    return [row["Document Name"]["value"].removesuffix(".pdf") for row in rows]


@pytest.mark.parametrize("preset", LOCAL_PRESETS)
@pytest.mark.parametrize("doc_id", benchmark_doc_ids())
def test_benchmark_document_fits_with_every_page_image(doc_id, preset):
    prefs = load_extraction_preferences()
    longest = max((pdf_query.build_columns_prompt(batch, prefs) for batch in build_batches(load_groups(), None, done=set())), key=len)
    model = CATALOG.models[CATALOG.presets[preset]["pdf_query"]]
    budget = pdf_query.document_token_budget(
        model.context_tokens, pdf_query.system_prompt_text() + longest, MAX_TOKENS["pdf_query"]
    )

    info = pdf_query.build_document_input(doc_id, "markdown_images", budget, model.image_tokens).info

    assert info["fallback"] is None and info["image_pages"] == list(range(1, info["pages"] + 1)), info
    assert not info["warnings"], info
