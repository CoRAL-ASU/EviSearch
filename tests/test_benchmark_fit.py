"""Every benchmark paper fits Arm A's markdown_images input on the local model, with every page image and no fallback."""
from __future__ import annotations

import json

import pytest

import src.evisearch.services.pdf_query as pdf_query
from src.config.config import CATALOG, GOLD_TABLE_JSON_PATH, MAX_TOKENS
from src.evisearch.knowledge.preferences import load_extraction_preferences
from src.evisearch.pipelines.batching import build_batches, load_groups


def benchmark_doc_ids():
    if not GOLD_TABLE_JSON_PATH.exists():
        return []
    rows = json.loads(GOLD_TABLE_JSON_PATH.read_text(encoding="utf-8"))["data"]
    return [row["Document Name"]["value"].removesuffix(".pdf") for row in rows]


@pytest.mark.parametrize("doc_id", benchmark_doc_ids())
def test_benchmark_document_fits_with_every_page_image(doc_id):
    prefs = load_extraction_preferences()
    longest = max((pdf_query.build_columns_prompt(batch, prefs) for batch in build_batches(load_groups(), None, done=set())), key=len)
    local_model = CATALOG.models[CATALOG.presets["local"]["pdf_query"]]  # the tightest context we run Arm A on
    budget = pdf_query.document_token_budget(
        local_model.context_tokens, pdf_query.SYSTEM_PROMPT + pdf_query.IMAGE_RULES + longest, MAX_TOKENS["pdf_query"]
    )

    info = pdf_query.build_document_input(doc_id, "markdown_images", budget).info

    assert info["fallback"] is None and info["image_pages"] == list(range(1, info["pages"] + 1)), info
    assert not info["warnings"], info
