from __future__ import annotations

import json
import sys
import types

import src.LLMProvider.provider as provider_module
import src.evisearch.services.markdown_pdf_query as markdown_agent
import src.evisearch.pipelines.unified_extraction as unified_extraction


def test_local_provider_uses_openai_compatible_endpoint(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr(provider_module, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "local-key")
    monkeypatch.setenv("LOCAL_OPENAI_MODEL", "local-model")

    provider = provider_module.LLMProvider(provider="local")

    assert provider.model == "local-model"
    assert captured == {"base_url": "http://localhost:9000/v1", "api_key": "local-key"}


def test_markdown_pdf_query_agent_matches_agent_output_contract(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    chunk_dir = results_root / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "parsed_markdown.md").write_text("# Page 1\nTrial used ADT plus docetaxel.", encoding="utf-8")
    monkeypatch.setattr(markdown_agent, "RESULTS_ROOT", results_root)

    captured = {}

    class FakeProvider:
        def __init__(self, provider, model):
            captured["provider"] = provider
            captured["model"] = model

        def generate(self, prompt, temperature, max_tokens, **kwargs):
            captured["prompt"] = prompt
            return types.SimpleNamespace(
                success=True,
                text=json.dumps(
                    {
                        "columns": {
                            "Treatment Arm": {
                                "value": "ADT plus docetaxel",
                                "reasoning": "Reported on page 1.",
                                "found": True,
                                "attribution": [{"page": "1", "modality": "text"}],
                            }
                        }
                    }
                ),
                input_tokens=11,
                output_tokens=7,
            )

    monkeypatch.setattr(markdown_agent, "LLMProvider", FakeProvider)
    monkeypatch.setattr(markdown_agent, "load_extraction_preferences", lambda: "")

    results, usage = markdown_agent.run_markdown_pdf_query_agent(
        "doc-1",
        [
            {"column_name": "Treatment Arm", "definition": "Experimental arm"},
            {"column_name": "Control Arm", "definition": "Comparator arm"},
        ],
        provider_name="local",
        model="local-model",
    )

    assert captured["provider"] == "local"
    assert captured["model"] == "local-model"
    assert "DOCUMENT MARKDOWN" in captured["prompt"]
    assert results["Treatment Arm"] == {
        "value": "ADT plus docetaxel",
        "reasoning": "Reported on page 1.",
        "found": True,
        "attribution": [{"page": 1, "modality": "text"}],
        "tried": True,
    }
    assert results["Control Arm"] == {
        "value": "Not reported",
        "reasoning": "Not reported in provided markdown",
        "found": False,
        "attribution": [],
        "tried": True,
    }
    assert usage == {"input_tokens": 11, "output_tokens": 7, "api_calls": 1, "total_tokens": 18}


def test_unified_extraction_uses_markdown_pdf_query_agent_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_QUERY_AGENT_METHOD", "markdown_full_text")
    monkeypatch.setenv("PDF_QUERY_PROVIDER", "local")
    monkeypatch.setenv("PDF_QUERY_MODEL", "local-model")
    monkeypatch.setattr(unified_extraction, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(
        unified_extraction,
        "load_definitions",
        lambda: {"Arms": [{"Column Name": "Treatment Arm", "Definition": "Experimental arm"}]},
    )

    pdf_path = tmp_path / "doc-1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    fake_agent_extraction = types.ModuleType("src.evisearch.services.agent_extraction")
    fake_agent_extraction.resolve_pdf_path = lambda doc_id: pdf_path
    fake_agent_extraction.load_previous_extraction = lambda doc_id: None
    fake_agent_extraction.extract_batch = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native PDF path should not run"))
    fake_agent_extraction.LLMProvider = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native provider should not initialize"))
    fake_agent_extraction._attribution_to_page_modality = lambda attr: attr or []
    monkeypatch.setitem(sys.modules, "src.evisearch.services.agent_extraction", fake_agent_extraction)

    fake_search = types.ModuleType("src.evisearch.services.search")
    fake_search.run_search_agent = lambda doc_id, batch, definitions_map, log_path=None: (
        {"Treatment Arm": {"value": "search-value", "reasoning": "", "found": True, "attribution": []}},
        {"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
    )
    monkeypatch.setitem(sys.modules, "src.evisearch.services.search", fake_search)

    fake_markdown = types.ModuleType("src.evisearch.services.markdown_pdf_query")
    fake_markdown.run_markdown_pdf_query_agent = lambda doc_id, batch, provider_name=None, model=None: (
        {"Treatment Arm": {"value": f"{provider_name}:{model}", "reasoning": "markdown", "found": True, "attribution": [{"page": 1, "modality": "text"}], "tried": True}},
        {"input_tokens": 1, "output_tokens": 1, "api_calls": 1},
    )
    monkeypatch.setitem(sys.modules, "src.evisearch.services.markdown_pdf_query", fake_markdown)

    events = []
    result = unified_extraction.run_unified_extraction("doc-1", on_event=events.append)

    assert result["agent"]["Treatment Arm"]["value"] == "local:local-model"
    agent_path = unified_extraction.RESULTS_ROOT / "doc-1" / "agent_extractor" / "extraction_results.json"
    saved = json.loads(agent_path.read_text(encoding="utf-8"))
    assert saved["columns"]["Treatment Arm"]["attribution"] == [{"page": 1, "modality": "text"}]
    assert any(event.get("type") == "batch_complete" for event in events)
