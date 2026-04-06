from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

import src.retrieval.openai_embedding_retriever as retriever
import web.reconciliation_agent as reconciliation_agent


def _load_run_search_agent_module():
    module_path = Path(__file__).resolve().parents[1] / "experiment-scripts" / "run_search_agent.py"
    spec = importlib.util.spec_from_file_location("test_run_search_agent_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embed_chunks_uses_disk_cache(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    chunk_dir = results_root / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "parsed_markdown.md").write_text("dummy markdown", encoding="utf-8")

    monkeypatch.setattr(retriever, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(retriever, "PARSED_MARKDOWN_BASELINES", tmp_path / "baselines")
    monkeypatch.setattr(retriever, "EMBEDDINGS_CACHE", tmp_path / "chunk_embeddings")
    monkeypatch.setattr(
        retriever,
        "_load_page_chunks",
        lambda doc_id: [("page_1", 1, "alpha"), ("page_2", 2, "beta")],
    )

    calls = {"count": 0}

    def fake_embed_texts(client, texts):
        calls["count"] += 1
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key

    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(retriever, "_embed_texts", fake_embed_texts)

    chunk_ids_1, embeddings_1 = retriever.embed_chunks("doc-1", force=True)
    chunk_ids_2, embeddings_2 = retriever.embed_chunks("doc-1", force=False)

    assert calls["count"] == 1
    assert list(chunk_ids_1) == ["page_1", "page_2"]
    assert list(chunk_ids_2) == ["page_1", "page_2"]
    assert embeddings_1.shape == (2, 2)
    assert embeddings_2.shape == (2, 2)


def test_search_chunks_ranks_by_similarity(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "embed_chunks",
        lambda doc_id: (["page_1", "page_2"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)),
    )
    monkeypatch.setattr(
        retriever,
        "_load_page_chunks",
        lambda doc_id: [("page_1", 1, "alpha evidence"), ("page_2", 2, "beta evidence")],
    )
    monkeypatch.setattr(
        retriever,
        "_embed_texts",
        lambda client, texts: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key

    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    hits = retriever.search_chunks("doc-1", query="alpha", top_k=2)

    assert [hit["page"] for hit in hits] == [1, 2]
    assert hits[0]["text"] == "alpha evidence"
    assert hits[0]["score"] > hits[1]["score"]


def test_reconciliation_extract_source_output_normalizes_invalid_data():
    result = reconciliation_agent._extract_source_output(
        {
            "value": "67",
            "reasoning": "from table",
            "attribution": [{"page": "bad", "modality": "weird"}],
        }
    )

    assert result == {
        "value": "67",
        "page": None,
        "modality": "text",
        "reasoning": "from table",
    }


def test_reconciliation_run_get_page_returns_function_response_parts(monkeypatch):
    monkeypatch.setattr(
        reconciliation_agent,
        "_ensure_genai",
        lambda: (
            None,
            types.SimpleNamespace(
                Part=types.SimpleNamespace(
                    from_function_response=lambda name, response: {"kind": "function", "name": name, "response": response},
                    from_text=lambda text: {"kind": "text", "text": text},
                    from_bytes=lambda data, mime_type: {"kind": "bytes", "mime_type": mime_type, "size": len(data)},
                )
            ),
        ),
    )
    monkeypatch.setattr(
        retriever,
        "get_page_content",
        lambda doc_id, page_numbers: {1: "page one", 3: "page three"},
    )

    response, parts = reconciliation_agent._run_get_page(
        doc_id="doc-1",
        page_numbers=[1, 2, 3],
        pages_sent={2},
        total_pages=3,
        pdf_path=None,
    )

    assert response["pages_returned"] == [1, 3]
    assert "[Page 2] already provided" in response["formatted_chunks"]
    assert parts[0]["name"] == "get_page"
    assert parts[0]["response"]["pages_returned"] == [1, 3]


def test_run_search_agent_pipeline_writes_outputs_and_emits_events(tmp_path, monkeypatch):
    module = _load_run_search_agent_module()
    monkeypatch.setattr(module, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(
        module,
        "load_definitions",
        lambda: {
            "Outcomes": [
                {"Column Name": "Overall Survival", "Definition": "Median OS"},
                {"Column Name": "Treatment Arm", "Definition": "Intervention"},
            ]
        },
    )

    fake_search_agent = types.ModuleType("web.search_agent")
    fake_search_agent.run_search_agent = lambda doc_id, batch, definitions_map, log_path=None: (
        {
            col["column_name"]: {
                "value": f"value:{col['column_name']}",
                "reasoning": "ok",
                "found": True,
                "attribution": [{"page": 1, "modality": "text"}],
            }
            for col in batch
        },
        {"input_tokens": 5, "output_tokens": 2, "api_calls": 1},
    )
    monkeypatch.setitem(sys.modules, "web.search_agent", fake_search_agent)

    events = []
    result = module.run_search_agent_pipeline("doc-1", on_event=events.append)

    assert result["filled"] == 2
    extraction_path = module.RESULTS_ROOT / "doc-1" / "search_agent" / "extraction_results.json"
    metadata_path = module.RESULTS_ROOT / "doc-1" / "search_agent" / "extraction_metadata.json"
    assert extraction_path.exists()
    assert metadata_path.exists()
    assert any(event["type"] == "phase_start" for event in events)
    assert any(event["type"] == "search_batch_done" for event in events)
    assert any(event["type"] == "phase_done" for event in events)
