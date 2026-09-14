from __future__ import annotations

import json
import sys
import types

import src.inference as inference
from src.config.catalog import Capabilities, ModelSpec
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage


def test_api_extract_unified_stream_emits_sse_events(client, isolated_app, monkeypatch):
    monkeypatch.setattr(isolated_app, "_ensure_pdf_for_extraction", lambda doc_id: None)

    fake_unified = types.ModuleType("src.evisearch.pipelines.unified_extraction")
    calls = []

    def run_unified_extraction(doc_id, group_names=None, resume=True, on_event=None):
        calls.append({"doc_id": doc_id, "resume": resume})
        on_event({"type": "extraction_start", "total": 1, "column_names": ["Overall Survival"], "batches": [["Overall Survival"]]})
        on_event({"type": "batch_complete", "batch": 1, "total_batches": 1, "columns": [{"column": "Overall Survival", "candidate_a": "12.1", "candidate_b": "12.1"}]})
        on_event({"type": "done", "filled": 1, "total": 1})
        return {"agent": {}, "search": {}}

    fake_unified.run_unified_extraction = run_unified_extraction
    monkeypatch.setitem(sys.modules, "src.evisearch.pipelines.unified_extraction", fake_unified)

    response = client.post("/api/extract/unified/stream", json={"doc_id": "doc-1", "no_resume": True})

    body = response.data.decode("utf-8")
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert '"type": "extraction_start"' in body
    assert '"type": "batch_complete"' in body
    assert '"type": "done"' in body
    assert calls == [{"doc_id": "doc-1", "resume": False}]


def test_api_qa_prepare_document_stream_reports_success(client, isolated_app, monkeypatch, tmp_path):
    pdf_path = tmp_path / "trial.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 qa")

    monkeypatch.setattr(isolated_app, "_ensure_pdf_for_extraction", lambda doc_id: None)
    monkeypatch.setattr(isolated_app, "resolve_pdf_path", lambda doc_id: pdf_path)

    fake_parse = types.ModuleType("src.evisearch.services.preparation")

    def parse_pdf_for_qa(doc_id, pdf_path, on_event=None):
        chunk_dir = isolated_app.RESULTS_ROOT / doc_id / "chunking"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "landing_ai_parse_output.json").write_text(json.dumps({"chunks": []}), encoding="utf-8")
        (chunk_dir / "parsed_markdown.md").write_text("# parsed", encoding="utf-8")
        return {"success": True}

    fake_parse.parse_pdf_for_qa = parse_pdf_for_qa
    monkeypatch.setitem(sys.modules, "src.evisearch.services.preparation", fake_parse)

    fake_retriever = types.ModuleType("src.retrieval.embedding_retriever")
    fake_retriever.has_embedding_cache = lambda doc_id: False
    fake_retriever.embed_chunks = lambda doc_id, force=False: (["page_1"], [[1.0, 0.0]])
    monkeypatch.setitem(sys.modules, "src.retrieval.embedding_retriever", fake_retriever)

    response = client.post("/api/qa/prepare-document", json={"doc_id": "doc-qa"})

    body = response.data.decode("utf-8")
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert '"stage": "parsing"' in body
    assert '"stage": "embedding_done"' in body
    assert '"type": "ready"' in body


def test_api_run_reconciliation_uses_reconciliation_pipeline(client, isolated_app, monkeypatch):
    doc_dir = isolated_app.RESULTS_ROOT / "doc-1"
    agent_dir = doc_dir / "agent_extractor"
    search_dir = doc_dir / "search_agent"
    agent_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    (agent_dir / "extraction_results.json").write_text(
        json.dumps({"columns": {"Overall Survival": {"value": "12.1"}}}),
        encoding="utf-8",
    )
    (search_dir / "extraction_results.json").write_text(
        json.dumps({"columns": {"Overall Survival": {"value": "12.1"}}}),
        encoding="utf-8",
    )

    fake_module = types.ModuleType("src.evisearch.pipelines.reconciliation_pipeline")
    fake_module.run_reconciliation_pipeline = lambda doc_id, group_names=None, resume=True: {
        "columns": {"Overall Survival": {"value": "12.1"}}
    }
    monkeypatch.setitem(sys.modules, "src.evisearch.pipelines.reconciliation_pipeline", fake_module)

    response = client.post(
        "/api/documents/doc-1/run-reconciliation",
        json={"group_names": ["Outcomes"]},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["columns_count"] == 1


class _QuickChat(ChatModel):
    def __init__(self):
        super().__init__("qa-model", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities()))
        self.messages = None

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        self.messages = messages
        return ChatResult(text="Median OS was 76.6 months.", tool_calls=[], usage=Usage(1, 1, 1), message=Message(role="assistant", parts=[TextPart("x")]), model=self.key)


def test_api_qa_ask_quick_uses_qa_role_with_parsed_markdown(client, isolated_app, monkeypatch, tmp_path):
    import src.retrieval.embedding_retriever as retriever

    chunk_dir = tmp_path / "results" / "doc-qa" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "parsed_markdown.md").write_text("Median overall survival 76.6 months", encoding="utf-8")
    monkeypatch.setattr(retriever, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(isolated_app, "_ensure_pdf_for_extraction", lambda doc_id: None)
    monkeypatch.setattr(isolated_app, "resolve_pdf_path", lambda doc_id: None)
    chat = _QuickChat()
    monkeypatch.setattr(inference, "get_chat", lambda role, model=None: chat if role == "qa" else None)

    response = client.post("/api/qa/ask", json={"doc_id": "doc-qa", "question": "What was median OS?", "mode": "quick"})

    body = response.data.decode("utf-8")
    assert '"type": "done"' in body and "76.6 months" in body
    document, prompt = chat.messages[0].parts
    assert "Median overall survival 76.6 months" in document.text
    assert "Current question: What was median OS?" in prompt.text
