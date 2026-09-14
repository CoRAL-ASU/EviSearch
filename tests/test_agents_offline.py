"""The three agents and their pipelines, run end to end against a scripted model (no network, no GPUs)."""
from __future__ import annotations

import json

import fitz
import numpy as np
import pytest

import src.evisearch.pipelines.pdf_query_pipeline as pdf_query_pipeline
import src.evisearch.pipelines.reconciliation_pipeline as reconciliation_pipeline
import src.evisearch.pipelines.results_store as store
import src.evisearch.pipelines.search_pipeline as search_pipeline
import src.evisearch.pipelines.unified_extraction as unified_extraction
import src.evisearch.services.pdf_query as pdf_query
import src.evisearch.services.reconciliation as reconciliation
import src.evisearch.services.search as search
import src.retrieval.embedding_retriever as retriever
from src.config.catalog import Capabilities, ModelSpec
from src.evisearch.columns import column_result
from src.inference.base import ChatModel
from src.inference.types import ChatResult, ImagePart, InferenceError, Message, PdfPart, TextPart, ToolCall, Usage

PAGES = [
    "STAMPEDE: abiraterone acetate and prednisolone added to ADT",
    "Table 2. Median overall survival 76.6 months with abiraterone versus 45.7 months with ADT",
    "Adverse events grade 3 or higher occurred in 47 percent",
]
TRIAL = "Trial"
MEDIAN_OS = "Median OS (mo) | Overall | Treatment"
BATCH = [{"column_name": TRIAL, "definition": "Trial name"}, {"column_name": MEDIAN_OS, "definition": "Median OS"}]


class ScriptedChat(ChatModel):
    """Each queued turn is either reply text or a list of (tool name, arguments)."""

    def __init__(self, turns, images=False, pdf=False, page_image_scale=None):
        spec = ModelSpec(
            kind="chat",
            endpoint="fake",
            name="fake",
            capabilities=Capabilities(tools=True, json_schema=True, images=images, pdf=pdf),
            page_image_scale=page_image_scale,
        )
        super().__init__("fake-model", spec)
        self.turns = list(turns)
        self.requests = []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        self.requests.append({"messages": list(messages), "tools": [t.name for t in tools], "schema": response_schema})
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            calls, text = [], turn
        else:
            calls, text = [ToolCall(f"c{len(self.requests)}_{i}", name, args) for i, (name, args) in enumerate(turn)], ""
        message = Message(role="assistant", parts=[TextPart(text)] if text else [], tool_calls=calls)
        return ChatResult(text=text, tool_calls=calls, usage=Usage(100, 10, 1), message=message, model=self.key)


class KeywordEmbedder:
    vocab = ["stampede", "survival", "adverse"]

    def embed(self, texts, kind="document"):
        return np.array([[1.0 if w in t.lower() else 0.0 for w in self.vocab] + [0.01] for t in texts], dtype=np.float32)


@pytest.fixture
def doc(tmp_path, monkeypatch):
    results = tmp_path / "results"
    chunk_dir = results / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "parsed_markdown.md").write_text("\n<!-- PAGE BREAK -->\n".join(PAGES), encoding="utf-8")
    pdf_path = results / "doc-1" / "doc-1.pdf"
    with fitz.open() as pdf:
        for text in PAGES:
            pdf.new_page().insert_text((72, 72), text)
        pdf.save(pdf_path)

    monkeypatch.setattr(retriever, "RESULTS_ROOT", results)
    monkeypatch.setattr(store, "RESULTS_ROOT", results)
    monkeypatch.setattr(retriever, "PARSED_MARKDOWN_BASELINES", tmp_path / "baselines")
    monkeypatch.setattr(retriever, "EMBEDDINGS_CACHE", tmp_path / "cache")
    monkeypatch.setattr(retriever, "get_embedder", lambda: KeywordEmbedder())
    monkeypatch.setattr(retriever, "get_reranker", lambda: None)
    monkeypatch.setattr(retriever, "embedding_model_id", lambda: "fake-embed")
    monkeypatch.setattr(pdf_query, "resolve_pdf_path", lambda doc_id: pdf_path)
    monkeypatch.setattr(reconciliation, "resolve_pdf_path", lambda doc_id: pdf_path)
    monkeypatch.setattr(unified_extraction, "resolve_pdf_path", lambda doc_id: pdf_path)
    monkeypatch.setattr(pdf_query, "load_extraction_preferences", lambda: "")
    return {"results": results, "pdf": pdf_path}


def _use(module, chat, monkeypatch):
    monkeypatch.setattr(module, "get_chat", lambda role, model=None: chat)


# ---- Arm A -----------------------------------------------------------------------------------------

def test_pdf_query_markdown_input_uses_structured_output(doc, monkeypatch):
    reply = json.dumps({"columns": [{"column": TRIAL, "value": "STAMPEDE", "reasoning": "title", "found": True, "attribution": [{"page": 1, "modality": "text"}]}]})
    chat = ScriptedChat([reply])
    _use(pdf_query, chat, monkeypatch)
    raw_path = doc["results"] / "raw.json"

    results, usage = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown", raw_response_path=raw_path)

    request = chat.requests[0]
    document_text = request["messages"][1].parts[0].text
    assert "DOCUMENT MARKDOWN" in document_text and "76.6 months" in document_text
    assert request["schema"]["properties"]["columns"]["items"]["properties"]["column"]["enum"] == [TRIAL, MEDIAN_OS]
    assert results[TRIAL] == {"value": "STAMPEDE", "reasoning": "title", "found": True, "attribution": [{"page": 1, "modality": "text"}], "tried": True}
    assert results[MEDIAN_OS]["found"] is False
    assert usage["api_calls"] == 1
    assert json.loads(raw_path.read_text())["response_text"] == reply


def test_pdf_query_pdf_input_needs_a_pdf_capable_model(doc, monkeypatch):
    _use(pdf_query, ScriptedChat([]), monkeypatch)
    results, _ = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="pdf")
    assert "cannot read PDFs" in results[TRIAL]["reasoning"]

    reader = ScriptedChat(['{"columns": []}'], pdf=True)
    _use(pdf_query, reader, monkeypatch)
    pdf_query.run_pdf_query("doc-1", BATCH, input_mode="pdf")
    assert isinstance(reader.requests[0]["messages"][1].parts[1], PdfPart)


# ---- Arm B -----------------------------------------------------------------------------------------

def test_search_agent_runs_real_tools_and_logs_the_conversation(doc, monkeypatch):
    submission = {
        "results": [
            {"column": TRIAL, "value": "STAMPEDE", "reasoning": "page 1", "found": True, "attribution": [{"page": 1, "modality": "text"}]},
            {"column": MEDIAN_OS, "value": "76.6", "reasoning": "Table 2", "found": True, "attribution": [{"page": 2, "modality": "table"}]},
        ]
    }
    chat = ScriptedChat([
        [("get_chunks_by_page", {"page_numbers": [1, 1, 9]})],
        [("search_chunks", {"query": "median overall survival"})],
        [("get_chunks_by_page", {"page_numbers": [2]}), ("submit_extraction", submission)],
    ])
    _use(search, chat, monkeypatch)
    log_path = doc["results"] / "doc-1" / "search_agent" / "verification_logs" / "batch_0.txt"

    results, usage = search.run_search_agent("doc-1", BATCH, {}, log_path=log_path)

    assert chat.requests[0]["tools"] == ["search_chunks", "get_chunks_by_page", "submit_extraction"]
    first_pages = chat.requests[1]["messages"][3].tool_results[0].content
    assert first_pages["pages_returned"] == [1]
    assert "Page 9 does not exist" in first_pages["formatted_chunks"]
    search_result = chat.requests[2]["messages"][5].tool_results[0].content
    assert search_result["pages_returned"] == [2, 3]
    assert "[Page 1, score=" in search_result["formatted_chunks"] and "already provided" in search_result["formatted_chunks"]

    assert results[TRIAL]["value"] == "STAMPEDE"
    assert results[MEDIAN_OS]["attribution"] == [{"page": 2, "modality": "table"}]
    assert usage["api_calls"] == 3
    log = json.loads((log_path.parent / "batch_0_conversation.json").read_text())
    assert log["stopped_by"] == "finish_tool"
    assert [c["name"] for c in log["tool_calls_sequence"]] == ["get_chunks_by_page", "search_chunks", "get_chunks_by_page", "submit_extraction"]


def test_search_agent_reports_retrieval_errors_to_the_model_instead_of_crashing(doc, monkeypatch):
    def unavailable():
        raise InferenceError("embedding server down")

    monkeypatch.setattr(retriever, "get_embedder", unavailable)
    chat = ScriptedChat([[("search_chunks", {"query": "survival"})], "Nothing found."])
    _use(search, chat, monkeypatch)

    results, _ = search.run_search_agent("doc-1", BATCH, {})

    assert "embedding server down" in chat.requests[1]["messages"][3].tool_results[0].content["error"]
    assert results[TRIAL]["found"] is False
    assert "did not submit (no_tool_call)" in results[TRIAL]["reasoning"]


# ---- Reconciliation ---------------------------------------------------------------------------------

def test_reconciliation_reads_pages_with_images_until_every_column_is_submitted(doc, monkeypatch):
    source_a = {
        TRIAL: {"value": "STAMPEDE", "attribution": [{"page": 1, "modality": "text"}]},
        MEDIAN_OS: {"value": "76.6", "reasoning": "Table 2", "attribution": [{"page": 2, "modality": "table"}]},
    }
    source_b = {TRIAL: {"value": "STAMPEDE"}, MEDIAN_OS: {"value": "Not reported"}}
    chat = ScriptedChat(
        [
            [("submit_verification", {"results": [{"column": TRIAL, "value": "STAMPEDE", "reasoning": "match", "verification": "both_correct", "source": {"page": 1, "modality": "text", "verbatim_quote": "STAMPEDE"}}]})],
            [("get_page", {"page_numbers": [2]})],
            [("submit_verification", {"results": [{"column": MEDIAN_OS, "value": "76.6", "reasoning": "Table 2", "verification": "A_correct_B_wrong", "source": {"page": 2, "modality": "table"}}]})],
        ],
        images=True,
        page_image_scale=1,
    )
    _use(reconciliation, chat, monkeypatch)

    results, usage = reconciliation.run_reconciliation_agent("doc-1", BATCH, {}, source_a, source_b)

    prompt = chat.requests[0]["messages"][1].text
    assert 'A: value="76.6" | page=2 | modality=table' in prompt
    assert 'B: value="Not reported" | page=None' in prompt
    page_result = chat.requests[2]["messages"][5].tool_results[0]
    assert page_result.content["pages_returned"] == [2]
    assert "76.6 months" in page_result.content["formatted_chunks"]
    assert isinstance(page_result.attachments[1], ImagePart) and page_result.attachments[1].data.startswith(b"\x89PNG")
    assert results[TRIAL]["attribution"] == [{"page": 1, "modality": "text", "verbatim_quote": "STAMPEDE"}]
    assert results[MEDIAN_OS]["verification"] == "A_correct_B_wrong"
    assert len(chat.requests) == 3 and usage["api_calls"] == 3


def test_reconciliation_source_output_normalizes_invalid_attribution():
    assert reconciliation._extract_source_output({"value": "67", "reasoning": "from table", "attribution": [{"page": "bad", "modality": "weird"}]}) == {
        "value": "67",
        "page": None,
        "modality": "text",
        "reasoning": "from table",
    }


# ---- Pipelines --------------------------------------------------------------------------------------

GROUPS = {"ID": [{"Column Name": TRIAL, "Definition": "Trial name"}], "Outcomes": [{"Column Name": MEDIAN_OS, "Definition": "Median OS"}]}
USAGE = {"input_tokens": 5, "output_tokens": 1, "api_calls": 1}


def _fake_arm(prefix, calls):
    def run(doc_id, batch, *args, **kwargs):
        calls.append([c["column_name"] for c in batch])
        return {c["column_name"]: column_result(f"{prefix}-{c['column_name']}") for c in batch}, dict(USAGE)

    return run


def test_pipelines_write_results_resume_and_reconcile(doc, monkeypatch):
    for module in (pdf_query_pipeline, search_pipeline, reconciliation_pipeline):
        monkeypatch.setattr(module, "load_groups", lambda: GROUPS)
    agent_calls, search_calls = [], []
    monkeypatch.setattr(pdf_query, "run_pdf_query", _fake_arm("A", agent_calls))
    monkeypatch.setattr(search, "run_search_agent", _fake_arm("B", search_calls))

    missing = reconciliation_pipeline.run_reconciliation_pipeline("doc-1")
    assert "agent_extractor results not found" in missing["error"]

    events = []
    first = pdf_query_pipeline.run_pdf_query_pipeline("doc-1", on_event=events.append)
    assert first["filled"] == 2 and agent_calls == [[TRIAL, MEDIAN_OS]]
    assert [e["type"] for e in events] == ["phase_start", "columns_written", "phase_done"]
    metadata = json.loads((store.method_dir("doc-1", "agent") / "extraction_metadata.json").read_text())
    assert metadata["method"] == "pdf_query" and metadata["usage"]["api_calls"] == 1
    assert pdf_query_pipeline.run_pdf_query_pipeline("doc-1")["usage"]["api_calls"] == 0  # resumed: nothing left
    assert len(agent_calls) == 1

    search_events = []
    search_pipeline.run_search_agent_pipeline("doc-1", on_event=search_events.append)
    assert store.load_columns("doc-1", "search")[MEDIAN_OS]["value"] == f"B-{MEDIAN_OS}"
    assert {"phase_start", "search_columns_written", "search_batch_done", "phase_done"} <= {e["type"] for e in search_events}

    reconciled = []

    def fake_reconcile(doc_id, batch, definitions, source_a, source_b, log_path=None, model=None):
        reconciled.append((sorted(source_a), sorted(source_b)))
        return {c["column_name"]: {"value": source_a[c["column_name"]]["value"], "verification": "both_correct"} for c in batch}, dict(USAGE)

    monkeypatch.setattr(reconciliation, "run_reconciliation_agent", fake_reconcile)
    result = reconciliation_pipeline.run_reconciliation_pipeline("doc-1")
    assert result["error"] is None and reconciled == [([MEDIAN_OS, TRIAL], [MEDIAN_OS, TRIAL])]
    saved = store.load_columns("doc-1", "reconciliation")
    assert saved[TRIAL] == {"value": f"A-{TRIAL}", "verification": "both_correct", "tried": True}


def test_unified_extraction_runs_both_arms_per_batch(doc, monkeypatch):
    monkeypatch.setattr(unified_extraction, "load_groups", lambda: GROUPS)
    monkeypatch.setattr(pdf_query, "run_pdf_query", _fake_arm("A", []))
    monkeypatch.setattr(search, "run_search_agent", _fake_arm("B", []))

    events = []
    result = unified_extraction.run_unified_extraction("doc-1", on_event=events.append)

    batch = next(e for e in events if e["type"] == "batch_complete")
    assert batch["columns"][0] == {"column": TRIAL, "candidate_a": f"A-{TRIAL}", "candidate_b": f"B-{TRIAL}"}
    assert events[-1]["type"] == "done"
    assert store.load_columns("doc-1", "agent")[TRIAL]["value"] == f"A-{TRIAL}"
    assert result["search"][MEDIAN_OS]["value"] == f"B-{MEDIAN_OS}"


def test_clis_reject_unknown_group_names(doc, monkeypatch, capsys):
    for module in (pdf_query_pipeline, search_pipeline):
        monkeypatch.setattr(module, "load_groups", lambda: GROUPS)

    assert pdf_query_pipeline.main(["doc-1", "--groups", "ID,12345", "--dry-run"]) == 2
    assert search_pipeline.main(["doc-1", "--groups", "Nope", "--dry-run"]) == 2
    errors = capsys.readouterr().err
    assert "unknown group(s) ['12345']" in errors and "unknown group(s) ['Nope']" in errors
    assert pdf_query_pipeline.main(["doc-1", "--groups", "ID", "--dry-run", "--no-resume"]) == 0


def test_unified_extraction_requires_prepared_document(doc, monkeypatch):
    (doc["results"] / "doc-1" / "chunking" / "parsed_markdown.md").unlink()
    events = []
    result = unified_extraction.run_unified_extraction("doc-1", on_event=events.append)
    assert "prepare the document" in result["error"]
    assert events == [{"type": "error", "error": result["error"]}]
