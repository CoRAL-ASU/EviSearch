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
from src.config.catalog import Capabilities, ModelSpec, load_catalog
from src.evisearch.columns import column_result
from src.inference.base import ChatModel
from src.inference.types import ChatResult, ImagePart, InferenceError, Message, TextPart, ToolCall, Usage

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

    def __init__(self, turns, images=False):
        spec = ModelSpec(
            kind="chat",
            endpoint="fake",
            name="fake",
            capabilities=Capabilities(tools=True, json_schema=True, images=images),
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
    parts = request["messages"][1].parts
    assert all(isinstance(part, TextPart) for part in parts)
    assert parts[2].text.startswith("=== PAGE 2: parsed text ===") and "76.6 months" in parts[2].text
    assert "trust the image" not in request["messages"][0].text
    assert request["schema"]["properties"]["columns"]["items"]["properties"]["column"]["enum"] == [TRIAL, MEDIAN_OS]
    assert results[TRIAL] == {"value": "STAMPEDE", "reasoning": "title", "found": True, "attribution": [{"page": 1, "modality": "text"}], "tried": True}
    assert results[MEDIAN_OS]["found"] is False
    assert usage["api_calls"] == 1
    log = json.loads(raw_path.read_text())
    assert log["response_text"] == reply
    assert log["started_at"] and log["duration_s"] == usage["model_seconds"]


def test_pdf_query_sends_each_page_text_followed_by_its_image(doc, monkeypatch):
    _use(pdf_query, ScriptedChat([]), monkeypatch)
    results, _ = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown_images")
    assert "cannot read images" in results[TRIAL]["reasoning"]

    reader = ScriptedChat(['{"columns": []}'], images=True)
    _use(pdf_query, reader, monkeypatch)
    details = {}
    pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown_images", details=details)

    system, user = reader.requests[0]["messages"]
    assert "trust the image" in system.text
    kinds = ["image" if isinstance(part, ImagePart) else part.text.split("\n")[0] for part in user.parts]
    assert kinds[:7] == [kinds[0], "=== PAGE 1: parsed text ===", "image", "=== PAGE 2: parsed text ===", "image", "=== PAGE 3: parsed text ===", "image"]
    assert kinds[7] == "END OF DOCUMENT." and "COLUMNS TO EXTRACT" in user.parts[-1].text  # document first, columns last
    assert user.parts[3].text.endswith("=== PAGE 2: image ===") and user.parts[4].data.startswith(b"\x89PNG")
    assert details["image_pages"] == [1, 2, 3] and details["fallback"] is None and details["page_image_scale"] == 2.0


def test_document_input_strips_anchors_and_falls_back_when_images_do_not_fit(doc, monkeypatch):
    pages = ["<a id='x1'></a>\n\nSTAMPEDE trial", "<a id='x2'></a>\n\n<table><tr><td>76.6</td></tr></table>", "Discussion"]
    (doc["results"] / "doc-1" / "chunking" / "parsed_markdown.md").write_text("\n<!-- PAGE BREAK -->\n".join(pages), encoding="utf-8")

    full = pdf_query.build_document_input("doc-1", "markdown_images")
    assert full.info["image_pages"] == [1, 2, 3] and full.info["fallback"] is None
    assert not any("<a id" in part.text for part in full.parts if isinstance(part, TextPart))

    text_only = pdf_query.build_document_input("doc-1", "markdown").info["estimated_tokens"]
    one_image = (full.info["estimated_tokens"] - text_only) // 3
    tables = pdf_query.build_document_input("doc-1", "markdown_images", token_budget=text_only + one_image)
    assert tables.info["fallback"] == "figure_table_pages" and tables.info["image_pages"] == [2]
    assert sum(isinstance(part, ImagePart) for part in tables.parts) == 1

    none = pdf_query.build_document_input("doc-1", "markdown_images", token_budget=text_only)
    assert none.info["fallback"] == "markdown_only" and not any(isinstance(part, ImagePart) for part in none.parts)

    monkeypatch.setattr(pdf_query, "PDF_QUERY_MAX_PAGE_IMAGES", 2)
    assert pdf_query.build_document_input("doc-1", "markdown_images").info["fallback"] == "figure_table_pages"


def test_image_token_estimate_follows_the_model(doc):
    catalog = load_catalog()
    text_only = pdf_query.build_document_input("doc-1", "markdown").info["estimated_tokens"]

    def image_tokens(model_key):
        spec = catalog.models[model_key].image_tokens
        return pdf_query.build_document_input("doc-1", "markdown_images", image_tokens=spec).info["estimated_tokens"] - text_only

    # Three A4 pages at scale 2 (1190x1684 px). Qwen: 38 x 53 tokens of 32 px. Mistral (Pixtral): scaled to 1088x1540,
    # 39 x 55 tokens of 28 px plus one break token per row.
    assert image_tokens("qwen3.6-27b") == 3 * 38 * 53
    assert image_tokens("mistral-small-3.2-24b") == 3 * 55 * (39 + 1)


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
    chat = ScriptedChat([[("search_chunks", {"query": "survival"})], "Nothing found.", [("submit_extraction", {"results": []})]])
    _use(search, chat, monkeypatch)
    log_path = doc["results"] / "doc-1" / "search_agent" / "verification_logs" / "batch_0.txt"

    results, _ = search.run_search_agent("doc-1", BATCH, {}, log_path=log_path)

    assert "embedding server down" in chat.requests[1]["messages"][3].tool_results[0].content["error"]
    assert chat.requests[2]["tools"] == ["submit_extraction"]  # replying without a tool call forces a final submit
    assert results[TRIAL]["found"] is False
    assert json.loads((log_path.parent / "batch_0_conversation.json").read_text())["stopped_by"] == "forced_finish"


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

    for method in ("agent", "search", "reconciliation"):  # one batch each, one fake call per batch
        timing = store.load_metadata("doc-1", method)["timing"]
        assert timing["started_at"] <= timing["finished_at"] and timing["duration_s"] >= 0, method
        assert (timing["n_calls"], timing["input_tokens"], timing["output_tokens"]) == (1, 5, 1), method
        assert {"cached_input_tokens", "input_images", "model_seconds", "resumed_columns"} <= set(timing), method


def test_named_runs_keep_results_apart_and_resume_refuses_other_settings(doc, monkeypatch):
    monkeypatch.setattr(pdf_query_pipeline, "load_groups", lambda: GROUPS)
    monkeypatch.setattr(pdf_query, "run_pdf_query", _fake_arm("A", []))
    monkeypatch.setattr(store, "_run", "")

    pdf_query_pipeline.run_pdf_query_pipeline("doc-1", input_mode="markdown_images")
    with pytest.raises(store.ResumeError, match="input_mode: saved 'markdown_images', now 'markdown'"):
        pdf_query_pipeline.run_pdf_query_pipeline("doc-1", input_mode="markdown")

    store.use_run("text_only")
    assert store.method_dir("doc-1", "agent") == doc["results"] / "doc-1" / "runs" / "text_only" / "agent_extractor"
    pdf_query_pipeline.run_pdf_query_pipeline("doc-1", input_mode="markdown")
    assert json.loads((store.method_dir("doc-1", "agent") / "extraction_metadata.json").read_text())["run"] == "text_only"
    with pytest.raises(ValueError):
        store.use_run("../elsewhere")


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
    assert "document: 3 pages, images for 3" in capsys.readouterr().out


def test_unified_extraction_requires_prepared_document(doc, monkeypatch):
    (doc["results"] / "doc-1" / "chunking" / "parsed_markdown.md").unlink()
    events = []
    result = unified_extraction.run_unified_extraction("doc-1", on_event=events.append)
    assert "prepare the document" in result["error"]
    assert events == [{"type": "error", "error": result["error"]}]
