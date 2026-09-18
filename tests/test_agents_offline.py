"""The three agents and their pipelines, run end to end against a scripted model (no network, no GPUs)."""
from __future__ import annotations

import json
import re

import fitz
import numpy as np
import pytest

import src.evisearch.pipelines.pdf_query_pipeline as pdf_query_pipeline
import src.evisearch.pipelines.reconciliation_pipeline as reconciliation_pipeline
import src.evisearch.pipelines.results_store as store
import src.evisearch.pipelines.search_pipeline as search_pipeline
import src.evisearch.pipelines.unified_extraction as unified_extraction
import src.evisearch.services.evidence_check as evidence_check
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
    chat = ScriptedChat([reply, '{"columns": []}'])  # the follow-up for the column left out returns nothing
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
    assert results[MEDIAN_OS]["found"] is False and results[MEDIAN_OS]["reasoning"] == "Not returned by the model"
    assert usage["api_calls"] == 2
    log = json.loads(raw_path.read_text())
    assert log["response_text"] == reply and log["follow_ups"][0]["columns"] == [MEDIAN_OS]
    columns_schema = request["schema"]["properties"]["columns"]
    assert columns_schema["minItems"] == columns_schema["maxItems"] == 2  # one entry per requested column
    assert log["started_at"] and log["duration_s"] == usage["model_seconds"]


class CutOffChat(ScriptedChat):
    """Replies ending in "<cut>" come back with finish_reason="length" and the marker removed; max_tokens is recorded."""

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        result = super()._chat(messages, tools, tool_choice, response_schema, temperature, max_tokens)
        self.requests[-1]["max_tokens"] = max_tokens
        if result.text.endswith("<cut>"):
            result.text, result.finish_reason = result.text[: -len("<cut>")], "length"
        return result


def _column(name, value):
    return {"column": name, "value": value, "reasoning": "p2", "found": True, "attribution": [{"page": 2, "modality": "text"}]}


def test_pdf_query_asks_again_only_for_the_columns_a_reply_left_out(doc, monkeypatch):
    chat = CutOffChat([json.dumps({"columns": [_column(TRIAL, "STAMPEDE")]}), json.dumps({"columns": [_column(MEDIAN_OS, "76.6")]})])
    _use(pdf_query, chat, monkeypatch)
    raw_path = doc["results"] / "raw.json"

    results, usage = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown", raw_response_path=raw_path)

    follow_up = chat.requests[1]
    columns_text = follow_up["messages"][1].parts[-1].text
    assert f"Column 1: {MEDIAN_OS}" in columns_text and f"Column 1: {TRIAL}" not in columns_text
    assert follow_up["schema"]["properties"]["columns"]["items"]["properties"]["column"]["enum"] == [MEDIAN_OS]
    assert follow_up["schema"]["properties"]["columns"]["minItems"] == 1
    assert follow_up["messages"][1].parts[:-1] == chat.requests[0]["messages"][1].parts[:-1]  # same document prefix
    assert follow_up["max_tokens"] == chat.requests[0]["max_tokens"]
    assert results[TRIAL]["value"] == "STAMPEDE" and results[MEDIAN_OS]["value"] == "76.6"
    assert usage["api_calls"] == 2
    log = json.loads(raw_path.read_text())
    assert [f["columns"] for f in log["follow_ups"]] == [[MEDIAN_OS]] and log["usage"]["api_calls"] == 2


def test_pdf_query_asks_a_cut_off_batch_again_in_two_halves(doc, monkeypatch):
    # The same prompt at temperature 0 replays the same loop, so the follow-up must be a different prompt.
    cut = json.dumps({"columns": [_column(TRIAL, "STAMPEDE")]})[:40] + "<cut>"  # unreadable JSON, as when the budget runs out
    chat = CutOffChat([cut, json.dumps({"columns": [_column(TRIAL, "STAMPEDE")]}), json.dumps({"columns": [_column(MEDIAN_OS, "76.6")]})])
    _use(pdf_query, chat, monkeypatch)
    raw_path = doc["results"] / "raw.json"

    results, usage = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown", raw_response_path=raw_path)

    asked = [r["schema"]["properties"]["columns"]["items"]["properties"]["column"]["enum"] for r in chat.requests]
    assert asked == [[TRIAL, MEDIAN_OS], [TRIAL], [MEDIAN_OS]]
    assert results[TRIAL]["value"] == "STAMPEDE" and results[MEDIAN_OS]["value"] == "76.6" and usage["api_calls"] == 3
    log = json.loads(raw_path.read_text())
    assert log["finish_reason"] == "length" and [f["finish_reason"] for f in log["follow_ups"]] == [None, None]


def test_pdf_query_makes_one_follow_up_at_most(doc, monkeypatch):
    chat = CutOffChat(['{"columns": []}', '{"columns": []}'])
    _use(pdf_query, chat, monkeypatch)

    results, usage = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown")

    assert usage["api_calls"] == 2 and results[TRIAL]["reasoning"] == "Not returned by the model"


def test_pdf_query_sends_each_page_text_followed_by_its_image(doc, monkeypatch):
    _use(pdf_query, ScriptedChat([]), monkeypatch)
    results, _ = pdf_query.run_pdf_query("doc-1", BATCH, input_mode="markdown_images")
    assert "cannot read images" in results[TRIAL]["reasoning"]

    reader = ScriptedChat(['{"columns": []}', '{"columns": []}'], images=True)
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


def _entry(column, value):
    return {"column": column, "value": value, "reasoning": "p1", "found": True, "attribution": [{"page": 1, "modality": "text"}]}


def test_search_agent_asks_for_the_columns_a_submission_left_out(doc, monkeypatch):
    chat = ScriptedChat([
        [("submit_extraction", {"results": [_entry(TRIAL, "STAMPEDE")]})],
        [("submit_extraction", {"results": [_entry(MEDIAN_OS, "76.6")]})],
    ])
    _use(search, chat, monkeypatch)

    results, _ = search.run_search_agent("doc-1", BATCH, {})

    assert f"still missing: {MEDIAN_OS}" in chat.requests[1]["messages"][3].tool_results[0].content["error"]
    assert results[TRIAL]["value"] == "STAMPEDE" and results[MEDIAN_OS]["value"] == "76.6"


def test_search_agent_keeps_received_columns_when_the_loop_ends_before_accepting(doc, monkeypatch):
    monkeypatch.setattr(search, "AGENT_MAX_TURNS", 1)
    chat = ScriptedChat([
        [("submit_extraction", {"results": [_entry(TRIAL, "STAMPEDE")]})],  # partial: sent back, then turns run out
        "No further columns.",  # the forced final turn without a tool call
    ])
    _use(search, chat, monkeypatch)

    results, _ = search.run_search_agent("doc-1", BATCH, {})

    assert results[TRIAL]["value"] == "STAMPEDE"
    assert results[MEDIAN_OS]["found"] is False


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

AE = "Adverse Events | Grade >=3"
CLAIM_RE = re.compile(r"id: (c\d+)\nColumn: [^\n]*\nDefinition: [^\n]*\nClaimed value: ([^\n]*)")


class VerifyingChat(ScriptedChat):
    """Agent turns come from the script. Tool-internal calls (structured output, no tools) are answered here: reader
    calls (schema "answers") from `answers`, verifier calls (schema "results") from the page text, where a claim is
    supported when its value appears in the page's parsed text."""

    def __init__(self, turns, answers=None, images=True):
        super().__init__(turns, images=images)
        self.answers = list(answers or [])
        self.reader_requests = []
        self.verifier_requests = []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        if tools:
            return super()._chat(messages, tools, tool_choice, response_schema, temperature, max_tokens)
        if "answers" in (response_schema or {}).get("properties", {}):
            self.reader_requests.append(messages)
            payload = {"answers": [{"id": f"q{i}", **answer} for i, answer in enumerate(self.answers.pop(0), 1)]}
        else:
            self.verifier_requests.append(messages)
            parts = messages[1].parts
            page_text = " ".join(p.text for p in parts if isinstance(p, TextPart) and "parsed text ===" in p.text).lower()
            payload = {"results": []}
            for claim_id, value in CLAIM_RE.findall(parts[-1].text):
                found = value.lower() in page_text
                payload["results"].append({
                    "id": claim_id, "verdict": "supported" if found else "not_supported", "page_value": value if found else "",
                    "evidence": f"... {value} ..." if found else "", "modality": "table", "reason": "on the page" if found else "not on this page",
                })
        text = json.dumps(payload)
        return ChatResult(text=text, tool_calls=[], usage=Usage(50, 5, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


def _claim(value, page, evidence=""):
    return {"value": value, "reasoning": f"read {value}", "found": True, "attribution": [{"page": page, "modality": "table", "evidence": evidence}]}


def _tool_response(chat, request_index):
    return chat.requests[request_index]["messages"][-1].tool_results[0].content


def test_verifier_checks_each_claim_on_its_page_with_the_page_image(doc):
    chat = VerifyingChat([])
    claims = [
        evidence_check.Claim(MEDIAN_OS, "76.6", 2, "Median overall survival 76.6 months"),
        evidence_check.Claim(MEDIAN_OS, "80.1", 2),
        evidence_check.Claim(AE, "47", 9),
    ]

    records, usage, calls = evidence_check.verify_claims(chat, "doc-1", claims, {MEDIAN_OS: "Median OS"}, pdf_path=doc["pdf"], image_scale=1.0)

    assert len(chat.verifier_requests) == 1 and usage.api_calls == 1  # both page-2 claims in one call; page 9 needs none
    request = chat.verifier_requests[0][1].parts
    assert request[0].text.startswith("=== PAGE 2: parsed text ===") and isinstance(request[2], ImagePart)
    supported = records[claims[0].key]
    assert supported["verdict"] == "supported" and supported["numbers_in_text"] is True and supported["evidence_in_text"] is True
    assert records[claims[1].key]["verdict"] == "not_supported" and records[claims[1].key]["numbers_in_text"] is False
    assert records[claims[2].key]["verdict"] == "not_supported" and "does not exist" in records[claims[2].key]["reason"]
    assert calls[0]["page"] == 2 and calls[0]["image"] is True


def test_reconciliation_keeps_the_paper_out_of_its_context_and_accepts_only_verified_values(doc, monkeypatch):
    source_a = {TRIAL: _claim("STAMPEDE", 1), MEDIAN_OS: _claim("76.6", 2, "OS 76.6 mo")}
    source_b = {TRIAL: _claim("STAMPEDE", 1), MEDIAN_OS: {"value": "Not reported", "reasoning": "no OS table", "found": False}}
    chat = VerifyingChat([
        [("submit_verification", {"results": [{"column": TRIAL, "value": "STAMPEDE", "reasoning": "agree", "verification": "both_correct", "source": {"page": 1}}]})],
        [("verify_attribution", {"claims": [{"column": TRIAL, "value": "STAMPEDE", "page": 1}, {"column": MEDIAN_OS, "value": "76.6", "page": 2}]})],
        [("submit_verification", {"results": [
            {"column": TRIAL, "value": "STAMPEDE", "reasoning": "agree", "verification": "both_correct", "source": {"page": 1}},
            {"column": MEDIAN_OS, "value": "Not reported", "reasoning": "B", "verification": "B_correct_A_wrong"},
        ]})],
        [("submit_verification", {"results": [{"column": MEDIAN_OS, "value": "76.6", "reasoning": "Table 2", "verification": "A_correct_B_wrong", "source": {"page": 2}}]})],
    ])
    _use(reconciliation, chat, monkeypatch)

    results, usage = reconciliation.run_reconciliation_agent("doc-1", BATCH, {}, source_a, source_b)

    prompt = chat.requests[0]["messages"][1].text
    assert 'A: "76.6"' in prompt and 'cites page 2 (table), evidence "OS 76.6 mo"' in prompt and "reasoning: no OS table" in prompt
    assert "Median overall survival 76.6 months" not in prompt  # page text reaches the agent only through tools
    assert chat.requests[0]["tools"] == ["ask_document", "search_pages", "verify_attribution", "submit_verification"]
    assert "verify_attribution" in _tool_response(chat, 1)["rejected"][0]["reason"]  # submitted before verifying
    second = _tool_response(chat, 3)
    assert second["accepted"] == [TRIAL] and "76.6" in second["rejected"][0]["reason"]  # "Not reported" over a verified value
    assert results[TRIAL]["verified"] and results[TRIAL]["verification"] == "both_correct" and results[TRIAL]["attribution"][0]["page"] == 1
    final = results[MEDIAN_OS]
    assert final["value"] == "76.6" and final["verified"] and not final["needs_review"] and final["decided_by"] == "agent"
    assert final["source"]["page"] == 2 and final["attribution"][0]["verified"] is True
    assert [c["verdict"] for c in final["checks"]] == ["supported"]
    assert len(chat.verifier_requests) == 2 and usage["api_calls"] == 6  # 4 agent turns + one verifier call per page


def test_reconciliation_reads_the_paper_through_the_reader_and_flags_what_it_cannot_verify(doc, monkeypatch):
    batch = BATCH + [{"column_name": AE, "definition": "Grade 3 or higher adverse events"}]
    source_a = {TRIAL: {"value": "Not reported"}, MEDIAN_OS: _claim("80.1", 2), AE: _claim("52", 3)}
    source_b = {TRIAL: {"value": "Not reported"}, MEDIAN_OS: {"value": "Not reported"}, AE: {"value": "Not reported"}}
    chat = VerifyingChat(
        [
            [("ask_document", {"questions": [{"column": TRIAL, "question": "What is the trial called?"}]})],
            [("verify_attribution", {"claims": [
                {"column": TRIAL, "value": "STAMPEDE", "page": 1}, {"column": MEDIAN_OS, "value": "80.1", "page": 2}, {"column": AE, "value": "52", "page": 3},
            ]})],
            [("submit_verification", {"results": [
                {"column": TRIAL, "value": "STAMPEDE", "reasoning": "reader, page 1", "verification": "both_wrong", "source": {"page": 1}},
                {"column": MEDIAN_OS, "value": "80.1", "reasoning": "A", "verification": "A_correct_B_wrong", "source": {"page": 2}, "review": True, "review_reason": "not on page 2"},
                {"column": AE, "value": "52", "reasoning": "A", "verification": "A_correct_B_wrong", "source": {"page": 3}},
            ]})],
            "I am done.",
            [("submit_verification", {"results": [{"column": AE, "value": "52", "reasoning": "A", "verification": "A_correct_B_wrong", "source": {"page": 3}}]})],
        ],
        answers=[[{"answer": "STAMPEDE", "pages": [1], "evidence": "STAMPEDE: abiraterone", "modality": "text"}]],
    )
    _use(reconciliation, chat, monkeypatch)

    results, _ = reconciliation.run_reconciliation_agent("doc-1", batch, {TRIAL: "Trial name"}, source_a, source_b)

    reader = chat.reader_requests[0][1].parts
    assert reader[0].text.startswith("DOCUMENT: 3 pages") and any(isinstance(p, ImagePart) for p in reader)  # the whole paper
    assert "Column: Trial\nDefinition: Trial name\nQuestion: What is the trial called?" in reader[-1].text
    answer = _tool_response(chat, 1)["answers"][0]
    assert answer["answer"] == "STAMPEDE" and answer["pages"] == [1]
    submit = _tool_response(chat, 3)
    assert sorted(submit["accepted"]) == sorted([TRIAL, MEDIAN_OS]) and submit["rejected"][0]["column"] == AE
    assert submit["rejected"][0]["verifier"]["verdict"] == "not_supported"
    assert results[TRIAL]["verified"] and results[TRIAL]["verification"] == "both_wrong"
    assert results[MEDIAN_OS]["needs_review"] and not results[MEDIAN_OS]["verified"] and results[MEDIAN_OS]["value"] == "80.1"
    ae = results[AE]  # rejected again in the forced final turn: the value is kept and flagged
    assert ae["value"] == "52" and ae["needs_review"] and not ae["verified"] and ae["decided_by"] == "unsubmitted"
    assert "forced_finish" in ae["review_reason"]


def test_reconciliation_accepts_absence_answers_without_a_page_unless_a_value_is_verified(doc):
    batch = [{"column_name": "QoL reported", "definition": "Yes/No"}, {"column_name": MEDIAN_OS, "definition": "Median OS"}]
    session = reconciliation._ReconciliationSession(VerifyingChat([]), "doc-1", batch, {}, {}, {}, None)
    skipped = session.verify_attribution({"claims": [{"column": "QoL reported", "value": "No", "page": 1}]}).content
    assert skipped["checks"] == [] and "claims absence" in skipped["problems"][0]
    session.verify_attribution({"claims": [{"column": MEDIAN_OS, "value": "76.6", "page": 2}]})
    response = session.submit_verification({"results": [
        {"column": "QoL reported", "value": "No", "reasoning": "none", "verification": "both_correct"},
        {"column": MEDIAN_OS, "value": "No", "reasoning": "?", "verification": "both_correct"},
    ]}).content
    assert response["accepted"] == ["QoL reported"] and response["rejected"][0]["column"] == MEDIAN_OS
    qol = session.submitted["QoL reported"]
    assert qol["value"] == "No" and qol["attribution"] == [] and not qol["verified"] and not qol["needs_review"]


def test_reconciliation_verifies_values_across_pages_and_never_blanks_an_extracted_value_silently(doc):
    batch = [{"column_name": AE, "definition": "Grade 3 or higher adverse events"}, {"column_name": MEDIAN_OS, "definition": "Median OS"}]
    source_a = {AE: _claim("47", 2), MEDIAN_OS: _claim("80.1", 2)}
    chat = VerifyingChat([])
    session = reconciliation._ReconciliationSession(chat, "doc-1", batch, {}, source_a, {}, 1.0)

    checks = session.verify_attribution({"claims": [{"column": AE, "value": "47", "pages": [3, 2]}]}).content["checks"]
    assert checks[0]["verdict"] == "supported" and checks[0]["pages"] == [2, 3]
    request = chat.verifier_requests[0][1].parts
    assert [p.text.split(":")[0] for p in request if isinstance(p, TextPart)][:4] == ["=== PAGE 2", "=== PAGE 2", "=== PAGE 3", "=== PAGE 3"]
    assert sum(isinstance(p, ImagePart) for p in request) == 2  # both pages with their images, in one call

    response = session.submit_verification({"results": [
        {"column": AE, "value": "47", "reasoning": "p3", "verification": "A_correct_B_wrong", "source": {"page": 3}},
        {"column": MEDIAN_OS, "value": "Not reported", "reasoning": "80.1 is not on p2", "verification": "B_correct_A_wrong"},
    ]}).content
    assert response["accepted"] == [AE]  # page 3 lies inside the verified page set [2, 3]
    assert "'80.1'" in response["rejected"][0]["reason"] and "review=true" in response["rejected"][0]["reason"]
    assert [a["page"] for a in session.submitted[AE]["attribution"]] == [2, 3] and session.submitted[AE]["verified"]

    session.submit_verification({"results": [{"column": MEDIAN_OS, "value": "Not reported", "reasoning": "a hazard ratio, not a median",
                                               "verification": "B_correct_A_wrong", "review": True, "review_reason": "answers another statistic"}]})
    final = session.submitted[MEDIAN_OS]
    assert final["value"] == "Not reported" and final["needs_review"] and final["review_reason"] == "answers another statistic"


class LoopingVerifier(VerifyingChat):
    """The first verifier call is cut off at max_tokens (as a structured-output loop would be); later calls answer."""

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        result = super()._chat(messages, tools, tool_choice, response_schema, temperature, max_tokens)
        if not tools and len(self.verifier_requests) == 1:
            result.text, result.finish_reason = result.text[:40], "length"
        return result


def test_verifier_retries_a_cut_off_call_as_two_halves(doc):
    chat = LoopingVerifier([])
    claims = [evidence_check.Claim(MEDIAN_OS, "76.6", 2), evidence_check.Claim(MEDIAN_OS, "45.7", 2)]

    records, usage, calls = evidence_check.verify_claims(chat, "doc-1", claims, {}, pdf_path=doc["pdf"], image_scale=None)

    assert [c["claims"] for c in calls] == [2, 1, 1] and calls[0]["recovered_by_split"] and "cut off" in calls[0]["error"]
    assert all(records[c.key]["verdict"] == "supported" for c in claims) and usage.api_calls == 3


def test_reconciliation_tools_read_list_arguments_sent_as_json_strings(doc):
    session = reconciliation._ReconciliationSession(VerifyingChat([]), "doc-1", BATCH, {}, {}, {}, None)
    checks = session.verify_attribution({"claims": json.dumps([{"column": MEDIAN_OS, "value": "76.6", "page": 2}])}).content
    assert checks["checks"][0]["verdict"] == "supported" and "JSON string" in checks["problems"][0]
    broken = '[{"column": "Trial", "value": "Not reported", "reasoning": "none", "verification": "both_correct"}, ' \
             '"column": "' + MEDIAN_OS + '", "value": "76.6", "reasoning": "p2", "verification": "A_correct_B_wrong", "source": {"page": 2}}'
    response = session.submit_verification({"results": broken}).content
    assert sorted(response["accepted"]) == sorted([TRIAL, MEDIAN_OS]) and "recovered 2 item(s)" in response["note"]


def test_reconciliation_search_returns_pages_with_their_relevant_lines(doc, monkeypatch):
    session = reconciliation._ReconciliationSession(VerifyingChat([]), "doc-1", BATCH, {}, {}, {}, None)
    matches = session.search_pages({"query": "median overall survival"}).content["matches"]
    assert matches[0]["page"] == 2 and "76.6 months" in matches[0]["lines"][0]


def test_reconciliation_source_output_normalizes_invalid_attribution():
    assert reconciliation._extract_source_output({"value": "67", "reasoning": "from table", "attribution": [{"page": "bad", "modality": "weird"}, {"page": 4, "modality": "weird", "evidence": "Age 67"}]}) == {
        "value": "67",
        "reasoning": "from table",
        "pages": [{"page": 4, "modality": "text", "evidence": "Age 67"}],
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


def test_resumed_pipelines_keep_the_logs_of_earlier_batches(doc, monkeypatch):
    def logging_arm(prefix, path_key):
        def run(doc_id, batch, *args, **kwargs):
            names = [c["column_name"] for c in batch]
            path = kwargs[path_key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(names))
            return {name: column_result(f"{prefix}-{name}") for name in names}, dict(USAGE)
        return run

    for module in (pdf_query_pipeline, search_pipeline):
        monkeypatch.setattr(module, "load_groups", lambda: GROUPS)
    monkeypatch.setattr(pdf_query, "run_pdf_query", logging_arm("A", "raw_response_path"))
    monkeypatch.setattr(search, "run_search_agent", logging_arm("B", "log_path"))

    for run_stage, method in ((pdf_query_pipeline.run_pdf_query_pipeline, "agent"), (search_pipeline.run_search_agent_pipeline, "search")):
        run_stage("doc-1", group_names=["ID"])  # stopped after its first batch
        run_stage("doc-1")  # resumed: the rest
        logs = sorted(store.logs_dir("doc-1", method).glob("batch_*"))
        assert [json.loads(p.read_text()) for p in logs] == [[TRIAL], [MEDIAN_OS]], method
    assert [p.name for p in sorted(store.logs_dir("doc-1", "agent").glob("batch_*"))] == ["batch_001.json", "batch_002.json"]
    assert [p.name for p in sorted(store.logs_dir("doc-1", "search").glob("batch_*"))] == ["batch_0.txt", "batch_1.txt"]


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
