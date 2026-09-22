"""The schema + feedback loop through the web API: draft a schema from a spreadsheet (fake model), review, revise, lock,
propose a convention through the gate, store and approve it, and see every step in the feedback log."""
from __future__ import annotations

import json
import time

import pytest

from src.config import runtime_paths
from src.config.catalog import Capabilities, ModelSpec
from src.evisearch.schema import generator, store
from src.evisearch.schema.ingest import write_sheet
from src.evisearch.services import feedback
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage

HEADERS = ["Document Name", "Median PFS (mo) | Overall | Treatment", "Type of Therapy"]
PAGES = {1: "ARASENS: darolutamide plus ADT and docetaxel", 2: "Time to castration resistance 16.4 months"}


class RoutedChat(ChatModel):
    """Answers by the shape of the requested JSON schema: drafts, revisions, proposals, gate relations."""

    def __init__(self):
        super().__init__("fake", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        props = response_schema.get("properties", {})
        if "is_knowledge" in props:
            payload = {"is_knowledge": True, "note": "endpoints", "heading": "Progression-free survival",
                       "text": "- Time to castration resistance is not progression-free survival: answer Not reported.",
                       "why": "the definition did not exclude the other endpoint", "new_note": False, "role": "definitions"}
        elif "columns" in props:
            item = props["columns"]["items"]["properties"]
            names = item["column"]["enum"]
            if "answer_format" in item:
                cols = [{"column": n, "definition": f"What is {n}? Use 'Not reported' if missing.", "answer_format": "text",
                         "eval_category": "structured_text", "not_reported_policy": "not stated", "reading": "p1",
                         "questions": [{"question": "Which label?", "options": ["Triplet therapy", "Combination"]}] if n == "Type of Therapy" else [],
                         "confidence": "medium"} for n in names]
            else:
                cols = [{"column": n, "definition": f"What type of therapy is given ({n}), as the table's label (e.g. 'Triplet therapy')?",
                         "change": "uses the owner's label"} for n in names]
            payload = {"columns": cols}
        else:
            payload = {"relation": "independent", "reason": "different subject"}
        text = json.dumps(payload)
        return ChatResult(text=text, tool_calls=[], usage=Usage(1, 1, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


@pytest.fixture
def api(isolated_app, client, tmp_path, monkeypatch):
    import web.schema_routes as routes

    monkeypatch.setattr(runtime_paths, "SCHEMAS_DIR", tmp_path / "schemas")
    monkeypatch.setattr(runtime_paths, "KNOWLEDGE_DIR", tmp_path / "kb")
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback" / "feedback.jsonl")
    monkeypatch.setattr(feedback, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(isolated_app, "record_feedback", feedback.record_feedback)
    monkeypatch.setattr(routes, "_chat", lambda: RoutedChat())
    monkeypatch.setattr(generator.retriever, "get_total_pages", lambda doc_id: 2)
    monkeypatch.setattr(generator.retriever, "get_page_content", lambda doc_id, wanted: {p: PAGES[p] for p in wanted})
    return client


def _wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_schema_loop_through_the_api(api, tmp_path):
    sheet = write_sheet(tmp_path / "t.xlsx", HEADERS, [{"Document Name": "Smith_ARASENS.pdf", "Median PFS (mo) | Overall | Treatment": "",
                                                        "Type of Therapy": "Triplet therapy (ADT + docetaxel + AR inhibitor)"}])
    job = _wait(api, api.post("/api/schemas", json={"sheet_path": str(sheet), "doc_id": "doc-1", "name": "mHSPC", "by": "human"}).get_json()["job_id"])
    assert job["status"] == "done", job
    sid = job["result"]["schema_id"]
    schema = api.get(f"/api/schemas/{sid}").get_json()["schema"]
    assert [f["name"] for f in schema["fields"]] == HEADERS[1:]
    therapy = schema["fields"][1]["x-evisearch"]
    assert therapy["example"]["grounding"]["status"] == "not_in_paper" and therapy["questions"][0]["options"][0] == "Triplet therapy"

    r = api.post(f"/api/schemas/{sid}/review", json={"column": "Type of Therapy", "action": "answer", "question_id": "q1",
                                                     "answer": "Triplet therapy", "by": "human"})
    assert r.get_json()["success"]
    assert api.post(f"/api/schemas/{sid}/review", json={"column": "Median PFS (mo) | Overall | Treatment", "action": "accept", "by": "human"}).get_json()["success"]
    assert api.post(f"/api/schemas/{sid}/review", json={"column": "Nope", "action": "accept"}).status_code == 400

    job = _wait(api, api.post(f"/api/schemas/{sid}/revise", json={}).get_json()["job_id"])
    assert job["status"] == "done" and job["result"]["revised"] == ["Type of Therapy"]
    assert job["result"]["changed"] == ["Type of Therapy"]  # it looked at one column and its text moved
    field = api.get(f"/api/schemas/{sid}").get_json()["schema"]["fields"][1]
    assert "Triplet therapy" in field["description"] and field["x-evisearch"]["review"]["state"] == "revised"

    locked = api.post(f"/api/schemas/{sid}/lock", json={"by": "human"}).get_json()
    assert locked["version"] == 1 and locked["run"] == f"schema-{sid}-v1"
    assert api.get("/api/schemas").get_json()["schemas"][0]["locked_versions"] == [1]

    # a reviewer's correction becomes an edit to the note that already governs the column, not a rule beside it
    proposal = api.post("/api/notes/propose", json={"column": "Median PFS (mo) | Overall | Treatment",
                                                   "definition": "Median PFS",
                                                   "feedback": "16.4 months is time to castration resistance, not PFS",
                                                   "before": "16.4", "after": "Not reported", "reason": "wrong endpoint",
                                                   "schema_id": sid, "doc_id": "doc-1", "by": "human"}).get_json()
    assert proposal["is_knowledge"] and proposal["proposal"]["note"] == "endpoints"
    applied = api.post("/api/notes/apply", json={"note": proposal["proposal"]["note"], "text": proposal["proposal"]["text"],
                                                "heading": proposal["proposal"]["heading"], "role": "definitions",
                                                "by": "human", "why": proposal["proposal"]["why"],
                                                "schema_id": sid}).get_json()
    assert applied["success"] and applied["fingerprint"]
    tree = api.get("/api/notes").get_json()
    assert any("castration resistance" in n["body"] for n in tree["notes"])
    log = api.get("/api/notes/log").get_json()["log"]
    assert log[-1]["note"] == "endpoints" and log[-1]["by"] == "human" and log[-1]["fingerprint"] == tree["fingerprint"]

    events = [e["event"] for e in api.get(f"/api/feedback/events?schema_id={sid}").get_json()["events"]]
    for expected in ("schema_draft", "definition_answer", "definition_accept", "definition_revise", "schema_lock",
                     "note_edit"):
        assert expected in events, (expected, events)


def test_human_correction_keeps_the_reason_and_is_logged(api, isolated_app):
    body = {"columns": {"Median PFS (mo) | Overall | Treatment": {"value": "Not reported", "reason": "wrong endpoint", "note": "16.4 is TTCR"}},
            "by": "human", "run": "schema-x-v1", "schema_id": "x"}
    assert api.post("/api/documents/doc-1/human-edited", json=body).get_json()["success"]
    saved = json.loads((isolated_app.RESULTS_ROOT / "doc-1" / "human-edited" / "human_edited_results.json").read_text())
    cell = saved["columns"]["Median PFS (mo) | Overall | Treatment"]
    assert cell["reason"] == "wrong endpoint" and cell["by"] == "human" and cell["edited_at"]
    event = api.get("/api/feedback/events?source=correction").get_json()["events"][0]
    assert event["event"] == "cell_correct" and event["after"] == "Not reported" and event["schema_id"] == "x"


def test_the_old_page_urls_redirect_into_the_workspace(client):
    """/schema, /feedback and the other old pages are now tabs of the table workspace (or Learning)."""
    # with no table yet they land on the table list; with one they land on its matching tab
    for path, target in [("/schema", "/tables"), ("/extract", "/tables"), ("/comparison-report", "/tables"),
                         ("/attribution", "/tables"), ("/feedback", "/learning"), ("/method-comparison-report", "/benchmark")]:
        response = client.get(path)
        assert response.status_code == 302, path
        assert response.headers["Location"].startswith(target), (path, response.headers["Location"])
    home = client.get("/").data
    assert b'href="/tables"' in home and b'href="/learning"' in home  # the home page leads into the new pages


def test_a_note_edit_reaches_only_the_columns_its_note_governs(api, tmp_path, monkeypatch):
    """What the old integrity gate was for. A one-line rule had to have its scope negotiated because it arrived with
    no context; a note carries its own scope in frontmatter, so the check is which prompts receive it."""
    from src.config import runtime_paths
    from src.evisearch.knowledge import notes as notes_kb

    kb = tmp_path / "kb"
    monkeypatch.setattr(runtime_paths, "KNOWLEDGE_DIR", kb)
    names = ["Mode of metastases - N (%) | Synchronous | Treatment", "Mode of metastases - N (%) | Metachronous | Treatment",
             "OS Rate (%) | Metachronous | Treatment"]
    path = kb / "notes" / "definitions" / "presentation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nid: presentation\nrole: definitions\nscope: family\nfamily: Mode of metastases\n---\n\n"
                    "# Presentation\n\n- Use the paper's own words.\n", encoding="utf-8")

    notes_kb.apply_edit("presentation", "- Prior local therapy is not evidence of presentation.",
                        heading="Presentation", by="reviewer")
    governed = {n.id for n in notes_kb.governing(names[:2])}
    assert "presentation" in governed
    assert "presentation" not in {n.id for n in notes_kb.governing([names[2]])}  # a different family
    assert "not evidence of presentation" in (kb / "notes" / "definitions" / "presentation.md").read_text()
    assert api.get("/static/js/note_edit.js").status_code == 200 and api.get("/static/js/rule_scope.js").status_code == 404


def test_a_revise_that_rewrites_nothing_is_reported_as_no_change(api, monkeypatch):
    """The agent returns every column it read; only the ones whose text moved are worth reviewing."""
    import web.schema_routes as routes

    fields = [{"name": "NCT", "title": "NCT", "description": "What is the NCT id?", "type": "string", "x-evisearch": {
        "group": "Trial", "facets": {}, "eval_category": "structured_text", "example": {"doc": "d", "value": "", "grounding": {}},
        "questions": [], "review": {"state": "proposed", "by": None, "at": None}, "history": []}}]
    sid = store.create("T", fields, source={}, by="h")["id"]
    monkeypatch.setattr(routes.generator, "revise_fields",
                        lambda *a, **k: ({"NCT": {"definition": "What  is the NCT id?", "change": "none", "revised": True}}, []))
    job = _wait(api, api.post(f"/api/schemas/{sid}/revise", json={}).get_json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["revised"] == ["NCT"] and job["result"]["changed"] == []
    field = api.get(f"/api/schemas/{sid}").get_json()["schema"]["fields"][0]
    assert field["x-evisearch"]["review"]["state"] == "proposed"  # a no-op leaves the review state alone
    assert field["x-evisearch"]["history"][-1]["unchanged"] is True
