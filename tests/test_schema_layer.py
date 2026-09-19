"""Schema layer: spreadsheet ingest, header facets, grounding, the schema agent (fake model), and the schema store."""
import csv
import json

import pytest

from src.config import runtime_paths
from src.config.catalog import Capabilities, ModelSpec
from src.evisearch.schema import generator, store
from src.evisearch.schema.facets import parse_header
from src.evisearch.schema.grounding import ground
from src.evisearch.schema.ingest import read_sheet, write_sheet
from src.evisearch.services import feedback
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage

PAGES = {
    1: "ARASENS: darolutamide plus androgen-deprivation therapy and docetaxel in 1,306 patients",
    2: "Median overall survival was not reached; 4-year overall survival 62.7% with darolutamide versus 50.4% with placebo",
}


class JsonChat(ChatModel):
    """Answers each structured call with a scripted function of the requested column names."""

    def __init__(self, reply):
        super().__init__("fake", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))
        self.reply, self.requests = reply, []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        names = response_schema["properties"]["columns"]["items"]["properties"]["column"]["enum"]
        self.requests.append({"messages": messages, "names": names})
        text = json.dumps({"columns": self.reply(names, len(self.requests))})
        return ChatResult(text=text, tool_calls=[], usage=Usage(10, 5, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "SCHEMAS_DIR", tmp_path / "schemas")
    monkeypatch.setattr(feedback, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback" / "feedback.jsonl")
    monkeypatch.setattr(generator.retriever, "get_total_pages", lambda doc_id: len(PAGES))
    monkeypatch.setattr(generator.retriever, "get_page_content", lambda doc_id, wanted: {p: PAGES[p] for p in wanted})
    return tmp_path


def test_sheet_ingest_skips_section_and_blank_rows_and_strips_page_notes(tmp_path):
    path = tmp_path / "table.xlsx"
    write_sheet(path, ["TRIAL CHARACTERISTICS", "", ""], [])
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ws.append(["Document Name", "NCT", "Median OS (mo) | Overall | Treatment", "_SYS_id"])
    ws.append([None, None, None, None])
    ws.append(["Smith_ARASENS.pdf", "NCT02799602\n(pg3, text)", "Not reached (pg4, text)", "x"])
    wb.save(path)

    sheet = read_sheet(path)
    assert sheet.headers == ["Document Name", "NCT", "Median OS (mo) | Overall | Treatment"]
    row = sheet.row_for("smith_arasens")
    assert row["NCT"] == "NCT02799602" and row["Median OS (mo) | Overall | Treatment"] == "Not reached"


def test_header_facets():
    f = parse_header("Median OS (mo) | High volume | Treatment")
    assert (f["characteristic"], f["statistic"], f["unit"], f["subgroup"], f["arm"]) == ("Median OS", "median", "months", "High volume", "Treatment")
    f = parse_header("Race - N (%) | White | Control")
    assert (f["characteristic"], f["statistic"], f["category"], f["arm"]) == ("Race", "count (percent)", "White", "Control")
    assert parse_header("OS Rate (%) | Overall | Control")["statistic"] == "rate (percent)"
    assert parse_header("COE_RCT_IND_OVERALL_RJ")["cryptic"] == ["COE_RCT_IND_OVERALL_RJ"]
    assert parse_header("TTPSA (mo) | Treatment")["cryptic"] == ["TTPSA"]


def test_grounding_finds_numbers_and_flags_values_the_paper_never_prints():
    found = ground("d", "62.7%", PAGES)
    assert found["status"] == "found" and found["pages"] == [2] and "62.7" in found["snippets"][0]["text"]
    assert ground("d", "1306", PAGES)["pages"] == [1]  # commas in the paper are ignored
    assert ground("d", "Triplet therapy (ADT + docetaxel + AR inhibitor)", PAGES)["status"] == "not_in_paper"
    assert ground("d", "Yes", PAGES)["status"] == "short" and ground("d", "", PAGES)["status"] == "empty"


def test_batches_keep_header_families_together():
    headers = [f"Race - N (%) | {r} | {a}" for r in ("White", "Asian") for a in ("Treatment", "Control")] + ["NCT", "Year"]
    assert generator.batches(headers, size=4) == [headers[:4], headers[4:]]


def test_schema_agent_drafts_every_column_asks_about_unprinted_values_and_stores_the_draft(paths):
    headers = ["NCT", "OS Rate (%) | Overall | Treatment", "Type of Therapy"]
    example = {"NCT": "NCT02799602", "OS Rate (%) | Overall | Treatment": "62.7% at 4 yr", "Type of Therapy": "Triplet therapy"}

    def reply(names, call):
        out = []
        for n in names:
            if call == 1 and n == "Type of Therapy":
                continue  # left out: asked again in a follow-up
            q = [{"question": "Is 'Triplet therapy' a label you want for ADT + docetaxel + an AR inhibitor?", "options": ["Yes", "No"]}] if n == "Type of Therapy" else []
            out.append({"column": n, "definition": f"What is {n}? Use 'Not reported' if missing.", "answer_format": "text",
                        "eval_category": "exact_match", "not_reported_policy": "not stated", "reading": "p2", "questions": q, "confidence": "high"})
        return out

    chat = JsonChat(reply)
    fields, logs = generator.draft_fields(chat, "doc-1", headers, example)
    assert [f["name"] for f in fields] == headers and all(f["description"] for f in fields)
    assert len(chat.requests) == 2 and chat.requests[1]["names"] == ["Type of Therapy"] and "follow_up" in logs[0]
    prompt = chat.requests[0]["messages"][1].text
    assert 'Example value (doc-1): "62.7% at 4 yr" — printed on page(s) [2]' in prompt
    assert '"Triplet therapy" — NOT printed in the paper' in prompt
    therapy = fields[2]["x-evisearch"]
    assert therapy["example"]["grounding"]["status"] == "not_in_paper" and therapy["questions"][0]["id"] == "q1"

    schema = store.create("Test table", fields, source={"sheet": "t.xlsx", "example_doc": "doc-1"}, by="claude-as-human")
    assert store.list_schemas()[0]["fields"] == 3 and store.list_schemas()[0]["open_questions"] == 1


def test_review_history_events_and_lock_export(paths):
    fields = [{"name": n, "title": n, "description": f"What is {n}?", "type": "string", "x-evisearch": {
        "group": n.split(" |")[0], "eval_category": "exact_match", "example": {"doc": "d", "value": "", "grounding": {}},
        "questions": [{"id": "q1", "question": "Which arm?", "options": ["A", "B"], "answer": None}],
        "review": {"state": "proposed", "by": None, "at": None}, "history": []}} for n in ("NCT", "Year")]
    schema = store.create("T", fields, source={}, by="h")
    store.review_field(schema["id"], "NCT", "accept", by="h")
    store.review_field(schema["id"], "Year", "answer", by="h", question_id="q1", answer="A")
    store.review_field(schema["id"], "Year", "edit", by="h", definition="What year was the paper published?", reason="wording")
    with pytest.raises(ValueError):
        store.review_field(schema["id"], "Year", "edit", by="h", definition=" ")

    locked = store.lock(schema["id"], by="h")
    assert locked["version"] == 1 and locked["run"] == f"schema-{schema['id']}-v1"
    with open(locked["csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["Column Name"] for r in rows] == ["NCT", "Year"] and rows[1]["Definition"] == "What year was the paper published?"
    current = store.load(schema["id"])
    assert current["version"] == 2 and current["status"] == "locked" and store.load(schema["id"], version=1)["version"] == 1
    year = store.get_field(current, "Year")["x-evisearch"]
    assert [h["action"] for h in year["history"]] == ["answer", "edit"] and year["review"]["state"] == "edited"
    events = [json.loads(line)["event"] for line in (paths / "feedback" / "feedback.jsonl").read_text().splitlines()]
    assert events == ["schema_draft", "definition_accept", "definition_answer", "definition_edit", "schema_lock"]
