"""The knowledge notes: loading, role gating, scoped delivery, editing, the audit log, and the shared_rules hook.

There is one knowledge format. The conventions log and its integrity gate are gone: a one-line rule arrived with no
context, so overlaps and contradictions had to be found by comparing sentence pairs, and every wrong rule in that base
was confirmed unanimously because the same text went to all six prompts. A note is a document about one topic,
addressed to a role, and edited in place - which is why these tests check that an edit lands inside the note that
already governs the column rather than beside it.
"""
import json

import pytest

from src.config import runtime_paths
from src.config.catalog import Capabilities, ModelSpec
from src.evisearch.knowledge import notes as notes_kb
from src.evisearch.knowledge import proposer
from src.evisearch.services import extraction_rules
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage

PFS = ["Median PFS (mo) | Overall | Treatment", "Median PFS (mo) | Overall | Control"]
AGE = ["Median Age (years) | Overall | Treatment"]


class ReplyChat(ChatModel):
    def __init__(self, replies):
        super().__init__("fake", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))
        self.replies, self.calls, self.seen = list(replies), 0, []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        self.calls += 1
        self.seen.append("\n".join(p.text for m in messages for p in m.parts if isinstance(p, TextPart)))
        text = json.dumps(self.replies.pop(0))
        return ChatResult(text=text, tool_calls=[], usage=Usage(1, 1, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


def _write(kb, role, name, body, **meta):
    path = kb / "notes" / role / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    front = "".join(f"{k}: {json.dumps(v) if isinstance(v, list) else v}\n" for k, v in meta.items())
    path.write_text(f"---\nid: {name}\nrole: {role}\n{front}---\n\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def kb_dir(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    monkeypatch.setattr(runtime_paths, "KNOWLEDGE_DIR", kb)
    _write(kb, "definitions", "endpoints", "# Progression-free survival\n\n- Give the median in months.",
           scope="family", family="Median PFS (mo)")
    _write(kb, "definitions", "statistics-and-units", "# Units\n\n- Convert years to months.", scope="global")
    _write(kb, "extraction", "tables-and-subgroups", "# Baseline tables\n\n- Read the arm column, not the total.", scope="global")
    return kb


def test_notes_load_general_before_specific_so_the_specific_one_is_read_last():
    ids = [n.id for n in notes_kb.load_notes("all")]
    assert ids.index("endpoints") > ids.index("statistics-and-units")


def test_the_auditor_gets_definitions_only_so_it_does_not_inherit_the_agents_method():
    """Every wrong rule in the previous knowledge base was confirmed unanimously, because one text went to every
    prompt including the checker. The auditor is the reconciliation stage's own reading pass."""
    agent = {n.id for n in notes_kb.for_role(notes_kb.load_notes("all"), "agent")}
    auditor = {n.id for n in notes_kb.for_role(notes_kb.load_notes("all"), "auditor")}
    assert "tables-and-subgroups" in agent and "tables-and-subgroups" not in auditor
    assert "endpoints" in auditor


def test_a_family_note_reaches_only_the_prompts_holding_its_columns():
    """The failure Q10 measured: an eligibility rule for the docetaxel columns was generalised to previous local
    therapy because every rule reached every prompt."""
    assert "endpoints" in {n.id for n in notes_kb.select_for(notes_kb.load_notes("all"), PFS)}
    assert "endpoints" not in {n.id for n in notes_kb.select_for(notes_kb.load_notes("all"), AGE)}
    assert "statistics-and-units" in {n.id for n in notes_kb.select_for(notes_kb.load_notes("all"), AGE)}


def test_governing_returns_the_notes_a_reviewer_would_have_to_contradict():
    ids = [n.id for n in notes_kb.governing(PFS)]
    assert ids[-1] == "endpoints"  # most specific last, so it is read last


def test_an_edit_lands_under_its_heading_inside_the_note_that_already_governs_the_column(kb_dir):
    before = notes_kb.fingerprint(notes_kb.load_notes("all"))
    out = notes_kb.apply_edit("endpoints", "- When only variants are reported, give each with its label.",
                              heading="Progression-free survival", by="reviewer", why="variants were being collapsed")
    body = (kb_dir / "notes" / "definitions" / "endpoints.md").read_text()
    assert "each with its label" in body
    assert body.index("Give the median") < body.index("each with its label")  # appended within the section
    assert out["fingerprint"] != before and not out["created"]


def test_an_edit_under_an_unknown_heading_grows_the_note_a_new_section(kb_dir):
    notes_kb.apply_edit("endpoints", "- Time to PSA progression is not progression-free survival.",
                        heading="What does not count", by="reviewer")
    body = (kb_dir / "notes" / "definitions" / "endpoints.md").read_text()
    assert "# What does not count" in body and "not progression-free survival" in body


def test_a_genuinely_new_topic_becomes_a_new_note_rather_than_being_forced_into_one(kb_dir):
    out = notes_kb.apply_edit("adverse-events", "- Grade 5 means death from any cause unless the paper says otherwise.",
                              heading="Grades", by="reviewer", role="extraction")
    assert out["created"] and (kb_dir / "notes" / "extraction" / "adverse-events.md").exists()
    assert "adverse-events" in {n.id for n in notes_kb.load_notes("all")}


def test_every_edit_is_recorded_with_who_asked_why_and_the_resulting_fingerprint():
    """notes_log.jsonl is what replaced the conventions log: the chain from a correction to a cell."""
    notes_kb.apply_edit("endpoints", "- Give each variant with its label.", heading="Progression-free survival",
                        by="reviewer", why="variants collapsed", event="ev-1")
    log = notes_kb.log_entries()
    assert len(log) == 1
    entry = log[0]
    assert entry["note"] == "endpoints" and entry["by"] == "reviewer" and entry["event"] == "ev-1"
    assert entry["why"] == "variants collapsed" and entry["fingerprint"] == notes_kb.fingerprint(notes_kb.load_notes("all"))


def test_an_edit_needs_text():
    with pytest.raises(ValueError):
        notes_kb.apply_edit("endpoints", "   ")


def test_the_proposer_sends_the_governing_notes_so_it_edits_instead_of_adding_a_rival_rule():
    chat = ReplyChat([{"is_knowledge": True, "note": "endpoints", "heading": "Progression-free survival",
                       "text": "- Give each variant with its label.", "why": "the definition did not say",
                       "new_note": False, "role": "definitions"}])
    out = proposer.propose(chat, column=PFS[0], definition="Median PFS in months", feedback="both variants are needed",
                           before="68.0", after="68.0 (PSA); 81.0 (clinical)")
    assert out["is_knowledge"] and out["note"] == "endpoints" and "endpoints" in out["governing"]
    assert "Give the median in months" in chat.seen[0]  # the existing note text was in the request


def test_the_proposer_refuses_a_fact_about_one_paper():
    chat = ReplyChat([{"is_knowledge": False, "why": "a misread number on one page"}])
    out = proposer.propose(chat, column=PFS[0], definition="Median PFS in months", feedback="it read the control row")
    assert out["is_knowledge"] is False and "misread" in out["why"]


def test_shared_rules_delivers_the_notes_and_records_their_fingerprint(monkeypatch):
    monkeypatch.setenv("EVISEARCH_KB", "notes")
    text = extraction_rules.shared_rules(columns=PFS, role="agent")
    assert "Give the median in months" in text and "Read the arm column" in text
    auditor = extraction_rules.shared_rules(columns=PFS, role="auditor")
    assert "Read the arm column" not in auditor
    fp = extraction_rules.rules_setting()["extraction_rules"]
    assert fp.startswith("notes:")


def test_kb_on_is_a_synonym_for_notes_so_older_launch_commands_keep_working(monkeypatch):
    monkeypatch.setenv("EVISEARCH_KB", "on")
    assert extraction_rules.rules_setting()["extraction_rules"].startswith("notes:")
