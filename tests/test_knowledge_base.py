"""Conventions knowledge base: seeding reproduces the rules text exactly, lifecycle from the log, the integrity gate,
the rule proposer, and the shared_rules hook."""
import json

import pytest

from src.config import runtime_paths
from src.config.catalog import Capabilities, ModelSpec
from src.evisearch.knowledge import conventions as kb
from src.evisearch.knowledge import gate, proposer
from src.evisearch.services import extraction_rules
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage

PFS = ["Median PFS (mo) | Overall | Treatment", "Median PFS (mo) | Overall | Control"]


class ReplyChat(ChatModel):
    def __init__(self, replies):
        super().__init__("fake", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))
        self.replies, self.calls = list(replies), 0

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        self.calls += 1
        text = json.dumps(self.replies.pop(0))
        return ChatResult(text=text, tool_calls=[], usage=Usage(1, 1, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


@pytest.fixture(autouse=True)
def kb_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "KNOWLEDGE_DIR", tmp_path / "kb")
    return tmp_path / "kb"


def _variants(scope="family", columns=None, instruction="- Give each PFS variant with its label."):
    return {"trigger": {"scope": scope, "family": "Median PFS (mo)", "columns": columns or [], "facets": {}, "condition": "only variants reported"},
            "action": {"type": "enumerate", "params": {}}, "instruction": instruction,
            "source": {"kind": "extraction_review", "by": "reviewer"}}


def test_seed_renders_exactly_the_rules_text_and_is_idempotent():
    seeds = kb.seed_from_rules("v5")
    assert len(seeds) == extraction_rules.RULES["v5"].count("\n- ") and all(s["status"] == "approved" for s in seeds)
    assert kb.render() == extraction_rules.RULES["v5"]
    assert kb.seed_from_rules("v5") == [] and len(kb.load_all()) == len(seeds)


def test_lifecycle_is_rebuilt_from_the_log_and_learned_conventions_render_after_the_seed():
    kb.seed_from_rules("v5")
    c = kb.create(_variants(), by="reviewer")
    assert c["status"] == "proposed" and kb.render() == extraction_rules.RULES["v5"]  # proposals are not used yet
    kb.decide(c["id"], "approve", by="reviewer")
    kb.merge_into(c["id"], [{"doc": "p2", "column": PFS[0]}], by="reviewer")
    rec = kb.load_all()[c["id"]]
    assert rec["status"] == "approved" and rec["support"] == 2 and [h["op"] for h in rec["history"]] == ["create", "approve", "merge"]
    text = kb.render()
    assert text.startswith(extraction_rules.RULES["v5"]) and "- [Median PFS (mo) columns] Give each PFS variant with its label." in text
    assert text.endswith("the specific one applies to those columns.")
    before = kb.fingerprint()
    relabelled = kb.annotate_source(c["id"], by="reviewer", note="came from the schema review", kind="schema_review")
    assert relabelled["source"] == {**c["source"], "kind": "schema_review"} and relabelled["history"][-1]["op"] == "update"
    assert kb.fingerprint() == before and kb.render() == text  # provenance only: the prompt text is unchanged
    kb.decide(c["id"], "retire", by="reviewer")
    assert kb.render() == extraction_rules.RULES["v5"] and kb.fingerprint() != before
    with pytest.raises(ValueError):
        kb.create({**_variants(), "action": {"type": "made_up", "params": {}}})


def test_overlap_is_decided_by_scope_family_columns_and_facets():
    fam = {"scope": "family", "family": "Median PFS (mo)"}
    assert gate.triggers_overlap(fam, {"scope": "column", "columns": [PFS[0]]})
    assert not gate.triggers_overlap(fam, {"scope": "family", "family": "Median OS (mo)"})
    assert gate.triggers_overlap(fam, {"scope": "global"})
    assert not gate.triggers_overlap({**fam, "facets": {"arm": "Treatment"}}, {**fam, "facets": {"arm": "Control"}})
    fields = [{"name": n, "x-evisearch": {"facets": {"arm": n.rsplit("| ", 1)[1]}}} for n in PFS + ["Median OS (mo) | Overall | Control"]]
    assert gate.impact({**fam, "facets": {"arm": "Control"}}, fields) == [PFS[1]]


def test_gate_merges_exact_duplicates_without_the_model_and_blocks_conflicts():
    kb.decide(kb.create(_variants())["id"], "approve")
    chat = ReplyChat([{"relation": "conflict", "reason": "one says give variants, the other says Not reported"}])
    dup = gate.check(chat, _variants())
    assert dup["verdict"] == "duplicate" and dup["duplicate_of"] == "cv-0001" and chat.calls == 0
    clash = gate.check(chat, _variants(instruction="- Answer Not reported when only PFS variants are reported."))
    assert clash["verdict"] == "blocked" and clash["conflicts"] == ["cv-0001"] and chat.calls == 1
    other = gate.check(chat, {**_variants(), "trigger": {"scope": "family", "family": "Median OS (mo)"}})
    assert other["verdict"] == "new" and other["relations"] == []


def test_a_more_specific_convention_that_disagrees_is_an_exception_not_a_conflict():
    kb.decide(kb.create({**_variants(scope="global", instruction="- Give a subgroup value with the paper's own label."),
                         "trigger": {"scope": "global", "facets": {}, "condition": ""}})["id"], "approve")
    chat = ReplyChat([{"relation": "conflict", "reason": "Region columns would get different values"}])
    region = {**_variants(instruction="- Write 'Included in \"<category>\"' for a region inside a broader category."),
              "trigger": {"scope": "family", "family": "Region - N (%)", "columns": [], "facets": {}, "condition": ""}}
    out = gate.check(chat, region)
    assert out["verdict"] == "new" and out["relations"][0]["relation"] == "exception"
    assert "the proposal (the more specific) applies" in out["relations"][0]["reason"]


def test_proposer_generalises_or_declines_paper_specific_feedback():
    chat = ReplyChat([
        {"is_convention": True, "why_not": "", "scope": "family", "family": "Median PFS (mo)", "columns": [], "condition": "only variants",
         "action_type": "enumerate", "instruction": "Give each PFS variant with its label."},
        {"is_convention": False, "why_not": "a misread number on one page", "scope": "column", "family": "", "columns": [],
         "condition": "", "action_type": "answer_format", "instruction": ""},
    ])
    out = proposer.propose(chat, column=PFS[0], definition="Median PFS", feedback="PFS here is reported as bPFS and rPFS; use both",
                           before="Not reported", after="bPFS 22.9; rPFS 23.5", reason="convention", columns=PFS, paper="p1")
    assert out["is_convention"] and out["record"]["instruction"].startswith("- ") and out["record"]["action"]["type"] == "enumerate"
    assert out["record"]["trigger"]["columns"] == [PFS[0]] and out["record"]["examples"][0]["after"] == "bPFS 22.9; rPFS 23.5"
    assert not proposer.propose(chat, column=PFS[0], definition="", feedback="16.4 is 16.2", columns=PFS)["is_convention"]


def test_shared_rules_reads_the_knowledge_base_when_it_is_on(monkeypatch):
    kb.seed_from_rules("v5")
    monkeypatch.setenv("EVISEARCH_KB", "on")
    assert extraction_rules.shared_rules() == extraction_rules.RULES["v5"]
    assert extraction_rules.rules_setting() == {"extraction_rules": f"kb:{kb.fingerprint()}"}
    assert extraction_rules.shared_rules("v1") == extraction_rules.RULES["v1"]  # an explicit version is never replaced
    monkeypatch.setenv("EVISEARCH_KB", "off")
    assert extraction_rules.rules_setting()["extraction_rules"] != f"kb:{kb.fingerprint()}"
