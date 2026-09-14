from __future__ import annotations

import json
import threading

import src.evaluation.evaluator_v2 as evaluator_module
from src.config.catalog import Capabilities, ModelSpec
from src.evaluation.evaluator_v2 import EvaluatorV2
from src.inference.base import ChatModel
from src.inference.types import ChatResult, Message, TextPart, Usage


class FakeJudge(ChatModel):
    """Answers every batch with full marks, but returns invalid JSON the first time it sees the exact-match batch."""

    def __init__(self):
        super().__init__("fake-judge", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))
        self.lock = threading.Lock()
        self.failed_once = False
        self.schemas = []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        prompt = messages[0].text
        with self.lock:
            self.schemas.append(response_schema)
            if "exact match columns" in prompt and not self.failed_once:
                self.failed_once = True
                text = "Here are my thoughts, no JSON yet"
            else:
                columns = [c for c in ("Trial", "NCT") if f". {c}:" in prompt]
                text = json.dumps({"results": [{"column": c, "correctness": 1.0, "completeness": 1.0, "reason": "match"} for c in columns]})
        return ChatResult(text=text, tool_calls=[], usage=Usage(100, 10, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


def test_evaluator_scores_with_the_judge_role_retrying_invalid_json(tmp_path, monkeypatch):
    extraction = tmp_path / "extraction_metadata.json"
    extraction.write_text(json.dumps({"Trial": {"value": "STAMPEDE"}, "NCT": {"value": "NCT00268476"}}))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"data": [{"Document Name": {"value": "doc-1.pdf"}, "Trial": {"value": "STAMPEDE"}, "NCT": {"value": "NCT00268476"}}]}))
    definitions = tmp_path / "definitions.csv"
    definitions.write_text("Column Name,Definition,Label,eval_category\nTrial,Trial name,ID,structured_text\nNCT,NCT id,ID,exact_match\n")

    judge = FakeJudge()
    monkeypatch.setattr(evaluator_module, "get_chat", lambda role, model=None: judge)
    evaluator = EvaluatorV2(str(extraction), str(gold), str(definitions), "doc-1", str(tmp_path / "out"))
    evaluator.run()

    summary = json.loads((tmp_path / "out" / "summary_metrics.json").read_text())
    assert summary["overall"]["total_columns"] == 2
    assert summary["overall"]["avg_overall"] == 1.0
    assert summary["judge_model"] == "gemini-2.5-flash"
    assert all(schema["properties"]["results"]["type"] == "array" for schema in judge.schemas)

    calls = [json.loads(line) for line in (tmp_path / "out" / "llm_logs" / "judge_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 3
    assert [c["success"] for c in calls].count(False) == 1

    usage = evaluator.get_usage()
    assert usage["input_tokens"] == 200 and usage["model"] == "gemini-2.5-flash" and usage["cost_usd"] > 0
