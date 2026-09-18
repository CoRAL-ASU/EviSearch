"""Benchmark runner (experiment-scripts/run_benchmark.py) and system B1, with the stages replaced by fakes (no models)."""
from __future__ import annotations

import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import pytest

import src.evisearch.pipelines.results_store as store
import src.evisearch.services.markdown_baseline as markdown_baseline
from src.config.catalog import Capabilities, ModelSpec
from src.inference.base import ChatModel
from src.inference.types import ChatResult, InferenceError, Message, TextPart, Usage

SCRIPT = Path(__file__).resolve().parents[1] / "experiment-scripts" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
run_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmark)

DOC = "NCT00309985_Sweeney_CHAARTED_NEJM'15"
OTHER = "NCT00104715_Gravis_GETUG_EU'15"
COLUMNS = ["Trial", "Median OS"]


@pytest.fixture
def results(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    monkeypatch.setattr(store, "RESULTS_ROOT", root)
    monkeypatch.setattr(store, "_run", "")
    monkeypatch.setattr(run_benchmark, "benchmark_columns", lambda: list(COLUMNS))
    monkeypatch.setattr(run_benchmark, "git_state", lambda: {"commit": "abc123", "branch": "test", "dirty": False})
    return root


def fake_stages(monkeypatch, fail=None):
    """run_stage replacement: writes each stage's results and metadata like the pipelines; records the calls."""
    calls = []

    def run_stage(stage, doc_id):
        calls.append((stage, doc_id))
        if fail and fail(stage, doc_id):
            raise store.ResumeError(f"{stage} refused for {doc_id}")
        usage = {"api_calls": 1, "input_tokens": 10, "output_tokens": 2}
        store.save_columns(doc_id, stage, {name: {"value": f"{stage}-{name}", "reasoning": "r"} for name in COLUMNS})
        store.save_metadata(doc_id, stage, {"method": stage, "run": store.current_run(), "usage": usage, "timing": {"duration_s": 1.5, "n_calls": 1}})
        return {"filled": 2, "total": 2, "usage": usage, "columns": {"not": "kept"}}

    monkeypatch.setattr(run_benchmark, "run_stage", run_stage)
    monkeypatch.setattr(run_benchmark, "run_check", lambda doc_id, system, run: {"result": "PASS", "summary": "[check_run] PASS"})
    return calls


def manifest(run):
    return json.loads((store.RESULTS_ROOT.parent / "benchmark_runs" / f"{run}.json").read_text())


def test_doc_selection_splits_the_ten_gold_documents():
    gold = run_benchmark.gold_doc_ids()
    assert len(gold) == 10 and DOC in gold
    assert run_benchmark.select_docs("all", gold) == gold
    heldout = run_benchmark.select_docs("heldout", gold)
    dev = run_benchmark.select_docs("dev", gold)
    assert sorted(heldout) == sorted(run_benchmark.HELDOUT)
    assert len(dev) == 7 and set(dev) | set(heldout) == set(gold) and not set(dev) & set(heldout)
    assert run_benchmark.select_docs(f"{DOC}.pdf, {OTHER},{DOC}", gold) == [DOC, OTHER]
    with pytest.raises(ValueError, match="unknown doc id"):
        run_benchmark.select_docs("NCT0_nope", gold)


def test_e_reuses_arm_a_from_a_b2_run_instead_of_running_it(results, monkeypatch):
    calls = fake_stages(monkeypatch)
    assert run_benchmark.main(["--system", "B2", "--docs", DOC, "--run", "b2"]) == 0
    assert calls == [("agent", DOC)]
    b2 = store.load_metadata(DOC, "agent")
    # the fake stage writes no settings; give the B2 results the settings the real Arm A pipeline saves
    store.save_metadata(DOC, "agent", {**{k: v for k, v in b2.items() if k != "doc_id"}, **run_benchmark.stage_settings("agent")})
    (store.method_dir(DOC, "agent") / "raw_llm_responses").mkdir()
    (store.method_dir(DOC, "agent") / "raw_llm_responses" / "batch_001.json").write_text("{}")

    calls.clear()
    assert run_benchmark.main(["--system", "E", "--docs", DOC, "--run", "e", "--reuse-a-from", "b2"]) == 0

    assert calls == [("search", DOC), ("reconciliation", DOC)]  # Arm A was copied, not run
    copied = store.RESULTS_ROOT / DOC / "runs" / "e" / "agent_extractor"
    assert (copied / "raw_llm_responses" / "batch_001.json").exists()
    metadata = json.loads((copied / "extraction_metadata.json").read_text())
    assert metadata["run"] == "e" and metadata["reused_from"] == "b2" and metadata["timing"]["n_calls"] == 1
    record = json.loads((store.RESULTS_ROOT / DOC / "runs" / "e" / "benchmark_manifest.json").read_text())
    assert record["stages"]["agent"]["status"] == "reused" and record["stages"]["search"]["status"] == "ok"
    assert "columns" not in record["stages"]["search"] and record["stages"]["search"]["timing"]["duration_s"] == 1.5
    assert manifest("e")["reuse_a_from"] == "b2" and manifest("e")["failures"] == []

    # a B2 run made with another Arm A model is refused before anything is copied or run
    other = store.RESULTS_ROOT / DOC / "runs" / "b2_qwen8b" / "agent_extractor"
    other.mkdir(parents=True)
    (other / "extraction_results.json").write_text(json.dumps({"columns": {name: {} for name in COLUMNS}}))
    (other / "extraction_metadata.json").write_text(json.dumps({**run_benchmark.stage_settings("agent"), "model": "qwen3-8b"}))
    calls.clear()
    assert run_benchmark.main(["--system", "E", "--docs", DOC, "--run", "e2", "--reuse-a-from", "b2_qwen8b"]) == 1
    assert calls == [] and not (store.RESULTS_ROOT / DOC / "runs" / "e2" / "agent_extractor").exists()
    assert "made with other settings" in manifest("e2")["failures"][0]["error"]


def test_manifests_record_git_models_timings_and_failures_across_invocations(results, monkeypatch):
    fake_stages(monkeypatch, fail=lambda stage, doc_id: doc_id == OTHER)
    assert run_benchmark.main(["--system", "B2", "--docs", f"{DOC},{OTHER}", "--run", "b2", "--parallel", "2"]) == 1

    top = manifest("b2")
    assert top["git"] == {"commit": "abc123", "branch": "test", "dirty": False}
    assert top["system"] == "B2" and top["preset"] == "local" and top["input_mode"] == "markdown_images"
    assert top["models"]["pdf_query"] == "qwen3.6-27b" and top["models"]["embedding"] == "qwen3-embedding-8b"
    assert top["stage_models"] == {"agent": "qwen3.6-27b"} and top["page_image_scale"] == 2.0
    assert top["docs"] == [OTHER, DOC]  # gold-table order
    assert top["failures"] == [{"doc_id": OTHER, "error": f"agent: ResumeError: agent refused for {OTHER}"}]
    assert set(top["timings"][DOC]) == {"duration_s", "agent"} and top["invocations"][0]["finished_at"]
    ok = json.loads((store.RESULTS_ROOT / DOC / "runs" / "b2" / "benchmark_manifest.json").read_text())
    assert ok["status"] == "ok" and ok["check"]["result"] == "PASS" and ok["started_at"] <= ok["finished_at"]
    failed = json.loads((store.RESULTS_ROOT / OTHER / "runs" / "b2" / "benchmark_manifest.json").read_text())
    assert failed["status"] == "failed" and failed["check"] == {"result": "skipped"}

    fake_stages(monkeypatch)  # fixed: run the failed document again under the same run name
    assert run_benchmark.main(["--system", "B2", "--docs", OTHER, "--run", "b2"]) == 0
    top = manifest("b2")
    assert top["docs"] == [OTHER, DOC] and top["failures"] == [] and len(top["invocations"]) == 2

    assert run_benchmark.main(["--system", "E", "--docs", DOC, "--run", "b2"]) == 2  # a run name keeps one system


def test_dry_run_calls_no_model_and_writes_nothing(results, monkeypatch, capsys):
    calls = fake_stages(monkeypatch)
    assert run_benchmark.main(["--system", "B2", "--docs", DOC, "--run", "dry", "--dry-run"]) == 0
    assert run_benchmark.main(["--system", "B1", "--docs", DOC, "--run", "dry_b1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert calls == [] and not (store.RESULTS_ROOT.parent / "benchmark_runs").exists() and not list(store.RESULTS_ROOT.iterdir())
    assert "agent: model=qwen3.6-27b batches=" in out and "document: 10 pages, images for 10" in out and "fallback=None" in out
    assert f"check_run.py \"{DOC}\" --run dry --stages agent" in out
    assert "baseline: model=gemini-2.5-flash" in out and "parsed_markdown.md" in out


class GroupChat(ChatModel):
    """Answers each definition group from `answers` (label -> reply dict); a missing label fails the call."""

    def __init__(self, answers):
        super().__init__("gemini-2.5-flash", ModelSpec(kind="chat", endpoint="fake", name="fake", capabilities=Capabilities(json_schema=True)))
        self.answers = answers
        self.labels = []

    def _chat(self, messages, tools, tool_choice, response_schema, temperature, max_tokens):
        label = messages[0].text.split("(Label: ")[1].split(")")[0]
        self.labels.append(label)
        if label not in self.answers:
            raise InferenceError("server down")
        text = json.dumps(self.answers[label])
        return ChatResult(text=text, tool_calls=[], usage=Usage(1000, 50, 1), message=Message(role="assistant", parts=[TextPart(text)]), model=self.key)


def test_b1_writes_agent_shaped_results_timing_and_resumes_failed_groups(results, tmp_path, monkeypatch):
    markdown = tmp_path / "md" / DOC / "parsed_markdown.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("STAMPEDE ... median OS 76.6 months", encoding="utf-8")
    definitions = {
        "Trial": {"definition": "Trial name", "label": "ID", "eval_category": "x", "index": 0},
        "Median OS": {"definition": "Median OS", "label": "Outcomes", "eval_category": "x", "index": 1},
    }
    monkeypatch.setattr(markdown_baseline, "load_definitions_with_metadata", lambda path: definitions)
    store.use_run("b1")
    chat = GroupChat({"ID": {"Trial": {"value": "STAMPEDE", "reasoning": "title page"}}})
    monkeypatch.setattr(markdown_baseline, "get_chat", lambda role, model=None: chat)

    result = markdown_baseline.run_baseline_stage(DOC, parsed_markdown_root=tmp_path / "md", workers=2)

    saved = json.loads(store.results_path(DOC, "baseline").read_text())
    assert store.results_path(DOC, "baseline") == results / DOC / "runs" / "b1" / "markdown_baseline" / "extraction_results.json"
    assert saved["doc_id"] == DOC
    assert saved["columns"]["Trial"] == {"value": "STAMPEDE", "reasoning": "title page"}
    assert saved["columns"]["Median OS"]["value"] == "Extraction error" and "server down" in saved["columns"]["Median OS"]["reasoning"]
    metadata = store.load_metadata(DOC, "baseline")
    assert metadata["model"] == "gemini-2.5-flash" and metadata["preset"] == "local" and metadata["run"] == "b1"
    assert metadata["failed_groups"] == ["Outcomes"] and result["failed_groups"] == ["Outcomes"]
    assert (metadata["timing"]["n_calls"], metadata["timing"]["input_tokens"]) == (1, 1000)
    assert metadata["timing"]["started_at"] <= metadata["timing"]["finished_at"] and len(metadata["calls"]) == 1
    monkeypatch.setattr(run_benchmark, "benchmark_columns", lambda: list(definitions))
    assert run_benchmark.check_baseline(DOC)["result"] == "FAIL"

    chat.answers["Outcomes"] = {"Median OS": {"value": "", "reasoning": ""}}
    chat.labels.clear()
    markdown_baseline.run_baseline_stage(DOC, parsed_markdown_root=tmp_path / "md")
    assert chat.labels == ["Outcomes"]  # only the failed group is asked again
    columns = store.load_columns(DOC, "baseline")
    assert columns["Trial"]["value"] == "STAMPEDE" and columns["Median OS"] == {"value": "not found", "reasoning": "not found"}
    assert run_benchmark.check_baseline(DOC)["result"] == "PASS"
    raw = json.loads((store.method_dir(DOC, "baseline") / "raw_llm_responses.json").read_text())
    assert set(raw) == {"ID", "Outcomes"}

    with pytest.raises(store.ResumeError, match="gemini-2.5-flash"):
        markdown_baseline.run_baseline_stage(DOC, model="gemini-2.5-pro", parsed_markdown_root=tmp_path / "md")


def test_baseline_columns_match_extract_once_value_rules():
    groups = OrderedDict([("ID", [{"column": "Trial"}, {"column": "Arms"}]), ("Outcomes", [{"column": "Median OS"}])])
    raw = {"ID": {"Trial": {"value": " STAMPEDE", "reasoning": None}, "Arms": "bad"}, "Outcomes": {"_raw": "{", "_error": "JSON decode failed"}}
    assert markdown_baseline.baseline_columns(raw, groups) == {
        "Trial": {"value": " STAMPEDE", "reasoning": "not found"},
        "Arms": {"value": "not found", "reasoning": "not found"},
        "Median OS": {"value": "Extraction error", "reasoning": "JSON decode failed"},
    }
