from __future__ import annotations

import csv
import re
import importlib.util
from pathlib import Path

import pytest

import src.evisearch.pipelines.results_store as store
from src.config.config import DEFINITIONS_CSV_PATH
from src.evisearch.pipelines import pdf_query_pipeline
from src.evisearch.services import extraction_rules, markdown_baseline, pdf_query

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["GETUG", "STAMPEDE", "CHAARTED", "SWOG", "PEACE", "ENZAMET", "ARASENS", "LATITUDE", "TITAN", "ARCHES",
          "Gravis", "Attard", "James", "Kriayako", "Sweeney", "Aggarwal", "Fizazi", "Hussain", "Smith",
          "docetaxel", "darolutamide", "abiraterone", "enzalutamide"]


def _runner():
    spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "experiment-scripts" / "run_benchmark.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
def test_rules_are_generic(version):
    text = extraction_rules.RULES[version]
    with open(DEFINITIONS_CSV_PATH, newline="") as handle:
        columns = [row["Column Name"] for row in csv.DictReader(handle)]
    assert not [c for c in columns if c in text], "rules must not name benchmark columns"
    assert not [t for t in TRIALS if t.lower() in text.lower()], "rules must not name trials, papers or drugs"
    assert not re.search(r"\d+\.\d", text), "rules must not contain values (decimals) that could come from a paper"


def test_v2_adds_named_subtypes_to_v1():
    assert extraction_rules.RULES["v2"].startswith(extraction_rules.RULES["v1"].split('answer "Not reported".')[0])
    assert "biochemical" in extraction_rules.RULES["v2"] and "biochemical" not in extraction_rules.RULES["v1"]


def test_v4_is_v3_without_the_every_patient_bullet_and_with_an_unquoted_example():
    v3 = extraction_rules.RULES["v3"].replace('(for example "bPFS X months; rPFS Y months")', "(for example: bPFS X months; rPFS Y months)")
    v4 = extraction_rules.RULES["v4"]
    removed = [line for line in v3.splitlines() if line not in v4.splitlines()]
    assert removed[0].startswith("- A value the paper states for every patient") and len(removed) == 4
    assert [line for line in v3.splitlines() if line not in removed] == v4.splitlines()
    assert "0 (0%)" not in v4 and "An endpoint keeps its identity" in v4 and "events/N" in v4 and '"bPFS' not in v4


def test_v5_adds_three_bullets_to_v4_before_the_arm_size_rule():
    v4, v5 = extraction_rules.RULES["v4"], extraction_rules.RULES["v5"]
    added = [line for line in v5.splitlines() if line not in v4.splitlines()]
    assert sum(line.startswith("- ") for line in added) == 3
    assert [line for line in v5.splitlines() if line not in added] == v4.splitlines()  # nothing in v4 changed
    order = [v5.index(s) for s in ("- Counts of patients", "- Report a value in the column's unit", "- A subgroup column",
                                   "- When the trial design gives a treatment", "- Total-participant")]
    assert order == sorted(order)
    new_text = "\n".join(added)
    assert '"' not in new_text.replace('"Not reported"', "")  # no quoted examples (a quote copied into JSON looped Agent A)
    assert "not patient characteristics or eligibility criteria" in " ".join(new_text.split())


def test_v3_adds_three_bullets_to_v1():
    v1, v3 = extraction_rules.RULES["v1"], extraction_rules.RULES["v3"]
    v1_lines = v1.splitlines()
    added = [line for line in v3.splitlines() if line not in v1_lines]
    assert sum(line.startswith("- ") for line in added) == 3
    assert [line for line in v3.splitlines() if line in v1_lines] == v1_lines  # every v1 line kept, in order
    order = [v3.index(s) for s in ("- The column name says", "- An endpoint keeps its identity", "- A median that",
                                   "- A per-arm column", "- A value the paper states", "- Counts of patients",
                                   "- Total-participant")]
    assert order == sorted(order)
    # The variant rule is its own bullet, not part of the clause that ends in "Not reported" (the v2 sentence was).
    assert "A named subtype" not in v3
    # Zero is only for treatments an arm did not receive; other categories of a characteristic stay empty.
    assert "no such treatment, give 0 (0%)" in " ".join(v3.split()) and v3.count("0 (0%)") == 1


def test_none_reproduces_the_e0_prompts():
    assert extraction_rules.shared_rules("none") == ""
    assert pdf_query.SYSTEM_PROMPT + pdf_query.IMAGE_RULES + extraction_rules.shared_rules("none") == pdf_query.SYSTEM_PROMPT + pdf_query.IMAGE_RULES
    assert markdown_baseline.build_prompt("ID", [{"column": "c", "definition": "d"}]) == markdown_baseline.build_prompt("ID", [{"column": "c", "definition": "d"}], "")


def test_settings_record_the_rules_for_every_stage(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(extraction_rules, "rules_version", lambda: "v1")
    assert pdf_query_pipeline.run_settings("qwen3.6-27b", "markdown_images")["extraction_rules"] == "v1"
    assert all(runner.stage_settings(stage)["extraction_rules"] == "v1" for stage in ("agent", "search", "baseline"))
    monkeypatch.setattr(extraction_rules, "rules_version", lambda: "none")
    assert all(runner.stage_settings(stage)["extraction_rules"] is None for stage in ("agent", "search", "baseline"))


@pytest.mark.parametrize("saved,current,refused", [
    ({"extraction_rules": "v1"}, "none", True),   # v1 results are never resumed under none
    ({"extraction_rules": None}, "v1", True),     # none results are never resumed under v1
    ({}, "v1", True),                             # results made before the option existed are E0 prompts
    ({}, "none", False),
    ({"extraction_rules": "v1"}, "v1", False),
])
def test_resume_refuses_mixing_rule_versions_both_ways(tmp_path, monkeypatch, saved, current, refused):
    monkeypatch.setattr(store, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(store, "_run", "")
    store.save_columns("doc", "search", {"c": {"value": "1"}})
    store.save_metadata("doc", "search", {"model": "m", **saved})
    settings = {"model": "m", **extraction_rules.rules_setting(current)}
    if refused:
        with pytest.raises(store.ResumeError):
            store.check_resume("doc", "search", settings)
    else:
        store.check_resume("doc", "search", settings)


def test_rules_reach_every_extraction_prompt():
    text =pdf_query.SYSTEM_PROMPT + pdf_query.IMAGE_RULES + extraction_rules.RULES["v1"]
    assert pdf_query.system_prompt_text() == pdf_query.SYSTEM_PROMPT + pdf_query.IMAGE_RULES + extraction_rules.shared_rules()
    assert text.index("Pages come with") < text.index("COLUMN AND TRIAL CONVENTIONS")  # image rules stay under "Rules:"
    source = {name: (ROOT / "src" / "evisearch" / "services" / f"{name}.py").read_text() for name in ("pdf_query", "search", "markdown_baseline")}
    assert "system = system_prompt_text(" in source["pdf_query"]
    assert "SYSTEM_PROMPT + shared_rules(columns=names)" in source["search"]  # the batch's columns (scoped delivery)
    assert "query_label_groups(provider, markdown_text, pending, workers, shared_rules())" in source["markdown_baseline"]
    prompt = markdown_baseline.build_prompt("ID", [{"column": "c", "definition": "d"}], extraction_rules.RULES["v1"])
    assert prompt.index("COLUMN AND TRIAL CONVENTIONS") < prompt.index("Pay special attention")
