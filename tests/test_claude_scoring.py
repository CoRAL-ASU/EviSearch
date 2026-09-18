from __future__ import annotations

import json

import pytest

from src.evaluation import claude_scoring as cs

COLUMNS = {
    "Control Arm - N": cs.Column("Control Arm - N", "numeric_tolerance", "Number randomised to control"),
    "Region": cs.Column("Region", "structured_text", "Where the trial ran"),
    "NCT": cs.Column("NCT", "exact_match", "Trial identifier"),
}
GOLD = {"docA": {"Control Arm - N": "454", "Region": "", "NCT": "NCT1"}}


def _write_output(root, run, stage, doc, columns):
    path = root / doc / "runs" / run / stage / cs.STAGE_FILES[stage]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"doc_id": doc, "columns": columns}))


def test_empty_equivalents_follow_the_rubric():
    for value in ["", None, "Not reported", "not found.", "N/A", "NaN", "not present", " na "]:
        assert cs.is_empty(value)
    assert not cs.is_empty("0")


def test_item_id_ignores_whitespace_but_not_content():
    assert cs.item_id("d", "c", "12.9  mo") == cs.item_id("d", "c", " 12.9 mo")
    assert cs.item_id("d", "c", "12.9") != cs.item_id("d", "c", "12.8")


def test_system_spec_parsing():
    s = cs.System.parse("E0=e0_qwen/reconciliation_agent")
    assert (s.name, s.run, s.stage) == ("E0", "e0_qwen", "reconciliation_agent")
    with pytest.raises(ValueError):
        cs.System.parse("E0=e0_qwen/unknown_stage")


def test_cells_treat_missing_columns_as_empty_predictions(tmp_path):
    _write_output(tmp_path, "r", "agent_extractor", "docA", {"Control Arm - N": {"value": "454"}})
    got = cs.cells(cs.System("A", "r", "agent_extractor"), ["docA"], tmp_path, COLUMNS, GOLD)
    by_col = {c.column: c for c in got}
    assert by_col["NCT"].missing and by_col["NCT"].pred == ""
    assert by_col["Region"].mechanical  # gold empty, prediction empty -> 1/1 by rule
    assert not by_col["NCT"].mechanical  # gold has a value -> must be scored (a miss)


def test_queue_is_blinded_deduplicated_and_skips_scored_and_mechanical(tmp_path):
    _write_output(tmp_path, "r1", "agent_extractor", "docA", {"Control Arm - N": {"value": "454"}, "Region": {"value": "Europe"}, "NCT": {"value": "NCT1"}})
    _write_output(tmp_path, "r2", "search_agent", "docA", {"Control Arm - N": {"value": "454"}, "Region": {"value": "Not reported"}, "NCT": {"value": "NCT2"}})
    systems = [cs.System("A", "r1", "agent_extractor"), cs.System("B", "r2", "search_agent")]
    all_cells = [c for s in systems for c in cs.cells(s, ["docA"], tmp_path, COLUMNS, GOLD)]
    already = {cs.item_id("docA", "NCT", "NCT1"): {"correctness": 1.0, "completeness": 1.0}}
    written = cs.build_queue(all_cells, already, tmp_path / "q")
    items = [i for p in written for i in json.loads(p.read_text())["items"]]
    # "454" appears in both systems but is queued once; NCT1 already scored; Region/Not reported is mechanical
    assert sorted((i["column"], i["pred"]) for i in items) == [("Control Arm - N", "454"), ("NCT", "NCT2"), ("Region", "Europe")]
    assert all(set(i) == {"id", "column", "definition", "gold", "pred"} for i in items)


def test_ingest_validates_and_appends(tmp_path):
    cell = cs.Cell("docA", "Control Arm - N", "numeric_tolerance", "def", "454", "502")
    [batch_path] = cs.build_queue([cell], {}, tmp_path / "q")
    batch = json.loads(batch_path.read_text())
    store = tmp_path / "labels.jsonl"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"batch": batch["batch"], "results": [{"id": cell.id, "correctness": 0.3, "completeness": 0, "reason": "x"}]}))
    with pytest.raises(ValueError, match="not 0, 0.5 or 1"):
        cs.ingest(bad, tmp_path / "q", store, "claude")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"batch": batch["batch"], "results": [{"id": cell.id, "correctness": 0, "completeness": 0, "reason": "502 vs 454"}]}))
    assert cs.ingest(good, tmp_path / "q", store, "claude") == 1
    assert cs.load_labels(store)[cell.id]["gold"] == "454"


def test_review_stats_compute_agreement_accuracy_and_flags():
    def scored(v, c, k):
        return cs.Scored(cs.Cell("d", f"c{v}{c}{k}", "numeric_tolerance", "", "1", "1", verification=v), c, k, "label")
    rows = [scored("both_correct", 1, 1), scored("both_correct", 1, 0), scored("both_wrong", 0, 0),
            scored("both_wrong", 1, 1), scored("A_correct_B_wrong", 0, 0)]
    stats = cs.review_stats(rows)
    assert stats["agreed"]["accuracy"] == 75.0
    assert stats["flag_rate"] == 40.0
    assert stats["flag_precision"] == 50.0  # 1 of 2 flagged cells is wrong
    assert stats["flag_recall"] == pytest.approx(33.33)  # 1 of 3 imperfect cells was flagged
    assert stats["accuracy_after_simulated_review"] == 70.0  # (1 + 0.5 + 1 + 1 + 0) / 5: flagged cells counted as fixed


def test_agreement_kappa():
    a = {"1": {"correctness": 1, "completeness": 1}, "2": {"correctness": 0, "completeness": 0}}
    b = {"1": {"correctness": 1, "completeness": 1}, "2": {"correctness": 0, "completeness": 0.5}}
    got = cs.agreement(a, b)
    assert got["n"] == 2 and got["exact_pair_agreement"] == 50.0


def test_rubric_is_the_evaluator_prompt_verbatim():
    text = cs.rubric_text("numeric_tolerance")
    assert "Tolerance**: ±0.1 for absolute values, ±2% relative for percentages" in text
    assert "When GT is empty and Pred has a number or substantive value → correctness=0.0" in text


def test_real_gold_and_definitions_load():
    columns, gold = cs.load_columns(), cs.load_gold()
    assert len(columns) == 133 and len(gold) == 10
    assert len(cs.select_docs("heldout", gold)) == 3 and len(cs.select_docs("dev", gold)) == 7
