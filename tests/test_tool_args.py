from __future__ import annotations

import json

from src.evisearch.services.search import MAX_SUBMIT_RETRIES, _SearchSession
from src.evisearch.tool_args import decode_items

A = {"column": "Control Arm", "value": "ADT alone", "reasoning": "p2", "found": True,
     "attribution": [{"page": 2, "modality": "text", "evidence": "randomized to ADT plus D or ADT alone"}]}
B = {"column": "Control Arm - N", "value": "193", "reasoning": "p3", "found": True,
     "attribution": [{"page": 3, "modality": "table", "evidence": "ADT alone (n = 193)"}]}


def test_lists_dicts_and_valid_strings_pass_through():
    assert decode_items([A]) == ([A], None)
    assert decode_items({"x": 1}) == ({"x": 1}, None)
    assert decode_items(json.dumps([A, B]))[0] == [A, B]


def test_extra_closing_brackets_between_items_are_recovered():
    # shape seen in real runs: an extra "]}" after the first item's attribution list
    broken = json.dumps([A])[:-1] + "]}, " + json.dumps(B) + "]"
    items, note = decode_items(broken)
    assert items == [A, B] and "recovered 2" in note


def test_items_that_lost_their_opening_brace_are_recovered():
    broken = "[" + json.dumps(A) + "]}], " + json.dumps(B)[1:] + "]"
    items, _ = decode_items(broken)
    assert [i["column"] for i in items] == ["Control Arm", "Control Arm - N"]


def test_unreadable_strings_say_why():
    items, note = decode_items('[{"value": "x", ')
    assert items is None and "not valid JSON" in note


def _session():
    return _SearchSession("doc", ["Control Arm", "Control Arm - N"], 7)


def test_submit_reads_a_string_argument():
    s = _session()
    out = s.submit_extraction({"results": json.dumps([A, B])})
    assert out.stop and set(s.submitted) == {"Control Arm", "Control Arm - N"}
    assert s.submitted["Control Arm - N"]["value"] == "193"


def test_submit_asks_again_for_missing_columns_then_merges():
    s = _session()
    degenerate = json.dumps([A])[:-1] + ']}], "found": true, "attribution": [{"page": 1'  # loop cut off mid-item
    first = s.submit_extraction({"results": degenerate})
    assert not first.stop and "Control Arm - N" in first.content["error"]
    second = s.submit_extraction({"results": [B]})
    assert second.stop and set(s.submitted) == {"Control Arm", "Control Arm - N"}


def test_submit_gives_up_after_the_retry_budget():
    s = _session()
    for _ in range(MAX_SUBMIT_RETRIES):
        assert not s.submit_extraction({"results": "not json"}).stop
    final = s.submit_extraction({"results": "not json"})
    assert final.stop and s.submitted == {}
