"""The table workspace API: runs read from disk, reviews kept per run (a correction on one run never shows on another),
undo, the review queue order, the filled-in table and its export, run comparison, and jobs that survive on disk."""
from __future__ import annotations

import json
import sys
import time

import pytest

from src.config import runtime_paths
from src.evisearch.services import feedback, jobs, reviews
from src.evisearch.services import runs as runs_service

DOC, OTHER = "NCT01_Smith_ARASENS", "NCT02_Kriayako_CHAARTED"
COLS = ["Median OS | Treatment", "Region | Asia", "Trial Name"]


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _run(root, doc, run, recon, a=None, b=None, manifest=True):
    base = root / doc / "runs" / run
    _write(base / "reconciliation_agent" / "reconciled_results.json", {"doc_id": doc, "columns": recon})
    _write(base / "agent_extractor" / "extraction_results.json", {"columns": a or {}})
    _write(base / "search_agent" / "extraction_results.json", {"columns": b or {}})
    if manifest:
        _write(base / "benchmark_manifest.json", {"doc_id": doc, "run": run, "status": "ok", "stages": {"agent": {"status": "ok"}}})


@pytest.fixture
def ws(isolated_app, client, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback" / "feedback.jsonl")
    monkeypatch.setattr(feedback, "FEEDBACK_DIR", tmp_path / "feedback")
    monkeypatch.setattr(isolated_app, "record_feedback", feedback.record_feedback)
    monkeypatch.setattr(runtime_paths, "SCHEMAS_DIR", tmp_path / "schemas")
    root = runtime_paths.RESULTS_ROOT
    cell = lambda v, **kw: {"value": v, "verified": True, "needs_review": False, **kw}
    for run in ("r1", "r2"):
        _run(root, DOC, run, {
            COLS[0]: cell("62.1", source={"page": 4, "modality": "table", "verbatim_quote": "median OS 62.1 months"}),
            COLS[1]: cell("Not reported", needs_review=True, review_reason="every value failed the page check"),
            COLS[2]: cell("ARASENS" if run == "r1" else "ARASENS trial"),
        }, a={COLS[0]: {"value": "62.1"}, COLS[2]: {"value": "ARASENS"}}, b={COLS[0]: {"value": "57.6"}, COLS[2]: {"value": "ARASENS"}})
    _run(root, OTHER, "r1", {COLS[0]: cell("57.6")}, manifest=False)
    return client


def test_runs_are_listed_with_their_papers(ws):
    listed = {r["run"]: r for r in runs_service.list_runs()}
    assert listed["r1"]["docs"] == sorted([DOC, OTHER]) and listed["r2"]["docs"] == [DOC]
    assert runs_service.doc_progress(OTHER, "r1")["status"] == "running"  # outputs but no manifest yet
    assert runs_service.parse_run_name("schema-mhspc-20260919-v3-kboff-r2") == {"schema_id": "mhspc-20260919", "version": 3, "variant": "kboff-r2"}


def test_a_review_stays_on_its_own_run_and_can_be_undone(ws):
    saved = ws.post(f"/api/runs/r1/docs/{DOC}/review", json={"column": COLS[1], "value": "Included in Rest of the world",
                                                           "reason": "missed value on the page", "note": "p2 footnote", "by": "rev"}).get_json()
    assert saved["success"] and saved["state"] == "corrected" and saved["event"]["machine_value"] == "Not reported"
    r1 = {c["column"]: c for c in ws.get(f"/api/runs/r1/docs/{DOC}/cells").get_json()["columns"]}
    r2 = {c["column"]: c for c in ws.get(f"/api/runs/r2/docs/{DOC}/cells").get_json()["columns"]}
    assert r1[COLS[1]]["review"]["value"] == "Included in Rest of the world"
    assert r1[COLS[1]]["value"] == "Not reported"  # the machine value is never replaced
    assert r2[COLS[1]]["review"] is None and r2[COLS[1]]["state"] == "flagged"
    # the legacy per-run view (/api/documents/<doc>/reconciled?run=) reads the same per-run reviews
    undo = ws.post(f"/api/runs/r1/docs/{DOC}/undo", json={"event_id": saved["event"]["event_id"], "by": "rev"}).get_json()
    assert undo["success"] and undo["review"]["value"] is None and undo["state"] == "flagged"


def test_a_correction_needs_a_reason_and_a_name(ws):
    body = {"column": COLS[0], "value": "57.6", "by": "rev"}
    assert "reason" in ws.post(f"/api/runs/r1/docs/{DOC}/review", json=body).get_json()["error"]
    body.update(reason="wrong arm", by="")
    assert "name" in ws.post(f"/api/runs/r1/docs/{DOC}/review", json=body).get_json()["error"]
    ok = ws.post(f"/api/runs/r1/docs/{DOC}/review", json={"column": COLS[0], "value": "62.1", "by": "rev"}).get_json()
    assert ok["state"] == "accepted" and ok["event"]["reason"] == reviews.ACCEPT_REASON


def test_queue_puts_flagged_then_disputed_cells_first(ws):
    q = ws.get("/api/runs/r1/queue").get_json()
    kinds = [(i["kind"], i["column"]) for i in q["items"] if i["doc_id"] == DOC]
    assert kinds[0] == ("flagged", COLS[1]) and kinds[1] == ("disputed", COLS[0])
    assert q["counts"]["flagged"] == {"total": 1, "reviewed": 0}


def test_table_grid_states_and_csv_export(ws):
    ws.post(f"/api/runs/r1/docs/{DOC}/review", json={"column": COLS[2], "value": "ARASENS", "by": "rev"})
    grid = ws.get("/api/runs/r1/table").get_json()
    row = next(d for d in grid["docs"] if d["doc_id"] == DOC)["cells"]
    assert row[COLS[0]]["s"] == "verified" and row[COLS[0]]["p"] == 4
    assert row[COLS[1]]["s"] == "flagged" and row[COLS[2]]["s"] == "accepted"
    csv_text = ws.get("/api/runs/r1/export.csv").get_data(as_text=True)
    assert csv_text.splitlines()[0].startswith("Document,") and "62.1" in csv_text


def test_compare_counts_changes_and_review_matches(ws):
    ws.post(f"/api/runs/r1/docs/{DOC}/review", json={"column": COLS[2], "value": "ARASENS trial", "reason": "unit or format", "by": "rev"})
    out = ws.get("/api/runs/compare?a=r1&b=r2").get_json()
    changed = out["docs"][0]["changed"]
    assert [c["column"] for c in changed] == [COLS[2]] and changed[0]["b_matches_review"]
    assert out["totals"]["fixed_by_review"] == 1


def test_legacy_human_edited_endpoint_records_the_runs_machine_value(ws):
    body = {"run": "r1", "by": "script", "columns": {COLS[1]: {"value": "Included", "reason": "missed value on the page"}}}
    assert ws.post(f"/api/documents/{DOC}/human-edited", json=body).get_json()["success"]
    event = [e for e in feedback.all_events() if e.get("event") == "cell_correct"][-1]
    assert event["machine_value"] == "Not reported" and event["state"] == "corrected" and event["event_id"]
    view = ws.get(f"/api/documents/{DOC}/reconciled?run=r2").get_json()
    assert view["success"] is False or not any(c.get("human_edited") for c in view.get("columns", []))


def test_bad_names_are_rejected(ws):
    assert ws.get("/api/runs/..%2Fx/table").status_code in (400, 404)
    assert ws.post("/api/runs/r1/docs/..%2Fetc/review", json={"column": "x"}).status_code in (400, 404)


def test_jobs_survive_on_disk_and_processes_can_be_cancelled(ws, tmp_path):
    job = jobs.run_thread("demo", lambda: {"answer": 42}, by="rev", table="t1")
    for _ in range(100):
        if jobs.get(job["id"])["status"] == "done":
            break
        time.sleep(0.02)
    assert jobs.get(job["id"])["result"] == {"answer": 42}
    stale = jobs.create("draft_schema", boot="an-earlier-server")
    jobs.update(stale["id"], status="running")
    assert jobs.get(stale["id"])["status"] == "interrupted"
    proc = jobs.run_process("extract", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "x.log", table="t1", run="rx")
    assert jobs.get(proc["id"])["status"] == "running"
    assert ws.post(f"/api/jobs/{proc['id']}/cancel", json={"by": "rev"}).get_json()["job"]["status"] == "cancelled"
    assert [j["id"] for j in ws.get("/api/jobs?table=t1").get_json()["jobs"]][:1] == [proc["id"]]


def test_unique_run_names_never_resume_an_existing_run(ws):
    from web.workspace_routes import unique_run_name

    assert unique_run_name("r1") == "r1-r2"
    assert unique_run_name("fresh") == "fresh"
