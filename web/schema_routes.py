"""Schema and human-feedback routes: generate a schema from a spreadsheet, review and lock it, extract under it, and
turn reviewers' feedback into conventions through the knowledge base's integrity gate.

Pages: /schema, /feedback. APIs (JSON, {"success": ...} envelope like the rest of the app):
  GET  /api/schemas                         list
  POST /api/schemas                         create a draft (multipart sheet + pdf, or JSON {sheet_path, doc_id}); returns a job
  GET  /api/schemas/<id>                    the schema (current state; ?version=N for a locked snapshot)
  POST /api/schemas/<id>/review             {column, action: accept|edit|answer, definition?, question_id?, answer?, reason?, note?, by}
  POST /api/schemas/<id>/revise             the agent rewrites the definitions that got answers or notes; returns a job
  POST /api/schemas/<id>/lock               snapshot as the next version
  POST /api/schemas/<id>/extract            {docs, system?, kb?} runs experiment-scripts/run_schema.py in a child process
  GET  /api/schemas/<id>/extractions        runs under the schema's versions and their papers
  GET  /api/conventions                     the knowledge base (?status=)
  POST /api/conventions/propose             {column, definition, feedback, before?, after?, reason?, doc_id?, schema_id?}: proposer + gate + impact, stores nothing
  POST /api/conventions                     {record, by}: gate again; duplicate -> merged, conflict -> 409, else stored as proposed
  POST /api/conventions/<cid>/decide        {op: approve|reject|retire, by, note?}
  GET  /api/feedback/events                 feedback log (?source=&schema_id=&doc_id=&event=&limit=)
  GET  /api/jobs/<job_id>                   status of a long-running job
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict

from flask import Blueprint, jsonify, render_template, request

from src.config import runtime_paths
from src.evisearch.knowledge import conventions as kb
from src.evisearch.knowledge import gate, proposer
from src.evisearch.schema import generator, store
from src.evisearch.services import feedback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
bp = Blueprint("schema_layer", __name__)
JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _chat():
    from src.inference import get_chat

    return get_chat("reconciliation")


def _ok(**payload: Any):
    return jsonify({"success": True, **payload}), 200


def _err(message: str, status: int = 400, **payload: Any):
    return jsonify({"success": False, "error": message, **payload}), status


def _start_job(kind: str, fn: Callable[[], Any], **meta: Any) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "status": "running", "started": time.time(), **meta}
    with _JOBS_LOCK:
        JOBS[job_id] = job

    def run():
        try:
            job["result"] = fn()
            job["status"] = "done"
        except Exception as exc:  # reported to the page; the job never kills the server
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
        job["ended"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ---- pages -------------------------------------------------------------------------------------------------------
@bp.route("/schema")
def schema_page():
    return render_template("schema.html")


@bp.route("/feedback")
def feedback_page():
    return render_template("feedback.html")


@bp.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    return _ok(job=job) if job else _err("unknown job", 404)


# ---- schemas -----------------------------------------------------------------------------------------------------
@bp.route("/api/schemas", methods=["GET"])
def api_schemas():
    return _ok(schemas=store.list_schemas())


@bp.route("/api/schemas", methods=["POST"])
def api_create_schema():
    if request.files:
        sheet = request.files.get("sheet")
        if sheet is None or not sheet.filename:
            return _err("sheet file required")
        folder = runtime_paths.SCHEMAS_DIR / "_uploads"
        folder.mkdir(parents=True, exist_ok=True)
        sheet_path = folder / f"{uuid.uuid4().hex[:8]}_{Path(sheet.filename).name}"
        sheet.save(sheet_path)
        form = request.form
        doc_id = form.get("doc_id", "").strip()
        pdf = request.files.get("pdf")
        if pdf is not None and pdf.filename and not doc_id:
            from src.documents.pdf_registry import register_uploaded_pdf

            doc_id = register_uploaded_pdf(pdf.read(), original_filename=pdf.filename, uploads_dir=runtime_paths.UPLOADS_DIR,
                                           results_root=runtime_paths.RESULTS_ROOT, dataset_dir=runtime_paths.DATASET_DIR)["canonical_doc_id"]
        body = dict(form)
    else:
        body = request.get_json() or {}
        sheet_path, doc_id = Path(str(body.get("sheet_path", ""))), str(body.get("doc_id", "")).strip()
        if not sheet_path.exists():
            return _err("sheet_path does not exist")
    if not doc_id:
        return _err("doc_id (or a pdf) required: the example row's paper")
    name = str(body.get("name") or Path(sheet_path).stem)
    by, hint, description = str(body.get("by", "")), body.get("row_hint") or None, str(body.get("description", ""))

    def draft():
        schema = generator.draft_schema_from_sheet(_chat(), str(sheet_path), doc_id, name, row_hint=hint, by=by,
                                                   conventions=kb.render() if kb.active() else "", description=description)
        return {"schema_id": schema["id"]}

    return _ok(job_id=_start_job("draft_schema", draft, name=name, doc_id=doc_id))


@bp.route("/api/schemas/<schema_id>", methods=["GET"])
def api_schema(schema_id):
    try:
        version = request.args.get("version", type=int)
        return _ok(schema=store.load(schema_id, version))
    except FileNotFoundError as exc:
        return _err(str(exc), 404)


@bp.route("/api/schemas/<schema_id>/review", methods=["POST"])
def api_review(schema_id):
    body = request.get_json() or {}
    try:
        field = store.review_field(schema_id, str(body.get("column", "")), str(body.get("action", "")), by=str(body.get("by", "")),
                                   definition=body.get("definition"), question_id=body.get("question_id"), answer=body.get("answer"),
                                   reason=str(body.get("reason", "")), note=str(body.get("note", "")))
        return _ok(field=field)
    except (KeyError, ValueError) as exc:
        return _err(str(exc))
    except FileNotFoundError as exc:
        return _err(str(exc), 404)


@bp.route("/api/schemas/<schema_id>/revise", methods=["POST"])
def api_revise(schema_id):
    body = request.get_json() or {}
    by = str(body.get("by", "schema-agent"))
    schema = store.load(schema_id)

    def revise():
        revised, logs = generator.revise_fields(_chat(), schema["fields"], conventions=kb.render() if kb.active() else "")
        for column, item in revised.items():
            store.review_field(schema_id, column, "revise", by=by, definition=item.get("definition", ""), note=item.get("change", ""))
        (store.schema_dir(schema_id) / f"revise_calls_{int(time.time())}.json").write_text(json.dumps(logs, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"revised": sorted(revised)}

    return _ok(job_id=_start_job("revise_schema", revise, schema_id=schema_id))


@bp.route("/api/schemas/<schema_id>/lock", methods=["POST"])
def api_lock(schema_id):
    body = request.get_json() or {}
    try:
        return _ok(**store.lock(schema_id, by=str(body.get("by", ""))))
    except FileNotFoundError as exc:
        return _err(str(exc), 404)


@bp.route("/api/schemas/<schema_id>/extract", methods=["POST"])
def api_extract(schema_id):
    body = request.get_json() or {}
    docs = body.get("docs")
    docs = ",".join(docs) if isinstance(docs, list) else str(docs or "")
    if not docs:
        return _err("docs required")
    schema = store.load(schema_id)
    if not schema.get("locked_versions"):
        return _err("lock the schema before extracting")
    version = int(body.get("version") or max(schema["locked_versions"]))
    system, use_kb = str(body.get("system", "E")), str(body.get("kb", "on"))
    run = store.run_name(schema_id, version) + ("" if use_kb == "on" else "-kboff") + ("" if system == "E" else f"-{system.lower()}")
    log = runtime_paths.RESULTS_ROOT.parent / "benchmark_runs" / f"{run}.driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "experiment-scripts" / "run_schema.py"), "--schema", schema_id, "--version", str(version),
           "--system", system, "--docs", docs, "--kb", use_kb, "--run", run]
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    return _ok(run=run, pid=proc.pid, log=str(log))


@bp.route("/api/schemas/<schema_id>/extractions", methods=["GET"])
def api_extractions(schema_id):
    prefix = f"schema-{schema_id}-v"
    runs: Dict[str, Dict[str, Any]] = {}
    root = runtime_paths.RESULTS_ROOT
    for run_dir in root.glob(f"*/runs/{prefix}*") if root.exists() else []:
        doc = run_dir.parent.parent.name
        manifest = run_dir / "benchmark_manifest.json"
        status = json.loads(manifest.read_text(encoding="utf-8")).get("status") if manifest.exists() else "running"
        entry = runs.setdefault(run_dir.name, {"run": run_dir.name, "papers": []})
        entry["papers"].append({"doc_id": doc, "status": status,
                                "reconciled": (run_dir / "reconciliation_agent" / "reconciled_results.json").exists()})
    return _ok(runs=sorted(runs.values(), key=lambda r: r["run"]))


# ---- conventions (knowledge base) ---------------------------------------------------------------------------------
@bp.route("/api/conventions", methods=["GET"])
def api_conventions():
    status = request.args.get("status")
    items = [c for c in kb.load_all().values() if not status or c["status"] == status]
    return _ok(conventions=items, fingerprint=kb.fingerprint(), active=sum(c["status"] == "approved" for c in items))


def _schema_fields(schema_id: str | None):
    if not schema_id:
        return []
    try:
        return store.load(schema_id)["fields"]
    except FileNotFoundError:
        return []


@bp.route("/api/conventions/propose", methods=["POST"])
def api_propose():
    body = request.get_json() or {}
    column = str(body.get("column", ""))
    if not column or not str(body.get("feedback", "")).strip():
        return _err("column and feedback required")
    fields = _schema_fields(body.get("schema_id"))
    chat = _chat()
    out = proposer.propose(chat, column=column, definition=str(body.get("definition", "")), feedback=str(body["feedback"]),
                           before=str(body.get("before", "")), after=str(body.get("after", "")), reason=str(body.get("reason", "")),
                           columns=[f["name"] for f in fields] or [column], paper=str(body.get("doc_id", "")),
                           source={"kind": str(body.get("kind", "extraction_review")), "by": str(body.get("by", "")),
                                   "schema_id": body.get("schema_id"), "doc_id": body.get("doc_id")})
    if not out["is_convention"]:
        return _ok(is_convention=False, why_not=out["why_not"])
    record = out["record"]
    return _ok(is_convention=True, record=record, gate=gate.check(chat, record),
               impact=gate.impact(record["trigger"], fields, body.get("schema_id")))


@bp.route("/api/conventions", methods=["POST"])
def api_add_convention():
    body = request.get_json() or {}
    record, by = body.get("record") or {}, str(body.get("by", ""))
    try:
        verdict = gate.check(_chat(), record)
        if verdict["verdict"] == "duplicate":
            merged = kb.merge_into(verdict["duplicate_of"], record.get("examples", []), by=by, note="duplicate proposal merged")
            store.record_event("convention_merge", record.get("source", {}).get("schema_id"), by=by, convention=merged["id"])
            return _ok(merged_into=merged["id"], convention=merged, gate=verdict)
        if verdict["verdict"] == "blocked":
            return _err("conflicts with existing conventions", 409, gate=verdict)
        created = kb.create(record, by=by)
        store.record_event("convention_propose", record.get("source", {}).get("schema_id"), by=by, convention=created["id"],
                           instruction=created["instruction"], relations=verdict["relations"])
        return _ok(convention=created, gate=verdict)
    except ValueError as exc:
        return _err(str(exc))


@bp.route("/api/conventions/<cid>/decide", methods=["POST"])
def api_decide(cid):
    body = request.get_json() or {}
    try:
        rec = kb.decide(cid, str(body.get("op", "")), by=str(body.get("by", "")), note=str(body.get("note", "")))
    except KeyError:
        return _err("unknown convention", 404)
    except ValueError as exc:
        return _err(str(exc))
    store.record_event("convention_decide", rec.get("source", {}).get("schema_id"), by=str(body.get("by", "")), convention=cid,
                       op=str(body.get("op")), note=str(body.get("note", "")))
    return _ok(convention=rec)


# ---- feedback log ------------------------------------------------------------------------------------------------
@bp.route("/api/feedback/events", methods=["GET"])
def api_feedback_events():
    args = request.args
    events = feedback.load_feedback(doc_id=args.get("doc_id") or None, source=args.get("source") or None, limit=args.get("limit", 2000, type=int))
    if args.get("schema_id"):
        events = [e for e in events if e.get("schema_id") == args["schema_id"]]
    if args.get("event"):
        events = [e for e in events if e.get("event") == args["event"]]
    return _ok(events=events)
