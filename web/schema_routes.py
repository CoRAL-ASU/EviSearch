"""Schema and human-feedback routes: generate a schema from a spreadsheet, review and lock it, extract under it, and
turn reviewers' feedback into edits to the knowledge notes.

Pages: /schema, /feedback. APIs (JSON, {"success": ...} envelope like the rest of the app):
  GET  /api/schemas                         list
  POST /api/schemas                         create a draft (multipart sheet + pdf, or JSON {sheet_path, doc_id}); returns a job
  GET  /api/schemas/<id>                    the schema (current state; ?version=N for a locked snapshot)
  POST /api/schemas/<id>/review             {column, action: accept|edit|answer, definition?, question_id?, answer?, reason?, note?, by}
  GET  /api/notes                           the notes tree and its fingerprint
  GET  /api/notes/log                       every edit ever made to the notes
  POST /api/notes/propose                   {column, definition, feedback, before?, after?, reason?}: a proposed edit, stores nothing
  POST /api/notes/apply                     {note, text, heading?, by, why?, role?}: write it and record it
  POST /api/schemas/<id>/revise             the agent rewrites the definitions that got answers or notes; returns a job
  POST /api/schemas/<id>/lock               snapshot as the next version
  POST /api/schemas/<id>/extract            {docs, system?, kb?} runs experiment-scripts/run_schema.py in a child process
  GET  /api/schemas/<id>/extractions        runs under the schema's versions and their papers
  GET  /api/feedback/events                 feedback log (?source=&schema_id=&doc_id=&event=&limit=)
  GET  /api/jobs/<job_id>                   status of a long-running job
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict

from flask import Blueprint, jsonify, request

from src.config import runtime_paths
from src.evisearch.knowledge import notes as notes_kb
from src.evisearch.knowledge import proposer
from src.evisearch.schema import generator, store
from src.evisearch.services import feedback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
bp = Blueprint("schema_layer", __name__)


def _chat():
    from src.inference import get_chat

    return get_chat("reconciliation")


def _ok(**payload: Any):
    return jsonify({"success": True, **payload}), 200


def _err(message: str, status: int = 400, **payload: Any):
    return jsonify({"success": False, "error": message, **payload}), status


def _start_job(kind: str, fn: Callable[[], Any], **meta: Any) -> str:
    """Run `fn` as a job kept on disk (services/jobs.py): its status survives reloads, and a restart reports it as
    interrupted instead of losing it."""
    from src.evisearch.services import jobs

    return jobs.run_thread(kind, fn, **meta)["id"]


# ---- pages -------------------------------------------------------------------------------------------------------
# /schema and /feedback now redirect into the table workspace and Learning (web/workspace_routes.py); this file keeps
# the APIs those pages used, which the workspace still calls.


@bp.route("/api/jobs/<job_id>")
def api_job(job_id):
    from src.evisearch.services import jobs

    try:
        job = jobs.get(job_id)
    except ValueError as exc:
        return _err(str(exc))
    if job and job.get("status") == "queued":
        job = dict(job, status="running")  # the old Schema page polls until the status leaves "running"
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
                                                   conventions=notes_kb.render(notes_kb.load_notes('all')), description=description)
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
        revised, logs = generator.revise_fields(_chat(), schema["fields"], conventions=notes_kb.render(notes_kb.load_notes('all')))
        changed = []
        for column, item in revised.items():
            field = store.review_field(schema_id, column, "revise", by=by, definition=item.get("definition", ""),
                                       note=item.get("change", ""), claimed_revised=item.get("revised"))
            # the agent returns every column it looked at; only the ones whose text moved are worth reviewing
            if not (field["x-evisearch"].get("history") or [{}])[-1].get("unchanged"):
                changed.append(column)
        (store.schema_dir(schema_id) / f"revise_calls_{int(time.time())}.json").write_text(json.dumps(logs, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"revised": sorted(revised), "changed": sorted(changed)}

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
    docs = [d.strip() for d in docs] if isinstance(docs, list) else [d.strip() for d in str(docs or "").split(",")]
    docs = [d for d in docs if d]
    if not docs:
        return _err("docs required")
    from web.workspace_routes import start_extraction  # one launcher: job registry, unique run names, one run at a time

    try:
        job = start_extraction(schema_id, docs, version=body.get("version"), system=str(body.get("system", "E")),
                               kb=str(body.get("kb", "on")), by=str(body.get("by", "")))
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    except RuntimeError as exc:
        return _err(str(exc), 409)
    except ValueError as exc:
        return _err(str(exc))
    return _ok(run=job["run"], job_id=job["id"], pid=job.get("pid"))


@bp.route("/api/schemas/<schema_id>/extractions", methods=["GET"])
def api_extractions(schema_id):
    from src.evisearch.services import runs as runs_service

    out = []
    for run in runs_service.list_runs(schema_id):
        papers = []
        for doc in run["docs"]:
            progress = runs_service.doc_progress(doc, run["run"])
            papers.append({"doc_id": doc, "status": progress["status"],
                           "reconciled": runs_service.stage_of(doc, run["run"]) == "reconciliation_agent"})
        out.append({"run": run["run"], "papers": papers})
    return _ok(runs=out)


# ---- knowledge notes ------------------------------------------------------------------------------------------------
# One knowledge format: markdown notes. There is no proposed/approved lifecycle and no integrity gate, because both
# existed to manage one-line rules that arrived without context. A note is a document about one topic, so a reviewer
# edits the text that is already there and can see what they are contradicting.
@bp.route("/api/notes", methods=["GET"])
def api_notes():
    items = [{"id": n.id, "role": n.role, "scope": n.scope, "family": n.family, "columns": list(n.columns),
              "supersedes": list(n.supersedes), "body": n.body, "path": n.path} for n in notes_kb.load_notes("all")]
    return _ok(notes=items, fingerprint=notes_kb.fingerprint(), count=len(items))


@bp.route("/api/notes/log", methods=["GET"])
def api_notes_log():
    """Every edit ever made to the notes, newest last: what changed, who asked, why, and the resulting fingerprint."""
    return _ok(log=notes_kb.log_entries())


def _schema_fields(schema_id: str | None):
    if not schema_id:
        return []
    try:
        return store.load(schema_id)["fields"]
    except FileNotFoundError:
        return []


@bp.route("/api/notes/propose", methods=["POST"])
def api_propose():
    """A reviewer's correction -> a proposed edit to a named note. Stores nothing; the reviewer applies it."""
    body = request.get_json() or {}
    column = str(body.get("column", ""))
    if not column or not str(body.get("feedback", "")).strip():
        return _err("column and feedback required")
    governing = notes_kb.governing([column])
    out = proposer.propose(_chat(), column=column, definition=str(body.get("definition", "")),
                           feedback=str(body["feedback"]), before=str(body.get("before", "")),
                           after=str(body.get("after", "")), reason=str(body.get("reason", "")),
                           governing=governing)
    if not out.get("is_knowledge"):
        return _ok(is_knowledge=False, why=out.get("why", ""), governing=[n.id for n in governing])
    return _ok(is_knowledge=True, proposal=out,
               governing=[{"id": n.id, "role": n.role, "scope": n.scope, "body": n.body} for n in governing])


@bp.route("/api/notes/apply", methods=["POST"])
def api_apply_note():
    """Write an edit into a note and record it. The reviewer may have rewritten the proposed text first."""
    body = request.get_json() or {}
    note_id, text = str(body.get("note", "")).strip(), str(body.get("text", "")).strip()
    if not note_id or not text:
        return _err("note and text required")
    try:
        result = notes_kb.apply_edit(note_id, text, heading=str(body.get("heading", "")) or None,
                                    by=str(body.get("by", "")), why=str(body.get("why", "")),
                                    event=str(body.get("event", "")),
                                    role=str(body.get("role", "extraction")))
    except ValueError as exc:
        return _err(str(exc))
    store.record_event("note_edit", body.get("schema_id"), by=str(body.get("by", "")), note=result["note"],
                       created=result["created"], fingerprint=result["fingerprint"])
    return _ok(**result)



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
