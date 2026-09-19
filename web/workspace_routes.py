"""The table workspace: one table (a schema) with its papers, runs, reviews and the filled-in table.

Pages
  /tables                         all tables
  /tables/<id>                    workspace: Overview, Schema, Papers, Runs, Table
  /tables/<id>/review             cell review, flagged cells first (?run=&doc=&column=)
  /knowledge  /learning  /benchmark
APIs (JSON, {"success": ...} envelope)
  GET  /api/tables                               tables with counts
  GET  /api/tables/<id>                          table, papers, runs, steps, next action
  POST /api/tables/<id>/papers                   add a paper (JSON {doc_id} or multipart file); starts a prepare job if needed
  GET  /api/tables/<id>/runs                     runs of the table with per-paper progress and review counts
  POST /api/tables/<id>/runs                     {docs, version?, system?, kb?, by} start an extraction (job)
  GET  /api/jobs                                 jobs (?table=&kind=)      POST /api/jobs/<id>/cancel     GET /api/jobs/<id>/log
  GET  /api/runs/<run>/table                     papers x fields with each cell's value and state
  GET  /api/runs/<run>/export.<csv|xlsx>         the filled-in table in the schema's column order (+ evidence sheet)
  GET  /api/runs/<run>/queue                     cells to review: flagged, then disputed, then the rest
  GET  /api/runs/<run>/docs/<doc>/cells          one paper's cells: definition, A, B, arbiter, evidence, review
  POST /api/runs/<run>/docs/<doc>/review         {column, value, reason, note, by}
  POST /api/runs/<run>/docs/<doc>/undo           {event_id, by}
  GET  /api/runs/compare?a=&b=                   cells that changed between two runs, per paper
Reviews live in the feedback log per run (services/reviews.py); nothing here reads the legacy per-paper file.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, Response, jsonify, render_template, request

from src.config import runtime_paths
from src.evisearch.schema import store
from src.evisearch.services import jobs, reviews
from src.evisearch.services import runs as runs_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]
bp = Blueprint("workspace", __name__)


def _ok(**payload: Any):
    return jsonify({"success": True, **payload}), 200


def _err(message: str, status: int = 400, **payload: Any):
    return jsonify({"success": False, "error": message, **payload}), status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- pages -------------------------------------------------------------------------------------------------------
@bp.route("/tables")
def tables_page():
    return render_template("tables.html")


@bp.route("/tables/<table_id>")
def table_page(table_id):
    return render_template("table.html", table_id=table_id)


@bp.route("/tables/<table_id>/review")
def review_page(table_id):
    return render_template("review.html", table_id=table_id)


@bp.route("/knowledge")
def knowledge_page():
    return render_template("knowledge.html")


@bp.route("/learning")
def learning_page():
    return render_template("learning.html")


@bp.route("/benchmark")
def benchmark_page():
    return render_template("benchmark.html")


# ---- definitions ---------------------------------------------------------------------------------------------------
def _hand_written_fields() -> List[Dict[str, Any]]:
    from src.config.config import HUMAN_DEFINITIONS_CSV_PATH

    out = []
    with open(HUMAN_DEFINITIONS_CSV_PATH, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Column Name"):
                out.append({"name": row["Column Name"], "group": row.get("Label") or row["Column Name"],
                            "description": row.get("Definition", ""), "eval_category": row.get("eval_category", "")})
    return out


def _schema_fields(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"name": f["name"], "group": f.get("x-evisearch", {}).get("group") or f["name"], "description": f.get("description", ""),
             "eval_category": f.get("x-evisearch", {}).get("eval_category", "")} for f in schema.get("fields", [])]


def run_fields(run: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The columns (name, group, definition) a run extracted, in table order, and where they came from."""
    info = runs_service.parse_run_name(run or "")
    if info and isinstance(info.get("version"), int):
        try:
            return _schema_fields(store.load(info["schema_id"], info["version"])), {"source": "schema", **info}
        except FileNotFoundError:
            pass
    if info:  # an unlocked draft rung (v0draft): the current schema's column list, labelled as such
        try:
            return _schema_fields(store.load(info["schema_id"])), {"source": "schema draft", **info}
        except FileNotFoundError:
            pass
    return _hand_written_fields(), {"source": "hand-written definitions"}


# ---- tables and papers ---------------------------------------------------------------------------------------------
def _project_path(table_id: str) -> Path:
    return store.schema_dir(table_id) / "project.json"


def _doc_name(doc_id: str) -> str:
    from src.documents.pdf_registry import get_registered_document

    reg = get_registered_document(doc_id, results_root=runtime_paths.RESULTS_ROOT) or {}
    return str(reg.get("display_name") or doc_id)


def _gold_docs() -> List[str]:
    try:
        from src.config.config import GOLD_TABLE_JSON_PATH

        rows = json.loads(Path(GOLD_TABLE_JSON_PATH).read_text(encoding="utf-8"))["data"]
        return [r["Document Name"]["value"].removesuffix(".pdf") for r in rows]
    except Exception:
        return []


def _prepared(doc_id: str) -> Dict[str, bool]:
    from src.evisearch.services.highlight import resolve_pdf_path
    from src.retrieval.embedding_retriever import parsed_markdown_path

    pdf = resolve_pdf_path(doc_id)
    return {"pdf": bool(pdf and Path(pdf).exists()), "parsed": parsed_markdown_path(doc_id).exists()}


def table_papers(table_id: str, schema: Dict[str, Any], table_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The table's papers: its project file, plus the example paper and every paper a run of the table covered."""
    try:
        saved = json.loads(_project_path(table_id).read_text(encoding="utf-8")).get("papers", [])
    except (OSError, ValueError):
        saved = []
    papers: Dict[str, Dict[str, Any]] = {}
    example = (schema.get("source") or {}).get("example_doc")
    if example:
        papers[example] = {"doc_id": example, "role": "example", "added_by": schema.get("created_by", ""), "added_at": schema.get("created_at")}
    for p in saved:
        papers.setdefault(p["doc_id"], p)
    for r in table_runs:
        for doc in r["docs"]:
            papers.setdefault(doc, {"doc_id": doc, "role": "target", "added_by": "", "added_at": None})
    gold = set(_gold_docs())
    out = []
    for doc, p in papers.items():
        out.append({**p, "name": _doc_name(doc), "gold": doc in gold, **_prepared(doc)})
    return out


def _save_paper(table_id: str, doc_id: str, role: str, by: str) -> Dict[str, Any]:
    path = _project_path(table_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"papers": []}
    if not any(p["doc_id"] == doc_id for p in data["papers"]):
        data["papers"].append({"doc_id": doc_id, "role": role, "added_by": by, "added_at": _now()})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
    return next(p for p in data["papers"] if p["doc_id"] == doc_id)


def _machine(doc_id: str, run: str, cells: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    cells = runs_service.final_cells(doc_id, run) if cells is None else cells
    return {c: v["value"] for c, v in cells.items()}


def run_summary(run: Dict[str, Any], with_reviews: bool = True) -> Dict[str, Any]:
    """A run with each paper's progress, flags and reviews."""
    papers, flagged, reviewed, corrected = [], 0, 0, 0
    for doc in run["docs"]:
        progress = runs_service.doc_progress(doc, run["run"])
        cells = runs_service.final_cells(doc, run["run"]) if progress["status"] != "waiting" else {}
        n_flag = sum(1 for c in cells.values() if c["flagged"])
        counts = reviews.counts(doc, run["run"], _machine(doc, run["run"], cells)) if with_reviews and cells else {"reviewed": 0, "corrected": 0}
        flagged, reviewed, corrected = flagged + n_flag, reviewed + counts["reviewed"], corrected + counts["corrected"]
        papers.append({**progress, "name": _doc_name(doc), "cells": len(cells), "flagged": n_flag, **counts})
    done = sum(1 for p in papers if p["status"] == "ok")
    status = "ok" if papers and done == len(papers) else ("failed" if any(p["status"] not in ("ok", "running", "waiting") for p in papers) and not any(p["status"] == "running" for p in papers) else "running")
    return {**run, "papers": papers, "done": done, "flagged": flagged, "reviewed": reviewed, "corrected": corrected, "status": status}


def _steps(schema: Dict[str, Any], summaries: List[Dict[str, Any]], learned: int, table_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fields = schema.get("fields", [])
    reviewed = sum(1 for f in fields if f["x-evisearch"]["review"]["state"] != "proposed")
    open_q = sum(1 for f in fields for q in f["x-evisearch"].get("questions", []) if not q.get("answer"))
    locked = schema.get("locked_versions") or []
    latest = summaries[-1] if summaries else None
    steps = [
        {"key": "draft", "label": "Schema drafted", "done": bool(fields), "detail": f"{len(fields)} columns"},
        {"key": "review", "label": "Schema reviewed", "done": bool(fields) and reviewed == len(fields) and open_q == 0,
         "detail": f"{reviewed}/{len(fields)} reviewed · {open_q} open questions"},
        {"key": "lock", "label": "Version locked", "done": bool(locked),
         "detail": f"v{max(locked)}" + (" · draft has unsaved changes" if schema.get("status") == "draft" and locked else "") if locked else "not locked"},
        {"key": "extract", "label": "Papers extracted", "done": bool(latest) and latest["status"] == "ok",
         "detail": f"{latest['run']}: {latest['done']}/{len(latest['papers'])} papers" if latest else "no run yet"},
        {"key": "cells", "label": "Flagged cells reviewed", "done": bool(latest) and latest["status"] == "ok" and latest["reviewed"] >= latest["flagged"],
         "detail": f"{latest['reviewed']} reviewed · {latest['flagged']} flagged" if latest else "—"},
        {"key": "learn", "label": "Rules learned", "done": learned > 0, "detail": f"{learned} approved from reviews"},
    ]
    targets = {"draft": ("Create the schema", f"/tables/{table_id}#schema"), "review": ("Review the schema", f"/tables/{table_id}#schema"),
               "lock": ("Lock a version", f"/tables/{table_id}#schema"), "extract": ("Start or follow a run", f"/tables/{table_id}#runs"),
               "cells": ("Review flagged cells", f"/tables/{table_id}/review" + (f"?run={latest['run']}" if latest else "")),
               "learn": ("See the knowledge base", "/knowledge")}
    first = next((s for s in steps if not s["done"]), None)
    label, href = targets[first["key"]] if first else ("Compare runs", "/learning")
    return steps, {"label": label, "href": href, "step": first["key"] if first else None}


def _learned_count() -> int:
    from src.evisearch.knowledge import conventions as kb

    return sum(1 for c in kb.load_all().values()
               if c["status"] == "approved" and (c.get("source") or {}).get("kind") in ("schema_review", "extraction_review"))


@bp.route("/api/tables")
def api_tables():
    out = []
    for s in store.list_schemas():
        table_runs = runs_service.list_runs(s["id"])
        out.append({**s, "runs": len(table_runs), "last_run": table_runs[-1]["run"] if table_runs else None,
                    "papers": len({d for r in table_runs for d in r["docs"]})})
    return _ok(tables=out)


@bp.route("/api/tables/<table_id>")
def api_table(table_id):
    try:
        schema = store.load(table_id)
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    table_runs = runs_service.list_runs(table_id)
    summaries = [run_summary(r) for r in table_runs]
    steps, next_action = _steps(schema, summaries, _learned_count(), table_id)
    info = {k: schema.get(k) for k in ("id", "name", "status", "version", "locked_versions", "created_at", "created_by", "locked_at")}
    info["description"] = (schema.get("source") or {}).get("description", "")
    info["example_doc"] = (schema.get("source") or {}).get("example_doc")
    info["fields"] = len(schema.get("fields", []))
    return _ok(table=info, papers=table_papers(table_id, schema, table_runs), runs=summaries, steps=steps, next_action=next_action)


@bp.route("/api/tables/<table_id>/papers", methods=["POST"])
def api_add_paper(table_id):
    try:
        store.load(table_id)
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    by = str((request.form if request.files else (request.get_json(silent=True) or {})).get("by", ""))
    if request.files:
        pdf = request.files.get("file")
        if pdf is None or not pdf.filename.lower().endswith(".pdf"):
            return _err("a PDF file is required")
        from src.documents.pdf_registry import register_uploaded_pdf

        doc_id = register_uploaded_pdf(pdf.read(), original_filename=pdf.filename, uploads_dir=runtime_paths.UPLOADS_DIR,
                                       results_root=runtime_paths.RESULTS_ROOT, dataset_dir=runtime_paths.DATASET_DIR)["canonical_doc_id"]
    else:
        doc_id = str((request.get_json(silent=True) or {}).get("doc_id", "")).strip()
        if not doc_id:
            return _err("doc_id or a PDF required")
    try:
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    paper = _save_paper(table_id, doc_id, "target", by)
    state = _prepared(doc_id)
    job = None
    if not state["parsed"]:
        job = start_prepare(doc_id, by=by, table=table_id)
    return _ok(paper={**paper, "name": _doc_name(doc_id), **state}, job=job)


def start_prepare(doc_id: str, by: str = "", table: Optional[str] = None) -> Dict[str, Any]:
    """Parse the PDF (LandingAI) and build page embeddings, as a job; the same steps as the Ask page's Prepare."""
    existing = [j for j in jobs.list_jobs(kind="prepare", doc_id=doc_id) if j["status"] in ("queued", "running")]
    if existing:
        return existing[0]

    def work():
        from src.evisearch.services.highlight import resolve_pdf_path
        from src.evisearch.services.preparation import parse_pdf_for_qa
        from src.retrieval.embedding_retriever import embed_chunks, has_embedding_cache, parsed_markdown_path

        steps = []
        if not parsed_markdown_path(doc_id).exists():
            pdf = resolve_pdf_path(doc_id)
            if not pdf or not Path(pdf).exists():
                raise RuntimeError(f"PDF not found for {doc_id}")
            result = parse_pdf_for_qa(doc_id, Path(pdf), on_event=lambda e: None)
            if not result.get("success"):
                raise RuntimeError(result.get("error", "parse failed"))
            steps.append("parsed")
        if not has_embedding_cache(doc_id):
            if not embed_chunks(doc_id, force=False):
                raise RuntimeError("no parsed content to embed")
            steps.append("embedded")
        return {"doc_id": doc_id, "steps": steps}

    return jobs.run_thread("prepare", work, by=by, doc_id=doc_id, table=table)


@bp.route("/api/tables/<table_id>/runs", methods=["GET"])
def api_table_runs(table_id):
    return _ok(runs=[run_summary(r) for r in runs_service.list_runs(table_id)],
               jobs=jobs.list_jobs(table=table_id, kind="extract")[:20])


def unique_run_name(base: str) -> str:
    taken = {r["run"] for r in runs_service.list_runs()} | {j.get("run") for j in jobs.list_jobs(kind="extract")}
    if base not in taken and not (runs_service.headers_dir() / f"{base}.json").exists():
        return base
    k = 2
    while f"{base}-r{k}" in taken or (runs_service.headers_dir() / f"{base}-r{k}.json").exists():
        k += 1
    return f"{base}-r{k}"


def start_extraction(table_id: str, docs: List[str], *, version: Optional[int] = None, system: str = "E", kb: str = "on",
                     by: str = "", preset: Optional[str] = None) -> Dict[str, Any]:
    schema = store.load(table_id)
    locked = schema.get("locked_versions") or []
    if not locked:
        raise ValueError("lock the schema before extracting")
    version = int(version or max(locked))
    if version not in locked:
        raise ValueError(f"v{version} is not a locked version (locked: {locked})")
    if system not in ("E", "B1", "B2") or kb not in ("on", "off"):
        raise ValueError("system must be E, B1 or B2 and kb on or off")
    for doc in docs:
        runs_service.check_doc(doc)
    running = [j for j in jobs.list_jobs(kind="extract") if j["status"] == "running"]
    if running:
        raise RuntimeError(f"run {running[0].get('run')} is still running; one extraction at a time")
    base = store.run_name(table_id, version) + ("" if kb == "on" else "-kboff") + ("" if system == "E" else f"-{system.lower()}")
    run = unique_run_name(base)
    log = runs_service.headers_dir() / f"{run}.driver.log"
    cmd = [sys.executable, str(PROJECT_ROOT / "experiment-scripts" / "run_schema.py"), "--schema", table_id, "--version", str(version),
           "--system", system, "--docs", ",".join(docs), "--kb", kb, "--run", run]
    env = {"EVISEARCH_PRESET": preset} if preset else None
    return jobs.run_process("extract", cmd, log, by=by, env=env, table=table_id, run=run, docs=docs, version=version,
                            system=system, kb=kb)


@bp.route("/api/tables/<table_id>/runs", methods=["POST"])
def api_start_run(table_id):
    body = request.get_json(silent=True) or {}
    docs = body.get("docs") or []
    if isinstance(docs, str):
        docs = [d for d in docs.split(",") if d.strip()]
    if not docs:
        return _err("pick at least one paper")
    try:
        job = start_extraction(table_id, [str(d).strip() for d in docs], version=body.get("version"), system=str(body.get("system", "E")),
                               kb=str(body.get("kb", "on")), by=str(body.get("by", "")))
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    except RuntimeError as exc:
        return _err(str(exc), 409)
    except ValueError as exc:
        return _err(str(exc))
    return _ok(job=job, run=job["run"])


# ---- jobs ----------------------------------------------------------------------------------------------------------
@bp.route("/api/jobs")
def api_jobs():
    return _ok(jobs=jobs.list_jobs(table=request.args.get("table") or None, kind=request.args.get("kind") or None)[:50])


@bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel_job(job_id):
    try:
        return _ok(job=jobs.cancel(job_id, by=str((request.get_json(silent=True) or {}).get("by", ""))))
    except KeyError:
        return _err("unknown job", 404)
    except ValueError as exc:
        return _err(str(exc))


@bp.route("/api/jobs/<job_id>/log")
def api_job_log(job_id):
    try:
        job = jobs.get(job_id)
    except ValueError as exc:
        return _err(str(exc))
    if not job:
        return _err("unknown job", 404)
    return _ok(log=jobs.log_tail(job, request.args.get("lines", 80, type=int)), status=job["status"])


# ---- runs: the filled-in table, the review queue, one paper's cells --------------------------------------------------
def _run_or_404(run: str) -> Optional[Dict[str, Any]]:
    runs_service.check_name(run)
    return next((r for r in runs_service.list_runs() if r["run"] == run), None)


def cell_state(cell: Dict[str, Any], review: Optional[Dict[str, Any]]) -> str:
    if review and review.get("value") is not None:
        return {"accepted": "accepted", "corrected": "corrected", "not_reported": "corrected"}.get(review.get("state", ""), "corrected")
    if cell["flagged"]:
        return "flagged"
    if runs_service.is_not_reported(cell["value"]):
        return "not_reported"
    if cell.get("verified") is True:
        return "verified"
    return "unverified"


def _grid(run: Dict[str, Any]) -> Dict[str, Any]:
    fields, source = run_fields(run["run"])
    per_doc = {doc: runs_service.final_cells(doc, run["run"]) for doc in run["docs"]}
    known = {f["name"] for f in fields}
    extra = [c for cells in per_doc.values() for c in cells if c not in known]
    fields = fields + [{"name": c, "group": "Other", "description": ""} for c in dict.fromkeys(extra)]  # columns the definitions lack
    docs = []
    for doc in run["docs"]:
        cells = per_doc[doc]
        if not cells:
            continue
        revs = reviews.cell_reviews(doc, run["run"], _machine(doc, run["run"], cells))
        row = {}
        for f in fields:
            cell = cells.get(f["name"])
            if cell is None:
                continue
            rv = revs.get(f["name"])
            value = rv["value"] if rv and rv.get("value") is not None else cell["value"]
            ev = cell["evidence"][0] if cell["evidence"] else None
            row[f["name"]] = {"v": value, "s": cell_state(cell, rv), "p": ev["page"] if ev else None,
                              "q": (ev["quote"][:240] if ev else ""), "m": cell["value"] if rv and rv.get("value") is not None else None}
        docs.append({"doc_id": doc, "name": _doc_name(doc), "cells": row})
    return {"run": run["run"], "info": {k: run.get(k) for k in ("schema_id", "version", "variant", "system", "preset", "models", "started_at")},
            "definitions": source, "columns": [{"name": f["name"], "group": f["group"], "definition": f["description"]} for f in fields],
            "docs": docs}


@bp.route("/api/runs/<run>/table")
def api_run_table(run):
    try:
        found = _run_or_404(run)
    except ValueError as exc:
        return _err(str(exc))
    if not found:
        return _err(f"no run {run!r}", 404)
    return _ok(**_grid(found))


@bp.route("/api/runs/<run>/export.<fmt>")
def api_run_export(run, fmt):
    try:
        found = _run_or_404(run)
    except ValueError as exc:
        return _err(str(exc))
    if not found:
        return _err(f"no run {run!r}", 404)
    grid = _grid(found)
    columns = [c["name"] for c in grid["columns"]]
    values = [[d["name"]] + [(d["cells"].get(c) or {}).get("v", "") for c in columns] for d in grid["docs"]]
    evidence = [[d["name"], c, (d["cells"].get(c) or {}).get("v", ""), (d["cells"].get(c) or {}).get("s", ""),
                 (d["cells"].get(c) or {}).get("p") or "", (d["cells"].get(c) or {}).get("q", "")]
                for d in grid["docs"] for c in columns if c in d["cells"]]
    head_values = ["Document"] + columns
    head_evidence = ["Document", "Column", "Value", "State", "Page", "Quote"]
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(head_values)
        writer.writerows(values)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{run}.csv"'})
    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError:
            return _err("openpyxl is not installed; download CSV instead", 501)
        wb = Workbook()
        ws = wb.active
        ws.title = "Table"
        ws.append(head_values)
        for row in values:
            ws.append(row)
        ev = wb.create_sheet("Evidence")
        ev.append(head_evidence)
        for row in evidence:
            ev.append(row)
        buf = io.BytesIO()
        wb.save(buf)
        return Response(buf.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{run}.xlsx"'})
    return _err("format must be csv or xlsx", 404)


@bp.route("/api/runs/<run>/queue")
def api_run_queue(run):
    try:
        found = _run_or_404(run)
    except ValueError as exc:
        return _err(str(exc))
    if not found:
        return _err(f"no run {run!r}", 404)
    fields, _ = run_fields(run)
    order = {f["name"]: i for i, f in enumerate(fields)}
    group = {f["name"]: f["group"] for f in fields}
    items = []
    for doc in found["docs"]:
        cells = runs_service.final_cells(doc, run)
        revs = reviews.cell_reviews(doc, run, _machine(doc, run, cells))
        for column, cell in cells.items():
            rv = revs.get(column)
            reviewed = bool(rv and rv.get("value") is not None)
            kind = "flagged" if cell["flagged"] else ("disputed" if runs_service.disputed(cell) else "other")
            items.append({"doc_id": doc, "column": column, "group": group.get(column, ""), "kind": kind,
                          "value": rv["value"] if reviewed else cell["value"], "reviewed": reviewed,
                          "state": cell_state(cell, rv), "order": order.get(column, 10_000)})
    rank = {"flagged": 0, "disputed": 1, "other": 2}
    docs_order = {d: i for i, d in enumerate(found["docs"])}
    items.sort(key=lambda it: (rank[it["kind"]], docs_order[it["doc_id"]], it["order"]))
    counts = {k: {"total": sum(1 for it in items if it["kind"] == k), "reviewed": sum(1 for it in items if it["kind"] == k and it["reviewed"])}
              for k in rank}
    return _ok(run=run, docs=[{"doc_id": d, "name": _doc_name(d)} for d in found["docs"]], items=items, counts=counts)


@bp.route("/api/runs/<run>/docs/<path:doc_id>/cells")
def api_doc_cells(run, doc_id):
    try:
        found = _run_or_404(run)
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    if not found or doc_id not in found["docs"]:
        return _err(f"{doc_id} is not in run {run}", 404)
    fields, source = run_fields(run)
    cells = runs_service.final_cells(doc_id, run)
    revs = reviews.cell_reviews(doc_id, run, _machine(doc_id, run, cells))
    out = []
    known = set()
    for f in fields + [{"name": c, "group": "", "description": ""} for c in cells if c not in {f["name"] for f in fields}]:
        if f["name"] in known or f["name"] not in cells:
            continue
        known.add(f["name"])
        cell = cells[f["name"]]
        out.append({"column": f["name"], "group": f["group"], "definition": f["description"], **cell,
                    "disputed": runs_service.disputed(cell), "review": revs.get(f["name"]),
                    "state": cell_state(cell, revs.get(f["name"]))})
    return _ok(run=run, doc_id=doc_id, name=_doc_name(doc_id), definitions=source, schema_id=found.get("schema_id"),
               columns=out, reasons=list(reviews.REASONS))


@bp.route("/api/runs/<run>/docs/<path:doc_id>/review", methods=["POST"])
def api_review_cell(run, doc_id):
    body = request.get_json(silent=True) or {}
    try:
        found = _run_or_404(run)
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    if not found or doc_id not in found["docs"]:
        return _err(f"{doc_id} is not in run {run}", 404)
    column = str(body.get("column", ""))
    cells = runs_service.final_cells(doc_id, run)
    if column not in cells:
        return _err(f"column {column!r} is not in this run's output", 404)
    by = str(body.get("by", "")).strip()
    if not by:
        return _err("enter your name (top right) before saving reviews")
    try:
        event = reviews.record(doc_id, run, column, body.get("value", ""), machine_value=cells[column]["value"],
                               reason=str(body.get("reason", "")), note=str(body.get("note", "")), by=by,
                               schema_id=found.get("schema_id"))
    except ValueError as exc:
        return _err(str(exc))
    review = reviews.cell_reviews(doc_id, run, _machine(doc_id, run, cells)).get(column)
    return _ok(event=event, review=review, state=cell_state(cells[column], review))


@bp.route("/api/runs/<run>/docs/<path:doc_id>/undo", methods=["POST"])
def api_undo_review(run, doc_id):
    body = request.get_json(silent=True) or {}
    try:
        runs_service.check_name(run)
        runs_service.check_doc(doc_id)
        event = reviews.undo(doc_id, run, str(body.get("event_id", "")), by=str(body.get("by", "")))
    except ValueError as exc:
        return _err(str(exc))
    except KeyError as exc:
        return _err(str(exc), 404)
    cells = runs_service.final_cells(doc_id, run)
    column = event.get("column")
    review = reviews.cell_reviews(doc_id, run, _machine(doc_id, run, cells)).get(column)
    cell = cells.get(column)
    return _ok(event=event, review=review, state=cell_state(cell, review) if cell else None)


@bp.route("/api/runs/compare")
def api_compare_runs():
    a, b = request.args.get("a", ""), request.args.get("b", "")
    try:
        ra, rb = _run_or_404(a), _run_or_404(b)
    except ValueError as exc:
        return _err(str(exc))
    if not ra or not rb:
        return _err("both runs must exist", 404)
    norm = lambda v: " ".join(str(v or "").lower().split())
    out, totals = [], {"changed": 0, "same": 0, "papers": 0}
    for doc in [d for d in ra["docs"] if d in rb["docs"]]:
        ca, cb = runs_service.final_cells(doc, a), runs_service.final_cells(doc, b)
        if not ca or not cb:
            continue
        rva = reviews.cell_reviews(doc, a, _machine(doc, a, ca))
        changed = []
        for column in [c for c in ca if c in cb]:
            if norm(ca[column]["value"]) == norm(cb[column]["value"]):
                totals["same"] += 1
                continue
            rv = rva.get(column)
            changed.append({"column": column, "a": ca[column]["value"], "b": cb[column]["value"],
                            "a_flagged": ca[column]["flagged"], "b_flagged": cb[column]["flagged"],
                            "reviewed_in_a": rv.get("value") if rv else None,
                            "b_matches_review": bool(rv and rv.get("value") is not None and norm(rv["value"]) == norm(cb[column]["value"]))})
        totals["changed"] += len(changed)
        totals["papers"] += 1
        out.append({"doc_id": doc, "name": _doc_name(doc), "changed": changed})
    totals["fixed_by_review"] = sum(1 for d in out for c in d["changed"] if c["b_matches_review"])
    return _ok(a=a, b=b, docs=out, totals=totals)
