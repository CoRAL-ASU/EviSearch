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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional, Tuple

import os

from flask import Blueprint, Response, jsonify, redirect, render_template, request

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


@bp.route("/review")
def review_redirect():
    """The Review tab: the demo table's showcase run (or the table list when there is none yet)."""
    table = demo_table_id()
    if not table:
        return redirect("/tables")
    run = showcase_run(table)
    return redirect(f"/tables/{table}/review" + (f"?run={run}" if run else ""))


def _demo_link(suffix: str = "") -> str:
    table = demo_table_id()
    return f"/tables/{table}{suffix}" if table else "/tables"


@bp.route("/schema")
def schema_redirect():
    """The Schema page is now a tab of the table workspace."""
    return redirect(_demo_link("#schema"))


@bp.route("/attribution")
@bp.route("/verify")
@bp.route("/comparison")
def attribution_redirect():
    """The old Verify page: same job, now per run, with a queue and the page beside the cell."""
    table, run = demo_table_id(), (request.args.get("run") or "").strip()
    if not table:
        return redirect("/tables")
    if run:
        info = runs_service.parse_run_name(run)
        table = info.get("schema_id") or table
    else:
        run = showcase_run(table) or ""
    args = {k: v for k, v in (("run", run), ("doc", request.args.get("doc")), ("column", request.args.get("column"))) if v}
    return redirect(f"/tables/{table}/review" + ("?" + urlencode(args) if args else ""))


@bp.route("/comparison-report")
def report_redirect():
    """The old Report page: now the Table tab of a run."""
    return redirect(_demo_link("#table"))


@bp.route("/extract")
def extract_redirect():
    """Extraction now runs from a table's Runs tab, under a locked schema version."""
    return redirect(_demo_link("#runs"))


@bp.route("/method-comparison-report")
def method_report_redirect():
    return redirect("/benchmark")


@bp.route("/feedback")
def feedback_redirect():
    """The old Feedback page: the activity log and the rules now live on Learning and Knowledge."""
    return redirect("/learning")


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


def active_runs() -> set:
    """Run names with a job still going."""
    return {j.get("run") for j in jobs.list_jobs(kind="extract") if j["status"] in ("queued", "running")}


def run_summary(run: Dict[str, Any], with_reviews: bool = True, active: Optional[set] = None) -> Dict[str, Any]:
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
    if papers and done == len(papers):
        status = "ok"
    elif any(p["status"] not in ("ok", "running", "waiting") for p in papers):
        status = "failed"
    elif run["run"] in (active if active is not None else active_runs()) or _recently_written(run["run"]):
        status = "running"
    else:  # nothing written for a while and no job: the run stopped before finishing (e.g. cancelled or interrupted)
        status = "incomplete"
    return {**run, "papers": papers, "done": done, "flagged": flagged, "reviewed": reviewed, "corrected": corrected, "status": status}


def _recently_written(run: str, minutes: int = 30) -> bool:
    """Whether anything was written into the run lately (directory timestamps only, so this stays cheap)."""
    newest = 0.0
    for doc_dir in runtime_paths.RESULTS_ROOT.glob(f"*/runs/{run}"):
        for path in [doc_dir, *doc_dir.iterdir()]:
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    return bool(newest) and (time.time() - newest) < minutes * 60


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


def demo_table_id() -> Optional[str]:
    """EVISEARCH_DEMO_TABLE, else the table with the most runs."""
    wanted = os.getenv("EVISEARCH_DEMO_TABLE", "").strip()
    tables = store.list_schemas()
    if wanted and any(t["id"] == wanted for t in tables):
        return wanted
    if not tables:
        return None
    return max(tables, key=lambda t: (len(runs_service.list_runs(t["id"])), t.get("created_at") or ""))["id"]


def showcase_run(table_id: str) -> Optional[str]:
    """EVISEARCH_DEMO_RUN, else the table's latest main run (no variant suffix) whose papers all finished."""
    wanted = os.getenv("EVISEARCH_DEMO_RUN", "").strip()
    table_runs = runs_service.list_runs(table_id)
    if wanted and any(r["run"] == wanted for r in table_runs):
        return wanted
    finished = [r for r in table_runs if all(runs_service.doc_progress(d, r["run"])["status"] == "ok" for d in r["docs"])]
    main = [r for r in finished if not r.get("variant")] or finished
    return main[-1]["run"] if main else (table_runs[-1]["run"] if table_runs else None)


@bp.route("/api/stats")
def api_stats():
    """Live counts for the home page, computed from the stores."""
    from src.evisearch.knowledge import conventions as kb

    table = demo_table_id()
    run = showcase_run(table) if table else None
    cells = with_page = flagged = reviewed = 0
    papers: List[str] = []
    if run:
        found = next(r for r in runs_service.list_runs(table) if r["run"] == run)
        papers = found["docs"]
        for doc in papers:
            doc_cells = runs_service.final_cells(doc, run)
            reported = [c for c in doc_cells.values() if not runs_service.is_not_reported(c["value"])]
            cells += len(reported)
            with_page += sum(1 for c in reported if c["evidence"])
            flagged += sum(1 for c in doc_cells.values() if c["flagged"])
            reviewed += reviews.counts(doc, run, _machine(doc, run, doc_cells))["reviewed"]
    rules = kb.load_all().values()
    return _ok(demo_table=table, showcase_run=run, papers=len(papers), reported_values=cells, with_page=with_page,
               flagged=flagged, reviewed=reviewed, tables=len(store.list_schemas()),
               rules_learned=sum(1 for c in rules if c["status"] == "approved" and (c.get("source") or {}).get("kind") in ("schema_review", "extraction_review")),
               rules_total=sum(1 for c in rules if c["status"] == "approved"))


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
    active = active_runs()
    summaries = [run_summary(r, active=active) for r in table_runs]
    main = showcase_run(table_id)  # the run the steps describe: the latest finished main run, not a partial variant
    ordered = [s for s in summaries if s["run"] != main] + [s for s in summaries if s["run"] == main]
    steps, next_action = _steps(schema, ordered, _learned_count(), table_id)
    info = {k: schema.get(k) for k in ("id", "name", "status", "version", "locked_versions", "created_at", "created_by", "locked_at")}
    info["description"] = (schema.get("source") or {}).get("description", "")
    info["example_doc"] = (schema.get("source") or {}).get("example_doc")
    info["fields"] = len(schema.get("fields", []))
    return _ok(table=info, papers=table_papers(table_id, schema, table_runs), runs=summaries, steps=steps, next_action=next_action,
               showcase_run=main)


@bp.route("/api/tables/<table_id>/versions/<int:version>.<fmt>")
def api_version_download(table_id, version, fmt):
    """A locked version's definitions: the CSV the pipelines read, or a spreadsheet with one row per column."""
    try:
        schema = store.load(table_id, version)
    except FileNotFoundError as exc:
        return _err(str(exc), 404)
    fields = _schema_fields(schema)
    if fmt == "csv":
        path = store.schema_dir(table_id) / "versions" / f"v{version}.csv"
        if not path.exists():
            return _err("CSV not found", 404)
        return Response(path.read_text(encoding="utf-8"), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{table_id}-v{version}.csv"'})
    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError:
            return _err("openpyxl is not installed; download CSV instead", 501)
        wb = Workbook()
        ws = wb.active
        ws.title = f"v{version}"
        ws.append(["Column", "Group", "Definition", "Scoring"])
        for f in fields:
            ws.append([f["name"], f["group"], f["description"], f["eval_category"]])
        buf = io.BytesIO()
        wb.save(buf)
        return Response(buf.getvalue(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{table_id}-v{version}.xlsx"'})
    return _err("format must be csv or xlsx", 404)


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


@bp.route("/api/documents/<path:doc_id>/page/<int:page>.png")
def api_page_image(doc_id, page):
    """One rendered page of the paper. The Review page shows this instead of a PDF viewer in the browser: it needs no
    plugin, works for every paper, and reuses the renderer (and cache) the extraction agents already use."""
    from src.evisearch.services.highlight import resolve_pdf_path
    from src.evisearch.services.page_images import render_pages

    try:
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    pdf = resolve_pdf_path(doc_id)
    if not pdf or not Path(pdf).exists():
        return _err(f"no PDF for {doc_id}", 404)
    scale = min(max(request.args.get("scale", 1.6, type=float), 0.5), 3.0)
    images = render_pages(Path(pdf), [page], scale=scale)
    if not images:
        return _err(f"page {page} is not in this paper", 404)
    return Response(images[0].png, mimetype="image/png", headers={"Cache-Control": "private, max-age=86400"})


@bp.route("/api/documents/<path:doc_id>/page/<int:page>/find")
def api_page_find(doc_id, page):
    """Where a quote sits on a page: {pages, width, height, rects:[[x0,y0,x1,y1]…]} in the rendered image's pixels, so
    the page can be shown with the evidence highlighted. Falls back to the longest matching line when the whole quote
    is not found (parsed text and the PDF's own text differ in spacing and hyphenation)."""
    from src.evisearch.services.highlight import resolve_pdf_path

    try:
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    pdf = resolve_pdf_path(doc_id)
    if not pdf or not Path(pdf).exists():
        return _err(f"no PDF for {doc_id}", 404)
    quote = (request.args.get("q") or "").strip()
    scale = min(max(request.args.get("scale", 1.6, type=float), 0.5), 3.0)
    import fitz

    with fitz.open(str(pdf)) as doc:
        if not 1 <= page <= len(doc):
            return _err(f"page {page} is not in this paper", 404)
        target = doc[page - 1]
        rects: List[List[float]] = []
        if quote:
            for probe in _probes(quote):
                found = target.search_for(probe, quads=False)
                if found:
                    rects = [[r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale] for r in found]
                    break
            if not rects:  # the PDF's own text differs (tables, ligatures, line breaks): match word by word
                rects = _best_word_span(target, quote, scale)
        return _ok(page=page, pages=len(doc), width=target.rect.width * scale, height=target.rect.height * scale,
                   rects=rects, found=bool(rects))


def _best_word_span(page, quote: str, scale: float) -> List[List[float]]:
    """The run of words on the page that shares most words with the quote (at least 60% of them), as line rectangles."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    if not words:
        return []
    clean = lambda s: re.sub(r"[^a-z0-9.%]+", "", s.lower())
    wanted = [clean(w) for w in quote.split()]
    wanted = [w for w in wanted if len(w) > 1]
    if len(wanted) < 3:
        return []
    page_words = [clean(w[4]) for w in words]
    size = min(len(wanted), 60)
    target = set(wanted)
    best, best_at = 0, -1
    for start in range(0, max(1, len(page_words) - size + 1)):
        score = sum(1 for w in page_words[start:start + size] if w in target)
        if score > best:
            best, best_at = score, start
    if best_at < 0 or best < max(3, int(0.6 * min(size, len(wanted)))):
        return []
    chosen = words[best_at:best_at + size]
    lines: Dict[Any, List[Any]] = {}
    for w in chosen:
        lines.setdefault((w[5], w[6]), []).append(w)
    out = []
    for group in lines.values():
        x0 = min(w[0] for w in group) * scale
        y0 = min(w[1] for w in group) * scale
        x1 = max(w[2] for w in group) * scale
        y1 = max(w[3] for w in group) * scale
        out.append([x0, y0, x1, y1])
    return out


def _probes(quote: str) -> List[str]:
    """The quote, then its longest lines and sentences: PDF text often differs from the parsed text in line breaks."""
    text = " ".join(quote.split())
    parts = [text]
    for piece in sorted(re.split(r"(?<=[.;:])\s+|\n", quote), key=len, reverse=True):
        piece = " ".join(piece.split())
        if len(piece) >= 12:
            parts.append(piece)
    parts += [text[:120], text[:60], text[:30]]
    return [p for p in dict.fromkeys(parts) if len(p) >= 8]


@bp.route("/api/documents/prepare", methods=["POST"])
def api_prepare_document():
    """{doc_id, by}: parse + embed a paper as a job (the new-table wizard parses its example paper this way)."""
    body = request.get_json(silent=True) or {}
    doc_id = str(body.get("doc_id", "")).strip()
    try:
        runs_service.check_doc(doc_id)
    except ValueError as exc:
        return _err(str(exc))
    return _ok(job=start_prepare(doc_id, by=str(body.get("by", ""))))


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
    active = active_runs()
    return _ok(runs=[run_summary(r, active=active) for r in runs_service.list_runs(table_id)],
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


# ---- knowledge, learning, benchmark ---------------------------------------------------------------------------------
def run_kb_fingerprint(run: str, docs: List[str]) -> Optional[str]:
    """The knowledge-base snapshot a run used, as recorded in its stage metadata ("kb:<fingerprint>")."""
    for doc in docs:
        for folder in ("reconciliation_agent", "agent_extractor", "markdown_baseline"):
            meta = runs_service._read_json(runs_service.base_dir(doc, run) / folder / "extraction_metadata.json")
            rules = (meta or {}).get("extraction_rules") if isinstance(meta, dict) else None
            if isinstance(rules, str) and rules.startswith("kb:"):
                return rules[3:]
    return None


def _snapshot_ids(fingerprint: str) -> List[str]:
    data = runs_service._read_json(runtime_paths.KNOWLEDGE_DIR / "snapshots" / f"{fingerprint}.json")
    return [c["id"] for c in data] if isinstance(data, list) else []


@bp.route("/api/knowledge")
def api_knowledge():
    """Every rule with where it came from and which runs used it."""
    from src.evisearch.knowledge import conventions as kb
    from src.evisearch.services import feedback

    rules = list(kb.load_all().values())
    events = {e["event_id"]: e for e in feedback.all_events()}
    used: Dict[str, List[str]] = {}
    for run in runs_service.list_runs():
        fp = run_kb_fingerprint(run["run"], run["docs"])
        if not fp:
            continue
        for cid in _snapshot_ids(fp):
            used.setdefault(cid, []).append(run["run"])
    out = []
    for rule in rules:
        source = rule.get("source") or {}
        event = events.get(str(source.get("event_id") or source.get("feedback") or ""))
        out.append({**rule, "used_in_runs": sorted(used.get(rule["id"], [])),
                    "learned": source.get("kind") != "seed",
                    "source_event": {"doc_id": event.get("doc_id"), "column": event.get("column"), "run": event.get("run"),
                                     "before": event.get("before"), "after": event.get("after"),
                                     "reason": event.get("reason")} if event else None})
    return _ok(conventions=out, fingerprint=kb.fingerprint(), action_types=sorted(kb.ACTION_TYPES), scopes=list(kb.SCOPES))


@bp.route("/api/learning")
def api_learning():
    """What the table has learned and whether later runs changed: rules and reviews over time, and per-run counts."""
    from src.evisearch.knowledge import conventions as kb
    from src.evisearch.services import feedback

    table = request.args.get("table") or demo_table_id()
    if not table:
        return _ok(table=None, runs=[], series=[], effort={}, events=[])
    table_runs = runs_service.list_runs(table)
    active = active_runs()
    rows = []
    for run in table_runs:
        s = run_summary(run, active=active)
        fp = run_kb_fingerprint(run["run"], run["docs"])
        ids = _snapshot_ids(fp) if fp else []
        rows.append({"run": run["run"], "version": run.get("version"), "variant": run.get("variant"), "started_at": run.get("started_at"),
                     "papers": len(s["papers"]), "done": s["done"], "flagged": s["flagged"], "reviewed": s["reviewed"],
                     "corrected": s["corrected"], "status": s["status"], "kb": len(ids),
                     "kb_learned": sum(1 for cid in ids if cid not in _SEED_IDS()), "cells": sum(p["cells"] for p in s["papers"])})
    # cumulative series over time: rules approved, and cells reviewed
    rules_at = [c["history"][-1]["at"] for c in kb.load_all().values()
                if c["status"] == "approved" and (c.get("source") or {}).get("kind") != "seed" and c.get("history")]
    reviews_at = [e["timestamp"] for e in feedback.all_events()
                  if e.get("event") == "cell_correct" and (not table or e.get("schema_id") in (None, table))]
    # by the hour: this work happens over days, and a per-day line would be a single point
    bucket = lambda t: t[:13] + ":00Z"
    stamps = sorted({bucket(t) for t in rules_at} | {bucket(t) for t in reviews_at})
    series, rules_n, reviews_n = [], 0, 0
    for at in stamps:
        rules_n += sum(1 for t in rules_at if bucket(t) == at)
        reviews_n += sum(1 for t in reviews_at if bucket(t) == at)
        series.append({"at": at, "rules": rules_n, "reviews": reviews_n})
    counts: Dict[str, int] = {}
    for e in feedback.all_events():
        if e.get("schema_id") in (None, table):
            counts[str(e.get("event"))] = counts.get(str(e.get("event")), 0) + 1
    effort = {"definitions_accepted": counts.get("definition_accept", 0), "definitions_edited": counts.get("definition_edit", 0),
              "questions_answered": counts.get("definition_answer", 0), "cells_reviewed": counts.get("cell_correct", 0),
              "rules_proposed": counts.get("convention_propose", 0), "rules_decided": counts.get("convention_decide", 0)}
    return _ok(table=table, runs=rows, series=series, effort=effort)


def _SEED_IDS() -> set:
    from src.evisearch.knowledge import conventions as kb

    return {c["id"] for c in kb.load_all().values() if (c.get("source") or {}).get("kind") == "seed"}


@bp.route("/api/benchmark")
def api_benchmark():
    """The expert gold table released with the paper: its papers (with DOIs), its values, and how it was made."""
    from src.config.config import GOLD_TABLE_JSON_PATH

    data = runs_service._read_json(Path(GOLD_TABLE_JSON_PATH))
    rows = (data or {}).get("data") or []
    papers, values = [], []
    dois = _doi_index([r["Document Name"]["value"].removesuffix(".pdf") for r in rows])
    for row in rows:
        doc = row["Document Name"]["value"].removesuffix(".pdf")
        get = lambda key: str((row.get(key) or {}).get("value") or "")
        papers.append({"doc_id": doc, "name": _doc_name(doc), "nct": get("NCT"), "trial": get("Trial Name"), "author": get("Author"),
                       "year": get("Year"), "pmid": get("PubMed ID"), "doi": dois.get(doc, ""), **_prepared(doc)})
        values.append({"doc_id": doc, "cells": {k: {"v": str((v or {}).get("value") or ""), "loc": str((v or {}).get("location") or "")}
                                                for k, v in row.items() if k != "Document Name"}})
    columns = [c for c in (rows[0].keys() if rows else []) if c != "Document Name"]
    # "values" = cells the experts filled with a real value; "Not reported" and blanks are not values
    filled = sum(1 for row in values for c in row["cells"].values() if not runs_service.is_not_reported(c["v"]))
    return _ok(papers=papers, columns=columns, rows=values, filled=filled, cells=len(papers) * len(columns),
               license="Released for academic, non-commercial use with the paper.",
               protocol="Built by the study's authors: domain experts annotated every trial by hand, one row per paper. "
                        "Gold values are authoritative; cells the team disputes are listed in the paper's appendix.")


@bp.route("/api/benchmark/download.<fmt>")
def api_benchmark_download(fmt):
    """The gold table as released: CSV (one row per paper) or the annotated JSON with each value's location."""
    from src.config.config import GOLD_TABLE_JSON_PATH

    raw = Path(GOLD_TABLE_JSON_PATH).read_text(encoding="utf-8")
    if fmt == "json":
        return Response(raw, mimetype="application/json",
                        headers={"Content-Disposition": 'attachment; filename="evisearch-gold-table.json"'})
    if fmt == "csv":
        rows = json.loads(raw)["data"]
        columns = list(rows[0].keys()) if rows else []
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([str((row.get(c) or {}).get("value") or "") for c in columns])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="evisearch-gold-table.csv"'})
    return _err("format must be csv or json", 404)


_DOI_RE = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;:()/+-]*[A-Za-z0-9)]")
# "…2119115Copyright©2022" and "…75.3657DOI:" -> cut where the DOI runs into the words printed after it
_RUNS_INTO_TEXT = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z]{2,})|(?<=[0-9])(?=[A-Z]{2,})")


def _clean_doi(candidate: str) -> str:
    """Trim a DOI that ran into the words printed after it, and balance trailing brackets."""
    doi = _RUNS_INTO_TEXT.split(candidate, maxsplit=1)[0]
    while doi and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi.rstrip(".,;:")


def _doi_from_pdf(pdf: Optional[Path]) -> str:
    """The paper's DOI, from a doi.org link or a printed DOI on its first pages. Journals break DOIs across lines and
    add trailing punctuation, so the text is joined up before matching."""
    if not pdf or not Path(pdf).exists():
        return ""
    import fitz

    try:
        with fitz.open(str(pdf)) as handle:
            pages = [handle[p].get_text() for p in range(min(3, len(handle)))]
            links = [l.get("uri", "") for p in range(min(3, len(handle))) for l in handle[p].get_links() if l.get("uri")]
    except Exception:
        return ""
    text = "\n".join(pages)
    candidates: List[Tuple[int, str]] = []
    for start in (m.start() for m in re.finditer(r"10\.\d{4,9}/", text)):
        tail = text[start:start + 160]
        line = tail.split("\n", 1)[0]  # a DOI is printed on one line; the next line is other text (ISSN, copyright…)
        m = _DOI_RE.match(line)
        doi = _clean_doi(m.group(0)) if m else ""
        # broken across lines: unbalanced bracket, a trailing hyphen, or the line ending in the DOI's own dot
        # ("DOI: 10.1200/JCO.2017." then "77.4315"). A line that simply ends (ISSN printed underneath) is left alone.
        if doi and (doi.count("(") > doi.count(")") or doi.endswith("-") or line[m.end():].strip() == "."):
            joined = _DOI_RE.match(re.sub(r"-?\n\s*", "", tail))
            doi = _clean_doi(joined.group(0)) if joined else doi
        if len(doi) <= 12:  # "10.1200/JCO" alone is a prefix, not a DOI
            continue
        # the paper's own DOI is printed on a "DOI:" line of its front matter; DOIs inside sentences cite other papers
        before = text[max(0, start - 40):start]
        labelled = re.search(r"(?im)^\s*(?:article\s+)?doi:?\s*(?:https?://(?:dx\.)?doi\.org/)?$", before)
        candidates.append((0 if labelled else (2 if re.search(r"doi\.org/$", before, re.I) else 3), doi))
    for uri in links:  # a first-page link is usually the paper's own, but a reference list also links out
        m = _DOI_RE.search(uri.replace("%2F", "/"))
        if m:
            candidates.append((1, _clean_doi(m.group(0))))
    return min(candidates, key=lambda c: c[0])[1] if candidates else ""


def _doi_index(doc_ids: List[str]) -> Dict[str, str]:
    """DOIs read off the papers' own first pages (the gold sheet has none), cached next to the results."""
    cache_path = runtime_paths.RESULTS_ROOT / "_registry" / "dois.json"
    cache = runs_service._read_json(cache_path) or {}
    missing = [d for d in doc_ids if d not in cache]
    if missing:
        from src.evisearch.services.highlight import resolve_pdf_path
        import fitz

        for doc in missing:
            cache[doc] = _doi_from_pdf(resolve_pdf_path(doc))
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except OSError:
            pass
    return cache


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
