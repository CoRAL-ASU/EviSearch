#!/usr/bin/env python3
"""
main_app.py

Modern web interface for Clinical Trial Data Extraction.
Provides endpoints for PDF upload, query submission, and result retrieval.

Run from project root: python web/main_app.py
Then open http://127.0.0.1:8007
"""
import csv
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any
from urllib.parse import unquote

# Load .env from project root — must happen before any other imports that read env vars
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(dotenv_path=str(_env_path), override=False)
    except ImportError:
        import re as _re
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _rest = _line.partition("=")
                _k = _k.strip()
                _rest = _rest.strip()
                _m = _re.match(r'^(["\'])(.*?)\1', _rest)
                _v = _m.group(2) if _m else _re.sub(r'\s+#.*$', '', _rest).strip()
                if _k:
                    os.environ.setdefault(_k, _v)

from flask import Flask, request, jsonify, redirect, render_template, Response, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web.extraction_service import ExtractionService
from web.comparison_service import (
    get_document_status,
    load_comparison_data,
    get_report,
)
from web.highlight_service import (
    get_highlights_by_chunk_ids,
    resolve_pdf_path,
)
from web.feedback_service import record_feedback
from src.config.runtime_paths import (
    DATASET_DIR,
    RESULTS_ROOT,
    UPLOADS_DIR,
    ensure_runtime_dirs,
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ensure_runtime_dirs()
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['BOOT_ID'] = str(uuid.uuid4())  # Changes on each app restart; used to invalidate browser session

# Global extraction service instance
extraction_service = None
current_pdf_info = {}


def get_extraction_service() -> ExtractionService:
    """Get or create extraction service instance (lazy initialization)."""
    global extraction_service
    if extraction_service is None:
        try:
            extraction_service = ExtractionService()
        except Exception as e:
            # If initialization fails, return None - will be handled by routes
            print(f"Warning: Could not initialize ExtractionService: {e}")
            return None
    return extraction_service


@app.route('/')
def index():
    """Serve the home page with Ask a question and Extract full table cards."""
    return render_template('home.html')


@app.route('/qa')
def qa_page():
    """Serve the Ask a question page (single-query QA chatbot). Placeholder for now."""
    return render_template('qa.html')


@app.route('/comparison')
def comparison():
    """Redirect to attribution (unified view)."""
    return redirect('/attribution')


@app.route('/comparison-report')
def comparison_report():
    """Serve the tables report (reconciled data pivot view). Replaces old static comparison report."""
    return render_template('tables_report.html')


@app.route('/api/report/tables', methods=['GET'])
def api_report_tables():
    """
    Get all reconciled outputs in pivot format. Only documents with reconciliation.
    Returns: document_count, total_filled_values, documents, columns, column_groups, rows.
    """
    if not RESULTS_ROOT.exists():
        return jsonify({
            "success": True,
            "document_count": 0,
            "total_filled_values": 0,
            "documents": [],
            "columns": [],
            "column_groups": {},
            "rows": [],
        }), 200

    doc_ids = []
    for d in RESULTS_ROOT.iterdir():
        if d.is_dir() and (d / RECON_AGENT_DIR / "reconciled_results.json").exists():
            doc_ids.append(d.name)
    doc_ids = sorted(doc_ids)

    if not doc_ids:
        return jsonify({
            "success": True,
            "document_count": 0,
            "total_filled_values": 0,
            "documents": [],
            "columns": [],
            "column_groups": {},
            "rows": [],
        }), 200

    all_columns: set[str] = set()
    doc_cols: Dict[str, Dict[str, str]] = {}

    for doc_id in doc_ids:
        rec_path = RESULTS_ROOT / doc_id / RECON_AGENT_DIR / "reconciled_results.json"
        if not rec_path.exists():
            continue
        try:
            data = json.loads(rec_path.read_text(encoding="utf-8"))
            cols = data.get("columns") or {}
        except Exception:
            cols = {}

        human_edited = {}
        he_path = RESULTS_ROOT / doc_id / "human-edited" / "human_edited_results.json"
        if he_path.exists():
            try:
                he_data = json.loads(he_path.read_text(encoding="utf-8"))
                human_edited = (he_data.get("columns") or {}) if isinstance(he_data, dict) else {}
            except Exception:
                pass

        row: Dict[str, str] = {"doc_id": doc_id}
        for cn, v in cols.items():
            if not isinstance(v, dict):
                continue
            val = v.get("value", "")
            he = human_edited.get(cn)
            if he and isinstance(he, dict) and he.get("value") is not None:
                val = str(he.get("value", ""))
            row[cn] = str(val) if val is not None else ""
            all_columns.add(cn)
        doc_cols[doc_id] = row

    columns_sorted = sorted(all_columns)
    rows = [doc_cols[d] for d in doc_ids]

    empty_val = frozenset({"", "not reported", "not found", "n/a", "not applicable", "—", "-"})
    total_filled = sum(
        1 for r in rows for cn in columns_sorted
        if str((r.get(cn) or "")).strip().lower() not in empty_val
    )

    try:
        from src.table_definitions.definitions import load_definitions
        defs = load_definitions()
        col_to_group: Dict[str, str] = {}
        for gname, gcols in defs.items():
            for c in gcols or []:
                cn = c.get("Column Name", "")
                if cn:
                    col_to_group[cn] = gname
    except Exception:
        col_to_group = {}

    return jsonify({
        "success": True,
        "document_count": len(doc_ids),
        "total_filled_values": total_filled,
        "documents": doc_ids,
        "columns": columns_sorted,
        "column_groups": col_to_group,
        "rows": rows,
    }), 200


def _ensure_pdf_for_extraction(doc_id: str) -> str | None:
    """Ensure PDF exists for agent extraction. For upload_* doc_ids, copy from uploads to results. Returns error string or None."""
    if doc_id.startswith("upload_"):
        upload_path = app.config["UPLOAD_FOLDER"] / f"{doc_id}.pdf"
        if not upload_path.exists():
            return f"Uploaded PDF not found: {doc_id}"
        dest_dir = RESULTS_ROOT / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{doc_id}.pdf"
        if not dest_path.exists() or dest_path.stat().st_mtime < upload_path.stat().st_mtime:
            import shutil
            shutil.copy2(str(upload_path), str(dest_path))
    return None


@app.route('/extract')
def extract_page():
    """Serve the agentic extraction page (same as home, for direct links)."""
    return render_template('extract.html')


@app.route('/verify')
def verify_page():
    """Redirect to Attribution (verify page removed). Preserve ?doc= query."""
    return redirect(request.url.replace(request.path, '/attribution', 1))


@app.route('/api/documents/<path:doc_id>/verification-data', methods=['GET'])
def api_verification_data(doc_id):
    """Get agent, search, and reconciled data merged for the verify page."""
    doc_id = unquote(doc_id)
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    recon_path = RESULTS_ROOT / doc_id / "reconciliation_agent" / "reconciled_results.json"

    if not agent_path.exists():
        return jsonify({"success": False, "error": "No agent extraction found"}), 404

    try:
        agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
        agent_cols = agent_data.get("columns", {})
    except Exception:
        agent_cols = {}

    search_cols = {}
    if search_path.exists():
        try:
            search_data = json.loads(search_path.read_text(encoding="utf-8"))
            search_cols = search_data.get("columns", {})
        except Exception:
            pass

    recon_cols = {}
    if recon_path.exists():
        try:
            recon_data = json.loads(recon_path.read_text(encoding="utf-8"))
            recon_cols = recon_data.get("columns", {})
        except Exception:
            pass

    all_columns = set(agent_cols) | set(search_cols) | set(recon_cols)
    rows = []
    for col_name in sorted(all_columns):
        a = agent_cols.get(col_name, {})
        s = search_cols.get(col_name, {})
        r = recon_cols.get(col_name, {})
        val_a = a.get("value", "") if isinstance(a, dict) else str(a or "")
        val_b = s.get("value", "") if isinstance(s, dict) else str(s or "")
        val_recon = r.get("value", "") if isinstance(r, dict) else str(r or "")
        reasoning = r.get("reasoning", "") if isinstance(r, dict) else ""
        verification = r.get("verification", "") if isinstance(r, dict) else ""
        src = r.get("source", {}) if isinstance(r, dict) else {}
        verbatim = src.get("verbatim_quote", "") if isinstance(src, dict) else ""
        if not verbatim and isinstance(r, dict):
            attr = r.get("attribution") or []
            if attr and isinstance(attr[0], dict):
                verbatim = str(attr[0].get("verbatim_quote") or "").strip()
        rows.append({
            "column": col_name,
            "candidate_a": val_a,
            "candidate_b": val_b,
            "reconciled": val_recon,
            "reasoning": reasoning,
            "verification": verification,
            "verbatim_quote": verbatim,
        })
    return jsonify({
        "success": True,
        "doc_id": doc_id,
        "rows": rows,
        "has_reconciliation": bool(recon_cols),
        "has_search": bool(search_cols),
    }), 200


@app.route('/api/documents/<path:doc_id>/run-reconciliation', methods=['POST'])
def api_run_reconciliation(doc_id):
    """Run the reconciliation agent pipeline for this document."""
    doc_id = unquote(doc_id)
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
    if not agent_path.exists():
        return jsonify({"success": False, "error": "Agent extraction not found"}), 404
    if not search_path.exists():
        return jsonify({"success": False, "error": "Search agent results not found"}), 404

    body = request.get_json(silent=True) or {}
    no_resume = body.get("no_resume", False)
    group_names = body.get("group_names")

    # If no explicit groups given, infer from what was actually extracted —
    # only reconcile groups that have at least one column in extraction_results.json.
    if not group_names:
        try:
            agent_data = json.loads(agent_path.read_text(encoding="utf-8"))
            extracted_cols = set(agent_data.get("columns", {}).keys())
            from src.table_definitions.definitions import load_definitions as _ld
            _groups_raw = _ld()
            group_names = [
                g for g, cols in _groups_raw.items()
                if any(c.get("Column Name") in extracted_cols for c in cols)
            ] or None
        except Exception:
            group_names = None

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))
        from run_reconciliation_agent import run_reconciliation_pipeline
        result = run_reconciliation_pipeline(
            doc_id=doc_id,
            group_names=group_names,
            resume=not no_resume,
            no_resume=no_resume,
        )
        if result.get("error"):
            return jsonify({"success": False, "error": result["error"]}), 400
        return jsonify({"success": True, "doc_id": doc_id, "columns_count": len(result.get("columns", {}))}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/column-groups', methods=['GET'])
def api_column_groups():
    """Get column groups from definitions (for extract page)."""
    try:
        from src.table_definitions.definitions import load_definitions
        groups = load_definitions()
        out = [{"name": g, "columns": [{"name": c.get("Column Name", ""), "definition": c.get("Definition", "")} for c in cols]} for g, cols in groups.items()]
        return jsonify({"success": True, "groups": out}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _col_to_group(col_name: str, groups: list) -> str:
    """Return group name for a column from groups list."""
    for g in groups:
        if col_name in (g.get("columns") or []):
            return g.get("name", "")
    return ""


@app.route('/api/documents/<path:doc_id>/agent_extraction', methods=['GET'])
def api_agent_extraction(doc_id):
    """Get existing agent + search extraction results (for load-from-disk)."""
    doc_id = unquote(doc_id)
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    if not agent_path.exists():
        return jsonify({"success": False, "error": "No agent extraction found"}), 404
    try:
        from src.table_definitions.definitions import load_definitions
        groups_raw = load_definitions()
        groups_list = [{"name": g, "columns": [c.get("Column Name") for c in cols]} for g, cols in groups_raw.items()]

        data = json.loads(agent_path.read_text(encoding="utf-8"))
        agent_columns = data.get("columns", {})

        search_columns = {}
        search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
        if search_path.exists():
            try:
                search_data = json.loads(search_path.read_text(encoding="utf-8"))
                search_columns = search_data.get("columns", {})
            except Exception:
                pass

        rows = []
        for col_name, col_data in agent_columns.items():
            if isinstance(col_data, dict):
                val = col_data.get("value", "")
            else:
                val = col_data
            sc = search_columns.get(col_name)
            val_b = ""
            if sc is not None:
                val_b = sc.get("value", "") if isinstance(sc, dict) else str(sc)
            rows.append({
                "column": col_name,
                "value": str(val) if val is not None else "",
                "candidate_b": str(val_b) if val_b is not None else "",
                "group": _col_to_group(col_name, groups_list),
            })
        return jsonify({
            "success": True,
            "doc_id": doc_id,
            "columns": rows,
            "turns": data.get("turns", 0),
            "filled": len([r for r in rows if r["value"] and str(r["value"]).lower() not in ("not reported", "not found", "")]),
            "total": len(rows),
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/extract/unified/stream', methods=['POST'])
def api_extract_unified_stream():
    """Unified extraction: Agent + Search in parallel per batch. Emits batch_complete with both A and B."""
    data = request.get_json() or {}
    doc_id = (data.get("doc_id") or "").strip()
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400

    err = _ensure_pdf_for_extraction(doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    groups_filter = data.get("column_groups")
    resume = data.get("resume", True)
    no_resume = not resume or data.get("no_resume", False)

    def generate():
        import queue
        import threading

        sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))
        from unified_extraction import run_unified_extraction

        q = queue.Queue()

        def run():
            try:
                run_unified_extraction(
                    doc_id=doc_id,
                    group_names=groups_filter,
                    resume=resume,
                    no_resume=no_resume,
                    on_event=q.put,
                )
            except Exception as e:
                q.put({"type": "error", "error": str(e)})
            q.put(None)

        thread = threading.Thread(target=run)
        thread.start()

        while True:
            ev = q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/api/extract/agentic/stream', methods=['POST'])
def api_extract_agentic_stream():
    """Run agent_extractor and search_agent in parallel, stream SSE events."""
    data = request.get_json() or {}
    doc_id = (data.get("doc_id") or "").strip()
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400

    err = _ensure_pdf_for_extraction(doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    groups_filter = data.get("column_groups")
    max_turns = data.get("max_turns", 50)
    resume = data.get("resume", True)
    skip_if_done = data.get("skip_if_done", True)

    # Capture paths needed inside generate() (closures are fine here)
    _chunk_dir = RESULTS_ROOT / doc_id / "chunking"
    _md_path = _chunk_dir / "parsed_markdown.md"
    _pdf_path_for_prepare = resolve_pdf_path(doc_id)

    def generate():
        import queue
        import threading

        # For freshly uploaded PDFs, run Landing AI parse + embedding before extraction.
        # Without this, search_agent and reconciliation_agent see "Document has 0 pages."
        if doc_id.startswith("upload_") and not _md_path.exists():
            try:
                from web.landing_ai_parse_service import parse_pdf_for_qa
                from src.retrieval.openai_embedding_retriever import embed_chunks

                if not _pdf_path_for_prepare or not _pdf_path_for_prepare.exists():
                    raise FileNotFoundError(f"PDF not found for {doc_id}: {_pdf_path_for_prepare}")

                yield f"data: {json.dumps({'type': 'prepare_status', 'text': 'Parsing PDF with Landing AI… (this may take 30–60 s)'})}\n\n"
                parse_result = parse_pdf_for_qa(doc_id, _pdf_path_for_prepare, on_event=lambda e: None)
                if not parse_result.get("success"):
                    raise RuntimeError(parse_result.get("error", "PDF parse failed"))

                yield f"data: {json.dumps({'type': 'prepare_status', 'text': 'Parse complete. Building embeddings…'})}\n\n"
                embed_result = embed_chunks(doc_id, force=True)
                if not embed_result:
                    raise RuntimeError("embed_chunks returned None — no parsed content found")

                n_chunks = len(embed_result[0]) if embed_result and embed_result[0] else 0
                yield f"data: {json.dumps({'type': 'prepare_status', 'text': f'Ready: {n_chunks} page chunks indexed. Starting extraction…'})}\n\n"

            except Exception as _prep_ex:
                _err = f"[PREPARE FAILED] {_prep_ex}"
                yield f"data: {json.dumps({'type': 'prepare_error', 'text': _err})}\n\n"
                # Do not run agents — search agent will see 0 pages without embeddings
                return

        sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))
        from agent_extractor import run_extraction_loop_deterministic
        from run_search_agent import load_definitions, build_extraction_batches, run_search_agent_pipeline

        # Emit extraction_start with total and batches for empty table init
        groups = load_definitions()
        batches = build_extraction_batches(groups, group_names=groups_filter, resume_from=None)
        all_column_names = []
        for b in batches:
            all_column_names.extend(c.get("column_name", "") for c in b)
        total = len(all_column_names)
        batch_column_names = [[c.get("column_name", "") for c in b] for b in batches]

        first_batch_size = len(batch_column_names[0]) if batch_column_names else 0
        if total == 0:
            first_batch_size = 0
        yield f"data: {json.dumps({'type': 'extraction_start', 'total': total, 'column_names': all_column_names, 'batches': batch_column_names})}\n\n"
        yield f"data: {json.dumps({'type': 'stream_message', 'text': f'Loaded {first_batch_size} queries — ', 'show_columns': 0})}\n\n"

        q = queue.Queue()
        agent_done_payload = {}

        def on_agent_event(ev):
            if ev.get("type") == "done":
                agent_done_payload.update(ev)
                q.put({"type": "phase_done", "phase": "agent_extractor", **ev})
            else:
                q.put(ev)

        def run_agent():
            try:
                run_extraction_loop_deterministic(
                    doc_id=doc_id,
                    max_turns=max_turns,
                    groups_filter=groups_filter,
                    resume=resume,
                    skip_if_done=skip_if_done,
                    on_event=on_agent_event,
                )
            except Exception as e:
                q.put({"type": "error", "error": str(e)})
            finally:
                q.put({"type": "thread_done", "thread": "agent"})

        def run_search():
            try:
                run_search_agent_pipeline(
                    doc_id=doc_id,
                    group_names=groups_filter,
                    resume=resume,
                    no_resume=not resume or skip_if_done,
                    on_event=q.put,
                )
            except Exception as e:
                q.put({"type": "error", "error": str(e)})
            finally:
                q.put({"type": "thread_done", "thread": "search"})

        # Verify embedding cache exists before firing the search agent (warn loudly if missing)
        _md_check = RESULTS_ROOT / doc_id / "chunking" / "parsed_markdown.md"
        if not _md_check.exists():
            yield f"data: {json.dumps({'type': 'prepare_warning', 'text': f'WARNING: No parsed_markdown.md for {doc_id}. Search-Agent will see 0 pages and return Not Reported for everything. Restart server and re-upload the PDF to trigger Landing AI parsing.'})}\n\n"

        threading.Thread(target=run_agent).start()
        threading.Thread(target=run_search).start()
        yield f"data: {json.dumps({'type': 'stream_message', 'text': 'Running 2 methods: Agent Extractor + Search Agent.'})}\n\n"

        agent_done = search_done = False
        while True:
            ev = q.get()
            if ev.get("type") == "thread_done":
                if ev.get("thread") == "agent":
                    agent_done = True
                elif ev.get("thread") == "search":
                    search_done = True
                if agent_done and search_done:
                    yield f"data: {json.dumps({'type': 'done', **agent_done_payload})}\n\n"
                    break
                continue
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/attribution')
def attribution_index():
    """Serve attribution viewer — select doc and column, see highlighted chunks on PDF."""
    return render_template('attribution.html')


@app.route('/agent-viewer/<path:doc_id>')
def agent_viewer(doc_id):
    """Serve static conversation viewer HTML (generated by agent_extractor.py)."""
    doc_id = unquote(doc_id)
    path = RESULTS_ROOT / doc_id / "agent_extractor" / "conversation_viewer.html"
    if not path.exists():
        return f"Run: python experiment-scripts/agent_extractor.py \"{doc_id}\"", 404
    return send_from_directory(str(path.parent), path.name)


@app.route('/api/documents/reconciled', methods=['GET'])
def api_list_reconciled_documents():
    """List document IDs that have reconciled results or agent extraction (for attribution)."""
    if not RESULTS_ROOT.exists():
        return jsonify({"success": True, "documents": []}), 200
    docs = set()
    for d in RESULTS_ROOT.iterdir():
        if d.is_dir():
            if (d / "reconciliation_agent" / "reconciled_results.json").exists():
                docs.add(d.name)
            elif (d / "agent_extractor" / "extraction_results.json").exists():
                docs.add(d.name)
    return jsonify({"success": True, "documents": sorted(docs)}), 200


def _has_extraction(doc_id: str) -> bool:
    """True if document has agent extraction or reconciled results."""
    if not RESULTS_ROOT.exists():
        return False
    d = RESULTS_ROOT / doc_id
    return (
        (d / "reconciliation_agent" / "reconciled_results.json").exists()
        or (d / "agent_extractor" / "extraction_results.json").exists()
    )


@app.route('/api/documents/selectable', methods=['GET'])
def api_list_selectable_documents():
    """
    List all documents available for extraction: dataset PDFs, extracted results, and uploads.
    For extract page: user can choose from this list or upload new.
    """
    docs = {}  # doc_id -> {id, name, source, has_extraction}

    # 1. Dataset PDFs (dataset/*.pdf) - use stem as doc_id
    if DATASET_DIR.exists():
        for p in DATASET_DIR.glob("*.pdf"):
            doc_id = p.stem
            docs[doc_id] = {
                "id": doc_id,
                "name": doc_id,
                "source": "dataset",
                "has_extraction": _has_extraction(doc_id),
            }
        for p in DATASET_DIR.glob("**/*.pdf"):
            if p.parent == DATASET_DIR:
                continue  # already covered by *.pdf
            # e.g. dataset/subdir/foo.pdf -> doc_id = subdir/foo or just stem
            rel = p.relative_to(DATASET_DIR)
            doc_id = str(rel.with_suffix("")).replace("/", "_")
            if doc_id not in docs:
                docs[doc_id] = {
                    "id": doc_id,
                    "name": p.stem,
                    "source": "dataset",
                    "has_extraction": _has_extraction(doc_id),
                }

    # 2. Extracted docs from results (agent_extractor or reconciled)
    if RESULTS_ROOT.exists():
        for d in RESULTS_ROOT.iterdir():
            if d.is_dir() and d.name not in docs:
                has_ext = (
                    (d / "reconciliation_agent" / "reconciled_results.json").exists()
                    or (d / "agent_extractor" / "extraction_results.json").exists()
                )
                if has_ext:
                    docs[d.name] = {
                        "id": d.name,
                        "name": d.name,
                        "source": "extracted",
                        "has_extraction": True,
                    }

    # 3. Uploads (web/uploads/upload_*.pdf)
    upload_folder = app.config["UPLOAD_FOLDER"]
    if upload_folder.exists():
        for p in upload_folder.glob("upload_*.pdf"):
            doc_id = p.stem
            docs[doc_id] = {
                "id": doc_id,
                "name": f"{doc_id} (uploaded)",
                "source": "upload",
                "has_extraction": _has_extraction(doc_id),
            }

    out = sorted(docs.values(), key=lambda x: (x["name"].lower(), x["id"]))
    return jsonify({"success": True, "documents": out}), 200


@app.route('/api/documents/<path:doc_id>/attribution/refresh', methods=['POST'])
def api_refresh_attribution(doc_id):
    """Re-run attribution and save. Uses reconciliation_agent if present, else agent-only."""
    doc_id = unquote(doc_id)
    rec_path = RESULTS_ROOT / doc_id / "reconciliation_agent" / "reconciled_results.json"
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    try:
        from web.comparison_service import load_comparison_data
        from web.attribution_service import enrich_reconciled_with_attribution
        comparison = load_comparison_data(doc_id)
        rows = comparison.get("comparison") or []

        if rec_path.exists():
            data = json.loads(rec_path.read_text(encoding="utf-8"))
            columns = _reconciliation_agent_to_columns(data.get("columns") or {})
            out_path = rec_path.parent / "attribution_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        elif agent_path.exists():
            agent_cols = [r for r in rows if (r.get("methods") or {}).get("agent")]
            if not agent_cols:
                return jsonify({"success": False, "error": "No agent data for this document"}), 404
            columns = []
            for r in agent_cols:
                a = (r.get("methods") or {}).get("agent") or {}
                val = a.get("value") or a.get("primary_value", "")
                reasoning = (a.get("evidence") or a.get("reasoning", "") or "").strip()
                columns.append({
                    "column_name": r["column_name"],
                    "final_value": str(val) if val else "",
                    "contributing_methods": ["agent"],
                    "agent_reasoning": reasoning if reasoning else None,
                })
            out_path = RESULTS_ROOT / doc_id / "agent_extractor" / "attribution_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            return jsonify({"success": False, "error": f"No reconciled or agent results for {doc_id}"}), 404

        enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)
        out_path.write_text(json.dumps({"doc_id": doc_id, "columns": enriched}, indent=2), encoding="utf-8")
        return jsonify({"success": True, "doc_id": doc_id, "columns": enriched, "verification_stats": {}}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/reconciled', methods=['GET'])
def api_document_reconciled(doc_id):
    """Get reconciled results with attributed chunks. Uses reconciliation_agent, falls back to agent-only."""
    doc_id = unquote(doc_id)
    rec_path = RESULTS_ROOT / doc_id / RECON_AGENT_DIR / "reconciled_results.json"
    recon_attr_path = RESULTS_ROOT / doc_id / RECON_AGENT_DIR / "attribution_results.json"
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    agent_attr_path = RESULTS_ROOT / doc_id / "agent_extractor" / "attribution_results.json"

    try:
        # Serve cached attribution only if reconciled_results hasn't been updated since
        recon_attr_fresh = (
            recon_attr_path.exists()
            and not (rec_path.exists() and rec_path.stat().st_mtime > recon_attr_path.stat().st_mtime)
        )
        if recon_attr_fresh:
            data = json.loads(recon_attr_path.read_text(encoding="utf-8"))
        elif rec_path.exists():
            data = json.loads(rec_path.read_text(encoding="utf-8"))
            columns = _reconciliation_agent_to_columns(data.get("columns") or {})
            comparison = load_comparison_data(doc_id)
            rows = comparison.get("comparison") or []
            from web.attribution_service import enrich_reconciled_with_attribution
            enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)
            data["columns"] = enriched
        elif agent_attr_path.exists():
            data = json.loads(agent_attr_path.read_text(encoding="utf-8"))
        elif agent_path.exists():
            data = _build_agent_attribution(doc_id)
            if not data:
                return jsonify({"success": False, "error": "Could not build agent attribution"}), 500
        else:
            return jsonify({"success": False, "error": f"No reconciled or agent results for {doc_id}"}), 404

        comparison = load_comparison_data(doc_id)
        col_to_row = {r.get("column_name"): r for r in (comparison.get("comparison") or [])}

        agent_cols = {}
        agent_chunk_ids: Dict[str, list] = {}
        search_cols = {}
        search_chunk_ids: Dict[str, list] = {}
        if agent_path.exists():
            try:
                ad = json.loads(agent_path.read_text(encoding="utf-8"))
                for k, v in (ad.get("columns") or {}).items():
                    agent_cols[k] = str(v.get("value", "")) if isinstance(v, dict) else str(v or "")
                    if isinstance(v, dict):
                        pages = [a["page"] for a in (v.get("attribution") or []) if isinstance(a, dict) and a.get("page")]
                        agent_chunk_ids[k] = [f"page_{p}" for p in pages]
            except Exception:
                pass
        search_path = RESULTS_ROOT / doc_id / "search_agent" / "extraction_results.json"
        if search_path.exists():
            try:
                sd = json.loads(search_path.read_text(encoding="utf-8"))
                for k, v in (sd.get("columns") or {}).items():
                    search_cols[k] = str(v.get("value", "")) if isinstance(v, dict) else str(v or "")
                    if isinstance(v, dict):
                        pages = [a["page"] for a in (v.get("attribution") or []) if isinstance(a, dict) and a.get("page")]
                        search_chunk_ids[k] = [f"page_{p}" for p in pages]
            except Exception:
                pass

        columns = data.get("columns") or []

        human_edited_path = RESULTS_ROOT / doc_id / "human-edited" / "human_edited_results.json"
        human_edited = {}
        if human_edited_path.exists():
            try:
                he_data = json.loads(human_edited_path.read_text(encoding="utf-8"))
                human_edited = (he_data.get("columns") or {}) if isinstance(he_data, dict) else {}
            except Exception:
                pass
        for col in columns:
            cn = col.get("column_name", "")
            he_col = human_edited.get(cn)
            if he_col and isinstance(he_col, dict) and he_col.get("value") is not None:
                col["final_value"] = str(he_col.get("value", ""))
                col["human_edited"] = True
            cn = col.get("column_name", "")
            col["candidate_a"] = agent_cols.get(cn, "")
            col["candidate_b"] = search_cols.get(cn, "")
            col["chunk_ids_a"] = agent_chunk_ids.get(cn, [])
            col["chunk_ids_b"] = search_chunk_ids.get(cn, [])
            col["reconciliation_reasoning"] = col.get("agent_reasoning") or ""
            row = col_to_row.get(cn)
            if row and row.get("methods"):
                col["method_values"] = {
                    k: (v.get("value") or v.get("primary_value", ""))
                    for k, v in row["methods"].items()
                }

        data["columns"] = columns
        data["verification_stats"] = {}

        return jsonify({"success": True, **data}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/human-edited', methods=['POST'])
def api_save_human_edited(doc_id):
    """Save human-edited column values. Stores in new_pipeline_outputs/results/<doc_id>/human-edited/."""
    doc_id = unquote(doc_id)
    body = request.get_json() or {}
    columns = body.get("columns")
    if not isinstance(columns, dict):
        return jsonify({"success": False, "error": "columns object required"}), 400

    human_edited_dir = RESULTS_ROOT / doc_id / "human-edited"
    human_edited_dir.mkdir(parents=True, exist_ok=True)
    path = human_edited_dir / "human_edited_results.json"

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing_cols = existing.get("columns") or {}
    if not isinstance(existing_cols, dict):
        existing_cols = {}

    for cn, v in columns.items():
        if not cn or not isinstance(v, dict):
            continue
        val = v.get("value")
        existing_cols[str(cn)] = {"value": str(val) if val is not None else "", "human_edited": True}

    data = {"doc_id": doc_id, "columns": existing_cols}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"success": True, "doc_id": doc_id}), 200


RECON_AGENT_DIR = "reconciliation_agent"


def _reconciliation_agent_to_columns(cols_dict: Dict[str, Any]) -> list:
    """Convert reconciliation_agent columns dict to list format for enrich_reconciled_with_attribution."""
    out = []
    for col_name, r in (cols_dict or {}).items():
        if not isinstance(r, dict):
            continue
        src = r.get("source") or {}
        out.append({
            "column_name": col_name,
            "final_value": str(r.get("value", "")) or "",
            "contributing_methods": ["reconciliation_agent"],
            "page": src.get("page") if isinstance(src, dict) else None,
            "source_type": src.get("modality", "text") if isinstance(src, dict) else "text",
            "verbatim_quote": src.get("verbatim_quote", "") if isinstance(src, dict) else "",
            "agent_reasoning": str(r.get("reasoning", "") or "").strip() or None,
            "verification_label": str(r.get("verification", "") or "").strip() or None,
        })
    return out


def _build_agent_attribution(doc_id: str) -> Dict[str, Any] | None:
    """Build attribution columns from agent extraction, run enrich, return."""
    from web.attribution_service import enrich_reconciled_with_attribution

    # Load raw extraction_results.json directly so we can pass attribution hints
    # (page + modality) into enrich_reconciled_with_attribution.
    agent_raw: Dict[str, Any] = {}
    agent_path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    if agent_path.exists():
        try:
            agent_raw = json.loads(agent_path.read_text(encoding="utf-8")).get("columns", {})
        except Exception:
            pass

    comparison = load_comparison_data(doc_id)
    rows = comparison.get("comparison") or []

    # Build column list from raw extraction (covers uploaded docs with no comparison rows)
    columns = []
    if agent_raw:
        for col_name, v in agent_raw.items():
            if not isinstance(v, dict):
                continue
            val = v.get("value", "")
            reasoning = (v.get("reasoning") or "").strip()
            # Pull first attribution entry for page + source_type hints
            attr_list = v.get("attribution") or []
            first_attr = attr_list[0] if attr_list else {}
            col = {
                "column_name": col_name,
                "final_value": str(val) if val else "",
                "contributing_methods": ["agent"],
                "agent_reasoning": reasoning if reasoning else None,
                "page": first_attr.get("page"),
                "source_type": first_attr.get("modality"),
            }
            columns.append(col)
    else:
        # Fallback: build from comparison rows (pre-existing benchmark docs)
        agent_cols = [r for r in rows if (r.get("methods") or {}).get("agent")]
        if not agent_cols:
            return None
        for r in agent_cols:
            agent_data = (r.get("methods") or {}).get("agent") or {}
            val = agent_data.get("value") or agent_data.get("primary_value", "")
            reasoning = (agent_data.get("evidence") or agent_data.get("reasoning", "") or "").strip()
            columns.append({
                "column_name": r["column_name"],
                "final_value": str(val) if val else "",
                "contributing_methods": ["agent"],
                "agent_reasoning": reasoning if reasoning else None,
            })

    if not columns:
        return None

    enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)
    return {"doc_id": doc_id, "columns": enriched}


@app.route('/api/upload/extract', methods=['POST'])
def upload_pdf_for_extract():
    """Upload PDF for agentic extraction. Returns doc_id for use with /api/extract/agentic/stream."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files allowed"}), 400
    try:
        import uuid
        doc_id = "upload_" + uuid.uuid4().hex[:12]
        filename = f"{doc_id}.pdf"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        return jsonify({"success": True, "doc_id": doc_id, "filename": file.filename}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------------------------------------------------------
# QA (Ask a question) endpoints
# -----------------------------------------------------------------------------

@app.route('/api/qa/session-info', methods=['GET'])
def api_qa_session_info():
    """Return boot_id so the client can invalidate stored session when the app restarts."""
    return jsonify({"boot_id": app.config.get("BOOT_ID", "")})


@app.route('/api/qa/prepare-document', methods=['POST'])
def api_qa_prepare_document():
    """Parse PDF + build embeddings for QA. Streams SSE events: parsing → embedding → ready."""
    data = request.get_json() or {}
    doc_id = (data.get("doc_id") or "").strip()
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400

    err = _ensure_pdf_for_extraction(doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        return jsonify({"success": False, "error": f"PDF not found for {doc_id}"}), 400

    chunk_dir = RESULTS_ROOT / doc_id / "chunking"
    md_path = chunk_dir / "parsed_markdown.md"
    json_path = chunk_dir / "landing_ai_parse_output.json"

    def generate():
        from web.landing_ai_parse_service import parse_pdf_for_qa
        from src.retrieval.openai_embedding_retriever import embed_chunks

        # Require landing_ai_parse_output.json for attribution. Skip parse only if it exists and is fresh.
        # No baseline fallback: baseline markdown lacks chunk ids/grounding needed for attribution.
        need_parse = True
        if json_path.exists():
            pdf_mtime = pdf_path.stat().st_mtime
            json_mtime = json_path.stat().st_mtime
            need_parse = pdf_mtime > json_mtime

        if need_parse:
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'parsing', 'message': 'Parsing PDF with Landing AI…'})}\n\n"
            result = parse_pdf_for_qa(doc_id, pdf_path, on_event=lambda e: None)
            if not result.get("success"):
                yield f"data: {json.dumps({'type': 'error', 'error': result.get('error', 'Parse failed')})}\n\n"
                return
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'parsing_done', 'message': 'Parse complete'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'parsing_done', 'message': 'Using cached parse'})}\n\n"

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'embedding', 'message': 'Building embeddings…'})}\n\n"
        try:
            embed_result = embed_chunks(doc_id, force=False)
            if not embed_result:
                yield f"data: {json.dumps({'type': 'error', 'error': 'No parsed content; embedding failed'})}\n\n"
                return
        except Exception as ex:
            yield f"data: {json.dumps({'type': 'error', 'error': str(ex)})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'embedding_done', 'message': 'Embeddings ready'})}\n\n"

        # Ensure landing_ai_parse_output.json exists (required for attribution)
        if not json_path.exists():
            yield f"data: {json.dumps({'type': 'error', 'error': 'landing_ai_parse_output.json missing; attribution will not work. Re-run Prepare.'})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'ready', 'doc_id': doc_id})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/api/qa/ask', methods=['POST'])
def api_qa_ask():
    """QA: Quick mode (Gemini chat) or Full mode (Agent + Search + Reconcile)."""
    data = request.get_json() or {}
    doc_id = (data.get("doc_id") or "").strip()
    question = (data.get("question") or "").strip()
    history = data.get("history") or []
    mode = (data.get("mode") or "full").strip().lower()
    if mode not in ("quick", "full"):
        mode = "full"

    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400
    if not question:
        return jsonify({"success": False, "error": "question required"}), 400

    err = _ensure_pdf_for_extraction(doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    if mode == "quick":
        return _api_qa_ask_quick(doc_id, question, history)

    return _api_qa_ask_full(doc_id, question, history)


def _api_qa_ask_quick(doc_id: str, question: str, history: list):
    """Quick mode: Multi-turn Gemini chat with PDF. No attribution."""
    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        return jsonify({"success": False, "error": f"PDF not found for {doc_id}"}), 400

    def generate():
        from src.LLMProvider.provider import LLMProvider

        provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
        try:
            pdf_handle = provider.upload_pdf(str(pdf_path))
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'generating', 'message': 'Generating…'})}\n\n"

        context_block = ""
        if history:
            turns = history[-5:]
            context_block = "\n\nPrevious Q&A:\n" + "\n".join(
                f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}" for h in turns
            )

        prompt = f"""You are answering questions about this clinical trial research PDF. Use only the document to answer. Be concise and cite specific values when possible.{context_block}

Current question: {question}

Answer:"""

        try:
            response = provider.generate_with_pdf(
                prompt=prompt,
                pdf_handle=pdf_handle,
                temperature=0.2,
                max_tokens=4096,
            )
            provider.cleanup_pdf(pdf_handle)

            if response.success:
                answer = (response.text or "").strip()
                yield f"data: {json.dumps({'type': 'done', 'mode': 'quick', 'answer': answer})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': response.error or 'Generation failed'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _api_qa_ask_full(doc_id: str, question: str, history: list):
    """Full mode: Agent + Search + Reconcile with attribution."""
    def generate():
        import threading
        from web.qa_adapter import build_definition_with_context

        sys.path.insert(0, str(PROJECT_ROOT / "experiment-scripts"))
        from agent_extractor import extract_batch
        from src.LLMProvider.provider import LLMProvider
        from web.search_agent import run_search_agent

        col_name = "qa_query"
        definition = build_definition_with_context(question, history)
        batch = [{"column_name": col_name, "definition": definition}]
        definitions_map = {col_name: definition}

        agent_result = {}
        search_result = {}

        def run_agent():
            nonlocal agent_result
            try:
                pdf_path = resolve_pdf_path(doc_id)
                if not pdf_path or not pdf_path.exists():
                    agent_result = {col_name: {"value": "Not reported", "reasoning": "PDF not found", "found": False, "attribution": []}}
                    return
                provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
                pdf_handle = provider.upload_pdf(str(pdf_path))
                agent_result = extract_batch(doc_id, batch, pdf_handle, provider)
                provider.cleanup_pdf(pdf_handle)
            except Exception as e:
                agent_result = {col_name: {"value": "Not reported", "reasoning": str(e), "found": False, "attribution": []}}

        def run_search():
            nonlocal search_result
            try:
                search_result, _ = run_search_agent(doc_id, batch, definitions_map, log_path=None)
            except Exception as e:
                search_result = {col_name: {"value": "Not reported", "reasoning": str(e), "found": False, "attribution": []}}

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'direct_pdf', 'message': 'Extracting with Direct PDF…'})}\n\n"
        t_agent = threading.Thread(target=run_agent)
        t_search = threading.Thread(target=run_search)
        t_agent.start()
        t_search.start()
        t_agent.join()
        t_search.join()

        a_val = (agent_result.get(col_name) or {}).get("value", "Not reported")
        a_reason = (agent_result.get(col_name) or {}).get("reasoning", "")
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'direct_pdf_done', 'value': a_val, 'reasoning': a_reason})}\n\n"

        s_val = (search_result.get(col_name) or {}).get("value", "Not reported")
        s_reason = (search_result.get(col_name) or {}).get("reasoning", "")
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'search_done', 'value': s_val, 'reasoning': s_reason})}\n\n"

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'reconciling', 'message': 'Reconciling…'})}\n\n"
        from web.reconciliation_agent import run_reconciliation_agent

        rec_result, _ = run_reconciliation_agent(
            doc_id=doc_id,
            batch_columns=batch,
            definitions_map=definitions_map,
            source_a_data=agent_result,
            source_b_data=search_result,
            log_path=None,
        )
        rec_col = rec_result.get(col_name, {})
        rec_val = rec_col.get("value", "Not reported")
        rec_reason = rec_col.get("reasoning", "")
        rec_source = rec_col.get("source") or {}
        verbatim = rec_source.get("verbatim_quote", "") if isinstance(rec_source, dict) else ""
        rec_attr = rec_col.get("attribution", [])

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'reconciled', 'value': rec_val, 'reasoning': rec_reason})}\n\n"

        chunk_ids = []
        from web.attribution_service import resolve_chunks_from_reconciled_source, retrieve_chunks_for_evidence

        if isinstance(rec_source, dict) and rec_source.get("page"):
            raw = resolve_chunks_from_reconciled_source(
                doc_id,
                page=rec_source.get("page"),
                modality=rec_source.get("modality", "text"),
                verbatim_quote=verbatim,
            )
            chunk_ids = [c.get("chunk_id") for c in raw if c.get("chunk_id")]

        if not chunk_ids and rec_val and rec_val.lower() not in ("not reported", "not found", "n/a"):
            a_col = agent_result.get(col_name) or {}
            b_col = search_result.get(col_name) or {}
            a_attr = a_col.get("attribution") or []
            b_attr = b_col.get("attribution") or []
            attr_list = None
            fallback_page = None
            fallback_type = "text"
            if isinstance(a_attr, list) and len(a_attr) > 0 and isinstance(a_attr[0], dict):
                attr_list = [{"page": x.get("page"), "source_type": x.get("modality") or x.get("source_type") or "text"} for x in a_attr if x.get("page")]
                if attr_list:
                    fallback_page = attr_list[0].get("page")
                    fallback_type = attr_list[0].get("source_type") or "text"
            if not attr_list and isinstance(b_attr, list) and len(b_attr) > 0 and isinstance(b_attr[0], dict):
                attr_list = [{"page": x.get("page"), "source_type": x.get("modality") or x.get("source_type") or "text"} for x in b_attr if x.get("page")]
                if attr_list:
                    fallback_page = attr_list[0].get("page")
                    fallback_type = attr_list[0].get("source_type") or "text"
            raw = retrieve_chunks_for_evidence(
                doc_id,
                top_k=5,
                column_name=col_name,
                final_value=rec_val,
                pipeline_page=fallback_page,
                pipeline_source_type=fallback_type,
                method_values=[a_val, s_val] if a_val and s_val else None,
                attribution=attr_list if attr_list else None,
            )
            chunk_ids = [c.get("chunk_id") for c in raw if c.get("chunk_id")]

        payload = {
            "type": "done",
            "mode": "full",
            "candidate_a": a_val,
            "candidate_b": s_val,
            "reconciled": rec_val,
            "reconciliation_reasoning": rec_reason,
            "verbatim_quote": verbatim,
            "attribution": rec_attr,
            "chunk_ids": chunk_ids,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload a PDF file for extraction."""
    global current_pdf_info
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        # Load PDF into extraction service
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.upload_pdf(str(filepath))
        
        if result.get("success"):
            current_pdf_info = {
                "filename": filename,
                "filepath": str(filepath),
                "message": result.get("message")
            }
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500


@app.route('/api/columns', methods=['GET'])
def get_columns():
    """Get list of all available columns."""
    try:
        service = get_extraction_service()
        columns = service.get_available_columns()
        return jsonify({
            "success": True,
            "columns": columns,
            "total": len(columns)
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/extract/single', methods=['POST'])
def extract_single():
    """Extract a single column value."""
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    column_name = data.get('column_name')
    definition = data.get('definition')
    
    if not column_name:
        return jsonify({"success": False, "error": "column_name is required"}), 400
    
    try:
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.extract_single_column(column_name, definition)
        return jsonify(result), 200 if result.get("success") else 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<doc_id>/status', methods=['GET'])
def api_document_status(doc_id):
    """Get which extraction methods have run for this document."""
    try:
        status = get_document_status(doc_id)
        return jsonify({
            "success": True,
            "doc_id": doc_id,
            "status": status,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<doc_id>/report', methods=['GET'])
def api_document_report(doc_id):
    """Get document analysis report (summary stats)."""
    try:
        report = get_report(doc_id)
        return jsonify({
            "success": True,
            **report,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/highlights', methods=['GET'])
def api_document_highlights(doc_id):
    """Get highlight boxes for PDF overlay. Query params: chunk_ids (comma-separated)."""
    doc_id = unquote(doc_id)
    chunk_ids_str = request.args.get("chunk_ids")

    try:
        if not chunk_ids_str:
            return jsonify({
                "success": False,
                "error": "Provide chunk_ids (comma-separated)",
            }), 400
        chunk_ids = [x.strip() for x in chunk_ids_str.split(",") if x.strip()]
        result = get_highlights_by_chunk_ids(doc_id, chunk_ids)
        return jsonify({"success": True, **result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """
    Record user feedback. Supports source: chat | verify.
    Body: { source, doc_id, comment?, chat?: {...}, table?: {...} }
    """
    try:
        data = request.get_json() or {}
        source = data.get("source")
        doc_id = data.get("doc_id") or ""
        if not source or source not in ("chat", "attribution"):
            return jsonify({"success": False, "error": "Invalid source (use 'chat' or 'attribution')"}), 400
        if not doc_id:
            return jsonify({"success": False, "error": "doc_id required"}), 400

        payload = {
            "source": source,
            "doc_id": doc_id,
            "comment": (data.get("comment") or "").strip()[:500],
        }
        if source == "chat":
            chat = data.get("chat") or {}
            payload["chat"] = {
                "question": chat.get("question", ""),
                "mode": chat.get("mode", "quick"),
                "correct_sources": chat.get("correct_sources") or [],
                "answer": chat.get("answer"),
                "candidate_a": chat.get("candidate_a"),
                "candidate_b": chat.get("candidate_b"),
                "reconciled": chat.get("reconciled"),
            }
        elif source == "attribution":
            table = data.get("table") or {}
            payload["table"] = {
                "column_name": table.get("column_name", ""),
                "correct_sources": table.get("correct_sources") or [],
                "candidate_a": table.get("candidate_a"),
                "candidate_b": table.get("candidate_b"),
                "reconciled": table.get("reconciled"),
                "reasoning": table.get("reasoning"),
            }

        if record_feedback(payload):
            return jsonify({"success": True, "message": "Feedback recorded"}), 200
        return jsonify({"success": False, "error": "Failed to save feedback"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/pdf', methods=['GET'])
def api_document_pdf(doc_id):
    """Serve the PDF file for a document (for viewer)."""
    doc_id = unquote(doc_id)
    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        return jsonify({"success": False, "error": "PDF not found"}), 404
    try:
        return send_from_directory(
            pdf_path.parent,
            pdf_path.name,
            mimetype="application/pdf",
            as_attachment=False,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/available', methods=['GET'])
def get_available_documents():
    """Get list of documents with existing extractions."""
    try:
        results_dir = PROJECT_ROOT / 'experiment-scripts' / 'baselines_file_search_results' / 'gemini_native'
        
        documents = []
        
        if results_dir.exists():
            # Look for extraction_metadata.json files
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    for doc_dir in model_dir.iterdir():
                        if doc_dir.is_dir():
                            extraction_file = doc_dir / 'extraction_metadata.json'
                            if extraction_file.exists():
                                documents.append({
                                    'id': f"{model_dir.name}/{doc_dir.name}",
                                    'name': doc_dir.name,
                                    'model': model_dir.name,
                                    'path': str(extraction_file)
                                })
        
        # Sort by document name
        documents.sort(key=lambda x: x['name'])
        
        return jsonify({
            "success": True,
            "documents": documents,
            "count": len(documents)
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/extraction', methods=['GET'])
def get_document_extraction(doc_id):
    """Get extraction data for a specific document."""
    try:
        results_dir = PROJECT_ROOT / 'experiment-scripts' / 'baselines_file_search_results' / 'gemini_native'
        extraction_file = results_dir / doc_id / 'extraction_metadata.json'
        
        if not extraction_file.exists():
            return jsonify({
                "success": False,
                "error": f"Extraction file not found for document: {doc_id}"
            }), 404
        
        # Load extraction metadata
        with open(extraction_file, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        # Transform to web interface format
        results = {}
        for col_name, col_data in extraction_data.items():
            # Try to extract page number from evidence or page field
            page = col_data.get("page", "N/A")
            if page == "Not applicable" or page == "N/A":
                # Try to parse from evidence if available
                evidence = col_data.get("evidence", "")
                if "page" in evidence.lower():
                    import re
                    page_match = re.search(r'page\s+(\d+)', evidence, re.IGNORECASE)
                    if page_match:
                        page = page_match.group(1)
                    else:
                        page = "Unknown"
                else:
                    page = "Unknown"
            
            # Determine modality from plan_source_type or evidence
            modality = col_data.get("plan_source_type", "unknown")
            if modality == "Not applicable" or modality == "unknown":
                evidence = col_data.get("evidence", "").lower()
                if "table" in evidence:
                    modality = "table"
                elif "figure" in evidence or "chart" in evidence:
                    modality = "figure"
                else:
                    modality = "text"
            
            results[col_name] = {
                "value": col_data.get("value", "not found"),
                "page_number": page,
                "modality": modality,
                "evidence": col_data.get("evidence", ""),
                "definition": ""  # Could load from definitions if needed
            }
        
        # Try to load summary metrics if available
        summary_file = extraction_file.parent / 'evaluation' / 'summary_metrics.json'
        summary_info = None
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_info = json.load(f)
        
        return jsonify({
            "success": True,
            "document_id": doc_id,
            "results": results,
            "total_columns": len(results),
            "summary": summary_info
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/extract/csv', methods=['POST'])
def extract_from_csv():
    """Extract columns from uploaded CSV file."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No CSV file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"success": False, "error": "Only CSV files are allowed"}), 400
    
    try:
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF-8"), newline=None)
        csv_reader = csv.DictReader(stream)
        
        # Validate CSV headers
        headers = csv_reader.fieldnames
        if not headers:
            return jsonify({"success": False, "error": "CSV file is empty"}), 400
        
        # Check for required columns (case-insensitive)
        headers_lower = [h.lower() for h in headers]
        has_column_name = 'column_name' in headers_lower or 'column name' in headers_lower
        has_definition = 'definition' in headers_lower
        
        if not (has_column_name and has_definition):
            return jsonify({
                "success": False,
                "error": "CSV must contain 'column_name' (or 'Column Name') and 'definition' (or 'Definition') columns"
            }), 400
        
        # Read all rows
        csv_data = list(csv_reader)
        
        if not csv_data:
            return jsonify({"success": False, "error": "CSV file contains no data rows"}), 400
        
        # Extract columns
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.extract_from_csv(csv_data)
        
        return jsonify(result), 200 if result.get("success") else 400
        
    except Exception as e:
        return jsonify({"success": False, "error": f"CSV processing failed: {str(e)}"}), 500


@app.route('/api/pdf/info', methods=['GET'])
def get_pdf_info():
    """Get information about the currently loaded PDF."""
    global current_pdf_info
    
    if not current_pdf_info:
        return jsonify({
            "success": False,
            "error": "No PDF loaded"
        }), 404
    
    return jsonify({
        "success": True,
        **current_pdf_info
    }), 200


@app.route('/api/export/<format>', methods=['POST'])
def export_results(format):
    """Export extraction results in various formats."""
    data = request.get_json()
    
    if not data or 'results' not in data:
        return jsonify({"success": False, "error": "No results to export"}), 400
    
    results = data['results']
    
    try:
        if format == 'json':
            return jsonify(results), 200
        
        elif format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Column Name', 'Value', 'Page Number', 'Modality', 'Evidence', 'Definition'])
            
            # Write rows
            for col_name, col_data in results.items():
                writer.writerow([
                    col_name,
                    col_data.get('value', ''),
                    col_data.get('page_number', ''),
                    col_data.get('modality', ''),
                    col_data.get('evidence', ''),
                    col_data.get('definition', '')
                ])
            
            output.seek(0)
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=extraction_results.csv'
            }
        
        else:
            return jsonify({"success": False, "error": f"Unsupported format: {format}"}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Export failed: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({"success": False, "error": "File is too large. Maximum size is 50MB"}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Clinical Trial Data Extraction - Web Interface")
    print("=" * 60)
    port = int(os.getenv("PORT", "8007"))
    print(f"\nServer starting at: http://127.0.0.1:{port}")
    print("\nFeatures:")
    print("  • Upload PDF files for extraction")
    print("  • Extract single column or all 133 columns")
    print("  • Upload CSV with custom queries")
    print("  • View results with evidence and location")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)
