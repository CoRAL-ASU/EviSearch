#!/usr/bin/env python3
"""
main_app.py

Modern web interface for Clinical Trial Data Extraction.
Provides endpoints for PDF upload, query submission, and result retrieval.

Run from project root: python web/main_app.py
Then open http://127.0.0.1:8007
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evisearch.services.reports import (
    get_document_status,
    load_comparison_data,
    get_report,
    list_documents,
)
from src.evisearch.services.highlight import (
    get_highlights_by_chunk_ids,
    resolve_pdf_path,
)
from src.evisearch.services.feedback import record_feedback
from src.evisearch.services.result_transform import reconciliation_agent_to_columns
from src.config.runtime_paths import (
    DATASET_DIR,
    RESULTS_ROOT,
    UPLOADS_DIR,
    ensure_runtime_dirs,
)
from src.documents.pdf_registry import (
    get_registered_document,
    register_uploaded_pdf,
    resolve_canonical_doc_id,
)

FRONTEND_ROOT = PROJECT_ROOT / "apps" / "web" / "frontend"
app = Flask(
    __name__,
    template_folder=str(FRONTEND_ROOT / "templates"),
    static_folder=str(FRONTEND_ROOT / "static"),
)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ensure_runtime_dirs()
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['BOOT_ID'] = str(uuid.uuid4())  # Changes on each app restart; used to invalidate browser session

# the pages these blueprints serve: workspace_routes owns /tables, /review, /knowledge, /learning, /benchmark and the
# redirects from the old page URLs (/schema, /attribution, /extract, /comparison-report, /feedback)
from web.schema_routes import bp as schema_layer_bp  # noqa: E402  (schema generation, conventions, feedback log)
from web.workspace_routes import bp as workspace_bp  # noqa: E402  (table workspace, runs, reviews per run, jobs)

app.register_blueprint(schema_layer_bp)
app.register_blueprint(workspace_bp)

def _canonical_doc_id(doc_id: str) -> str:
    return resolve_canonical_doc_id(
        doc_id,
        uploads_dir=app.config["UPLOAD_FOLDER"],
        results_root=RESULTS_ROOT,
        dataset_dir=DATASET_DIR,
    )


def _document_runtime_state(doc_id: str) -> Dict[str, Any]:
    from src.retrieval.embedding_retriever import has_embedding_cache

    canonical_doc_id = _canonical_doc_id(doc_id)
    pdf_path = resolve_pdf_path(canonical_doc_id)
    chunk_dir = RESULTS_ROOT / canonical_doc_id / "chunking"
    md_path = chunk_dir / "parsed_markdown.md"
    json_path = chunk_dir / "landing_ai_parse_output.json"

    has_cached_parse = False
    if json_path.exists():
        if pdf_path and pdf_path.exists():
            has_cached_parse = json_path.stat().st_mtime >= pdf_path.stat().st_mtime
        else:
            has_cached_parse = True

    has_cached_embeddings = has_embedding_cache(canonical_doc_id)
    return {
        "canonical_doc_id": canonical_doc_id,
        "pdf_exists": bool(pdf_path and pdf_path.exists()),
        "has_parsed_markdown": md_path.exists(),
        "has_parse_json": json_path.exists(),
        "has_cached_parse": has_cached_parse,
        "has_cached_embeddings": has_cached_embeddings,
        "is_prepared": has_cached_parse and has_cached_embeddings,
    }


def _normalize_source_attribution(raw_attribution: Any) -> list[Dict[str, Any]]:
    """Normalize agent/search attribution to the resolver source shape."""
    if not isinstance(raw_attribution, list):
        return []

    out: list[Dict[str, Any]] = []
    for item in raw_attribution:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page")) if item.get("page") is not None else None
        except (TypeError, ValueError):
            page = None
        if page is None or page < 1:
            continue

        source_type = str(item.get("source_type") or item.get("modality") or "text").lower()
        if source_type not in ("text", "table", "figure"):
            source_type = "text"

        normalized: Dict[str, Any] = {"page": page, "source_type": source_type}
        snippet = item.get("snippet") or item.get("verbatim_quote")
        if snippet:
            normalized["snippet"] = str(snippet)
        for key in ("table_number", "figure_number", "caption"):
            if item.get(key):
                normalized[key] = str(item[key])
        out.append(normalized)
    return out


def _resolve_candidate_chunk_ids(doc_id: str, column_name: str, col_data: Any) -> list[str]:
    """Resolve raw A/B source attribution to real LandingAI highlight chunk IDs."""
    if not isinstance(col_data, dict):
        return []

    attribution = _normalize_source_attribution(col_data.get("attribution"))
    if not attribution:
        return []

    first = attribution[0]
    value = str(col_data.get("value") or col_data.get("primary_value") or "")
    reasoning = str(col_data.get("reasoning") or "").strip()

    try:
        from src.evisearch.services.attribution import retrieve_chunks_for_evidence

        chunks = retrieve_chunks_for_evidence(
            doc_id=doc_id,
            column_name=column_name,
            final_value=value,
            pipeline_page=first.get("page"),
            pipeline_source_type=first.get("source_type"),
            evidence_text=reasoning,
            attribution=attribution,
            top_k=2,
        )
    except Exception:
        return []

    return [c["chunk_id"] for c in chunks if isinstance(c, dict) and c.get("chunk_id")]


def _document_option_payload(doc_id: str, name: str, source: str, has_extraction: bool) -> Dict[str, Any]:
    state = _document_runtime_state(doc_id)
    registered = get_registered_document(state["canonical_doc_id"], results_root=RESULTS_ROOT) or {}
    upload_aliases = registered.get("upload_aliases") or []
    return {
        "id": state["canonical_doc_id"],
        "canonical_id": state["canonical_doc_id"],
        "name": str(registered.get("display_name") or name),
        "source": str(registered.get("source") or source),
        "has_extraction": has_extraction,
        "has_cached_parse": state["has_cached_parse"],
        "has_cached_embeddings": state["has_cached_embeddings"],
        "is_prepared": state["is_prepared"],
        "upload_count": len(upload_aliases),
    }


@app.route('/')
def index():
    """Serve the home page with Ask a question and Extract full table cards."""
    return render_template('home.html')


@app.route('/qa')
def qa_page():
    """Serve the Ask a question page (single-query QA chatbot). Placeholder for now."""
    return render_template('qa.html')


@app.route('/api/report/method-comparison', methods=['GET'])
def api_method_comparison_report():
    """Return row-per-document-column method comparison data."""
    default_methods = [
        "agent",
        "search_agent",
        "reconciliation_agent",
        "gemini_native",
        "landing_ai_baseline",
        "landing_ai_baseline_gpt4",
    ]
    method_labels = {
        "agent": "Agent extractor",
        "search_agent": "Search agent",
        "reconciliation_agent": "Reconciliation",
        "gemini_native": "Gemini file search",
        "landing_ai_baseline": "LandingAI + Gemini",
        "landing_ai_baseline_gpt4": "LandingAI + GPT-4",
    }
    requested = request.args.get("methods", "").strip()
    methods = [m.strip() for m in requested.split(",") if m.strip()] if requested else default_methods
    methods = [m for m in methods if m in default_methods]
    if not methods:
        methods = default_methods

    doc_query = request.args.get("docs", "").strip()
    if doc_query:
        doc_ids = [unquote(d.strip()) for d in doc_query.split(",") if d.strip()]
    else:
        doc_ids = [d["doc_id"] for d in list_documents()]

    rows = []
    documents = []
    columns_seen = set()
    methods_available = set()
    empty_values = {"", "not reported", "not found", "n/a", "not applicable", "-", "--"}

    for doc_id in sorted(dict.fromkeys(doc_ids)):
        comparison = load_comparison_data(doc_id)
        available = [m for m in comparison.get("methods_available", []) if m in methods]
        if not available:
            continue
        documents.append(doc_id)
        methods_available.update(available)
        for item in comparison.get("comparison", []):
            item_methods = item.get("methods") or {}
            if not any(m in item_methods for m in methods):
                continue
            row = {
                "doc_id": doc_id,
                "column_name": item.get("column_name", ""),
                "group_name": item.get("group_name", "") or "Other",
            }
            for method in methods:
                method_data = item_methods.get(method) or {}
                value = method_data.get("value") if isinstance(method_data, dict) else ""
                found = bool(method_data.get("found")) if isinstance(method_data, dict) else False
                evidence = (method_data.get("evidence") or method_data.get("attribution_snippet") or "") if isinstance(method_data, dict) else ""
                value_str = str(value) if value is not None else ""
                row[method] = value_str
                row[f"{method}__found"] = found and value_str.strip().lower() not in empty_values
                row[f"{method}__evidence"] = str(evidence or "")
            rows.append(row)
            if row["column_name"]:
                columns_seen.add(row["column_name"])

    return jsonify({
        "success": True,
        "documents": documents,
        "document_count": len(documents),
        "methods": methods,
        "methods_available": [m for m in methods if m in methods_available],
        "method_labels": method_labels,
        "columns": sorted(columns_seen),
        "rows": rows,
        "row_count": len(rows),
    }), 200


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
    doc_cols: Dict[str, Dict[str, Any]] = {}

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

        pdf_path = resolve_pdf_path(doc_id)
        row: Dict[str, Any] = {"doc_id": doc_id, "pdf_exists": bool(pdf_path and pdf_path.exists())}
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
    canonical_doc_id = _canonical_doc_id(doc_id)
    if canonical_doc_id != doc_id:
        pdf_path = resolve_pdf_path(canonical_doc_id)
        if pdf_path and pdf_path.exists():
            return None
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
        from src.evisearch.pipelines.reconciliation_pipeline import run_reconciliation_pipeline
        result = run_reconciliation_pipeline(
            doc_id=doc_id,
            group_names=group_names,
            resume=not no_resume,
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
        search_path = base / "search_agent" / "extraction_results.json"
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
    resume = bool(data.get("resume", True)) and not data.get("no_resume", False)

    def generate():
        import queue
        import threading

        from src.evisearch.pipelines.unified_extraction import run_unified_extraction

        q = queue.Queue()

        def run():
            try:
                run_unified_extraction(
                    doc_id=doc_id,
                    group_names=groups_filter,
                    resume=resume,
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


@app.route('/api/documents/reconciled', methods=['GET'])
def api_list_reconciled_documents():
    """List document IDs that have reconciled results or agent extraction (for attribution); ?run= lists the documents
    that run extracted."""
    if not RESULTS_ROOT.exists():
        return jsonify({"success": True, "documents": []}), 200
    docs = set()
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        try:
            base = _results_base(d.name)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        if (base / "reconciliation_agent" / "reconciled_results.json").exists() or (base / "agent_extractor" / "extraction_results.json").exists():
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
            docs[doc_id] = _document_option_payload(
                doc_id=doc_id,
                name=doc_id,
                source="dataset",
                has_extraction=_has_extraction(doc_id),
            )
        for p in DATASET_DIR.glob("**/*.pdf"):
            if p.parent == DATASET_DIR:
                continue  # already covered by *.pdf
            # e.g. dataset/subdir/foo.pdf -> doc_id = subdir/foo or just stem
            rel = p.relative_to(DATASET_DIR)
            doc_id = str(rel.with_suffix("")).replace("/", "_")
            if doc_id not in docs:
                docs[doc_id] = _document_option_payload(
                    doc_id=doc_id,
                    name=p.stem,
                    source="dataset",
                    has_extraction=_has_extraction(doc_id),
                )

    # 2. Extracted docs and canonical uploaded docs from results.
    if RESULTS_ROOT.exists():
        for d in RESULTS_ROOT.iterdir():
            if not d.is_dir() or d.name.startswith("_"):
                continue
            has_ext = (
                (d / "reconciliation_agent" / "reconciled_results.json").exists()
                or (d / "agent_extractor" / "extraction_results.json").exists()
            )
            has_pdf = (d / f"{d.name}.pdf").exists() or any(d.glob("*.pdf"))
            if not has_ext and not has_pdf:
                continue
            payload = _document_option_payload(
                doc_id=d.name,
                name=d.name,
                source="extracted" if has_ext else "upload",
                has_extraction=has_ext,
            )
            docs[payload["id"]] = payload

    # 3. Legacy uploads (web/uploads/upload_*.pdf) that do not yet resolve to a canonical doc.
    upload_folder = app.config["UPLOAD_FOLDER"]
    if upload_folder.exists():
        for p in upload_folder.glob("upload_*.pdf"):
            doc_id = p.stem
            canonical_doc_id = _canonical_doc_id(doc_id)
            if canonical_doc_id != doc_id:
                continue
            docs[doc_id] = _document_option_payload(
                doc_id=doc_id,
                name=f"{doc_id} (uploaded)",
                source="upload",
                has_extraction=_has_extraction(doc_id),
            )

    out = sorted(docs.values(), key=lambda x: (x["name"].lower(), x["id"]))
    return jsonify({"success": True, "documents": out}), 200


def _results_base(doc_id: str) -> Path:
    """Where a document's stage outputs live: runs/<run> for ?run=<run> (the runs a schema's extractions write), else the
    document's top-level stage folders."""
    run = (request.args.get("run") or "").strip()
    if not run:
        return RESULTS_ROOT / doc_id
    if "/" in run or "\\" in run or run.startswith("."):
        raise ValueError(f"bad run name {run!r}")
    return RESULTS_ROOT / doc_id / "runs" / run


@app.route('/api/documents/<path:doc_id>/attribution/refresh', methods=['POST'])
def api_refresh_attribution(doc_id):
    """Re-run attribution and save. Uses reconciliation_agent if present, else agent-only (?run= for a named run)."""
    doc_id = unquote(doc_id)
    base = _results_base(doc_id)
    rec_path = base / "reconciliation_agent" / "reconciled_results.json"
    agent_path = base / "agent_extractor" / "extraction_results.json"
    try:
        from src.evisearch.services.reports import load_comparison_data
        from src.evisearch.services.attribution import enrich_reconciled_with_attribution
        comparison = load_comparison_data(doc_id)
        rows = comparison.get("comparison") or []

        if rec_path.exists():
            data = json.loads(rec_path.read_text(encoding="utf-8"))
            columns = reconciliation_agent_to_columns(data.get("columns") or {})
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
            out_path = base / "agent_extractor" / "attribution_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            return jsonify({"success": False, "error": f"No reconciled or agent results for {doc_id}"}), 404

        enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)
        out_path.write_text(json.dumps({"doc_id": doc_id, "columns": enriched}, indent=2), encoding="utf-8")
        # Reuse the reconciled payload builder so refresh returns the same
        # frontend contract as the initial page load.
        return api_document_reconciled(doc_id)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/reconciled', methods=['GET'])
def api_document_reconciled(doc_id):
    """Get reconciled results with attributed chunks. Uses reconciliation_agent, falls back to agent-only.
    ?run=<run> reads that run's outputs (runs/<run>/...) instead of the document's top-level ones."""
    doc_id = unquote(doc_id)
    try:
        base = _results_base(doc_id)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    run = request.args.get("run") or None
    rec_path = base / RECON_AGENT_DIR / "reconciled_results.json"
    recon_attr_path = base / RECON_AGENT_DIR / "attribution_results.json"
    agent_path = base / "agent_extractor" / "extraction_results.json"
    agent_attr_path = base / "agent_extractor" / "attribution_results.json"

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
            columns = reconciliation_agent_to_columns(data.get("columns") or {})
            comparison = load_comparison_data(doc_id)
            rows = comparison.get("comparison") or []
            from src.evisearch.services.attribution import enrich_reconciled_with_attribution
            enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)
            data["columns"] = enriched
        elif agent_attr_path.exists():
            data = json.loads(agent_attr_path.read_text(encoding="utf-8"))
        elif agent_path.exists() and not run:
            data = _build_agent_attribution(doc_id)
            if not data:
                return jsonify({"success": False, "error": "Could not build agent attribution"}), 500
        else:
            return jsonify({"success": False, "error": f"No reconciled or agent results for {doc_id}" + (f" in run {run}" if run else "")}), 404

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
                        agent_chunk_ids[k] = _resolve_candidate_chunk_ids(doc_id, k, v)
            except Exception:
                pass
        search_path = base / "search_agent" / "extraction_results.json"
        if search_path.exists():
            try:
                sd = json.loads(search_path.read_text(encoding="utf-8"))
                for k, v in (sd.get("columns") or {}).items():
                    search_cols[k] = str(v.get("value", "")) if isinstance(v, dict) else str(v or "")
                    if isinstance(v, dict):
                        search_chunk_ids[k] = _resolve_candidate_chunk_ids(doc_id, k, v)
            except Exception:
                pass

        columns = data.get("columns") or []

        human_edited = {}
        if run:
            # a run's view shows only that run's reviews (stored per run in the feedback log), never another run's
            from src.evisearch.services import reviews as reviews_service
            for cn, rv in reviews_service.cell_reviews(doc_id, run).items():
                if rv.get("value") is not None:
                    human_edited[cn] = {"value": rv["value"], "reason": rv.get("reason"), "event_id": rv.get("event_id")}
        else:
            human_edited_path = RESULTS_ROOT / doc_id / "human-edited" / "human_edited_results.json"
            if human_edited_path.exists():
                try:
                    he_data = json.loads(human_edited_path.read_text(encoding="utf-8"))
                    human_edited = (he_data.get("columns") or {}) if isinstance(he_data, dict) else {}
                except Exception:
                    pass
        for col in columns:
            cn = col.get("column_name", "")
            he_col = human_edited.get(cn)
            col["machine_value"] = col.get("final_value", "")
            if he_col and isinstance(he_col, dict) and he_col.get("value") is not None:
                col["final_value"] = str(he_col.get("value", ""))
                col["human_edited"] = True
                col["human_reason"] = str(he_col.get("reason") or "")
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
        data["run"] = run

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

    by = str(body.get("by") or "")
    run = str(body.get("run") or "").strip() or None
    machine: Dict[str, str] = {}
    if run:
        from src.evisearch.services import runs as runs_service
        try:
            machine = {c: cell["value"] for c, cell in runs_service.final_cells(doc_id, run).items()}
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
    for cn, v in columns.items():
        if not cn or not isinstance(v, dict):
            continue
        val = v.get("value")
        previous = existing_cols.get(str(cn), {}).get("value", v.get("previous_value"))
        entry = {"value": str(val) if val is not None else "", "human_edited": True, "reason": str(v.get("reason") or ""),
                 "note": str(v.get("note") or ""), "previous_value": previous, "by": by,
                 "edited_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        existing_cols[str(cn)] = entry
        # every correction is also an append-only feedback event (shown on /feedback; source of proposed conventions);
        # with a run it also carries that run's own machine value, which is what per-run review views compare against
        event = {"source": "correction", "event": "cell_correct", "doc_id": doc_id, "column": str(cn),
                 "run": run, "schema_id": body.get("schema_id"), "by": by, "before": previous,
                 "after": entry["value"], "reason": entry["reason"], "note": entry["note"]}
        if run and str(cn) in machine:
            from src.evisearch.services.reviews import state_of
            event.update(machine_value=machine[str(cn)], state=state_of(entry["value"], machine[str(cn)]))
        record_feedback(event)

    data = {"doc_id": doc_id, "columns": existing_cols}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"success": True, "doc_id": doc_id}), 200


RECON_AGENT_DIR = "reconciliation_agent"


def _build_agent_attribution(doc_id: str) -> Dict[str, Any] | None:
    """Build attribution columns from agent extraction, run enrich, return."""
    from src.evisearch.services.attribution import enrich_reconciled_with_attribution

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
    """Upload a PDF for extraction. Returns the doc_id used by /api/qa/prepare-document and /api/extract/unified/stream."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files allowed"}), 400
    try:
        pdf_bytes = file.read()
        upload_info = register_uploaded_pdf(
            pdf_bytes,
            original_filename=file.filename,
            uploads_dir=app.config["UPLOAD_FOLDER"],
            results_root=RESULTS_ROOT,
            dataset_dir=DATASET_DIR,
        )
        state = _document_runtime_state(upload_info["canonical_doc_id"])
        return jsonify({
            "success": True,
            "doc_id": upload_info["canonical_doc_id"],
            "canonical_doc_id": upload_info["canonical_doc_id"],
            "upload_doc_id": upload_info["upload_doc_id"],
            "filename": file.filename,
            "display_name": upload_info["display_name"],
            "source": upload_info["source"],
            "sha256": upload_info["sha256"],
            "reused_existing_doc": upload_info["reused_existing_doc"],
            "has_cached_parse": state["has_cached_parse"],
            "has_cached_embeddings": state["has_cached_embeddings"],
            "is_prepared": state["is_prepared"],
        }), 200
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
    requested_doc_id = (data.get("doc_id") or "").strip()
    doc_id = _canonical_doc_id(requested_doc_id)
    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400

    err = _ensure_pdf_for_extraction(requested_doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        return jsonify({"success": False, "error": f"PDF not found for {doc_id}"}), 400

    chunk_dir = RESULTS_ROOT / doc_id / "chunking"
    md_path = chunk_dir / "parsed_markdown.md"
    json_path = chunk_dir / "landing_ai_parse_output.json"

    def generate():
        from src.evisearch.services.preparation import parse_pdf_for_qa
        from src.retrieval.embedding_retriever import embed_chunks, has_embedding_cache

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
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'parsing_done', 'message': 'Using cached parse', 'cached': True})}\n\n"

        if has_embedding_cache(doc_id):
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'embedding_done', 'message': 'Using cached embeddings', 'cached': True})}\n\n"
        else:
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
        state = _document_runtime_state(doc_id)
        yield f"data: {json.dumps({'type': 'ready', 'doc_id': doc_id, 'canonical_doc_id': doc_id, 'has_cached_parse': state['has_cached_parse'], 'has_cached_embeddings': state['has_cached_embeddings']})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route('/api/qa/ask', methods=['POST'])
def api_qa_ask():
    """QA: Quick mode (Gemini chat) or Full mode (Agent + Search + Reconcile)."""
    data = request.get_json() or {}
    requested_doc_id = (data.get("doc_id") or "").strip()
    doc_id = _canonical_doc_id(requested_doc_id)
    question = (data.get("question") or "").strip()
    history = data.get("history") or []
    mode = (data.get("mode") or "full").strip().lower()
    if mode not in ("quick", "full"):
        mode = "full"

    if not doc_id:
        return jsonify({"success": False, "error": "doc_id required"}), 400
    if not question:
        return jsonify({"success": False, "error": "question required"}), 400

    err = _ensure_pdf_for_extraction(requested_doc_id)
    if err:
        return jsonify({"success": False, "error": err}), 400

    if mode == "quick":
        return _api_qa_ask_quick(doc_id, question, history)

    return _api_qa_ask_full(doc_id, question, history)


def _api_qa_ask_quick(doc_id: str, question: str, history: list):
    """Quick mode: one call to the qa model with the whole document, given exactly as Arm A gets it (parsed text,
    plus page images when the model reads images). No attribution."""
    from src.retrieval.embedding_retriever import parsed_markdown_path

    pdf_path = resolve_pdf_path(doc_id)
    if not parsed_markdown_path(doc_id).exists():
        return jsonify({"success": False, "error": f"Parsed markdown not found for {doc_id}; prepare the document first."}), 400

    def generate():
        from src.config.config import MAX_TOKENS, SELECTION
        from src.evisearch.services.pdf_query import build_document_input, document_token_budget
        from src.inference import Message, get_chat

        try:
            chat = get_chat("qa")
            with_images = SELECTION.option("pdf_query_input") == "markdown_images" and chat.capabilities.images and pdf_path and pdf_path.exists()
            budget = document_token_budget(chat.spec.context_tokens, question + json.dumps(history), MAX_TOKENS["qa"])
            document = build_document_input(doc_id, "markdown_images" if with_images else "markdown", budget).parts
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

        prompt = f"""You are answering questions about this clinical trial research paper. Use only the document to answer. Be concise and cite specific values when possible.{context_block}

Current question: {question}

Answer:"""

        try:
            result = chat.chat([Message.user(*document, prompt)], temperature=0.2, max_tokens=MAX_TOKENS["qa"])
            yield f"data: {json.dumps({'type': 'done', 'mode': 'quick', 'answer': result.text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _api_qa_ask_full(doc_id: str, question: str, history: list):
    """Full mode: Arm A (pdf_query) + Arm B (search agent) + reconciliation, with attribution."""
    def generate():
        import threading
        from src.evisearch.services.pdf_query import run_pdf_query
        from src.evisearch.services.qa import build_definition_with_context
        from src.evisearch.services.search import run_search_agent

        col_name = "qa_query"
        definition = build_definition_with_context(question, history)
        batch = [{"column_name": col_name, "definition": definition}]
        definitions_map = {col_name: definition}

        agent_result = {}
        search_result = {}

        def run_agent():
            nonlocal agent_result
            try:
                agent_result, _ = run_pdf_query(doc_id, batch)
            except Exception as e:
                agent_result = {col_name: {"value": "Not reported", "reasoning": str(e), "found": False, "attribution": []}}

        def run_search():
            nonlocal search_result
            try:
                search_result, _ = run_search_agent(doc_id, batch, definitions_map, log_path=None)
            except Exception as e:
                search_result = {col_name: {"value": "Not reported", "reasoning": str(e), "found": False, "attribution": []}}

        yield f"data: {json.dumps({'type': 'stage', 'stage': 'direct_pdf', 'message': 'Extracting from the full document…'})}\n\n"
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
        from src.evisearch.pipelines.reconciliation_pipeline import arbiter_module

        rec_result, _ = arbiter_module().run_reconciliation_agent(
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
        from src.evisearch.services.attribution import resolve_chunks_from_reconciled_source, retrieve_chunks_for_evidence

        if isinstance(rec_source, dict) and rec_source.get("page"):
            raw = resolve_chunks_from_reconciled_source(
                doc_id,
                page=rec_source.get("page"),
                modality=rec_source.get("modality", "text"),
                verbatim_quote=verbatim,
                value=rec_val,
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
                top_k=2,
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
    doc_id = _canonical_doc_id(unquote(doc_id))
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
    doc_id = _canonical_doc_id(unquote(doc_id))
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
    from src.config.config import SELECTION
    print(f"\nInference preset: {SELECTION.preset}  (python -m src.config shows the models in use)")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)
