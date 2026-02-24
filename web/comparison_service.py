"""
comparison_service.py

Load and merge extraction results from multiple methods for display.
Read-only — no user confirmations or storage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths to scan for extraction results
RESULTS_PATHS = {
    "gemini_native": PROJECT_ROOT / "experiment-scripts" / "baselines_file_search_results" / "gemini_native",
    "landing_ai_baseline": PROJECT_ROOT / "experiment-scripts" / "baseline_landing_ai_w_gemini" / "results",
    "pipeline": PROJECT_ROOT / "new_pipeline_outputs" / "results",
}


def _normalize_gemini_result(col_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Gemini extraction_metadata format to unified shape."""
    value = data.get("value", "not found")
    evidence = data.get("evidence", "")
    return {
        "column_name": col_name,
        "group_name": data.get("group_name", ""),
        "value": value,
        "primary_value": value,
        "found": value not in ("not found", "Not reported", "Not applicable", ""),
        "page": data.get("page"),
        "source_type": data.get("plan_source_type") or "text",
        "candidates": [
            {
                "value": value,
                "evidence": evidence,
                "assumptions": None,
                "confidence": "medium",
            }
        ],
        "attribution": {
            "evidence": evidence,
            "sources": [],
            "confidence": "medium",
            "assumptions": None,
        },
    }


def _normalize_pipeline_result(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert extract_landing_ai extraction_results format to unified shape."""
    col_name = row.get("column_name", "")
    value = row.get("value") or row.get("primary_value", "")
    candidates = row.get("candidates", [])
    primary = row.get("primary_value") or (candidates[0].get("value") if candidates else "")
    evidence = candidates[0].get("evidence") if candidates else ""
    confidence = candidates[0].get("confidence", "low") if candidates else "low"
    return {
        "column_name": col_name,
        "group_name": row.get("group_name", ""),
        "value": value or primary,
        "primary_value": primary,
        "found": bool(row.get("found", False)),
        "page": row.get("page"),
        "source_type": row.get("source_type"),
        "retrieval_source": row.get("retrieval_source"),
        "chunk_count": row.get("chunk_count"),
        "candidates": candidates,
        "attribution": {
            "evidence": evidence,
            "sources": row.get("sources", []),
            "confidence": confidence,
            "assumptions": candidates[0].get("assumptions") if candidates else None,
        },
    }


def _normalize_plan_extract_result(col_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert plan_extract_columns extraction_results format to unified shape."""
    value = data.get("value", "")
    return {
        "column_name": col_name,
        "group_name": data.get("group_name", ""),
        "value": value,
        "primary_value": value,
        "found": bool(data.get("found", False)),
        "page": data.get("page"),
        "source_type": data.get("source_type"),
        "candidates": [
            {
                "value": value,
                "evidence": data.get("evidence"),
                "assumptions": None,
                "confidence": data.get("confidence", "medium"),
            }
        ],
        "attribution": {
            "evidence": data.get("evidence", ""),
            "sources": data.get("sources", []),
            "confidence": data.get("confidence", "medium"),
            "assumptions": None,
        },
    }


def _load_gemini_native(doc_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load Gemini native results. doc_id can be 'model/pdf_stem' or just pdf_stem."""
    base = RESULTS_PATHS["gemini_native"]
    if not base.exists():
        return None
    # Try model/pdf_stem or scan for pdf_stem in any model dir
    parts = doc_id.split("/", 1)
    if len(parts) == 2:
        model, pdf_stem = parts
        path = base / model / pdf_stem / "extraction_metadata.json"
    else:
        pdf_stem = doc_id
        path = None
        for model_dir in base.iterdir():
            if model_dir.is_dir():
                candidate = model_dir / pdf_stem / "extraction_metadata.json"
                if candidate.exists():
                    path = candidate
                    break
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: _normalize_gemini_result(k, v) for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return None


def _load_pipeline_extract_landing_ai(pdf_stem: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load extract_landing_ai results (group-wise pipeline)."""
    path = RESULTS_PATHS["pipeline"] / pdf_stem / "planning" / "extract_landing_ai" / "extraction_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", [])
        if isinstance(results, list):
            return {r["column_name"]: _normalize_pipeline_result(r) for r in results if r.get("column_name")}
        return {}
    except Exception:
        return None


def _expand_attribution_map(attribution_list: list, column_names: set) -> Dict[str, list]:
    """Inverted map (source -> columns) -> per-column attribution."""
    col_to_sources: Dict[str, list] = {c: [] for c in column_names}
    for src in attribution_list or []:
        if not isinstance(src, dict):
            continue
        cols = src.get("columns")
        if not isinstance(cols, list):
            continue
        src_copy = {k: v for k, v in src.items() if k != "columns"}
        for col in cols:
            if col in col_to_sources:
                col_to_sources[col].append(src_copy)
    return col_to_sources


def _load_agent(pdf_stem: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load agent extractor results. Expects {columns, attribution} format (modality->columns map)."""
    path = RESULTS_PATHS["pipeline"] / pdf_stem / "agent_extractor" / "extraction_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        columns = data.get("columns", {})
        if not isinstance(columns, dict):
            return None
        attribution_map = data.get("attribution")
        col_to_attr = _expand_attribution_map(attribution_map, set(columns)) if isinstance(attribution_map, list) else {}
        out = {}
        for col_name, col_data in columns.items():
            if isinstance(col_data, dict):
                v = col_data.get("value", "")
                evidence = (col_data.get("reasoning") or "").strip()
                attribution = col_to_attr.get(col_name, [])
                v = str(v) if v is not None else ""
            else:
                v = str(col_data) if col_data else ""
                evidence = ""
                attribution = col_to_attr.get(col_name, [])
            out[col_name] = {
                "column_name": col_name,
                "group_name": "",
                "value": v,
                "primary_value": v,
                "found": bool(v and v not in ("Not reported", "not found", "Not applicable", "")),
                "page": None,
                "source_type": "text",
                "evidence": evidence,
                "attribution_snippet": "",
                "attribution": attribution,
                "candidates": [{"value": v, "evidence": evidence, "assumptions": None, "confidence": "medium"}],
            }
        return out
    except Exception:
        return None


def _load_plan_extract(pdf_stem: str, with_keywords: bool) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load plan_extract_columns or plan_extract_columns_with_keywords results."""
    subdir = "plan_extract_columns_with_keywords" if with_keywords else "plan_extract_columns"
    path = RESULTS_PATHS["pipeline"] / pdf_stem / "planning" / subdir / "extraction_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", {})
        if isinstance(results, dict):
            return {k: _normalize_plan_extract_result(k, v) for k, v in results.items()}
        return {}
    except Exception:
        return None


def _load_landing_ai_baseline(pdf_stem: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load baseline_landing_ai_w_gemini results."""
    base = RESULTS_PATHS["landing_ai_baseline"]
    if not base.exists():
        return None
    # Structure: results/gemini-2.5-flash/pdf_stem/extraction_metadata.json
    for model_dir in base.iterdir():
        if model_dir.is_dir():
            path = model_dir / pdf_stem / "extraction_metadata.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return {k: _normalize_gemini_result(k, v) for k, v in data.items() if isinstance(v, dict)}
                except Exception:
                    pass
    return None


def get_document_status(pdf_stem: str) -> Dict[str, bool]:
    """Return which methods have results for this document."""
    return {
        "gemini_native": _load_gemini_native(pdf_stem) is not None,
        "landing_ai_baseline": _load_landing_ai_baseline(pdf_stem) is not None,
        "pipeline": _load_pipeline_extract_landing_ai(pdf_stem) is not None,
        "pipeline_plan_extract": _load_plan_extract(pdf_stem, with_keywords=False) is not None,
        "pipeline_keywords": _load_plan_extract(pdf_stem, with_keywords=True) is not None,
        "agent": _load_agent(pdf_stem) is not None,
    }


def load_comparison_data(pdf_stem: str) -> Dict[str, Any]:
    """
    Load and merge results from all available methods.
    Returns unified structure for comparison view.
    """
    methods: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_columns: set[str] = set()

    # Gemini native
    gemini = _load_gemini_native(pdf_stem)
    if gemini:
        methods["gemini_native"] = gemini
        all_columns.update(gemini.keys())

    # Landing AI baseline
    la_baseline = _load_landing_ai_baseline(pdf_stem)
    if la_baseline:
        methods["landing_ai_baseline"] = la_baseline
        all_columns.update(la_baseline.keys())

    # Pipeline (extract_landing_ai)
    pipeline = _load_pipeline_extract_landing_ai(pdf_stem)
    if pipeline:
        methods["pipeline"] = pipeline
        all_columns.update(pipeline.keys())

    # Pipeline plan_extract (no keywords)
    plan_extract = _load_plan_extract(pdf_stem, with_keywords=False)
    if plan_extract:
        methods["pipeline_plan_extract"] = plan_extract
        all_columns.update(plan_extract.keys())

    # Pipeline plan_extract with keywords
    plan_kw = _load_plan_extract(pdf_stem, with_keywords=True)
    if plan_kw:
        methods["pipeline_keywords"] = plan_kw
        all_columns.update(plan_kw.keys())

    # Agent extractor
    agent = _load_agent(pdf_stem)
    if agent:
        methods["agent"] = agent
        all_columns.update(agent.keys())

    # Build comparison rows: one per column, with values per method
    columns_sorted = sorted(all_columns)
    comparison_rows: List[Dict[str, Any]] = []
    for col_name in columns_sorted:
        row: Dict[str, Any] = {
            "column_name": col_name,
            "group_name": "",
            "methods": {},
        }
        for method_name, method_data in methods.items():
            col_data = method_data.get(col_name)
            if col_data:
                row["group_name"] = row["group_name"] or col_data.get("group_name", "")
                row["methods"][method_name] = col_data
        comparison_rows.append(row)

    # Group-wise view: group columns by group_name
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in comparison_rows:
        g = r.get("group_name") or "Other"
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(r)

    return {
        "pdf_stem": pdf_stem,
        "methods_available": list(methods.keys()),
        "status": get_document_status(pdf_stem),
        "total_columns": len(columns_sorted),
        "comparison": comparison_rows,
        "by_group": by_group,
        "method_results": methods,
    }


def list_documents() -> List[Dict[str, Any]]:
    """
    List all documents that have extraction results from any method.
    Returns list of {doc_id, pdf_stem, status, methods_available}.
    """
    seen: Dict[str, Dict[str, bool]] = {}
    pdf_stems: set[str] = set()

    # From new_pipeline_outputs/results
    pipeline_base = RESULTS_PATHS["pipeline"]
    if pipeline_base.exists():
        for doc_dir in pipeline_base.iterdir():
            if doc_dir.is_dir():
                pdf_stems.add(doc_dir.name)

    # From gemini_native
    gemini_base = RESULTS_PATHS["gemini_native"]
    if gemini_base.exists():
        for model_dir in gemini_base.iterdir():
            if model_dir.is_dir():
                for doc_dir in model_dir.iterdir():
                    if doc_dir.is_dir():
                        pdf_stems.add(doc_dir.name)

    # From landing_ai_baseline
    la_base = RESULTS_PATHS["landing_ai_baseline"]
    if la_base.exists():
        for model_dir in la_base.iterdir():
            if model_dir.is_dir():
                for doc_dir in model_dir.iterdir():
                    if doc_dir.is_dir():
                        pdf_stems.add(doc_dir.name)

    documents = []
    for stem in sorted(pdf_stems):
        status = get_document_status(stem)
        methods_available = [k for k, v in status.items() if v]
        if methods_available:
            documents.append({
                "doc_id": stem,
                "pdf_stem": stem,
                "status": status,
                "methods_available": methods_available,
            })
    return documents


def get_report(pdf_stem: str) -> Dict[str, Any]:
    """Generate document analysis report (summary stats)."""
    data = load_comparison_data(pdf_stem)
    methods = data.get("method_results", {})
    report: Dict[str, Any] = {
        "pdf_stem": pdf_stem,
        "methods_available": data.get("methods_available", []),
        "total_columns": data.get("total_columns", 0),
        "by_method": {},
    }
    for method_name, method_data in methods.items():
        found = sum(1 for c in method_data.values() if c.get("found"))
        report["by_method"][method_name] = {
            "total": len(method_data),
            "found": found,
            "not_found": len(method_data) - found,
        }
    return report
