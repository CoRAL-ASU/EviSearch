"""
explainability_service.py

Curates comparison data into a dashboard payload: highlights, core reasoning,
not-found-with-reasoning, agreement summary. Focus on interpretable content.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from web.comparison_service import load_comparison_data

# Column name patterns that are high priority (include in core_reasoning)
PRIORITY_PATTERNS = [
    "Treatment Arm 1 Regimen",
    "Treatment Arm(s)",
    "Control Arm",
    "Total Participants",
    "Treatment Arm - N",
    "Control Arm - N",
    "Primary Endpoint",
    "Secondary Endpoint",
    "Median OS",
    "Median PFS",
    "Adverse Events",
    "Add-on Treatment",
    "Class of Agent",
    "ID",
    "Trial",
]

# Values that mean "not found"
NOT_FOUND_VALUES = {"not found", "not reported", "not applicable", "—", "-", ""}


def _normalize_value(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    return s


def _is_not_found(v: Any) -> bool:
    s = _normalize_value(v)
    return s in NOT_FOUND_VALUES or not s


def _is_priority_column(col_name: str) -> bool:
    for p in PRIORITY_PATTERNS:
        if p.lower() in col_name.lower():
            return True
    return False


def _values_differ(row: Dict[str, Any], methods: List[str]) -> bool:
    """True if methods give different (non-empty) values."""
    vals: Set[str] = set()
    for m in methods:
        col = row.get("methods", {}).get(m)
        if not col:
            continue
        v = col.get("value") or col.get("primary_value")
        if v and not _is_not_found(v):
            vals.add(str(v).strip())
    return len(vals) > 1


def _has_multi_candidate(row: Dict[str, Any]) -> bool:
    for col in row.get("methods", {}).values():
        cands = col.get("candidates", [])
        if len(cands) > 1:
            return True
    return False


def _best_evidence(col: Dict[str, Any]) -> str:
    attr = col.get("attribution", {})
    if attr.get("evidence"):
        return str(attr["evidence"])
    cands = col.get("candidates", [])
    if cands and cands[0].get("evidence"):
        return str(cands[0]["evidence"])
    return ""


def _best_confidence(col: Dict[str, Any]) -> str:
    attr = col.get("attribution", {})
    if attr.get("confidence"):
        return str(attr["confidence"])
    cands = col.get("candidates", [])
    if cands and cands[0].get("confidence"):
        return str(cands[0]["confidence"])
    return "medium"


def _build_reasoning_block(
    row: Dict[str, Any],
    methods: List[str],
    method_labels: Dict[str, str],
) -> Dict[str, Any]:
    """Build one core_reasoning block from a comparison row."""
    col_name = row.get("column_name", "")
    group_name = row.get("group_name", "") or "Other"
    methods_data = row.get("methods", {})

    # Pick best value (first non-empty from any method)
    primary_value = ""
    best_evidence = ""
    best_confidence = "medium"
    best_source = ""
    why_not_found = ""
    where_we_looked = ""

    for m in methods:
        col = methods_data.get(m)
        if not col:
            continue
        ev = _best_evidence(col)
        if ev and len(ev) > len(best_evidence):
            best_evidence = ev
            best_confidence = _best_confidence(col)
        v = col.get("value") or col.get("primary_value")
        if v and not _is_not_found(v):
            primary_value = str(v)
            break
        # For not-found, use evidence as "where we looked" / "why"
        if ev and not primary_value:
            why_not_found = ev[:500] if len(ev) > 500 else ev
            if col.get("page") and col["page"] not in (-1, "Not applicable", "N/A"):
                where_we_looked = f"Page {col['page']}, {col.get('source_type', '')}"

    if not primary_value:
        primary_value = "Not reported"

    # Build source string
    for m in methods:
        col = methods_data.get(m)
        if col and col.get("page") and col["page"] not in (-1, "Not applicable", "N/A"):
            best_source = f"Page {col['page']}, {col.get('source_type', 'text')}"
            break

    by_method: Dict[str, Dict[str, str]] = {}
    for m in methods:
        col = methods_data.get(m)
        if not col:
            by_method[method_labels.get(m, m)] = {"value": "—", "evidence": ""}
            continue
        v = col.get("value") or col.get("primary_value") or "—"
        ev = _best_evidence(col)
        by_method[method_labels.get(m, m)] = {
            "value": str(v)[:200],
            "evidence": ev[:300] + "…" if len(ev) > 300 else ev,
        }

    reasoning: Dict[str, Any] = {
        "evidence": best_evidence[:800] + "…" if len(best_evidence) > 800 else best_evidence,
        "source": best_source,
        "confidence": best_confidence,
        "assumptions": None,
    }
    if _is_not_found(primary_value) or primary_value == "Not reported":
        reasoning["where_we_looked"] = where_we_looked or "Document text and tables"
        reasoning["why_not_found"] = why_not_found or "Value not explicitly reported"
    else:
        reasoning["how_we_got_it"] = "Extracted from " + (best_source or "document content")

    return {
        "column_name": col_name,
        "group_name": group_name,
        "primary_value": primary_value,
        "reasoning": reasoning,
        "by_method": by_method,
    }


def get_document_dashboard(pdf_stem: str) -> Dict[str, Any]:
    """
    Build dashboard payload: highlights, core_reasoning, not_found_with_reasoning,
    agreement_summary. Uses real comparison data.
    """
    data = load_comparison_data(pdf_stem)
    comparison = data.get("comparison", [])
    methods = data.get("methods_available", [])
    method_results = data.get("method_results", {})

    METHOD_LABELS = {
        "gemini_native": "Gemini",
        "landing_ai_baseline": "Landing AI",
        "pipeline": "Pipeline",
        "pipeline_plan_extract": "Pipeline (plan)",
        "pipeline_keywords": "Pipeline + KW",
    }

    highlights: List[Dict[str, Any]] = []
    core_reasoning: List[Dict[str, Any]] = []
    not_found_with_reasoning: List[Dict[str, Any]] = []
    agreement_summary: List[Dict[str, Any]] = []
    seen_in_core: Set[str] = set()

    # 1. Find disagreements
    for row in comparison:
        col_name = row.get("column_name", "")
        if _values_differ(row, methods):
            vals = {}
            for m in methods:
                col = row.get("methods", {}).get(m)
                if col:
                    v = col.get("value") or col.get("primary_value")
                    vals[METHOD_LABELS.get(m, m)] = str(v)[:100] if v else "—"
            highlights.append({
                "type": "disagreement",
                "column_name": col_name,
                "group_name": row.get("group_name", ""),
                "summary": "Methods report different values",
                "values_by_method": vals,
                "suggest_review": True,
            })
            if col_name not in seen_in_core:
                core_reasoning.append(_build_reasoning_block(row, methods, METHOD_LABELS))
                seen_in_core.add(col_name)

    # 2. Find multi-candidate
    for row in comparison:
        col_name = row.get("column_name", "")
        if col_name in seen_in_core:
            continue
        if _has_multi_candidate(row):
            cands = []
            for col in row.get("methods", {}).values():
                for c in col.get("candidates", [])[:3]:
                    cands.append({"value": c.get("value"), "evidence": (c.get("evidence") or "")[:200]})
            highlights.append({
                "type": "multi_candidate",
                "column_name": col_name,
                "group_name": row.get("group_name", ""),
                "summary": "Multiple plausible interpretations",
                "candidates": cands[:4],
            })
            if col_name not in seen_in_core:
                core_reasoning.append(_build_reasoning_block(row, methods, METHOD_LABELS))
                seen_in_core.add(col_name)

    # 3. Priority columns (found)
    for row in comparison:
        col_name = row.get("column_name", "")
        if col_name in seen_in_core:
            continue
        if not _is_priority_column(col_name):
            continue
        # Has at least one method with a found value
        has_found = False
        for col in row.get("methods", {}).values():
            if col.get("found") or (col.get("value") and not _is_not_found(col.get("value"))):
                has_found = True
                break
        if has_found:
            core_reasoning.append(_build_reasoning_block(row, methods, METHOD_LABELS))
            seen_in_core.add(col_name)

    # 4. Not found with good reasoning
    for row in comparison:
        col_name = row.get("column_name", "")
        if col_name in seen_in_core:
            continue
        all_not_found = True
        best_evidence = ""
        for col in row.get("methods", {}).values():
            if col.get("found") or (col.get("value") and not _is_not_found(col.get("value"))):
                all_not_found = False
                break
            ev = _best_evidence(col)
            if len(ev) > len(best_evidence):
                best_evidence = ev
        if all_not_found and best_evidence and len(best_evidence) > 50:
            not_found_with_reasoning.append({
                "column_name": col_name,
                "where_we_looked": "Document tables and text",
                "why": best_evidence[:400] + "…" if len(best_evidence) > 400 else best_evidence,
            })
            if col_name not in seen_in_core and _is_priority_column(col_name):
                core_reasoning.append(_build_reasoning_block(row, methods, METHOD_LABELS))
                seen_in_core.add(col_name)

    # 5. Agreement summary (all methods same value)
    for row in comparison:
        col_name = row.get("column_name", "")
        if col_name in seen_in_core:
            continue
        vals: Dict[str, str] = {}
        for m in methods:
            col = row.get("methods", {}).get(m)
            if col:
                v = col.get("value") or col.get("primary_value")
                vals[METHOD_LABELS.get(m, m)] = str(v)[:80] if v else "—"
        if len(vals) < 2:
            continue
        unique_vals = set(v for v in vals.values() if v and v != "—" and not _is_not_found(v))
        if len(unique_vals) == 1 and unique_vals:
            agreed = list(unique_vals)[0]
            ev = ""
            for col in row.get("methods", {}).values():
                ev = _best_evidence(col)
                if ev:
                    break
            agreement_summary.append({
                "column_name": col_name,
                "agreed_value": agreed,
                "methods_agreeing": list(vals.keys()),
                "evidence": (ev or "")[:200],
            })

    # Limit sizes
    core_reasoning = core_reasoning[:25]
    highlights = highlights[:10]
    not_found_with_reasoning = not_found_with_reasoning[:10]
    agreement_summary = agreement_summary[:5]

    return {
        "doc_id": pdf_stem,
        "methods_available": methods,
        "highlights": highlights,
        "core_reasoning": core_reasoning,
        "not_found_with_reasoning": not_found_with_reasoning,
        "agreement_summary": agreement_summary,
    }
