"""
attribution_service.py

Attribution retrieval: attribution (agent) → numeric match → planner location.
No semantic/embedding fallback; attribution will be done agentically.
See web/attribution_matcher.py for Phase 0, 1, 2.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from web.highlight_service import _chunk_text, _landing_type_to_pipeline, load_landing_ai_parse
from web.attribution_matcher import (
    extract_numeric_parts_from_values,
    extract_column_tokens,
    phase0_attribution_match,
    phase1_numeric_match,
    phase2_planner_location,
    chunks_to_attribution_output,
)


def _chunk_text_clean(chunk: Dict) -> str:
    text = _chunk_text(chunk)
    return re.sub(r"<::[^>]*::>", "", text).strip()[:4000]


def _parse_pages_from_evidence(evidence: str) -> List[int]:
    """Extract page numbers from evidence text, e.g. 'page 1', 'Table 1 on page 5', 'pages 5-6'."""
    pages = []
    for m in re.finditer(r"page\s+(\d+)", evidence or "", re.I):
        pages.append(int(m.group(1)))
    for m in re.finditer(r"pages?\s+(\d+)\s*[-–]\s*(\d+)", evidence or "", re.I):
        for p in range(int(m.group(1)), int(m.group(2)) + 1):
            pages.append(p)
    return list(dict.fromkeys(pages))  # dedupe preserving order


def retrieve_chunks_for_evidence(
    doc_id: str,
    top_k: int = 3,
    column_name: Optional[str] = None,
    final_value: Optional[str] = None,
    pipeline_page: Optional[int] = None,
    pipeline_source_type: Optional[str] = None,
    evidence_text: Optional[str] = None,
    method_values: Optional[List[str]] = None,
    attribution_snippet: Optional[str] = None,
    attribution: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Strategy: ATTRIBUTION (when provided) → numeric match → planner location.
    - attribution: list of {source_type, page, snippet?, table_number?, figure_number?, caption?} from agent
    - attribution_snippet: legacy single snippet (converted to text source)
    - When pipeline has page/type: prefer chunks on that page and type
    Returns list of {chunk_id, page, source_type, snippet, score}.
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []

    chunks_raw = parse_data.get("chunks") or []
    chunks = [c for c in chunks_raw if c.get("id")]
    valid = []
    for c in chunks:
        t = _chunk_text_clean(c)
        if t and len(t) >= 10:
            valid.append(c)

    if not valid:
        return []

    # Location hints: pipeline page/type, plus pages parsed from evidence
    hint_pages = set()
    if pipeline_page is not None and pipeline_page >= 1:
        hint_pages.add(int(pipeline_page))
    if evidence_text:
        hint_pages.update(_parse_pages_from_evidence(evidence_text))

    # Collect value strings: final_value + method values (exclude "not found" etc.)
    all_value_strs = [final_value or ""]
    if method_values:
        for v in method_values:
            v = str(v or "").strip()
            if v and v.lower() not in ("not found", "not reported", "not applicable", "n/a"):
                all_value_strs.append(v)

    required_parts, all_parts, has_numeric = extract_numeric_parts_from_values(all_value_strs)
    col_tokens = extract_column_tokens(column_name or "")

    # Resolve page: prefer pipeline_page, fallback to first from evidence
    hint_page = None
    if pipeline_page is not None and pipeline_page >= 1:
        try:
            hint_page = int(pipeline_page)
        except (TypeError, ValueError):
            pass
    if hint_page is None and hint_pages:
        hint_page = min(hint_pages) if hint_pages else None

    # Phase 0: Structured attribution (agent) or legacy snippet
    attr_to_use = attribution if attribution and isinstance(attribution, list) else None
    if not attr_to_use and attribution_snippet and attribution_snippet.strip():
        attr_to_use = [{"source_type": "text", "page": 1, "snippet": attribution_snippet}]
    if attr_to_use:
        matched = phase0_attribution_match(
            valid,
            attr_to_use,
            _chunk_text_clean,
            _landing_type_to_pipeline,
            top_k=top_k,
        )
        if matched:
            return chunks_to_attribution_output(
                matched,
                _chunk_text_clean,
                _landing_type_to_pipeline,
                score_label=1.0,
            )

    # Phase 1: Numeric match — chunks containing ALL required numeric parts
    if has_numeric and required_parts:
        matched = phase1_numeric_match(
            valid,
            _chunk_text_clean,
            required_parts,
            all_parts,
            col_tokens,
            hint_page,
            pipeline_source_type,
            _landing_type_to_pipeline,
            top_k=top_k,
        )
        if matched:
            return chunks_to_attribution_output(
                matched,
                _chunk_text_clean,
                _landing_type_to_pipeline,
                score_label=1.0,
            )

    # Phase 2: Planner location — chunks on page with matching type
    if hint_page and hint_page >= 1 and pipeline_source_type:
        st = str(pipeline_source_type or "").lower()
        if st not in ("not applicable", "not_applicable", "n/a", "na"):
            matched = phase2_planner_location(
                valid,
                _chunk_text_clean,
                hint_page,
                pipeline_source_type,
                col_tokens,
                _landing_type_to_pipeline,
                top_k=top_k,
            )
            if matched:
                return chunks_to_attribution_output(
                    matched,
                    _chunk_text_clean,
                    _landing_type_to_pipeline,
                    score_label=0.9,  # planner-based
                )

    return []


def enrich_reconciled_with_attribution(
    doc_id: str,
    reconciled_columns: List[Dict[str, Any]],
    comparison_rows: Optional[List[Dict]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    For each reconciled column: collate evidence, run attribution/numeric/planner matching, add attributed_chunks.
    """
    col_to_row = {r.get("column_name"): r for r in (comparison_rows or [])}

    enriched = []
    for col in reconciled_columns:
        col_name = col.get("column_name", "")
        final_value = col.get("final_value", "")

        # Collate evidence from contributing methods
        evidences = []
        row = col_to_row.get(col_name)
        if row and row.get("methods"):
            for m in col.get("contributing_methods") or []:
                meth = row["methods"].get(m)
                if meth:
                    ev = meth.get("evidence") or (meth.get("attribution") or {}).get("evidence", "")
                    if ev:
                        evidences.append(str(ev)[:400])

        # Pipeline location hints (page, source_type) and evidence for parsing pages
        pipeline_page = col.get("page")
        pipeline_source_type = col.get("source_type")
        if pipeline_page is not None and str(pipeline_page).lower() in ("not applicable", "n/a", "na"):
            pipeline_page = None
        evidence_combined = " ".join(evidences) if evidences else ""

        # Collect method values and attribution (e.g. from agent)
        method_values = []
        col_attribution = None
        col_attribution_snippet = None
        if row and row.get("methods"):
            for m in col.get("contributing_methods") or []:
                meth = row["methods"].get(m)
                if meth:
                    val = meth.get("value") or meth.get("primary_value", "")
                    if val and str(val).strip():
                        method_values.append(str(val).strip())
                    attr = meth.get("attribution")
                    if attr and isinstance(attr, list) and len(attr) > 0:
                        col_attribution = attr
                        break
                    snip = (meth.get("attribution_snippet") or "").strip()
                    if snip and len(snip) >= 10:
                        col_attribution_snippet = snip

        chunks_out = retrieve_chunks_for_evidence(
            doc_id,
            top_k=top_k,
            column_name=col_name,
            final_value=final_value,
            pipeline_page=pipeline_page,
            pipeline_source_type=pipeline_source_type,
            evidence_text=evidence_combined,
            method_values=method_values if method_values else None,
            attribution=col_attribution,
            attribution_snippet=col_attribution_snippet,
        )

        out = dict(col)
        out["attributed_chunks"] = chunks_out
        out["chunk_ids"] = [c["chunk_id"] for c in chunks_out]
        enriched.append(out)

    return enriched
