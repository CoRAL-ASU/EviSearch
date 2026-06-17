"""Result shape conversion helpers shared by routes and experiments."""
from __future__ import annotations

from typing import Any, Dict


def reconciliation_agent_to_columns(cols_dict: Dict[str, Any]) -> list:
    """Convert reconciliation_agent columns dict to list format for attribution enrichment."""
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
