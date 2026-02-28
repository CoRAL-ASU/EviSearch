#!/usr/bin/env python3
"""
Analyze columns with values but no chunks.
Runs the full enrich flow to see if attribution matching fails even when agent provides sources.

Usage: python experiment-scripts/analyze_attribution_gaps.py [doc_id]
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"


def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "NCT00309985_Kriayako_CHAARTED_JCO'18"

    from web.comparison_service import load_comparison_data
    from web.attribution_service import enrich_reconciled_with_attribution

    # Build agent attribution (same as _build_agent_attribution in main_app)
    comparison = load_comparison_data(doc_id)
    rows = comparison.get("comparison") or []
    agent_cols = [r for r in rows if (r.get("methods") or {}).get("agent")]
    if not agent_cols:
        print(f"No agent columns for {doc_id}")
        return

    columns = []
    for r in agent_cols:
        agent_data = (r.get("methods") or {}).get("agent") or {}
        val = agent_data.get("value") or agent_data.get("primary_value", "")
        columns.append({
            "column_name": r["column_name"],
            "final_value": str(val) if val else "",
            "contributing_methods": ["agent"],
        })

    # Check agent attribution per column (from comparison row)
    col_to_row = {r.get("column_name"): r for r in rows}
    with_attr = []
    without_attr = []
    for col in columns:
        row = col_to_row.get(col["column_name"], {})
        meth = (row.get("methods") or {}).get("agent") or {}
        attr = meth.get("attribution") or []
        if attr:
            with_attr.append((col["column_name"], len(attr)))
        else:
            without_attr.append(col["column_name"])

    print(f"\n=== Agent attribution (from comparison) ===")
    print(f"Columns with attribution: {len(with_attr)}")
    print(f"Columns WITHOUT attribution: {len(without_attr)}")
    if without_attr:
        print("  Sample (no attribution):", without_attr[:8])

    # Run enrich
    enriched = enrich_reconciled_with_attribution(doc_id, columns, comparison_rows=rows, top_k=3)

    # Analyze results
    with_chunks = []
    with_val_no_chunks = []
    with_attr_no_chunks = []  # Had attribution but still no chunks

    for c in enriched:
        val = (c.get("final_value") or "").strip()
        has_val = val and val.lower() not in ("not found", "not reported", "not applicable", "n/a")
        ids = c.get("attributed_chunks") or c.get("chunk_ids") or []
        row = col_to_row.get(c["column_name"], {})
        meth = (row.get("methods") or {}).get("agent") or {}
        had_attr = bool(meth.get("attribution"))

        if ids:
            with_chunks.append(c["column_name"])
        if has_val and not ids:
            with_val_no_chunks.append(c["column_name"])
            if had_attr:
                with_attr_no_chunks.append(c["column_name"])

    print(f"\n=== After enrich_reconciled_with_attribution ===")
    print(f"Columns with chunks: {len(with_chunks)}")
    print(f"Columns with value but NO chunks: {len(with_val_no_chunks)}")
    print(f"  Of those, had agent attribution but still no chunks: {len(with_attr_no_chunks)}")
    if with_attr_no_chunks:
        print("  Sample (had attribution, no chunks):", with_attr_no_chunks[:10])


if __name__ == "__main__":
    main()
