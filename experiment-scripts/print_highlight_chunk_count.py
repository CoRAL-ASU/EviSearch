#!/usr/bin/env python3
"""
Print how many chunks we're highlighting per column.
Run: python experiment-scripts/print_highlight_chunk_count.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web.highlight_service import get_highlights_for_column
from web.comparison_service import load_comparison_data

DOC_ID = "NCT00268476_Attard_STAMPEDE_Lancet'23"

def main():
    data = load_comparison_data(DOC_ID)
    comparison = data.get("comparison", [])
    
    print(f"Document: {DOC_ID}")
    print(f"Total columns: {len(comparison)}")
    print("-" * 70)
    
    # Sample: first 15 columns + a few known ones
    sample_cols = [
        "Add-on Treatment",
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Treatment",
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Control",
        "Median PFS (mo) | Overall | Treatment",
        "Median OS (mo) | Overall | Treatment",
        "Total Participants - N",
    ]
    
    for row in comparison[:12]:
        c = row.get("column_name") if isinstance(row, dict) else None
        if c and c not in sample_cols:
            sample_cols.append(c)
    
    for col_name in sample_cols:
        if not col_name:
            continue
        r = get_highlights_for_column(DOC_ID, col_name)
        n = len(r.get("highlights", []))
        src = r.get("match_source", "?")
        short = (col_name[:55] + "…") if len(col_name) > 55 else col_name
        print(f"  {n:2d} chunks  [{src:10s}]  {short}")


if __name__ == "__main__":
    main()
