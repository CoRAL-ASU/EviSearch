#!/usr/bin/env python3
"""
Print max value-match counts per column for each document.
Useful for assessing impact of highlighting ALL matched chunks (vs. just 1).

Run: python experiment-scripts/print_max_value_matches_per_doc.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web.highlight_service import count_value_matches_for_column
from web.comparison_service import load_comparison_data, list_documents


def main():
    docs = list_documents()
    if not docs:
        print("No documents found.")
        return

    print("Max value-match counts per document (if we highlighted ALL matched chunks)")
    print("=" * 80)

    for doc in docs:
        doc_id = doc.get("doc_id") or doc.get("pdf_stem", "?")
        try:
            data = load_comparison_data(doc_id)
        except Exception as e:
            print(f"\n{doc_id}: failed to load ({e})")
            continue

        comparison = data.get("comparison", [])
        if not comparison:
            print(f"\n{doc_id}: no comparison rows")
            continue

        max_count = 0
        max_columns: list[tuple[str, int]] = []  # (col_name, count)

        for row in comparison:
            col_name = row.get("column_name")
            if not col_name:
                continue
            n = count_value_matches_for_column(doc_id, col_name, data)
            if n > max_count:
                max_count = n
                max_columns = [(col_name, n)]
            elif n == max_count and n > 0:
                max_columns.append((col_name, n))

        # Also get distribution: how many columns have 0, 1, 2-5, 6-10, 11+ matches
        counts: list[int] = []
        for row in comparison:
            col_name = row.get("column_name")
            if not col_name:
                continue
            n = count_value_matches_for_column(doc_id, col_name, data)
            counts.append(n)

        n_cols = len(counts)
        n_zero = sum(1 for c in counts if c == 0)
        n_one = sum(1 for c in counts if c == 1)
        n_2_5 = sum(1 for c in counts if 2 <= c <= 5)
        n_6_10 = sum(1 for c in counts if 6 <= c <= 10)
        n_11_plus = sum(1 for c in counts if c >= 11)

        print(f"\n{doc_id}")
        print(f"  Columns: {n_cols} total")
        print(f"  Value-match distribution: 0 chunks={n_zero}, 1={n_one}, 2-5={n_2_5}, 6-10={n_6_10}, 11+={n_11_plus}")
        print(f"  MAX value matches: {max_count}")
        if max_columns:
            for col, cnt in max_columns[:5]:  # show up to 5 columns with max
                short = (col[:60] + "…") if len(col) > 60 else col
                print(f"    {cnt:3d}  {short}")
            if len(max_columns) > 5:
                print(f"    ... and {len(max_columns) - 5} more column(s) with {max_count} matches")


if __name__ == "__main__":
    main()
