#!/usr/bin/env python3
"""
Add section labels to Definitions_with_eval_category.csv based on row order.
- Rows 2-26: Trial characteristics
- Rows 27-86: Population characteristics
- Rows 87+: Results for the overall population and their prognostic groups
"""

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "Definitions_with_eval_category.csv"
OUTPUT_FILE = SCRIPT_DIR / "Definitions_with_eval_category_sectioned.csv"


def get_section(line_num: int) -> str:
    """Return section label for 1-based line number (line 1 = header)."""
    if line_num <= 26:
        return "Trial characteristics"
    if line_num <= 86:
        return "Population characteristics"
    return "Results for the overall population and their prognostic groups"


def main():
    rows = []
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header.append("section")
        rows.append(header)

        for line_num, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                row = row + [""]  # empty row, no section
            else:
                section = get_section(line_num)
                row = row + [section]
            rows.append(row)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {len(rows)-1} data rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
