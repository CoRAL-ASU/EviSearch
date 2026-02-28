"""
HTML table → Markdown conversion for Landing AI tables.
Uses markdownify + post-processing (single-label rows, colspan).
"""
from __future__ import annotations

import re
from typing import List


def html_table_to_markdown(html: str) -> str:
    """
    Convert HTML table to clean markdown.
    1. Use markdownify for initial conversion
    2. Fix single-label rows (fill empty cells with the one non-empty value)
    3. Fix colspan in header-like rows (copy previous into empty cells)
    """
    try:
        from markdownify import markdownify as md
    except ImportError:
        return f"ERROR: markdownify not installed. pip install markdownify"

    raw = md(html, strip=["a", "img"])
    if not raw or not raw.strip():
        return raw

    return _postprocess_markdown_table(raw)


def _parse_md_table(md: str) -> tuple[List[List[str]], List[int]]:
    """Parse markdown table into rows of cells. Returns (rows, separator_indices)."""
    lines = [ln.rstrip() for ln in md.strip().split("\n") if ln.strip()]
    rows: List[List[str]] = []
    sep_indices: List[int] = []

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        cells = parts

        is_sep = all(re.match(r"^[\s\-:]+$", c) for c in cells) and len(cells) >= 2
        if is_sep:
            sep_indices.append(len(rows))
        rows.append(cells)

    return rows, sep_indices


def _is_single_label_row(cells: List[str]) -> bool:
    non_empty = [c for c in cells if c and c.strip()]
    return len(non_empty) == 1


def _is_header_like_row(cells: List[str], row_index: int) -> bool:
    if row_index < 3:
        return True
    empty = sum(1 for c in cells if not c or not c.strip())
    return empty >= len(cells) * 0.5


def _fix_row(row: List[str], row_index: int) -> List[str]:
    if not row:
        return row
    if _is_single_label_row(row):
        label = next(c for c in row if c and c.strip())
        return [label] * len(row)
    if _is_header_like_row(row, row_index):
        out = list(row)
        last_non_empty = max(
            (i for i, c in enumerate(out) if c and c.strip()),
            default=-1,
        )
        prev = ""
        for j in range(len(out)):
            if out[j] and out[j].strip():
                prev = out[j]
            elif prev and j <= last_non_empty:
                out[j] = prev
        return out
    return row


def _serialize_rows(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    ncols = max(len(r) for r in rows) if rows else 0
    if ncols == 0:
        return ""
    lines = []
    for row in rows:
        padded = list(row) + [""] * (ncols - len(row))
        line = "| " + " | ".join(padded[:ncols]) + " |"
        lines.append(line)
    return "\n".join(lines)


def _postprocess_markdown_table(md: str) -> str:
    rows, sep_indices = _parse_md_table(md)
    if not rows:
        return md
    sep_set = set(sep_indices)
    fixed = [_fix_row(row, i) if i not in sep_set else row for i, row in enumerate(rows)]
    return _serialize_rows(fixed)
