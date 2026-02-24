#!/usr/bin/env python3
"""
Reconciliation dataset stats: columns-to-chunks ratio and total calls for any trial.

Shows how the chunk-centric batching would work for a given document.
Run: python experiment-scripts/reconciliation_dataset_stats.py [doc_id]
     python experiment-scripts/reconciliation_dataset_stats.py   # uses default doc
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web.highlight_service import load_landing_ai_parse
from web.comparison_service import load_comparison_data, list_documents


def parse_page_from_evidence(evidence: str) -> list[int]:
    """Extract page numbers mentioned in evidence text. Returns 1-based pages."""
    if not evidence:
        return []
    pages = set()
    for m in re.finditer(r'\bpage\s+(\d+)\b', evidence, re.IGNORECASE):
        pages.add(int(m.group(1)))
    for m in re.finditer(r'\bpages?\s+(\d+)(?:\s*[-–]\s*(\d+))?', evidence, re.IGNORECASE):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        for p in range(start, end + 1):
            pages.add(p)
    return sorted(pages)


def get_columns_by_page(comparison_rows: list, max_page: int = 999) -> dict[int, set[str]]:
    """Map page (1-based) -> set of column names that target it.
    Pages > max_page (e.g. from evidence typos like 'page 453') are excluded."""
    columns_by_page: dict[int, set[str]] = {}

    for row in comparison_rows:
        col_name = row.get("column_name")
        if not col_name:
            continue

        pages_for_col = set()

        for method_name, col_data in (row.get("methods") or {}).items():
            if not col_data:
                continue
            p = col_data.get("page")
            if p is not None and p != "Not applicable" and str(p).lower() not in ("n/a", "na"):
                try:
                    p_int = int(p)
                    if p_int >= 1:  # -1 means "not found" in pipeline
                        pages_for_col.add(p_int)
                except (ValueError, TypeError):
                    pass
            evidence = col_data.get("evidence") or col_data.get("attribution", {}).get("evidence", "")
            for pg in parse_page_from_evidence(str(evidence)):
                if 1 <= pg <= max_page:
                    pages_for_col.add(pg)

        for pg in pages_for_col:
            if pg not in columns_by_page:
                columns_by_page[pg] = set()
            columns_by_page[pg].add(col_name)

    return columns_by_page


def get_chunks_by_page(chunks: list) -> dict[int, list[dict]]:
    """Map page (1-based) -> list of chunks on that page."""
    chunks_by_page: dict[int, list[dict]] = {}
    for c in chunks:
        page_0 = c.get("grounding", {}).get("page", 0)
        try:
            page_1 = int(page_0) + 1
        except (ValueError, TypeError):
            page_1 = 1
        if page_1 not in chunks_by_page:
            chunks_by_page[page_1] = []
        chunks_by_page[page_1].append(c)
    return chunks_by_page


def estimate_chunk_tokens(chunk: dict) -> int:
    """Rough token estimate: ~4 chars per token."""
    md = chunk.get("markdown") or ""
    return max(1, len(md) // 4)


def main():
    parser = argparse.ArgumentParser(description="Reconciliation dataset stats for a trial")
    parser.add_argument("doc_id", nargs="?", help="Document ID (e.g. NCT00268476_Attard_STAMPEDE_Lancet'23)")
    parser.add_argument("--pages-per-batch", type=int, default=2, help="Pages per batch (default: 2)")
    parser.add_argument("--list", action="store_true", help="List available documents and exit")
    args = parser.parse_args()

    if args.list:
        docs = list_documents()
        print("Available documents:")
        for d in docs:
            print(f"  {d.get('doc_id', '?')}")
        return

    doc_id = args.doc_id
    if not doc_id:
        docs = list_documents()
        if not docs:
            print("No documents found. Run with --list to see available docs.")
            return
        doc_id = docs[0]["doc_id"]
        print(f"Using default doc: {doc_id}\n")

    # Load data
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        print(f"Error: No landing_ai_parse_output.json for {doc_id}")
        return

    try:
        comparison = load_comparison_data(doc_id)
    except Exception as e:
        print(f"Error loading comparison: {e}")
        return

    chunks = parse_data.get("chunks") or []
    comparison_rows = comparison.get("comparison") or []

    max_page = max((c.get("grounding", {}).get("page", 0) for c in chunks), default=0) + 1
    columns_by_page = get_columns_by_page(comparison_rows, max_page=max_page)
    chunks_by_page = get_chunks_by_page(chunks)

    all_pages_with_cols = sorted(p for p in columns_by_page.keys() if p >= 1)
    all_pages_with_chunks = sorted(chunks_by_page.keys())
    all_pages = sorted(set(all_pages_with_cols) | set(all_pages_with_chunks))

    total_columns = len(comparison_rows)
    total_chunks = len(chunks)
    cols_with_page = len(set().union(*columns_by_page.values())) if columns_by_page else 0
    cols_no_page = total_columns - cols_with_page

    print("=" * 70)
    print(f"RECONCILIATION DATASET: {doc_id}")
    print("=" * 70)
    print(f"\nTotals:")
    print(f"  Chunks:     {total_chunks}")
    print(f"  Columns:    {total_columns}")
    print(f"  Columns with page (from pipeline/evidence): {cols_with_page}")
    print(f"  Columns with no page (floating):           {cols_no_page}")
    print(f"  Pages with chunks:  {len(all_pages_with_chunks)} (1–{max(all_pages_with_chunks) if all_pages_with_chunks else 0})")
    print(f"  Pages with columns: {len(all_pages_with_cols)}")

    # Per-page breakdown
    print("\n" + "-" * 70)
    print("Per-page: chunks | columns")
    print("-" * 70)

    for page in all_pages[:25]:  # cap display
        n_chunks = len(chunks_by_page.get(page, []))
        n_cols = len(columns_by_page.get(page, set()))
        bar_c = "█" * min(n_chunks, 30) + ("…" if n_chunks > 30 else "")
        bar_p = "░" * min(n_cols, 30) + ("…" if n_cols > 30 else "")
        print(f"  Page {page:3d}:  {n_chunks:3d} chunks  {n_cols:3d} columns  | {bar_c}")
        if n_cols:
            print(f"           columns: {bar_p}")

    if len(all_pages) > 25:
        print(f"  ... ({len(all_pages) - 25} more pages)")

    # Batch analysis
    pages_per_batch = args.pages_per_batch
    batches: list[dict] = []
    max_page = max(all_pages) if all_pages else 0

    for start in range(1, max_page + 1, pages_per_batch):
        batch_pages = list(range(start, min(start + pages_per_batch, max_page + 1)))
        batch_pages = [p for p in batch_pages if p in all_pages or p in chunks_by_page]

        if not batch_pages:
            continue

        batch_chunks = []
        batch_columns = set()
        batch_tokens = 0

        for p in batch_pages:
            for c in chunks_by_page.get(p, []):
                batch_chunks.append(c)
                batch_tokens += estimate_chunk_tokens(c)
            batch_columns.update(columns_by_page.get(p, set()))

        if batch_chunks or batch_columns:
            batches.append({
                "pages": batch_pages,
                "n_chunks": len(batch_chunks),
                "n_columns": len(batch_columns),
                "tokens_est": batch_tokens,
            })

    print("\n" + "-" * 70)
    print(f"BATCHES (pages_per_batch={pages_per_batch})")
    print("-" * 70)
    print(f"{'Batch':<8} {'Pages':<20} {'Chunks':<8} {'Columns':<8} {'Est. tokens':<12} {'Cols/chunk':<10}")
    print("-" * 70)

    total_calls = 0
    for idx, b in enumerate(batches):
        total_calls += 1
        pages_str = f"{min(b['pages'])}–{max(b['pages'])}" if len(b["pages"]) > 1 else str(b["pages"][0])
        ratio = b["n_columns"] / b["n_chunks"] if b["n_chunks"] else 0
        print(f"{idx+1:<8} {pages_str:<20} {b['n_chunks']:<8} {b['n_columns']:<8} {b['tokens_est']:<12} {ratio:.1f}")

    # Floating batch
    if cols_no_page > 0:
        total_calls += 1
        float_chunks = []
        for p in (all_pages_with_chunks or [])[:5]:
            float_chunks.extend(chunks_by_page.get(p, []))
        float_tokens = sum(estimate_chunk_tokens(c) for c in float_chunks)
        print(f"{'float':<8} {'1–5 (first 5)':<20} {len(float_chunks):<8} {cols_no_page:<8} {float_tokens:<12} —")

    print("-" * 70)
    print(f"\nTOTAL CALLS: {total_calls}")
    print(f"Columns per call (avg): {total_columns / total_calls:.1f}" if total_calls else "")
    print(f"Chunks per call (avg): {total_chunks / total_calls:.1f}" if total_calls else "")


if __name__ == "__main__":
    main()
