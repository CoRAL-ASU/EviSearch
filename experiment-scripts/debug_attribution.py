#!/usr/bin/env python3
"""
Debug attribution: for sample columns, print value, retrieved chunks, and whether each chunk CONTAINS the value.
Run: python experiment-scripts/debug_attribution.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS = PROJECT_ROOT / "new_pipeline_outputs" / "results"


def _chunk_text(c):
    md = str(c.get("markdown") or "")
    return re.sub(r"<::[^>]*::>", "", re.sub(r"<a[^>]*>.*?</a>", "", md, flags=re.S)).strip()


def _value_in_chunk(value: str, chunk_text: str) -> tuple:
    """Check if value (or distinctive parts) appears in chunk. Return (any_match, matched_parts)."""
    value = str(value or "").strip()
    if not value or value.lower() in ("not reported", "not found"):
        return False, []
    # Extract distinctive tokens: numbers, decimals
    parts = re.findall(r"[\d.]+%?|%[\d.]+", value) + re.findall(r"[0-9]+", value)
    parts = list(dict.fromkeys(parts))  # dedupe preserving order
    matched = []
    for p in parts:
        if p in chunk_text:
            matched.append(p)
    # Also check if whole value is in (for short values)
    if len(value) >= 5 and value in chunk_text:
        matched.append("(full value)")
    return len(matched) > 0 or (len(value) >= 5 and value in chunk_text), matched


def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "NCT00268476_Attard_STAMPEDE_Lancet'23"
    path = RESULTS / doc_id / "reconciliation" / "reconciled_results.json"
    chunks_path = RESULTS / doc_id / "chunking" / "landing_ai_parse_output.json"

    if not path.exists() or not chunks_path.exists():
        print(f"Missing {path} or {chunks_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text())
    chunks_data = json.loads(chunks_path.read_text())
    chunk_by_id = {c["id"]: c for c in chunks_data.get("chunks", []) if c.get("id")}

    columns = data.get("columns", [])
    # Sample: first 5 with attributed chunks, plus a few specific ones
    sample_names = [
        "Add-on Treatment",
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Control",
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Treatment",
        "Median OS (mo) | Overall | Control",
        "Median OS (mo) | Overall | Treatment",
        "Number of Arms Included",
    ]
    sample_cols = [c for c in columns if c.get("column_name") in sample_names]
    if not sample_cols:
        sample_cols = [c for c in columns if (c.get("attributed_chunks") or c.get("chunk_ids"))][:6]

    for col in sample_cols:
        name = col.get("column_name", "")
        value = col.get("final_value", "")
        ac = col.get("attributed_chunks") or []
        print("\n" + "=" * 80)
        print(f"COLUMN: {name}")
        print(f"VALUE:  {value[:120]}{'…' if len(value) > 120 else ''}")
        print("-" * 80)

        for i, att in enumerate(ac[:4], 1):
            cid = att.get("chunk_id", "")
            snippet = att.get("snippet", "")
            score = att.get("score", 0)
            page = att.get("page", "?")
            chunk = chunk_by_id.get(cid)
            full_text = _chunk_text(chunk) if chunk else ""
            contains, matched = _value_in_chunk(value, full_text)
            status = "✓ CONTAINS" if contains else "✗ MISSING"
            if matched:
                status += f" (matched: {matched[:5]})"
            print(f"  #{i} [page {page}] score={score:.3f} {status}")
            print(f"      snippet: {snippet[:100]}…")
            if not contains and full_text:
                # Show a bit of chunk to see what we got
                print(f"      chunk preview: {full_text[:150]}…")

    print("\n" + "=" * 80)
    print("Summary: ✓ = chunk contains value (or key parts); ✗ = retriever returned wrong chunk")
    print("=" * 80)


if __name__ == "__main__":
    main()
