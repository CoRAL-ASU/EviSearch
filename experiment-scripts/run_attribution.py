#!/usr/bin/env python3
"""
Run attribution scoring on EXISTING reconciled results.
Does NOT run extractions or LLM reconciliation — only scores chunks and picks top 3-4.

Chunk embeddings are cached per doc at new_pipeline_outputs/chunk_embeddings/{doc_id}.npz.
First run builds the cache; subsequent runs load from cache (fast).

Usage:
  python experiment-scripts/run_attribution.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
  python experiment-scripts/run_attribution.py "NCT00268476_Attard_STAMPEDE_Lancet'23" -o attributed.json
  python experiment-scripts/run_attribution.py --list   # list docs with reconciliation
  python experiment-scripts/run_attribution.py DOC_ID --rebuild-cache  # force rebuild chunk embeddings
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"

from web.comparison_service import load_comparison_data, list_documents
from web.attribution_service import (
    CACHE_DIR,
    _safe_cache_name,
    enrich_reconciled_with_attribution,
)


def main():
    parser = argparse.ArgumentParser(description="Run attribution on existing reconciled results")
    parser.add_argument("doc_id", nargs="?", help="Document ID (e.g. NCT00268476_Attard_STAMPEDE_Lancet'23)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: overwrite reconciled_results.json)")
    parser.add_argument("--list", action="store_true", help="List docs that have reconciled results")
    parser.add_argument("--top-k", type=int, default=3, help="Top K chunks per column (default 3)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild chunk embeddings cache for doc")
    args = parser.parse_args()

    if args.list:
        recondir = RESULTS_ROOT
        if not recondir.exists():
            print("No results directory")
            return
        found = []
        for d in recondir.iterdir():
            if d.is_dir():
                f = d / "reconciliation" / "reconciled_results.json"
                if f.exists():
                    found.append(d.name)
        for x in sorted(found):
            print(x)
        return

    doc_id = args.doc_id
    if not doc_id:
        docs = list_documents()
        # Prefer docs that have reconciliation
        recondir = RESULTS_ROOT
        candidates = []
        for d in recondir.iterdir() if recondir.exists() else []:
            if d.is_dir() and (d / "reconciliation" / "reconciled_results.json").exists():
                candidates.append(d.name)
        doc_id = candidates[0] if candidates else (docs[0]["doc_id"] if docs else "")
    if not doc_id:
        print("No document. Use --list or provide doc_id.", file=sys.stderr)
        sys.exit(1)

    in_path = RESULTS_ROOT / doc_id / "reconciliation" / "reconciled_results.json"
    if not in_path.exists():
        print(f"No reconciled results at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {in_path}…", file=sys.stderr)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    columns = data.get("columns") or []

    comparison = load_comparison_data(doc_id)
    rows = comparison.get("comparison") or []

    if getattr(args, "rebuild_cache", False):
        cache_name = _safe_cache_name(doc_id)
        for p in CACHE_DIR.glob(f"{cache_name}*"):
            p.unlink(missing_ok=True)
            print(f"Cleared cache {p}", file=sys.stderr)

    print(f"Running attribution (top {args.top_k} chunks, semantic retrieval)…", file=sys.stderr)
    enriched = enrich_reconciled_with_attribution(
        doc_id,
        columns,
        comparison_rows=rows,
        top_k=args.top_k,
    )

    result = {"doc_id": doc_id, "columns": enriched}
    out_path = Path(args.output) if args.output else in_path
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)

    # Show sample for 2–3 columns
    print("\n--- Sample (first 3 columns with attributed chunks) ---", file=sys.stderr)
    for col in enriched[:10]:
        ac = col.get("attributed_chunks") or []
        if not ac:
            continue
        print(f"\n{col['column_name']}:", file=sys.stderr)
        print(f"  Value: {col.get('final_value', '')[:80]}…", file=sys.stderr)
        for i, c in enumerate(ac[:3], 1):
            print(f"  Chunk {i} (score={c['score']}): {c['snippet'][:70]}…", file=sys.stderr)


if __name__ == "__main__":
    main()
