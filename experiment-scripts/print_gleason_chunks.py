#!/usr/bin/env python3
"""Print chunk content for Gleason score columns. Run from project root."""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "new_pipeline_outputs" / "results"


def _chunk_text(c):
    md = str(c.get("markdown") or "")
    return re.sub(r"<::[^>]*::>", "", re.sub(r"<a[^>]*>.*?</a>", "", md, flags=re.S)).strip()


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
    gleason_cols = [c for c in columns if "Gleason" in (c.get("column_name") or "")]

    for col in gleason_cols:
        name = col.get("column_name", "")
        value = col.get("final_value", "")
        ac = col.get("attributed_chunks") or []
        chunk_ids = [x.get("chunk_id") for x in ac if isinstance(x, dict)] or col.get("chunk_ids") or []
        print("\n" + "=" * 80)
        print(f"COLUMN: {name}")
        print(f"VALUE:  {value[:150]}{'…' if len(value) > 150 else ''}")
        print("-" * 80)

        for i, cid in enumerate(chunk_ids[:5], 1):
            chunk = chunk_by_id.get(cid) if cid else None
            if not chunk:
                print(f"  #{i} chunk_id={cid} (not found)")
                continue
            text = _chunk_text(chunk)
            page = (chunk.get("grounding") or {}).get("page", 0)
            print(f"  #{i} [page {int(page)+1}] chunk_id={cid}")
            print(f"  CHUNK CONTENT:\n{text}\n")


if __name__ == "__main__":
    main()
