#!/usr/bin/env python3
"""Run markdown preprocessor on a couple docs and store output in temp."""
import json
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from src.retrieval.markdown_preprocessor import build_page_chunks_from_markdown

BASELINE_ROOT = repo / "experiment-scripts" / "baselines_landing_ai_new_results"
OUT_ROOT = repo / "temp" / "preprocessor_output"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DOCS = [
    "NCT02799602_Hussain_ARASENS_JCO'23",
    "NCT00268476_Attard_STAMPEDE_Lancet'23",
]

for doc_id in DOCS:
    md_path = BASELINE_ROOT / doc_id / "parsed_markdown.md"
    if not md_path.exists():
        print(f"Skip {doc_id}: no parsed_markdown.md")
        continue

    md = md_path.read_text(encoding="utf-8")
    chunks = build_page_chunks_from_markdown(md)

    safe_name = doc_id.replace("'", "_").replace(" ", "_")
    doc_out = OUT_ROOT / safe_name
    doc_out.mkdir(parents=True, exist_ok=True)

    summary = []
    for chunk_id, page, text in chunks:
        (doc_out / f"{chunk_id}.md").write_text(text, encoding="utf-8")
        summary.append({"chunk_id": chunk_id, "page": page, "len_chars": len(text)})

    (doc_out / "_summary.json").write_text(
        json.dumps({"doc_id": doc_id, "num_pages": len(chunks), "pages": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(chunks)} pages for {doc_id} -> {doc_out}")
