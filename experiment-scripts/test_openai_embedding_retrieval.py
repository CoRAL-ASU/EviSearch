#!/usr/bin/env python3
"""
Test OpenAI text-embedding-3-large semantic retrieval over Landing AI chunks.

Usage:
  python experiment-scripts/test_openai_embedding_retrieval.py "NCT02799602_Hussain_ARASENS_JCO'23"
  python experiment-scripts/test_openai_embedding_retrieval.py "NCT02799602_Hussain_ARASENS_JCO'23" --query "demographics by region"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv

load_dotenv()

from src.retrieval.openai_embedding_retriever import embed_chunks, search_chunks


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI embedding retrieval")
    parser.add_argument("doc_id", help="Document id, e.g. NCT02799602_Hussain_ARASENS_JCO'23")
    parser.add_argument("--query", default="demographics race region baseline characteristics", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings (ignore cache)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add to .env or export.")
        sys.exit(1)

    doc_id = args.doc_id.strip()
    print(f"Doc: {doc_id}")
    print(f"Query: {args.query}")
    print()

    print("Embedding chunks (or loading from cache)...")
    result = embed_chunks(doc_id, force=args.force)
    if not result:
        print("ERROR: No chunks found or embedding failed.")
        sys.exit(1)
    chunk_ids, embeddings = result
    print(f"  {len(chunk_ids)} chunks embedded, dim={embeddings.shape[1]}")
    print()

    print(f"Searching (top {args.top_k})...")
    hits = search_chunks(doc_id, args.query, top_k=args.top_k)
    if not hits:
        print("No results.")
        return

    for i, h in enumerate(hits, 1):
        print(f"\n--- Result {i} (score={h['score']:.4f}) ---")
        print(f"  chunk_id: {h['chunk_id'][:40]}...")
        print(f"  page={h['page']}, source_type={h['source_type']}")
        preview_len = 2000
        text_preview = h["text"][:preview_len]
        suffix = "..." if len(h["text"]) > preview_len else ""
        print(f"  text ({len(h['text'])} chars):\n{text_preview}{suffix}")
    print("\nDone.")


if __name__ == "__main__":
    main()
