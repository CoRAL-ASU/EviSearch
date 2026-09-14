#!/usr/bin/env python3
"""CLI for Arm A (pdf_query). Options: src/evisearch/pipelines/pdf_query_pipeline.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evisearch.pipelines.pdf_query_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
