#!/usr/bin/env python3
"""Compatibility CLI for src.evisearch.pipelines.search_pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evisearch.pipelines.search_pipeline import *  # noqa: F401,F403
from src.evisearch.pipelines.search_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
