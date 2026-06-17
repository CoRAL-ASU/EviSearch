#!/usr/bin/env python3
"""Compatibility entrypoint for the Gemini Landing-AI markdown baseline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evisearch.services.markdown_baseline import run_gemini_markdown_baseline


if __name__ == "__main__":
    run_gemini_markdown_baseline()
