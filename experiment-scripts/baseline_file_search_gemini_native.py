#!/usr/bin/env python3
"""Compatibility entrypoint for the Gemini-native file-search baseline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evisearch.services.file_search_gemini_native import run_file_search_gemini_native_baseline


if __name__ == "__main__":
    run_file_search_gemini_native_baseline()
