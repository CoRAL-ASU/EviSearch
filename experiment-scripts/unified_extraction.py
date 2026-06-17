#!/usr/bin/env python3
"""Compatibility wrapper for src.evisearch.pipelines.unified_extraction."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evisearch.pipelines.unified_extraction import *  # noqa: F401,F403
