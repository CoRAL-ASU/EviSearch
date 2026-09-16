"""
Rendered PDF pages for the agents that show page images to a model (Arm A and reconciliation).

Every provider gets the same PNGs at the same scale (PAGE_IMAGE_SCALE in src/config/config.py), so local
and API models see identical inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from src.config.config import PAGE_IMAGE_SCALE

# Patch-based vision encoders spend one token per 32x32 pixels after merging (measured on Qwen3.6-27B:
# a 1170x1566 page is ~1,815 tokens). Gemini and OpenAI tile images more cheaply, so this is an upper bound.
PIXELS_PER_IMAGE_TOKEN_SIDE = 32


@dataclass(frozen=True)
class PageImage:
    page: int  # 1-based
    width: int
    height: int
    png: bytes

    @property
    def estimated_tokens(self) -> int:
        return math.ceil(self.width / PIXELS_PER_IMAGE_TOKEN_SIDE) * math.ceil(self.height / PIXELS_PER_IMAGE_TOKEN_SIDE)


def pdf_page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return len(doc)


@lru_cache(maxsize=128)
def _render(pdf_path: str, modified_ns: int, page_number: int, scale: float) -> PageImage:
    import fitz

    with fitz.open(pdf_path) as doc:
        pixmap = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(scale, scale))
        return PageImage(page_number, pixmap.width, pixmap.height, pixmap.tobytes("png"))


def render_pages(pdf_path: Path, page_numbers: List[int], scale: float = PAGE_IMAGE_SCALE) -> List[PageImage]:
    """Pages that exist in the PDF, rendered once per (file version, page, scale) and reused across batches."""
    total = pdf_page_count(pdf_path)
    modified_ns = pdf_path.stat().st_mtime_ns
    return [_render(str(pdf_path), modified_ns, page, float(scale)) for page in page_numbers if 1 <= page <= total]


def render_pdf_pages_to_png(pdf_path: Path, page_numbers: List[int], scale: float = PAGE_IMAGE_SCALE) -> List[Tuple[int, bytes]]:
    return [(image.page, image.png) for image in render_pages(pdf_path, page_numbers, scale)]
