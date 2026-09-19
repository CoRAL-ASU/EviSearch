"""Find an example value in its paper: the pages where it is printed and the text around it.

Numbers carry most values: a value is found on a page when all its non-trivial numbers appear there (commas ignored,
"3.0" read as "3"). Text values are found when their longer words appear together on a page. A value that is not
found is often a convention (a label the paper never prints, such as "Triplet therapy"): the schema agent turns it
into a question for the reviewer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.retrieval import embedding_retriever as retriever

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TRIVIAL = {"0", "1", "2", "3", "4", "5"}
SNIPPET_CHARS = 250
MAX_SNIPPETS = 3


def _numbers(value: str) -> List[str]:
    out = []
    for n in NUMBER_RE.findall(value.replace(",", "")):
        n = n[:-2] if n.endswith(".0") else n
        if n not in TRIVIAL:
            out.append(n)
    return out


def _words(value: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]{5,}", value.lower())][:5]


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).lower()


def _snippet(text: str, needle: str) -> str:
    flat = re.sub(r"\s+", " ", text)
    at = flat.lower().replace(",", "").find(needle.lower())
    if at < 0:
        return flat[: 2 * SNIPPET_CHARS]
    start, end = max(0, at - SNIPPET_CHARS), min(len(flat), at + len(needle) + SNIPPET_CHARS)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def ground(doc_id: str, value: str, pages: Dict[int, str] | None = None) -> Dict[str, Any]:
    """{status: found | not_in_paper | empty | short, pages: [...], snippets: [{page, text}]}."""
    value = (value or "").strip()
    if not value:
        return {"status": "empty", "pages": [], "snippets": []}
    if pages is None:
        total = retriever.get_total_pages(doc_id)
        pages = retriever.get_page_content(doc_id, list(range(1, total + 1)))
    flat = {p: _flat(t) for p, t in pages.items()}
    numbers, words = _numbers(value), _words(value)
    if numbers:
        hits = [p for p, t in flat.items() if all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", t) for n in numbers)]
        needle = numbers[0]
    elif words:
        hits = [p for p, t in flat.items() if all(w in t for w in words)]
        needle = words[0]
    else:
        return {"status": "short", "pages": [], "snippets": []}
    if not hits:
        return {"status": "not_in_paper", "pages": [], "snippets": []}
    snippets = [{"page": p, "text": _snippet(pages[p], needle)} for p in sorted(hits)[:MAX_SNIPPETS]]
    return {"status": "found", "pages": sorted(hits), "snippets": snippets}
