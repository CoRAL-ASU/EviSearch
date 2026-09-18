"""
Evidence verifier: checks (column, value, pages, evidence) claims against the cited page(s) with a separate model call.

One structured call per page set (claims citing the same pages share it): each page's parsed text and, when the model
reads images, its rendered image, then the column definitions and the claims. A claim cites one page, or up to
MAX_CLAIM_PAGES pages when its value combines numbers from several (a table continued on the next page, subgroup
tables to add up). For each claim the model says whether the pages support the value for that column (population,
arm, timepoint, unit), what the pages state, and the supporting text copied from them. Two deterministic signals are
recorded next to each verdict (the value's numbers and the claimed evidence found in the parsed text); they do not
decide the verdict, because values read from page images are often missing from the parsed text.

The reconciler's verify_attribution tool calls this, and its submit tool accepts only values verified here.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.config.config import MAX_TOKENS
from src.evisearch.columns import MODALITIES
from src.evisearch.services.extraction_rules import shared_rules
from src.evisearch.services.page_images import render_pdf_pages_to_png
from src.inference import ImagePart, InferenceError, Message, TextPart, Usage
from src.inference.base import ChatModel
from src.retrieval import embedding_retriever as retriever

VERDICTS = ("supported", "partial", "not_supported")
CLAIMS_PER_CALL = 12
MAX_CLAIM_PAGES = 3
MAX_WORKERS = 4  # concurrent verifier calls; vLLM batches them

SYSTEM_PROMPT = """You check values extracted from a clinical trial paper against the page or pages they were taken from.

You get each page's parsed text and, when available, its image, then a list of claims. Each claim names a column with
its definition, a value someone extracted for it, and the evidence they quoted. Judge every claim only from these
pages.

For each claim return:
- verdict:
  "supported": the pages state this value for this column, for the population or subgroup, arm, timepoint and unit
    the definition asks for. Also supported:
    - the same number in another format or rounding;
    - the page's value converted to the unit the column asks for, when the conversion is right (for example years
      times 12 for months, within 0.1 after rounding);
    - the population, subgroup, arm or event named differently on the page (a synonym or abbreviation for the same
      thing);
    - a value fixed by a fact the pages state for every patient: all enrolled in one country gives 100% for the region
      containing it; all given the protocol treatment gives 100% for it; deaths attributed to the treatment are
      treatment-related grade 5 events; an arm's randomised count is its N;
    - a count made by adding printed counts of mutually exclusive subgroups into the population the column asks for,
      and, for an "N (%)" column, its percentage computed from that count and the printed arm size. The claimed
      evidence may spell out the sum ("349 + 113 = 462; 462 / 654 = 70.6%"): find each number on the pages, in the
      right rows and columns, and redo the arithmetic yourself before you decide.
  Never supported, whatever the arithmetic:
    - a rate or percentage of patients (survival, progression, response or event rate) computed from event counts,
      medians or curves: a rate column needs a percentage the paper states;
    - a number estimated from a curve, or "Not reached" / "Not estimable" that the paper does not state for that
      population;
    - a subgroup value given for the whole population, or a whole-population value given for a subgroup;
    - a statistic comparing arms (hazard ratio, odds ratio, difference, p value) given for one arm.
  "partial": the pages support only part of the value (for example the count but not the percentage, or one of two
    required items), or the value is right but the definition asks for more that the pages also state.
  "not_supported": the pages do not state this value for this column: a different number, a different population,
    arm or timepoint, or nothing about it on these pages.
- page_value: what the pages state for the column, written as printed ("" when they do not report it).
- evidence: the text that shows page_value, copied as printed: the sentence, or for a table the row label, column
  header and cell, or for a figure its label and the number read from it ("" when there is none).
- modality: "table", "figure" or "text": where page_value appears.
- reason: one short sentence.

When the parsed text and the image disagree, trust the image. Do not use knowledge from outside these pages."""

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TRIVIAL_NUMBERS = {"0", "1", "2", "3"}  # too common on any page to count as evidence

Pages = Tuple[int, ...]


def as_pages(page: Union[int, Sequence[int]]) -> Pages:
    """1-based page(s) as a sorted tuple of at most MAX_CLAIM_PAGES distinct pages."""
    pages = [page] if isinstance(page, int) else list(page)
    return tuple(sorted({int(p) for p in pages}))[:MAX_CLAIM_PAGES]


@dataclass(frozen=True)
class Claim:
    column: str
    value: str
    page: Union[int, Pages]  # one page, or the pages a combined value comes from
    evidence: str = ""

    @property
    def pages(self) -> Pages:
        return as_pages(self.page)

    @property
    def key(self) -> Tuple[str, str, Pages]:
        return claim_key(self.column, self.value, self.pages)


def normalize_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower().rstrip(".")


def claim_key(column: str, value: Any, page: Union[int, Sequence[int]]) -> Tuple[str, str, Pages]:
    return (column, normalize_value(value), as_pages(page))


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).strip().lower()


def numbers_in_text(value: str, page_text: str) -> Optional[bool]:
    """True when every non-trivial number of the value appears in the page text; None when the value has none."""
    numbers = [n for n in NUMBER_RE.findall(str(value).replace(",", "")) if n not in TRIVIAL_NUMBERS]
    if not numbers:
        return None
    flat = _flat(page_text)
    return all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d)", flat) for n in numbers)


def evidence_in_text(evidence: str, page_text: str) -> Optional[bool]:
    """True when the quoted evidence is in the page text (or 80% of its words are); None when there is no quote."""
    quote = _flat(evidence)
    if len(quote) < 4:
        return None
    flat = _flat(page_text)
    if quote in flat:
        return True
    words = [w for w in re.findall(r"[a-z0-9.%]+", quote) if len(w) > 1]
    if not words:
        return None
    page_words = set(re.findall(r"[a-z0-9.%]+", flat))
    return sum(w in page_words for w in words) / len(words) >= 0.8


def response_schema(ids: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "page_value": {"type": "string"},
                        "evidence": {"type": "string"},
                        "modality": {"type": "string", "enum": list(MODALITIES)},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "verdict", "page_value", "evidence", "modality", "reason"],
                },
            }
        },
        "required": ["results"],
    }


def _record(claim: Claim, verdict: str, reason: str, page_text: str = "", **found: Any) -> Dict[str, Any]:
    modality = str(found.get("modality") or "text").lower()
    return {
        "column": claim.column,
        "value": claim.value,
        "page": claim.pages[0],
        "pages": list(claim.pages),
        "claimed_evidence": claim.evidence,
        "verdict": verdict,
        "page_value": str(found.get("page_value") or "").strip(),
        "evidence": str(found.get("evidence") or "").strip(),
        "modality": modality if modality in MODALITIES else "text",
        "reason": reason,
        "numbers_in_text": numbers_in_text(claim.value, page_text) if page_text else None,
        "evidence_in_text": evidence_in_text(claim.evidence, page_text) if page_text else None,
    }


def _claims_block(claims: Sequence[Claim], definitions: Dict[str, str]) -> str:
    lines = ["CLAIMS:"]
    for index, claim in enumerate(claims, 1):
        lines.append(
            f"\n---\nid: c{index}\nColumn: {claim.column}\nDefinition: {definitions.get(claim.column, '')}\n"
            f"Claimed value: {claim.value}\nClaimed evidence: {claim.evidence or '(none given)'}"
        )
    lines.append('\nReturn JSON: {"results": [{"id": ..., "verdict": ..., "page_value": ..., "evidence": ..., "modality": ..., "reason": ...}]}')
    return "\n".join(lines)


def _verify_pages(
    chat: ChatModel, pages: Pages, texts: Dict[int, str], images: Dict[int, bytes], claims: Sequence[Claim], definitions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Usage, Dict[str, Any]]:
    ids = [f"c{i}" for i in range(1, len(claims) + 1)]
    parts: List[Any] = []
    for page in pages:
        parts.append(TextPart(f"=== PAGE {page}: parsed text ===\n{texts.get(page) or '(no parsed text for this page)'}"))
        if images.get(page):
            parts += [TextPart(f"=== PAGE {page}: image ==="), ImagePart(images[page])]
    parts.append(TextPart(_claims_block(claims, definitions)))
    messages = [Message.system(SYSTEM_PROMPT + shared_rules()), Message.user(*parts)]
    schema = response_schema(ids) if chat.capabilities.json_schema else None
    pages_text = "\n".join(texts.get(page, "") for page in pages)
    usage = Usage()
    call: Dict[str, Any] = {"page": pages[0], "pages": list(pages), "claims": len(claims), "image": all(images.get(p) for p in pages)}
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["verifier"])
        usage.add(result.usage)
        call.update(result.call_record())
        parsed = result.json()
    except (InferenceError, ValueError) as exc:
        call["error"] = str(exc)
        return [_record(c, "error", f"verifier call failed: {exc}", pages_text) for c in claims], usage, call
    by_id = {str(item.get("id")): item for item in (parsed or {}).get("results", []) if isinstance(item, dict)}
    records = []
    for claim_id, claim in zip(ids, claims):
        item = by_id.get(claim_id)
        if item is None:
            records.append(_record(claim, "error", "verifier returned no verdict for this claim", pages_text))
            continue
        verdict = str(item.get("verdict", "")).strip()
        found = {key: item.get(key) for key in ("page_value", "evidence", "modality")}
        records.append(_record(claim, verdict if verdict in VERDICTS else "error", str(item.get("reason", "")).strip(), pages_text, **found))
    return records, usage, call


def verify_claims(
    chat: ChatModel,
    doc_id: str,
    claims: Sequence[Claim],
    definitions: Dict[str, str],
    *,
    pdf_path: Optional[Path] = None,
    image_scale: Optional[float] = None,
) -> Tuple[Dict[Tuple[str, str, Pages], Dict[str, Any]], Usage, List[Dict[str, Any]]]:
    """Verify claims, one call per page set (at most CLAIMS_PER_CALL claims each). Returns ({claim key: record},
    usage, per-call logs). Claims citing a page outside the document are not_supported without a call; a failed call
    gives verdict "error" (treated as not verified)."""
    unique: Dict[Tuple[str, str, Pages], Claim] = {}
    for claim in claims:
        unique.setdefault(claim.key, claim)
    total_pages = retriever.get_total_pages(doc_id)
    records: Dict[Tuple[str, str, Pages], Dict[str, Any]] = {}
    by_pages: Dict[Pages, List[Claim]] = {}
    for key, claim in unique.items():
        missing = [page for page in claim.pages if not 1 <= page <= total_pages]
        if missing or not claim.pages:
            records[key] = _record(claim, "not_supported", f"page {missing} does not exist (document has {total_pages} pages)")
        else:
            by_pages.setdefault(claim.pages, []).append(claim)
    if not by_pages:
        return records, Usage(), []

    needed = sorted({page for pages in by_pages for page in pages})
    texts = retriever.get_page_content(doc_id, needed)
    images: Dict[int, bytes] = {}
    if image_scale and pdf_path and Path(pdf_path).exists():
        images = dict(render_pdf_pages_to_png(Path(pdf_path), needed, image_scale))
    jobs = [
        (pages, group[i : i + CLAIMS_PER_CALL])
        for pages, group in sorted(by_pages.items())
        for i in range(0, len(group), CLAIMS_PER_CALL)
    ]
    usage = Usage()
    calls: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as pool:
        futures = [pool.submit(_verify_pages, chat, pages, texts, images, group, definitions) for pages, group in jobs]
        for future in futures:
            page_records, page_usage, call = future.result()
            usage.add(page_usage)
            calls.append(call)
            for record in page_records:
                records[claim_key(record["column"], record["value"], record["pages"])] = record
    return records, usage, calls
