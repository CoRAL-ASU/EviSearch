"""
Evidence verifier: checks (column, value, pages, evidence) claims against the cited page(s) with a separate model call.

One structured call per page set (claims citing the same pages share it): each page's parsed text and, when the model
reads images, its rendered image, then the column definitions and the claims. A claim cites one page, or up to
MAX_CLAIM_PAGES pages when its value combines numbers from several (a table continued on the next page, subgroup
tables to add up). For each claim the model first writes the column's answer from the pages itself (page_value, with
every endpoint variant when the paper reports only variants), then its evidence and a reason tag, then the verdict:
supported only when the claimed value IS that answer (statistic, endpoint, population, arm, timepoint, unit); a number
printed on the page that answers another question is not_supported; right but incomplete is partial. Two deterministic signals are
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

SYSTEM_PROMPT = """You judge whether values extracted from a clinical trial paper are the CORRECT ANSWER for a table column, using
the page or pages they were taken from.

You get each page's parsed text and, when available, its image, then a list of claims. Each claim names a column with
its definition, a value someone extracted for it, and the evidence they quoted. Judge every claim only from these pages.

For each claim, work in this order:
1. Read the definition and name exactly what it asks for: the statistic (count, percentage, median, rate, hazard ratio,
   name, yes/no), the endpoint or characteristic, the population or subgroup, the arm, the timepoint and the unit.
2. page_value: find the correct answer on the pages YOURSELF, before you look at the claimed value, and copy it as
   printed. Include every part the column asks for that the pages state (a count and its percentage, each arm or
   trial the definition asks for). When the definition asks for an endpoint and the pages report it only under named
   variants (for example progression-free survival asked, and the pages give biochemical PFS and radiographic PFS, or
   PSA-PFS and clinical PFS), the answer is every variant with its label, joined with "; " (for example
   "bPFS 22.9 months; rPFS 23.5 months"). Write "" when the pages do not state an answer for this column.
3. evidence and modality for page_value, then reason, then the verdict comparing the claimed value with your answer:
  "supported": the claimed value IS the correct answer: the same statistic, endpoint, population or subgroup, arm,
    timepoint and unit as the definition asks, with the same numbers. Also supported:
    - the same number in another format or rounding;
    - the page's value converted to the unit the column asks for, when the conversion is right (for example years
      times 12 for months, within 0.1 after rounding);
    - the population, subgroup, arm or event named differently on the page (a synonym or abbreviation for the same
      thing);
    - a count made by adding printed counts of mutually exclusive subgroups into the population the column asks for,
      and, for an "N (%)" column, its percentage computed from that count and the printed arm size: find each number
      on the pages, in the right rows and columns, and redo the arithmetic yourself.
  "partial": the claimed value is right but incomplete: it gives only some of the parts of your answer (one of several
    endpoint variants, the count without the percentage, one of several arms or trials the definition asks for).
  "not_supported": anything else. A number printed on the pages that answers a DIFFERENT question is NOT supported:
    - another statistic (a median in months in a rate column; a hazard ratio, odds ratio or p value in a per-arm
      column);
    - another endpoint (for example time to castration resistance or time to PSA progression in a PFS column, when the
      paper does not call it progression-free survival);
    - another population or subgroup (a subgroup value given for the whole population, or the reverse), another arm,
      another timepoint;
    - a value computed from other numbers that the paper does not state (a rate from event counts, medians or curves;
      a number read off a curve; "Not reached" or "Not estimable" that the paper does not state for that population).
- reason: start with one tag in brackets, then one short sentence: [match], [incomplete], [statistic], [endpoint],
  [population], [arm], [timepoint], [unit] or [not on pages].

A number being printed on the pages is never enough: the claimed value must answer this column. When the parsed text
and the image disagree, trust the image. Do not use knowledge from outside these pages."""

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
                    # answer first: the checker writes what the pages state, its evidence and reason before the verdict
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "page_value": {"type": "string"},
                        "evidence": {"type": "string"},
                        "modality": {"type": "string", "enum": list(MODALITIES)},
                        "reason": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                    },
                    "required": ["id", "page_value", "evidence", "modality", "reason", "verdict"],
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
    lines.append('\nReturn JSON: {"results": [{"id": ..., "page_value": ..., "evidence": ..., "modality": ..., "reason": ..., "verdict": ...}]}')
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
    messages = [Message.system(SYSTEM_PROMPT + shared_rules(columns=[claim.column for claim in claims])), Message.user(*parts)]
    schema = response_schema(ids) if chat.capabilities.json_schema else None
    pages_text = "\n".join(texts.get(page, "") for page in pages)
    usage = Usage()
    call: Dict[str, Any] = {"page": pages[0], "pages": list(pages), "claims": len(claims), "image": all(images.get(p) for p in pages)}
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["verifier"])
        usage.add(result.usage)
        call.update(result.call_record(), finish_reason=result.finish_reason)
        if str(result.finish_reason).lower() == "length":
            raise ValueError("reply cut off at max_tokens")
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


def _verify_or_split(
    chat: ChatModel, pages: Pages, texts: Dict[int, str], images: Dict[int, bytes], claims: Sequence[Claim], definitions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Usage, List[Dict[str, Any]]]:
    """_verify_pages; a failed call with several claims is retried as two halves. Structured output can loop inside a
    string until max_tokens, and at temperature 0 an identical retry loops again, so the retry changes the prompt."""
    records, usage, call = _verify_pages(chat, pages, texts, images, claims, definitions)
    if "error" not in call or len(claims) < 2:
        return records, usage, [call]
    call["recovered_by_split"] = True
    half = (len(claims) + 1) // 2
    records, calls = [], [call]
    for part in (claims[:half], claims[half:]):
        part_records, part_usage, part_calls = _verify_or_split(chat, pages, texts, images, part, definitions)
        records += part_records
        usage.add(part_usage)
        calls += part_calls
    return records, usage, calls


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
        futures = [pool.submit(_verify_or_split, chat, pages, texts, images, group, definitions) for pages, group in jobs]
        for future in futures:
            page_records, page_usage, page_calls = future.result()
            usage.add(page_usage)
            calls += page_calls
            for record in page_records:
                records[claim_key(record["column"], record["value"], record["pages"])] = record
    return records, usage, calls
