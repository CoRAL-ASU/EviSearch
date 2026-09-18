"""
Evidence verifier: checks (column, value, page, evidence) claims against the cited page with a separate model call.

One structured call per page (claims on the same page share it): the page's parsed text, its rendered image when the
model reads images, the column definitions and the claims. For each claim the model says whether the page supports
the value for that column (population, arm, timepoint, unit), what the page itself states, and the supporting text
copied from the page. Two deterministic signals are recorded next to each verdict (the value's numbers and the claimed
evidence found in the parsed text); they do not decide the verdict, because values read from page images are often
missing from the parsed text.

The reconciler uses this to verify both arms' claims before it decides, and to check every value it submits.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.config import MAX_TOKENS
from src.evisearch.columns import MODALITIES
from src.evisearch.services.extraction_rules import shared_rules
from src.evisearch.services.page_images import render_pdf_pages_to_png
from src.inference import ImagePart, InferenceError, Message, TextPart, Usage
from src.inference.base import ChatModel
from src.retrieval import embedding_retriever as retriever

VERDICTS = ("supported", "partial", "not_supported")
CLAIMS_PER_CALL = 12
MAX_WORKERS = 4  # concurrent verifier calls; vLLM batches them

SYSTEM_PROMPT = """You check values extracted from a clinical trial paper against one page of that paper.

You get the page's parsed text and, when available, its image, then a list of claims. Each claim names a column with
its definition, a value someone extracted for it, and the evidence they quoted. Judge every claim only from this page.

For each claim return:
- verdict:
  "supported": the page states this value for this column, for the population or subgroup, arm, timepoint and unit the
    definition asks for. Different formatting or rounding of the same number is fine. A value the definition asks to
    derive (for example subgroups added up to the whole population) is supported when every number it uses is on this
    page and the arithmetic is right.
  "partial": the page supports only part of the value (for example the count but not the percentage, or one of two
    required items), or the value is right but the definition asks for more that the page also states.
  "not_supported": the page does not state this value for this column: a different number, a different population,
    arm or timepoint, or nothing about it on this page.
- page_value: what this page states for the column, written as printed ("" when the page does not report it).
- evidence: the text from the page that shows page_value, copied as printed: the sentence, or for a table the row
  label, column header and cell, or for a figure its label and the number read from it ("" when there is none).
- modality: "table", "figure" or "text": where page_value appears.
- reason: one short sentence.

When the parsed text and the image disagree, trust the image. Do not use knowledge from outside this page."""

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
TRIVIAL_NUMBERS = {"0", "1", "2", "3"}  # too common on any page to count as evidence


@dataclass(frozen=True)
class Claim:
    column: str
    value: str
    page: int
    evidence: str = ""

    @property
    def key(self) -> Tuple[str, str, int]:
        return claim_key(self.column, self.value, self.page)


def normalize_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower().rstrip(".")


def claim_key(column: str, value: Any, page: int) -> Tuple[str, str, int]:
    return (column, normalize_value(value), int(page))


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
        "page": claim.page,
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


def _verify_page(
    chat: ChatModel, page: int, page_text: str, png: Optional[bytes], claims: Sequence[Claim], definitions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Usage, Dict[str, Any]]:
    ids = [f"c{i}" for i in range(1, len(claims) + 1)]
    parts: List[Any] = [TextPart(f"=== PAGE {page}: parsed text ===\n{page_text or '(no parsed text for this page)'}")]
    if png:
        parts += [TextPart(f"=== PAGE {page}: image ==="), ImagePart(png)]
    parts.append(TextPart(_claims_block(claims, definitions)))
    messages = [Message.system(SYSTEM_PROMPT + shared_rules()), Message.user(*parts)]
    schema = response_schema(ids) if chat.capabilities.json_schema else None
    usage = Usage()
    call: Dict[str, Any] = {"page": page, "claims": len(claims), "image": bool(png)}
    try:
        result = chat.chat(messages, response_schema=schema, max_tokens=MAX_TOKENS["verifier"])
        usage.add(result.usage)
        call.update(result.call_record())
        parsed = result.json()
    except (InferenceError, ValueError) as exc:
        call["error"] = str(exc)
        return [_record(c, "error", f"verifier call failed: {exc}", page_text) for c in claims], usage, call
    by_id = {str(item.get("id")): item for item in (parsed or {}).get("results", []) if isinstance(item, dict)}
    records = []
    for claim_id, claim in zip(ids, claims):
        item = by_id.get(claim_id)
        if item is None:
            records.append(_record(claim, "error", "verifier returned no verdict for this claim", page_text))
            continue
        verdict = str(item.get("verdict", "")).strip()
        found = {key: item.get(key) for key in ("page_value", "evidence", "modality")}
        records.append(_record(claim, verdict if verdict in VERDICTS else "error", str(item.get("reason", "")).strip(), page_text, **found))
    return records, usage, call


def verify_claims(
    chat: ChatModel,
    doc_id: str,
    claims: Sequence[Claim],
    definitions: Dict[str, str],
    *,
    pdf_path: Optional[Path] = None,
    image_scale: Optional[float] = None,
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Usage, List[Dict[str, Any]]]:
    """Verify claims, one call per page (at most CLAIMS_PER_CALL claims each). Returns ({claim key: record}, usage,
    per-call logs). Claims on pages outside the document are not_supported without a call; a failed call gives
    verdict "error" (treated as not verified)."""
    unique: Dict[Tuple[str, str, int], Claim] = {}
    for claim in claims:
        unique.setdefault(claim.key, claim)
    total_pages = retriever.get_total_pages(doc_id)
    records: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    by_page: Dict[int, List[Claim]] = {}
    for key, claim in unique.items():
        if 1 <= claim.page <= total_pages:
            by_page.setdefault(claim.page, []).append(claim)
        else:
            records[key] = _record(claim, "not_supported", f"page {claim.page} does not exist (document has {total_pages} pages)")
    if not by_page:
        return records, Usage(), []

    texts = retriever.get_page_content(doc_id, sorted(by_page))
    images: Dict[int, bytes] = {}
    if image_scale and pdf_path and Path(pdf_path).exists():
        images = dict(render_pdf_pages_to_png(Path(pdf_path), sorted(by_page), image_scale))
    jobs = [
        (page, page_claims[i : i + CLAIMS_PER_CALL])
        for page, page_claims in sorted(by_page.items())
        for i in range(0, len(page_claims), CLAIMS_PER_CALL)
    ]
    usage = Usage()
    calls: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as pool:
        futures = [pool.submit(_verify_page, chat, page, texts.get(page, ""), images.get(page), group, definitions) for page, group in jobs]
        for future in futures:
            page_records, page_usage, call = future.result()
            usage.add(page_usage)
            calls.append(call)
            for record in page_records:
                records[claim_key(record["column"], record["value"], record["page"])] = record
    return records, usage, calls
