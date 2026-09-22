"""
The Reconciliation Agent's shared machinery: the session that holds a batch's two extractions, its tools, and the
submission bookkeeping. The agent itself - the two-phase reading and adjudication - is `reconciliation_v5`, which
subclasses the session here.

Tools defined on the session:
- verify_attribution: a separate model call reads claimed values on their cited pages, text and image
  (services/evidence_check.py), writes the column's answer itself, and returns the page, quotation and modality the
  viewer highlights: supported / partial / not_supported.
- submit_verification: final values, with the admission policy enforced in code rather than left to the prompt.

Per column output: value, reasoning, verification (both_correct | A_correct_B_wrong | B_correct_A_wrong | both_wrong:
which source had the final value), source {page, modality, verbatim_quote}, attribution (checked entries), verified,
needs_review, review_reason, decided_by (agent | auto_submit | unsubmitted), checks (verifier records for the column).
The model comes from the "reconciliation" role in src/config/config.py; the verifier uses the same model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evisearch.columns import MODALITIES, NOT_REPORTED, column_names, is_no_value
from src.evisearch.services import document_reader
from src.evisearch.services.evidence_check import MAX_CLAIM_PAGES, Claim, as_pages, normalize_value, verify_claims
from src.evisearch.services.highlight import resolve_pdf_path
from src.evisearch.tool_args import decode_items
from src.inference import ToolOutput, ToolSpec, Usage
from src.retrieval import embedding_retriever as retriever

VERIFICATIONS = ("A_correct_B_wrong", "B_correct_A_wrong", "both_correct", "both_wrong")
VERIFY_MAX_CLAIMS = 30  # per verify_attribution call; split into one verifier call per page
REASONING_CHARS = 2000  # of each arm's reasoning shown to the agent
SEARCH_LINES_PER_PAGE = 4
CONTINUED_RE = re.compile(r"\bcontinue[sd]?\b.{0,20}\b(next|following) page|\(\s*continued\s*\)|\bcont(?:inued|'d)\.?\s*\)|table \d+[^.\n]{0,20}\bcontinued", re.I)



def _page(raw: Any) -> Optional[int]:
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


def _items(args: Dict[str, Any], key: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The list argument `key` as a list of dicts, also when the model sent it as a (possibly malformed) JSON string;
    the note says what was recovered, for the tool response."""
    decoded, note = decode_items(args.get(key))
    if isinstance(decoded, dict):
        decoded = decoded.get(key, [decoded])
    items = [item for item in decoded if isinstance(item, dict)] if isinstance(decoded, list) else []
    return items, note


def _claim_pages(item: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """The page(s) a claim or submission cites: "pages" (a list) or "page"; None when neither is valid."""
    raw = item.get("pages")
    pages = [p for p in (_page(x) for x in raw) if p] if isinstance(raw, list) else []
    if not pages and _page(item.get("page")):
        pages = [_page(item.get("page"))]
    return as_pages(pages) if pages else None


def _extract_source_output(col_data: Any) -> Dict[str, Any]:
    """Normalize an A/B column result to {value, reasoning, pages: [{page, modality, evidence}]}."""
    if not isinstance(col_data, dict):
        return {"value": NOT_REPORTED, "reasoning": "", "pages": []}
    value = col_data.get("value")
    pages: List[Dict[str, Any]] = []
    for item in col_data.get("attribution") or []:
        if not isinstance(item, dict) or _page(item.get("page")) is None:
            continue
        modality = str(item.get("modality") or item.get("source_type") or "text").lower()
        pages.append({
            "page": _page(item.get("page")),
            "modality": modality if modality in MODALITIES else "text",
            "evidence": str(item.get("evidence") or item.get("verbatim_quote") or "").strip(),
        })
    return {
        "value": NOT_REPORTED if is_no_value(value) else str(value),
        "reasoning": str(col_data.get("reasoning") or "").strip(),
        "pages": pages,
    }


ABSENCE_ANSWERS = {"no", "n"}  # a negative yes/no answer claims absence, like "Not reported": no page can show it


def is_absence(value: Any) -> bool:
    return is_no_value(value) or str(value).strip().lower() in ABSENCE_ANSWERS


def _same(value: Any, other: Any) -> bool:
    if is_no_value(value) or is_no_value(other):
        return is_no_value(value) and is_no_value(other)
    return normalize_value(value) == normalize_value(other)


def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("column", "value", "pages", "verdict", "page_value", "evidence", "modality", "reason")
    return {key: record.get(key) for key in keys}


def _relevant_lines(text: str, query: str, limit: int = SEARCH_LINES_PER_PAGE) -> List[str]:
    """The page's lines (table rows, sentences) sharing the most words with the query."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    lines = [line.strip() for line in re.split(r"\n|(?<=[.;])\s+(?=[A-Z])", text) if len(line.strip()) > 3]
    scored = sorted(
        ((sum(w in line.lower() for w in words), index, line) for index, line in enumerate(lines)), key=lambda t: (-t[0], t[1])
    )
    return [line[:300] for score, _, line in scored[:limit] if score > 0]


def tool_specs(names: List[str]) -> List[ToolSpec]:
    source = {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "1-based page that shows the value"},
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "Instead of page: the pages of a value verified on several pages"},
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "evidence": {"type": "string", "description": "Supporting text as printed on that page"},
        },
    }
    return [
        ToolSpec(
            name="ask_document",
            description=f"Ask a reader that has the whole paper (text and page images). At most {document_reader.QUESTIONS_PER_CALL} questions per call; each answer comes with pages and evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names), "description": "Column the question is about (its definition is sent along)"},
                                "question": {"type": "string"},
                            },
                            "required": ["question"],
                        },
                    }
                },
                "required": ["questions"],
            },
        ),
        ToolSpec(
            name="search_pages",
            description="Semantic search over the paper's pages. Returns the best matching pages with their most relevant lines.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Specific terms, e.g. 'ECOG performance status baseline characteristics'"}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="verify_attribution",
            description=f"Check whether a page supports a value for a column (at most {VERIFY_MAX_CLAIMS} claims per call). Returns verdict, page_value and evidence per claim.",
            parameters={
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names)},
                                "value": {"type": "string"},
                                "page": {"type": "integer"},
                                "pages": {"type": "array", "items": {"type": "integer"}, "description": f"Instead of page: up to {MAX_CLAIM_PAGES} pages when the value combines numbers from several"},
                                "evidence": {"type": "string"},
                            },
                            "required": ["column", "value"],
                        },
                    }
                },
                "required": ["claims"],
            },
        ),
        ToolSpec(
            name="submit_verification",
            description="Submit final values for one or more columns. A value is accepted only when verify_attribution found it supported on its source page; a rejected value is never accepted; \"Not reported\" is accepted when every extracted value failed the check. The response lists accepted and rejected columns.",
            parameters={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "enum": list(names)},
                                "value": {"type": "string"},
                                "reasoning": {"type": "string"},
                                "verification": {"type": "string", "enum": list(VERIFICATIONS)},
                                "source": source,
                                "review": {"type": "boolean", "description": "true: send this value to a human reviewer (a partial value, or a value no page shows); a value the verifier rejected is not accepted even with review"},
                                "review_reason": {"type": "string"},
                            },
                            "required": ["column", "value", "reasoning", "verification"],
                        },
                    }
                },
                "required": ["results"],
            },
        ),
    ]


class _ReconciliationSession:
    def __init__(
        self,
        chat: Any,
        doc_id: str,
        batch_columns: List[Dict[str, Any]],
        definitions_map: Dict[str, str],
        source_a_data: Dict[str, Dict[str, Any]],
        source_b_data: Dict[str, Dict[str, Any]],
        image_scale: Optional[float],
    ):
        self.chat = chat
        self.doc_id = doc_id
        self.names = column_names(batch_columns)
        self.definitions = {
            col.get("column_name", ""): definitions_map.get(col.get("column_name", ""), "") or col.get("definition", "")
            for col in batch_columns
        }
        self.sources = {
            "A": {name: _extract_source_output(source_a_data.get(name)) for name in self.names},
            "B": {name: _extract_source_output(source_b_data.get(name)) for name in self.names},
        }
        self.total_pages = retriever.get_total_pages(doc_id)
        self.image_scale = image_scale
        self.pdf_path = resolve_pdf_path(doc_id) if image_scale else None
        self.checks: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self.attempts: Dict[str, Dict[str, Any]] = {}  # last rejected submission per column
        self.tool_usage = Usage()  # reader and verifier calls made inside tools
        self.reader_calls: List[Dict[str, Any]] = []
        self.verifier_calls: List[Dict[str, Any]] = []
        self.submitted: Dict[str, Dict[str, Any]] = {}

    # ---- prompt ------------------------------------------------------------------------------------------------

    def describe(self, name: str) -> str:
        lines = []
        for origin in ("A", "B"):
            source = self.sources[origin][name]
            lines.append(f'  {origin}: "{source["value"]}"')
            if source["reasoning"]:
                lines.append(f'     reasoning: {source["reasoning"][:REASONING_CHARS]}')
            for item in source["pages"]:
                evidence = f', evidence "{item["evidence"]}"' if item["evidence"] else ""
                lines.append(f'     cites page {item["page"]} ({item["modality"]}){evidence}')
            if not is_absence(source["value"]) and not source["pages"]:
                lines.append("     cites no page")
        return "\n".join(lines)

    def user_prompt(self) -> str:
        blocks = [
            f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}\n{self.describe(name)}"
            for i, name in enumerate(self.names, 1)
        ]
        return (
            f"Decide the following columns. The paper has {self.total_pages} pages. Sources are anonymous (A and B).\n"
            f"\nCOLUMNS:{''.join(blocks)}\n\nVerify, then submit every column."
        )

    # ---- helpers -----------------------------------------------------------------------------------------------

    def supported(self, name: str) -> List[Dict[str, Any]]:
        return [r for key, r in self.checks.items() if key[0] == name and r["verdict"] == "supported"]

    def standing(self, name: str) -> List[Dict[str, Any]]:
        """Checks that found a value for the column on the pages: supported first, then partial; A's value first."""
        records = [r for key, r in self.checks.items() if key[0] == name and r["verdict"] in ("supported", "partial")]
        a_value = self.sources["A"][name]["value"]
        return sorted(records, key=lambda r: (r["verdict"] != "supported", not _same(r["value"], a_value)))

    def arm_values(self, name: str) -> List[str]:
        return [self.sources[o][name]["value"] for o in ("A", "B") if not is_absence(self.sources[o][name]["value"])]

    def unchecked_values(self, name: str) -> List[str]:
        """Extracted values of the column that no verifier check has looked at."""
        checked = [r["value"] for key, r in self.checks.items() if key[0] == name]
        return [v for v in self.arm_values(name) if not any(_same(v, c) for c in checked)]

    def label(self, name: str, value: str) -> str:
        a_ok = _same(value, self.sources["A"][name]["value"])
        b_ok = _same(value, self.sources["B"][name]["value"])
        return "both_correct" if a_ok and b_ok else "A_correct_B_wrong" if a_ok else "B_correct_A_wrong" if b_ok else "both_wrong"

    def final(
        self, name: str, value: str, record: Optional[Dict[str, Any]], *, reasoning: str, decided_by: str,
        verification: Optional[str] = None, review_reason: str = "",
    ) -> Dict[str, Any]:
        value = NOT_REPORTED if is_no_value(value) else value
        verified = bool(record) and record["verdict"] == "supported" and not is_absence(value)
        source: Dict[str, Any] = {"page": None, "modality": "text"}
        attribution: List[Dict[str, Any]] = []
        if record and not is_absence(value):
            quote = record.get("evidence") or record.get("claimed_evidence") or ""
            source = {"page": record["page"], "modality": record["modality"], "verbatim_quote": quote}
            attribution = [
                {"page": page, "modality": record["modality"], **({"verbatim_quote": quote} if i == 0 else {}),
                 "verified": verified, "verdict": record["verdict"]}
                for i, page in enumerate(record.get("pages") or [record["page"]])
            ]
        return {
            "value": value,
            "reasoning": reasoning,
            "verification": verification if verification in VERIFICATIONS else self.label(name, value),
            "source": source,
            "attribution": attribution,
            "verified": verified,
            "needs_review": bool(review_reason),
            "review_reason": review_reason,
            "decided_by": decided_by,
        }

    def unsubmitted(self, name: str, reason: str) -> Dict[str, Any]:
        """A column the agent never got accepted: a verified value is submitted for it; a value the verifier rejected is
        not kept ("Not reported", flagged); anything else keeps the last attempt, flagged."""
        supported = [r for r in self.standing(name) if r["verdict"] == "supported"]
        if supported:
            best = supported[0]
            return self.final(name, best["value"], best, reasoning=f"{reason}; submitted the verified value", decided_by="auto_submit")
        attempt = self.attempts.get(name)
        record = (attempt or {}).get("record")
        if attempt and not (record and record["verdict"] == "not_supported"):
            return self.final(name, attempt["value"], record, reasoning=attempt["reasoning"], decided_by="unsubmitted",
                              verification=attempt.get("verification"), review_reason=f"{reason}; last value was not verified")
        why = f"{reason}; last value failed the page check" if attempt else reason
        return self.final(name, NOT_REPORTED, None, reasoning=why, decided_by="unsubmitted", review_reason=why)

    # ---- tools -------------------------------------------------------------------------------------------------

    def ask_document(self, args: Dict[str, Any]) -> ToolOutput:
        questions = []
        items, note = _items(args, "questions")
        for item in items:
            if str(item.get("question", "")).strip():
                column = item.get("column") if item.get("column") in self.names else ""
                questions.append({"column": column, "question": str(item["question"]).strip()})
        if not questions:
            return ToolOutput({"error": "questions is required: [{column, question}]" + (f" ({note})" if note else "")})
        answers, usage, calls = document_reader.answer_questions(self.chat, self.doc_id, questions, self.definitions)
        self.tool_usage.add(usage)
        self.reader_calls += calls
        dropped = len(questions) - len(answers)
        content: Dict[str, Any] = {"answers": answers}
        if dropped:
            content["note"] = f"only the first {len(answers)} questions were answered; ask the other {dropped} in another call"
        return ToolOutput(content)

    def search_pages(self, args: Dict[str, Any]) -> ToolOutput:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutput({"error": "query is required"})
        hits = retriever.search_chunks(self.doc_id, query)
        return ToolOutput({
            "matches": [
                {"page": hit["page"], "score": round(float(hit["score"]), 3), "lines": _relevant_lines(hit["text"], query)}
                for hit in hits if hit.get("page")
            ],
            "retrieval": hits[0].get("retrieval") if hits else None,
        })

    def _verify(self, claims: List[Claim]) -> None:
        if not claims:
            return
        pdf_path = Path(self.pdf_path) if self.pdf_path else None
        records, usage, calls = verify_claims(self.chat, self.doc_id, claims, self.definitions, pdf_path=pdf_path, image_scale=self.image_scale)
        self.checks.update(records)
        self.tool_usage.add(usage)
        self.verifier_calls += calls

    def _continued(self, claim: Claim) -> Optional[Claim]:
        """The same claim on its page and the next one, when it was rejected on a single page whose text says the table
        continues; None otherwise."""
        record = self.checks.get(claim.key)
        if not record or record["verdict"] != "not_supported" or len(claim.pages) != 1:
            return None
        page = claim.pages[0]
        if page >= self.total_pages:
            return None
        text = retriever.get_page_content(self.doc_id, [page]).get(page, "")
        if not CONTINUED_RE.search(text):
            return None
        return Claim(claim.column, claim.value, (page, page + 1), claim.evidence)

    def verify_attribution(self, args: Dict[str, Any]) -> ToolOutput:
        raw, note = _items(args, "claims")
        claims, problems = [], ([note] if note else [])
        for item in raw[:VERIFY_MAX_CLAIMS]:
            if is_absence(item.get("value")):
                problems.append(f"skipped {item.get('column')!r}: \"{item.get('value')}\" claims absence, which needs no verification; submit it as it is")
                continue
            pages = _claim_pages(item)
            if item.get("column") not in self.names or pages is None:
                problems.append(f"skipped {item!r}: needs a column of this batch, a value and a page (or pages)")
                continue
            claims.append(Claim(item["column"], str(item["value"]), pages, str(item.get("evidence") or "")))
        if len(raw) > VERIFY_MAX_CLAIMS:
            problems.append(f"only the first {VERIFY_MAX_CLAIMS} claims were checked")
        self._verify([claim for claim in claims if claim.key not in self.checks])
        # a table that continues on the next page: a value rejected on the cited page is checked on both pages
        continued = [self._continued(claim) for claim in claims]
        continued = [c for c in continued if c is not None and c.key not in self.checks]
        self._verify(continued)
        content: Dict[str, Any] = {"checks": [_compact(self.checks[claim.key]) for claim in claims + continued]}
        if problems:
            content["problems"] = problems
        return ToolOutput(content)

    def submit_verification(self, args: Dict[str, Any]) -> ToolOutput:
        entries, note = _items(args, "results")
        accepted: List[str] = []
        rejected: List[Dict[str, Any]] = []
        ignored: List[str] = []
        for item in entries:
            name = item.get("column")
            if name not in self.names or name in self.submitted:
                ignored.append(str(name))
                continue
            value = str(item.get("value") or "").strip()
            reasoning = str(item.get("reasoning") or "")
            label = item.get("verification")
            review = bool(item.get("review"))
            review_reason = str(item.get("review_reason") or "").strip()
            if is_absence(value):  # "Not reported", or "No" in a yes/no column: nothing on a page to verify
                standing = self.standing(name)  # supported or partial checks: a value for the column exists on the pages
                unchecked = self.unchecked_values(name)
                if standing:
                    best = standing[0]
                    rejected.append({"column": name, "reason": (
                        f'the verifier found "{best["value"]}" {best["verdict"]} on page(s) {best["pages"]} (they show '
                        f'"{best["page_value"]}"). Submit that value (complete it if partial) instead of "{value or NOT_REPORTED}".')})
                    continue
                if unchecked and not review:
                    rejected.append({"column": name, "reason": (
                        f"an extraction reported {unchecked[0]!r}, which has not been checked. Check it (or the right value) with "
                        f'verify_attribution: if every extracted value fails the check, "{value or NOT_REPORTED}" is accepted.')})
                    continue
                failed = self.arm_values(name)
                reason = review_reason if review else ""
                if failed and not reason:
                    reason = f"every extracted value failed the page check ({', '.join(repr(v) for v in failed[:2])})"
                self.submitted[name] = self.final(name, value, None, reasoning=reasoning, decided_by="agent", verification=label, review_reason=reason)
                accepted.append(name)
                continue

            source = item.get("source") if isinstance(item.get("source"), dict) else {}
            pages = _claim_pages(source)
            same_value = [r for r in self.checks.values() if r["column"] == name and _same(r["value"], value)]
            record = self.checks.get(Claim(name, value, pages).key) if pages else None
            if record is None or record["verdict"] != "supported":  # a check of this value on pages including the cited one
                covering = [r for r in same_value if r["verdict"] == "supported" and (pages is None or set(pages) <= set(r["pages"]))]
                record = covering[0] if covering else record
            if record is None and pages is None and same_value:
                record = same_value[0]
            attempt = {"value": value, "reasoning": reasoning, "verification": label, "record": record}
            if record and record["verdict"] == "supported":
                self.submitted[name] = self.final(name, value, record, reasoning=reasoning, decided_by="agent", verification=label)
                accepted.append(name)
            elif record and review and record["verdict"] in ("partial", "error"):  # right but incomplete, or the check failed to run
                reason = review_reason or f"verifier: {record['verdict']} on page(s) {record['pages']}"
                self.submitted[name] = self.final(name, value, record, reasoning=reasoning, decided_by="agent", verification=label, review_reason=reason)
                accepted.append(name)
            elif record and record["verdict"] == "partial":
                self.attempts[name] = attempt
                rejected.append({
                    "column": name,
                    "reason": f"the verifier found this value incomplete on page(s) {record['pages']}: the pages state \"{record['page_value']}\"",
                    "verifier": _compact(record),
                    "next": "Submit the complete value the pages state (verify it), or resubmit this one with review=true and a review_reason.",
                })
            elif record:
                self.attempts[name] = attempt
                rejected.append({
                    "column": name,
                    "reason": f"the verifier did not support this value on page(s) {record['pages']}; a rejected value is never accepted",
                    "verifier": _compact(record),
                    "next": "If the value is right, find the page(s) that show it (ask the reader; several pages together if it combines "
                    "them) and verify it there. If it answers a different question (statistic, endpoint, population, arm, timepoint), "
                    'drop it: submit another value the verifier supports, or "Not reported" when every extracted value fails.',
                })
            elif review and pages is None:  # no page found for it at all: flag without a check
                self.submitted[name] = self.final(name, value, None, reasoning=reasoning, decided_by="agent", verification=label,
                                                  review_reason=review_reason or "no page given and not verified")
                accepted.append(name)
            else:
                self.attempts[name] = attempt
                where = f"on page(s) {list(pages)}" if pages else "with its page (source.page)"
                rejected.append({"column": name, "reason": f"not verified: call verify_attribution for this value {where} first."})
        remaining = [name for name in self.names if name not in self.submitted]
        content: Dict[str, Any] = {"accepted": accepted, "rejected": rejected, "remaining": remaining}
        if ignored:
            content["ignored"] = ignored
        if note:
            content["note"] = note + ("; submit the remaining columns again as a JSON array" if remaining else "")
        return ToolOutput(content)

    def done(self) -> bool:
        return all(name in self.submitted for name in self.names)

    def results(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {**self.submitted[name], "checks": [_compact(r) for key, r in self.checks.items() if key[0] == name]}
            for name in self.names
        }


def _not_run(reason: str) -> Dict[str, Any]:
    return {
        "value": NOT_REPORTED, "reasoning": reason, "verification": "both_wrong", "source": {"page": None, "modality": "text"},
        "attribution": [], "verified": False, "needs_review": True, "review_reason": reason, "decided_by": "error", "checks": [],
    }
