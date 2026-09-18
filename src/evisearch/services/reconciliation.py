"""
Reconciliation agent: decide each column from Arm A (agent_extractor) and Arm B (search_agent), and verify the
attribution of every final value before it is accepted.

The agent's own context holds the column definitions and both arms' values, reasoning and cited evidence; the paper
itself stays out of it. It reads the paper through tools:
- ask_document: questions answered by a separate model call that has the whole paper (services/document_reader.py).
- search_pages: semantic search over the pages (embedding + reranker), returning the best pages and their most
  relevant lines.
- verify_attribution: a separate model call checks claimed values against one page, text and image
  (services/evidence_check.py): supported / partial / not_supported, with what the page states.
- submit_verification: final values. A value is accepted only when verify_attribution found it supported on the page
  given as its source; "Not reported" is refused while a value for the column is verified as supported. A value the
  verifier does not support is accepted only with review=true and is flagged for a human reviewer.
Columns left unsubmitted keep the agent's last rejected value (or "Not reported") and are flagged for review.

Per column output: value, reasoning, verification (both_correct | A_correct_B_wrong | B_correct_A_wrong | both_wrong:
which source had the final value), source {page, modality, verbatim_quote}, attribution (checked entries), verified,
needs_review, review_reason, decided_by (agent | unsubmitted), checks (verifier records for the column).
The model comes from the "reconciliation" role in src/config/config.py; the reader and verifier use the same model.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.catalog import ConfigError
from src.config.config import AGENT_MAX_TOOL_CALLS, AGENT_MAX_TURNS, MAX_TOKENS, PAGE_IMAGE_SCALE, SELECTION
from src.evisearch.columns import MODALITIES, NOT_REPORTED, column_names, is_no_value
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services import document_reader
from src.evisearch.services.evidence_check import Claim, normalize_value, verify_claims
from src.evisearch.services.extraction_rules import shared_rules
from src.evisearch.services.highlight import resolve_pdf_path
from src.inference import InferenceError, Tool, ToolOutput, ToolSpec, Usage, get_chat, run_tool_loop
from src.retrieval import embedding_retriever as retriever

RECONCILER_VERSION = "verified_tools_v1"  # part of the run settings: results of other versions are not resumed
VERIFICATIONS = ("A_correct_B_wrong", "B_correct_A_wrong", "both_correct", "both_wrong")
VERIFY_MAX_CLAIMS = 30  # per verify_attribution call; split into one verifier call per page
REASONING_CHARS = 2000  # of each arm's reasoning shown to the agent
SEARCH_LINES_PER_PAGE = 4

SYSTEM_PROMPT = """You decide the final value of clinical trial columns. For each column you get its definition and two
independent extractions of the same paper (A and B): the value, the reasoning, and the pages and evidence they cite.
You do not have the paper in front of you. Use your tools:
- ask_document: a reader that has the whole paper (every page's text and image) answers your questions with the
  answer, the pages and the evidence. Ask about several columns in one call.
- search_pages: semantic search over the paper; returns the best matching pages with their most relevant lines.
- verify_attribution: a checker looks at one page (text and image) and says whether it supports a value for a column
  (supported / partial / not_supported) and what the page states. Send many claims in one call.
- submit_verification: final values. A value is accepted only after verify_attribution found it supported on the page
  given as its source.

Deciding a column:
1. Read the definition first: population or subgroup, arm, timepoint, unit, and every part it asks for. The column's
   statistic governs: a rate column takes a rate, an "N (%)" column a count with its percentage, a "(mo)" column a
   duration. A statistic that compares arms (hazard ratio, p value) never goes into a per-arm column.
2. A and B agree: verify the value on its cited page and submit it.
3. A and B differ: check scope first (right population, arm, timepoint), then completeness. Answers that are compatible
   at different levels of detail (a drug class and the drug, a count and the same count with its percentage) are
   merged into the complete value the definition asks for. A different label for the same quantity, or a named subtype
   of the requested measure, is not absence.
4. "Not reported" wins over a value only when the page that value cites does not contain the quantity. Never replace
   a value the verifier supports with "Not reported". "Not reported", and "No" in a yes/no column, claim absence: they
   need no verification.
5. Both say "Not reported": accept it, unless either reasoning mentions a candidate (a number with %, months or n/N;
   "not reached"; "all patients" or a value fixed by the design; enrolment in one country or region). Then ask the
   reader and verify what it finds.
6. Verify the value you choose on the page that shows it, then submit it with that page. If the verifier supports
   none of the values you can find, submit your best value with review=true and a review_reason: a human reviewer
   checks it.

verification says which source had the final value: both_correct (A and B both), A_correct_B_wrong,
B_correct_A_wrong, both_wrong (neither). Work in few calls: verify the cited values of all columns together, ask the
reader about all open questions together, and submit settled columns together. Columns rejected by
submit_verification must be fixed and submitted again."""

FOLLOW_UP = "Continue. Fix rejected columns, verify what you still need, and submit every remaining column."


def _page(raw: Any) -> Optional[int]:
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


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
    keys = ("column", "value", "page", "verdict", "page_value", "evidence", "modality", "reason")
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
                                "evidence": {"type": "string"},
                            },
                            "required": ["column", "value", "page"],
                        },
                    }
                },
                "required": ["claims"],
            },
        ),
        ToolSpec(
            name="submit_verification",
            description="Submit final values for one or more columns. A value is accepted only when verify_attribution found it supported on its source page; the response lists accepted and rejected columns.",
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
                                "review": {"type": "boolean", "description": "true: the verifier did not support this value; send it to a human reviewer"},
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
            attribution = [{**source, "verified": verified, "verdict": record["verdict"]}]
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
        attempt = self.attempts.get(name)
        if attempt:
            return self.final(name, attempt["value"], attempt.get("record"), reasoning=attempt["reasoning"], decided_by="unsubmitted",
                              verification=attempt.get("verification"), review_reason=f"{reason}; last value was not verified")
        return self.final(name, NOT_REPORTED, None, reasoning=reason, decided_by="unsubmitted", review_reason=reason)

    # ---- tools -------------------------------------------------------------------------------------------------

    def ask_document(self, args: Dict[str, Any]) -> ToolOutput:
        questions = []
        for item in args.get("questions") or []:
            if isinstance(item, dict) and str(item.get("question", "")).strip():
                column = item.get("column") if item.get("column") in self.names else ""
                questions.append({"column": column, "question": str(item["question"]).strip()})
        if not questions:
            return ToolOutput({"error": "questions is required: [{column, question}]"})
        answers, usage, call = document_reader.answer_questions(self.chat, self.doc_id, questions, self.definitions)
        self.tool_usage.add(usage)
        self.reader_calls.append(call)
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

    def verify_attribution(self, args: Dict[str, Any]) -> ToolOutput:
        raw = args.get("claims") or []
        claims, problems = [], []
        for item in raw[:VERIFY_MAX_CLAIMS]:
            if isinstance(item, dict) and is_absence(item.get("value")):
                problems.append(f"skipped {item.get('column')!r}: \"{item.get('value')}\" claims absence, which needs no verification; submit it as it is")
                continue
            if not isinstance(item, dict) or item.get("column") not in self.names or _page(item.get("page")) is None:
                problems.append(f"skipped {item!r}: needs a column of this batch, a value and a page")
                continue
            claims.append(Claim(item["column"], str(item["value"]), _page(item["page"]), str(item.get("evidence") or "")))
        if len(raw) > VERIFY_MAX_CLAIMS:
            problems.append(f"only the first {VERIFY_MAX_CLAIMS} claims were checked")
        new = [claim for claim in claims if claim.key not in self.checks]
        if new:
            pdf_path = Path(self.pdf_path) if self.pdf_path else None
            records, usage, calls = verify_claims(self.chat, self.doc_id, new, self.definitions, pdf_path=pdf_path, image_scale=self.image_scale)
            self.checks.update(records)
            self.tool_usage.add(usage)
            self.verifier_calls += calls
        content: Dict[str, Any] = {"checks": [_compact(self.checks[claim.key]) for claim in claims]}
        if problems:
            content["problems"] = problems
        return ToolOutput(content)

    def submit_verification(self, args: Dict[str, Any]) -> ToolOutput:
        raw = args.get("results")
        entries = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
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
                supported = self.supported(name)
                if supported and not review:
                    best = supported[0]
                    rejected.append({
                        "column": name,
                        "reason": f'"{best["value"]}" is verified on page {best["page"]} (the page shows "{best["page_value"]}"). '
                        "Submit it, or set review=true with a review_reason if it answers a different question.",
                    })
                    continue
                reason = (review_reason or f"{value or NOT_REPORTED} although a value was verified") if review else ""
                self.submitted[name] = self.final(name, value, None, reasoning=reasoning, decided_by="agent", verification=label, review_reason=reason)
                accepted.append(name)
                continue

            source = item.get("source") if isinstance(item.get("source"), dict) else {}
            page = _page(source.get("page"))
            if page is None:
                match = [r for r in self.supported(name) if _same(r["value"], value)]
                page = match[0]["page"] if match else None
            record = self.checks.get(Claim(name, value, page).key) if page else None
            attempt = {"value": value, "reasoning": reasoning, "verification": label, "record": record}
            if record and record["verdict"] == "supported":
                self.submitted[name] = self.final(name, value, record, reasoning=reasoning, decided_by="agent", verification=label)
                accepted.append(name)
            elif record and review:
                reason = review_reason or f"verifier: {record['verdict']} on page {record['page']}"
                self.submitted[name] = self.final(name, value, record, reasoning=reasoning, decided_by="agent", verification=label, review_reason=reason)
                accepted.append(name)
            elif record:
                self.attempts[name] = attempt
                rejected.append({
                    "column": name,
                    "reason": f"the verifier did not support this value on page {page}",
                    "verifier": _compact(record),
                    "next": "Correct the value or the page (page_value is what that page states), or resubmit with review=true and a review_reason.",
                })
            else:
                self.attempts[name] = attempt
                where = f"on page {page}" if page else "with its page (source.page)"
                rejected.append({"column": name, "reason": f"not verified: call verify_attribution for this value {where} first."})
        remaining = [name for name in self.names if name not in self.submitted]
        content: Dict[str, Any] = {"accepted": accepted, "rejected": rejected, "remaining": remaining}
        if ignored:
            content["ignored"] = ignored
        return ToolOutput(content)

    def done(self) -> bool:
        return all(name in self.submitted for name in self.names)

    def results(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {**self.submitted[name], "checks": [_compact(r) for key, r in self.checks.items() if key[0] == name]}
            for name in self.names
        }


def run_reconciliation_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    source_a_data: Dict[str, Dict[str, Any]],
    source_b_data: Dict[str, Dict[str, Any]],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Reconcile one batch. Returns ({column: result}, usage), usage including the reader and verifier calls."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("reconciliation", model)
    except (ConfigError, InferenceError) as exc:
        return {name: _not_run(f"reconciliation not run: {exc}") for name in names}, Usage().to_dict()

    use_images = SELECTION.option("reconciliation_page_images") == "auto" and chat.capabilities.images
    session = _ReconciliationSession(
        chat, doc_id, batch_columns, definitions_map, source_a_data, source_b_data, PAGE_IMAGE_SCALE if use_images else None
    )
    specs = {spec.name: spec for spec in tool_specs(names)}
    loop = run_tool_loop(
        chat,
        system=SYSTEM_PROMPT + shared_rules(),
        user=session.user_prompt(),
        tools=[
            Tool(specs["ask_document"], session.ask_document),
            Tool(specs["search_pages"], session.search_pages),
            Tool(specs["verify_attribution"], session.verify_attribution),
            Tool(specs["submit_verification"], session.submit_verification),
        ],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["reconciliation"],
        follow_up=FOLLOW_UP,
        is_done=session.done,
        finish_tool="submit_verification",
    )
    reason = f"reconciler did not submit ({loop.stopped_by}{': ' + loop.error if loop.error else ''})"
    for name in names:
        if name not in session.submitted:
            session.submitted[name] = session.unsubmitted(name, reason)
    usage = Usage().add(loop.usage).add(session.tool_usage)
    results = session.results()

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "reconciler": RECONCILER_VERSION,
                "stopped_by": loop.stopped_by,
                "error": loop.error,
                "reader_calls": session.reader_calls,
                "verifier_calls": session.verifier_calls,
                "checks": list(session.checks.values()),
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in loop.transcript if e["role"] == "tool"],
                "conversation": loop.transcript,
                "calls": loop.calls,
                "tool_usage": session.tool_usage.to_dict(),
                "results": results,
            },
        )
    return results, usage.to_dict()


def _not_run(reason: str) -> Dict[str, Any]:
    return {
        "value": NOT_REPORTED, "reasoning": reason, "verification": "both_wrong", "source": {"page": None, "modality": "text"},
        "attribution": [], "verified": False, "needs_review": True, "review_reason": reason, "decided_by": "error", "checks": [],
    }
