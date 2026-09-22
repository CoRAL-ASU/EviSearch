"""The Reconciliation Agent, in two phases: an independent extraction, then an adjudication over three answers.

  phase 1  The columns the two extraction agents disagree on, or both leave empty, are answered from the paper with
           A's and B's answers hidden, so what this stage brings to the decision is a reading of its own rather than
           a preference between two it has already seen. Columns the agents agree on go straight to phase 2.
  phase 2  A's and B's answers are revealed alongside it. The stage decides each column and reports a verdict on
           all three answers, its own included.

What it reads with: Agent B's search (`search_chunks`, whole pages) and `get_pages`, which returns a page's text and
its rendered image, the only way to read a value printed inside a figure or a table captured as a picture.

What verify_attribution is for: a second reader opens the pages a value cites, writes the column's answer itself, and
returns it with the page, the quotation and the modality. Those fields are what the viewer highlights on the PDF. So
the submission gate is provenance, not agreement: a value is admitted once it has been read on its page, and when the
second reading does not reproduce it the value is admitted with the extracting agent's own quotation, marked, and
routed to review rather than dropped (Appendix B of the paper measures the agreement-gated alternative).

An absence is an answer and needs a basis: "Not reported" is admitted when the stage names the pages it read and what
they state instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.config.catalog import ConfigError
from src.config.config import (AGENT_MAX_TOOL_CALLS, AGENT_MAX_TURNS, MAX_TOKENS, PAGE_IMAGE_SCALE,
                               RECONCILIATION_MAX_PAGE_IMAGES, SELECTION)
from src.evisearch.columns import column_names, is_no_value
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services import page_images
from src.evisearch.services.extraction_rules import shared_rules
from src.evisearch.services.evidence_check import Claim
from src.evisearch.services.reconciliation import (
    REASONING_CHARS,
    _ReconciliationSession,
    _claim_pages,
    _items,
    _not_run,
    _same,
    is_absence,
    tool_specs,
)
from src.inference import ImagePart, InferenceError, Tool, ToolOutput, ToolSpec, Usage, get_chat, run_tool_loop
from src.retrieval import embedding_retriever as retriever

def own_reading_scope() -> str:
    """Which columns phase 1 answers for itself: `contested`, the columns the two agents disagree on plus the ones both
    left empty. Where both agents agree the stage adjudicates without a reading of its own."""
    return "contested"


def needs_own_reading(name: str, source_a: Dict[str, Any], source_b: Dict[str, Any], scope: str) -> bool:
    """Whether phase 1 answers this column for itself, under the run's scope."""
    a = _value_of(source_a.get(name))
    b = _value_of(source_b.get(name))
    both_silent = is_absence(a) and is_absence(b)
    if scope == "both_silent":
        return both_silent
    return both_silent or _squash(a) != _squash(b)  # contested


def contested(name: str, source_a: Dict[str, Any], source_b: Dict[str, Any]) -> bool:
    """Whether a column needs the stage's own reading: the agents differ, or neither of them answered."""
    return needs_own_reading(name, source_a, source_b, "contested")


def _value_of(col: Any) -> str:
    return str((col or {}).get("value") or "") if isinstance(col, dict) else ""


def _squash(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower().rstrip(".").replace("%", "")


# Part of the run settings, so saved results made by another reconciler are never resumed into a run.
RECONCILER_VERSION = "own_reading_v5_contested"

# what phase 2 says about each of the three answers it now holds
VERDICTS = ("correct", "incomplete", "wrong", "no_answer")
SOURCES = ("own", "A", "B", "merged")

FINDINGS_PROMPT = f"""You extract clinical trial values from a research paper.

You are given columns, each with a definition. Fill every column.

WORKFLOW
1. get_pages([1, 2]) first, and fill what those pages answer.
2. search_chunks with terms from a column's definition for the columns still open. Do not search for what you
   already have.
3. get_pages to read a page in full with its image. Use it for values printed in a figure, a Kaplan-Meier panel, or
   a table captured as a picture: the parsed text often loses those. At most {RECONCILIATION_MAX_PAGE_IMAGES} page
   images per batch, after which get_pages still returns the text.
4. verify_attribution on the values you found, to confirm the page and capture the quote the value will be cited
   with. It reads the page you name and writes the column's answer itself. Use it on values you have already
   located - it is not a way to find them.
5. submit_findings when every column has a value or has been established as not reported.

RULES
- Copy the value as the paper prints it.
- Every column gets either a value with the page it is on, or the list of pages you opened looking for it. Do not
  leave a column unattempted.
- Never write a value the paper does not state. Do not read a number off a curve. Do not compute one unless the
  knowledge notes license that computation; when they do, show the arithmetic and name the row and column of every
  number in it.
- Give the page(s) each value is on and the text or table cell you took it from, copied as printed.
- When the parsed text and the page image disagree, trust the image.
- Search where the answer would be, not only where a word matches: baseline tables for characteristics, results
  tables and figure panels for outcomes, the methods for design, the discussion for durations."""

RECONCILE_PROMPT = f"""You review two independent extractions of the same paper and decide each column's final value.

You have already extracted these columns yourself. For each column you now see your answer and two candidates, A and
B (anonymous), with their reasoning and the pages and evidence they cite.

WORKFLOW
1. Where all three agree, submit that value.
2. Where they differ, read the pages the differing values cite - with get_pages, or with verify_attribution, which
   gives you a second reader's answer for that column from those pages. You have a fresh budget of
   {RECONCILIATION_MAX_PAGE_IMAGES} page images here; the pages you opened earlier are not in this context.
3. Every value you submit must have been through verify_attribution, so the cell ships with the page and the quote a
   reader can check it against.
4. submit_verification with, per column: the final value, which answer it came from, and a verdict on each of the
   three answers.

RULES
- verify_attribution is a second reader confined to the pages you name. Where its answer differs from a claim, that
  is a disagreement for you to settle by looking - not a ruling. Its failure to find a value is not evidence the
  paper omits one.
- Answers compatible at different levels of detail (a drug class and the drug, one endpoint variant and both, a
  count and the same count with its percentage) are merged into the complete value the definition asks for; set
  final_source to "merged".
- "Not reported" is a valid final value when you have read the pages a candidate cites and they do not state one.
  Give absence_basis: the pages you read and what they say instead.
- Judge each of the three answers on its merits - own_verdict, a_verdict, b_verdict, each one of correct,
  incomplete, wrong or no_answer. Your own answer gets the same treatment as the other two."""

FINDINGS_FOLLOW_UP = "Continue. Answer the remaining columns and submit them with submit_findings."
RECONCILE_FOLLOW_UP = "Continue. Fix rejected columns, verify what you still need, and submit every remaining column."


def submit_spec_v5(names: List[str]) -> ToolSpec:
    """v4's submit_verification with two changes: the description no longer says a value needs a supporting verdict
    (the second reading informs the decision, it does not rule on it), and an absence carries the reading that
    justifies it."""
    spec = next(s for s in tool_specs(names) if s.name == "submit_verification")
    item = spec.parameters["properties"]["results"]["items"]
    item["properties"]["absence_basis"] = {
        "type": "object",
        "description": 'Required with "Not reported" when another answer stated a value: the reading that rules it out.',
        "properties": {
            "pages": {"type": "array", "items": {"type": "integer"}, "description": "pages you read"},
            "page_says": {"type": "string", "description": "what those pages state for this column instead"},
        },
    }
    item["properties"]["final_source"] = {"type": "string", "enum": list(SOURCES)}
    for key in ("own_verdict", "a_verdict", "b_verdict"):
        item["properties"][key] = {"type": "string", "enum": list(VERDICTS)}
    return ToolSpec(
        name=spec.name,
        description=("Submit final values for one or more columns. Every value must have been through "
                     "verify_attribution, so the cell ships with the page and quote a reader can check it against. "
                     '"Not reported" needs absence_basis when another answer stated a value. The response lists '
                     "accepted and rejected columns."),
        parameters=spec.parameters,
    )


def findings_spec(names: List[str]) -> ToolSpec:
    return ToolSpec(
        name="submit_findings",
        description="Your own answers for the columns, before you see anyone else's.",
        parameters={
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "enum": names},
                            "value": {"type": "string", "description": 'the answer as the paper prints it, or "" when the paper states none'},
                            "pages": {"type": "array", "items": {"type": "integer"}, "description": "page(s) the answer is on"},
                            "evidence": {"type": "string", "description": "the text or table cell the answer was taken from"},
                            "looked_at": {"type": "array", "items": {"type": "integer"}, "description": "pages you examined for this column"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["column", "value", "reasoning"],
                    },
                }
            },
            "required": ["findings"],
        },
    )


def paper_tool_specs() -> List[ToolSpec]:
    """The two tools v5 reads the paper with, in both phases. search_chunks is Agent B's, verbatim, so the arbiter sees
    exactly what the agent it is judging saw; get_pages is Agent B's get_chunks_by_page with the page image added."""
    return [
        ToolSpec(
            name="search_chunks",
            description="Semantic search over the paper's pages. Returns the best matching pages with full content. "
                        "Pages you already have show 'already provided; check your context'.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'median overall survival abiraterone months'"},
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="get_pages",
            description=f"Open pages by number: each page's full content and its rendered image. The image is the only "
                        f"way to read a value printed inside a figure or a table captured as a picture. At most "
                        f"{RECONCILIATION_MAX_PAGE_IMAGES} page images per batch; past that the text still comes back.",
            parameters={
                "type": "object",
                "properties": {
                    "page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "1-based page numbers, e.g. [1, 2, 3]"},
                },
                "required": ["page_numbers"],
            },
        ),
    ]


class _PaperSession(_ReconciliationSession):
    """v4's session with Agent B's reading tools bolted on, shared by both v5 phases.

    v4's own tools stay on the class (v4 uses them and is frozen), but v5 hands the model only these two plus its
    submission tool. What changed and why is in the module docstring: search_pages showed at most four query-matching
    lines per page, there was no way to open a page by number, and ask_document was Arm A under another name.

    search_chunks below is Agent B's `_SearchSession.search_chunks` (services/search.py) line for line - the same
    whole-page text, the same "[Page N, score=...]" framing, the same pages_sent set so a page is never sent twice, and
    the same on_evict callback so a page the loop drops to fit the context can be fetched again. Duplicated rather than
    imported because Agent B's session is built around its own submission state; tests assert the two payloads match.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pages_sent: Set[int] = set()
        self.images_attached: List[int] = []  # pages whose image was sent, in order, across the whole batch

    def _forget(self, pages: List[int]) -> Callable[[], None]:
        """Undo the bookkeeping of a tool output the loop had to drop to fit the context: the pages may be fetched
        again, and the page images that went with them no longer count against the batch's budget."""
        def undo() -> None:
            self.pages_sent.difference_update(pages)
            for page in pages:
                if page in self.images_attached:
                    self.images_attached.remove(page)
        return undo

    # ---- search_chunks: Agent B's tool, unchanged ---------------------------------------------------------------

    def search_chunks(self, args: Dict[str, Any]) -> ToolOutput:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutput({"error": "query is required"})
        hits = retriever.search_chunks(self.doc_id, query)
        if not hits:
            return ToolOutput({"matches": [], "formatted_chunks": "No matching chunks found. Try different search terms.", "pages_returned": []})
        parts, returned = [], []
        for hit in hits:
            page = hit["page"]
            if page in self.pages_sent:
                parts.append(f"[Page {page}, score={hit['score']:.2f}] already provided; check your context.")
            else:
                parts.append(f"[Page {page}, score={hit['score']:.2f}]\n{hit['text']}")
                returned.append(page)
        self.pages_sent.update(returned)
        formatted = (
            "\n\n---\n\n".join(parts)
            if returned
            else "All retrieved pages have already been provided. Try a different query or submit with what you have."
        )
        content = {
            "matches": [{"page": h["page"], "score": h["score"]} for h in hits],
            "retrieval": hits[0]["retrieval"],  # "rerank", or "embedding (reranker unavailable: ...)"
            "formatted_chunks": formatted,
            "pages_returned": returned,
        }
        return ToolOutput(content, on_evict=self._forget(returned))

    # ---- get_pages: Agent B's get_chunks_by_page, plus the page image -------------------------------------------

    def _images_for(self, pages: List[int]) -> Tuple[List[ImagePart], List[int], Optional[str]]:
        """Rendered images for the pages just returned, within the batch's budget: (attachments, pages shown, note).

        The budget is per session, not per call, because the images stay in that session's context for the rest of it.
        The two phases are separate conversations, so each gets its own budget. Whatever is left out still had its text
        returned, and the note says so - a silent drop would leave the model believing it had looked at a figure it
        never saw.
        """
        if not pages:
            return [], [], None
        if not self.pdf_path or not self.image_scale:
            return [], [], "page images are not available in this run; the text above is all this tool can show."
        room = RECONCILIATION_MAX_PAGE_IMAGES - len(self.images_attached)
        if room <= 0:
            return [], [], (f"the page-image budget for this batch is spent ({RECONCILIATION_MAX_PAGE_IMAGES} images), "
                            f"so page(s) {pages} came back as text only.")
        try:
            rendered = page_images.render_pdf_pages_to_png(Path(self.pdf_path), pages[:room], self.image_scale)
        except Exception as exc:  # a page that will not render must not lose the model the page's text
            return [], [], f"the page images could not be rendered ({exc}); the text above is all this tool can show."
        shown = [page for page, _ in rendered]
        self.images_attached += shown
        left_out = [page for page in pages if page not in shown]
        note = None
        if left_out:
            note = (f"page image(s) for {left_out} not attached: at most {RECONCILIATION_MAX_PAGE_IMAGES} page images "
                    f"per batch. Their text is above.")
        return [ImagePart(png) for _, png in rendered], shown, note

    def get_pages(self, args: Dict[str, Any]) -> ToolOutput:
        pages = sorted({int(p) for p in args.get("page_numbers") or [] if isinstance(p, (int, float))})
        content_map = retriever.get_page_content(self.doc_id, pages)
        parts, returned = [], []
        for page in pages:
            if page < 1 or page > self.total_pages:
                parts.append(content_map.get(page, f"Page {page} does not exist. Document has {self.total_pages} pages."))
            elif page in self.pages_sent:
                parts.append(f"[Page {page}] already provided; check your context.")
            else:
                parts.append(f"[Page {page}]\n{content_map.get(page, '')}")
                returned.append(page)
        self.pages_sent.update(returned)
        attachments, shown, note = self._images_for(returned)
        content: Dict[str, Any] = {
            "formatted_chunks": "\n\n---\n\n".join(parts),
            "pages_returned": returned,
            "page_images": shown,
        }
        if note:
            content["images_note"] = note
        return ToolOutput(content, attachments=attachments, on_evict=self._forget(returned))


class _FindingsSession(_PaperSession):
    """Phase 1. The paper-reading session without its sources: it searches and opens pages, and the checker is
    withheld - that ordering is the whole point of the design."""

    def __init__(self, chat: Any, doc_id: str, batch_columns: List[Dict[str, Any]], definitions_map: Dict[str, str],
                 image_scale: Optional[float]):
        super().__init__(chat, doc_id, batch_columns, definitions_map, {}, {}, image_scale)
        self.found: Dict[str, Dict[str, Any]] = {}

    def user_prompt(self) -> str:
        blocks = [f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}"
                  for i, name in enumerate(self.names, 1)]
        return (
            f"Answer the following columns from the paper. It has {self.total_pages} pages.\n"
            f"\nCOLUMNS:{''.join(blocks)}\n\nAnswer every column, then submit them with submit_findings."
        )

    def submit_findings(self, args: Dict[str, Any]) -> ToolOutput:
        entries, note = _items(args, "findings")
        accepted, ignored = [], []
        for item in entries:
            name = item.get("column")
            if name not in self.names or name in self.found:
                ignored.append(str(name))
                continue
            value = str(item.get("value") or "").strip()
            pages = _claim_pages(item)
            looked = [p for p in (item.get("looked_at") or []) if isinstance(p, int)]
            self.found[name] = {
                "value": "" if is_no_value(value) else value,
                "pages": list(pages) if pages else [],
                "evidence": str(item.get("evidence") or "").strip(),
                "looked_at": looked or (list(pages) if pages else []),
                "reasoning": str(item.get("reasoning") or "").strip(),
            }
            accepted.append(name)
        remaining = [name for name in self.names if name not in self.found]
        content: Dict[str, Any] = {"accepted": accepted, "remaining": remaining}
        if ignored:
            content["ignored"] = ignored
        if note:
            content["note"] = note
        return ToolOutput(content)

    def done(self) -> bool:
        return all(name in self.found for name in self.names)

    def findings(self) -> Dict[str, Dict[str, Any]]:
        """Every column's finding; a column the model never submitted is recorded as unread, not as an absence."""
        return {
            name: self.found.get(name) or {"value": "", "pages": [], "evidence": "", "looked_at": [],
                                           "reasoning": "not answered in phase 1", "unread": True}
            for name in self.names
        }


class _ReconcileSession(_PaperSession):
    """Phase 2. The same reading session, plus the stage's own findings, three-way verdicts, and absence treated as
    an answer."""

    def __init__(self, *args: Any, own: Dict[str, Dict[str, Any]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.own = own
        self.verdicts: Dict[str, Dict[str, Any]] = {}

    # ---- prompt ----------------------------------------------------------------------------------------------

    def describe_own(self, name: str) -> str:
        own = self.own.get(name) or {}
        if own.get("skipped"):
            return "  YOURS: (not read - the two extractions already agreed on this column)"
        if own.get("unread"):
            return "  YOURS: (you did not answer this column)"
        value = own.get("value") or ""
        head = f'  YOURS: "{value}"' if value else "  YOURS: the paper states no answer for this column"
        lines = [head]
        if own.get("pages"):
            lines.append(f"     on page(s) {own['pages']}" + (f", evidence \"{own['evidence']}\"" if own.get("evidence") else ""))
        if own.get("looked_at"):
            lines.append(f"     looked at page(s) {own['looked_at']}")
        if own.get("reasoning"):
            lines.append(f"     your reasoning: {own['reasoning'][:REASONING_CHARS]}")
        return "\n".join(lines)

    def user_prompt(self) -> str:
        blocks = [
            f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}\n"
            f"{self.describe_own(name)}\n{self.describe(name)}"
            for i, name in enumerate(self.names, 1)
        ]
        return (
            f"Decide the following columns. The paper has {self.total_pages} pages. YOURS is the answer you found "
            f"yourself; A and B are two other extractions and are anonymous.\n"
            f"\nCOLUMNS:{''.join(blocks)}\n\nDecide and submit every column."
        )

    # ---- verdicts --------------------------------------------------------------------------------------------

    def own_value(self, name: str) -> str:
        return (self.own.get(name) or {}).get("value") or ""

    def _verdict_block(self, name: str, item: Dict[str, Any], value: str) -> Dict[str, Any]:
        """The submitted verdicts, defaulted from the values themselves when the model left one out."""
        def clean(key: str, answer: str) -> str:
            given = str(item.get(key) or "").strip().lower()
            if given in VERDICTS:
                return given
            if is_no_value(answer):
                return "no_answer"
            return "correct" if _same(answer, value) else "wrong"

        source = str(item.get("final_source") or "").strip()
        if source not in SOURCES:
            source = ("own" if _same(value, self.own_value(name)) else
                      "A" if _same(value, self.sources["A"][name]["value"]) else
                      "B" if _same(value, self.sources["B"][name]["value"]) else "merged")
        return {
            "final_source": source,
            "own_verdict": clean("own_verdict", self.own_value(name)),
            "a_verdict": clean("a_verdict", self.sources["A"][name]["value"]),
            "b_verdict": clean("b_verdict", self.sources["B"][name]["value"]),
        }

    def citation_for(self, name: str, value: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The verify_attribution record this value will be cited with, or None if it has never been read.

        A record exists for exactly this value on the pages it names, or on pages covering them. The verdict does not
        enter here: what the record supplies is the page, the quote and the modality the cell ships with, and a
        second reader's own answer next to it. Whether that reader agreed is recorded, not enforced.
        """
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        pages = _claim_pages(source)
        if pages:
            exact = self.checks.get(Claim(name, value, pages).key)
            if exact:
                return exact
        same = [r for r in self.checks.values() if r["column"] == name and _same(r["value"], value)]
        if pages:
            covering = [r for r in same if set(pages) <= set(r["pages"])]
            if covering:
                return covering[0]
        return same[0] if same else None

    def final(self, name: str, value: str, record: Optional[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        result = super().final(name, value, record, **kwargs)
        # The record's `evidence` is the second reader's quote for its OWN answer. When we ship a value that reader
        # did not arrive at, that quote does not support this cell - cite the extraction's own quote instead.
        if record and not is_absence(value) and not (_same(record.get("value", ""), value)
                                                     and record.get("verdict") == "supported"):
            quote = str(record.get("claimed_evidence") or record.get("evidence") or "")
            if isinstance(result.get("source"), dict) and result["source"].get("page"):
                result["source"]["verbatim_quote"] = quote
            for i, entry in enumerate(result.get("attribution") or []):
                if i == 0:
                    entry["verbatim_quote"] = quote
        own = self.own.get(name) or {}
        result["own_finding"] = {k: own.get(k) for k in ("value", "pages", "evidence", "looked_at", "unread")}
        result.update(self.verdicts.get(name) or {})
        self._flag_for_review(name, result)
        return result

    def _flag_for_review(self, name: str, result: Dict[str, Any]) -> None:
        """Send a cell to the reviewer when the two extractions disagreed.

        Measured on R4: the cells where A and B differ are 20.7 per paper (16% of the table) and hold 38% of the
        table's errors, at 25% precision - one in four cells the reviewer opens is genuinely wrong, the best of any
        policy tried. The arbiter's own `needs_review` reached 2.9 cells per paper but only 9% of the errors, so a
        reviewer working it fixed almost nothing.

        Cells where both agents left the column empty are deliberately NOT flagged. They hold real errors - 26% of
        them - but they are 63.9 cells per paper at 5.6% precision, so nineteen of every twenty the reviewer opens
        are correctly empty. That is worse per cell read than reading the whole 133-column table, which at least
        catches everything else too. Those errors need better retrieval, not more human reading.
        """
        a, b = self.sources["A"][name]["value"], self.sources["B"][name]["value"]
        if _squash(a) == _squash(b):
            return
        note = f'the two extractions disagreed (A: "{a[:60]}" / B: "{b[:60]}")'
        existing = str(result.get("review_reason") or "").strip()
        result["needs_review"] = True
        result["review_reason"] = f"{existing}; {note}" if existing else note

    # ---- submission ------------------------------------------------------------------------------------------

    def submit_verification(self, args: Dict[str, Any]) -> ToolOutput:
        """Two gates, neither of them about who is right.

        A value ships once it has been through verify_attribution - not once that reading agreed with it. The gate is
        provenance: the record supplies the page, the quote and the modality the cell is cited with, and without one
        the cell reaches the viewer with nothing to click. Measured on R4, 674 of 695 shipped values carried a
        citation and every one of them came from a check; the 21 without one had no page and no quote at all. Where
        the second reading disagreed and we ship anyway, the cell goes to the reviewer rather than being refused -
        those refusals cost 25 correct values across two runs and saved 22 wrong ones.

        An absence ships once the stage says what it read instead. v4 refused an absence while a stated value stood
        unchecked, which let a *checked* value be blanked for free - six of the nine cells the checker broke went out
        that way. The condition is now unexplained rather than unchecked, and only a reading satisfies it.
        """
        entries, note = _items(args, "results")
        handled: List[Dict[str, Any]] = []
        for item in entries:
            name = item.get("column")
            value = str(item.get("value") or "").strip()
            if name not in self.names or name in self.submitted:
                handled.append({"column": str(name), "accepted": False, "reason": "not a column of this batch, or already submitted"})
                continue
            self.verdicts[name] = self._verdict_block(name, item, value)
            reasoning = str(item.get("reasoning") or "")
            review_reason = str(item.get("review_reason") or "").strip()

            if is_absence(value):
                stated = [v for v in (self.sources["A"][name]["value"], self.sources["B"][name]["value"],
                                      self.own_value(name)) if v and not is_absence(v)]
                basis = item.get("absence_basis") if isinstance(item.get("absence_basis"), dict) else {}
                pages = [p for p in (basis.get("pages") or []) if isinstance(p, int)]
                says = str(basis.get("page_says") or "").strip()
                if stated and not (pages and says) and not item.get("review"):
                    handled.append({"column": name, "accepted": False, "reason": (
                        f'"{stated[0][:80]}" was reported for this column. Give absence_basis - the page(s) you read '
                        f'and what they say for this column instead - or submit a value.')})
                    continue
                if stated and not review_reason:
                    review_reason = (f'shipped empty over "{stated[0][:60]}"'
                                     + (f"; read page(s) {pages}: {says[:120]}" if pages and says else ""))
                self.submitted[name] = self.final(
                    name, value, None, reasoning=reasoning, decided_by="own_reading",
                    verification=item.get("verification"), review_reason=review_reason,
                )
                handled.append({"column": name, "accepted": True})
                continue

            record = self.citation_for(name, value, item)
            if record is None:
                handled.append({"column": name, "accepted": False, "reason": (
                    f'"{value[:80]}" has not been through verify_attribution. Every value ships with the page and '
                    f"the quote a reader can check it against: verify it on the page it is printed on, then submit "
                    f"it again.")})
                self.attempts[name] = {"value": value, "reasoning": reasoning,
                                       "verification": item.get("verification"), "record": None}
                continue
            if record["verdict"] != "supported" and not review_reason:
                review_reason = (f'the second reading of page(s) {record["pages"]} gave '
                                 f'"{str(record.get("page_value") or "")[:60]}" ({record["verdict"]})')
            self.submitted[name] = self.final(
                name, value, record, reasoning=reasoning, decided_by="agent",
                verification=item.get("verification"), review_reason=review_reason,
            )
            handled.append({"column": name, "accepted": True})

        content: Dict[str, Any] = {
            "accepted": [h["column"] for h in handled if h.get("accepted")],
            "rejected": [{"column": h["column"], "reason": h["reason"]} for h in handled if not h.get("accepted")],
            "remaining": [n for n in self.names if n not in self.submitted],
        }
        if note:
            content["note"] = note
        return ToolOutput(content)

    def results(self) -> Dict[str, Dict[str, Any]]:
        out = super().results()
        for name, record in out.items():
            record.setdefault("own_finding", {k: (self.own.get(name) or {}).get(k)
                                              for k in ("value", "pages", "evidence", "looked_at", "unread")})
            record.update(self.verdicts.get(name) or {})
        return out


def run_reconciliation_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
    source_a_data: Dict[str, Dict[str, Any]],
    source_b_data: Dict[str, Dict[str, Any]],
    log_path: Optional[Path] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Reconcile one batch in two phases. Returns ({column: result}, usage)."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("reconciliation", model)
    except (ConfigError, InferenceError) as exc:
        return {name: _not_run(f"reconciliation not run: {exc}") for name in names}, Usage().to_dict()

    use_images = SELECTION.option("reconciliation_page_images") == "auto" and chat.capabilities.images
    scale = PAGE_IMAGE_SCALE if use_images else None
    specs = {spec.name: spec for spec in list(tool_specs(names)) + paper_tool_specs()}
    usage = Usage()

    # ---- phase 1: the stage's own reading, with no access to A or B ------------------------------------------
    # Which columns it reads for itself. Under `contested` the agreed ones are left to phase 2, where the agents'
    # matching answer stands: the R3 numbers say the arbiter is already at 99.9% on cells the agents agree and are
    # right about, so reading those again risks more than it can win.
    scope = own_reading_scope()
    to_read = batch_columns
    skipped: List[str] = []
    if scope != "all":
        to_read = [c for c in batch_columns
                   if needs_own_reading(c.get("column_name", ""), source_a_data, source_b_data, scope)]
        skipped = [c.get("column_name", "") for c in batch_columns if c not in to_read]
    not_read = {
        name: {"value": "", "pages": [], "evidence": "", "looked_at": [],
               "reasoning": "the two extractions agreed; this stage did not read it", "skipped": True}
        for name in skipped
    }
    phase1 = None
    if to_read:
        read_names = column_names(to_read)
        reading = _FindingsSession(chat, doc_id, to_read, definitions_map, scale)
        phase1 = run_tool_loop(
            chat,
            system=FINDINGS_PROMPT + shared_rules(columns=read_names, role="auditor"),
            user=reading.user_prompt(),
            tools=[
                Tool(specs["search_chunks"], reading.search_chunks),
                Tool(specs["get_pages"], reading.get_pages),
                Tool(specs["verify_attribution"], reading.verify_attribution),
                Tool(findings_spec(read_names), reading.submit_findings),
            ],
            max_turns=AGENT_MAX_TURNS,
            max_tool_calls=AGENT_MAX_TOOL_CALLS,
            max_tokens=MAX_TOKENS["reconciliation"],
            follow_up=FINDINGS_FOLLOW_UP,
            is_done=reading.done,
            finish_tool="submit_findings",
        )
        own = {**not_read, **reading.findings()}
        usage.add(phase1.usage).add(reading.tool_usage)
    else:  # every column in this batch was agreed: there is no reading pass to make
        own = not_read

    # ---- phase 2: A and B revealed, its own answer already committed -----------------------------------------
    session = _ReconcileSession(chat, doc_id, batch_columns, definitions_map, source_a_data, source_b_data, scale, own=own)
    if phase1 is not None:
        # a page phase 1 already had read carries its record forward: the citation is paid for once, and phase 2 sees
        # what that reading said rather than calling for it again
        session.checks.update(reading.checks)
        session.verifier_calls.extend(reading.verifier_calls)
    phase2 = run_tool_loop(
        chat,
        system=RECONCILE_PROMPT + shared_rules(columns=names, role="agent"),
        user=session.user_prompt(),
        tools=[
            Tool(specs["search_chunks"], session.search_chunks),
            Tool(specs["get_pages"], session.get_pages),
            Tool(specs["verify_attribution"], session.verify_attribution),
            Tool(submit_spec_v5(names), session.submit_verification),
        ],
        max_turns=AGENT_MAX_TURNS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
        max_tokens=MAX_TOKENS["reconciliation"],
        follow_up=RECONCILE_FOLLOW_UP,
        is_done=session.done,
        finish_tool="submit_verification",
    )
    reason = f"reconciler did not submit ({phase2.stopped_by}{': ' + phase2.error if phase2.error else ''})"
    for name in names:
        if name not in session.submitted:
            session.submitted[name] = session.unsubmitted(name, reason)
    usage.add(phase2.usage).add(session.tool_usage)
    results = session.results()

    if log_path:
        write_json(
            log_path.with_name(log_path.stem + "_conversation.json"),
            {
                "doc_id": doc_id,
                "model": chat.key,
                "reconciler": RECONCILER_VERSION,
                "phase1": {
                    "scope": scope,
                    "read": column_names(to_read),
                    "not_read_because_agreed": skipped,
                    "stopped_by": phase1.stopped_by if phase1 else "not run",
                    "error": phase1.error if phase1 else None,
                    "findings": own,
                    "page_images": len(reading.images_attached) if phase1 else 0,
                    "pages_with_images": reading.images_attached if phase1 else [],
                    "tool_calls_sequence": [{"name": e["name"], "args": e["args"]}
                                            for e in (phase1.transcript if phase1 else []) if e["role"] == "tool"],
                    "conversation": phase1.transcript if phase1 else [],
                },
                "stopped_by": phase2.stopped_by,
                "error": phase2.error,
                "page_images": len(session.images_attached),
                "pages_with_images": session.images_attached,
                "verifier_calls": session.verifier_calls,
                "checks": list(session.checks.values()),
                "tool_calls_sequence": [{"name": e["name"], "args": e["args"]} for e in phase2.transcript if e["role"] == "tool"],
                "conversation": phase2.transcript,
                "calls": phase2.calls,
                "tool_usage": session.tool_usage.to_dict(),
                "results": results,
            },
        )
    return results, usage.to_dict()
