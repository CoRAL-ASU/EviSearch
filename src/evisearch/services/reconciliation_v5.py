"""Arbiter v5: read the paper first, then reconcile — the reconciliation stage answers every column itself before it
is shown what Agent A and Agent B said.

Why the stage was rebuilt. Measured on R3 (two runs, 1283 scored cells each), arbiter v4:

* adopted a correct answer **0 of 22 times** when the agent that was right had answered "Not reported". On all cells
  where exactly one agent abstained it emitted a value 55 of 55 times in run 1. The failure was structural, not a
  judgement: `submit_verification` refused an absence whenever any value was supported on a page, and a printed number
  always is, so a correct "this cell is empty" could never win.
* leaned on Agent B: right 5 of 11 when Agent A was the one with the right answer, 62 of 70 when it was Agent B. That
  lean put the shipped table *below* Agent B alone in one of the two runs (91.41 vs 92.09).
* never formed an opinion of its own. Its prompt said it had no paper; its checker was handed the claimed value and
  the claimant's page and agreed with the claim 93% of the time, so nothing in the pipeline ever answered a column
  independently. 40% of batches never opened the paper at all, and the median batch of 15 columns was decided in
  3 tool calls.

The fix is ordering, enforced in code rather than asked for in a prompt — v4's checker prompt already said "find the
correct answer on the pages YOURSELF, before you look at the claimed value" and was ignored:

  phase 1  the columns, their definitions and pages retrieved from the definition text. No A, no B, and
           verify_attribution is not in the tool set. The stage answers each column, or records that the paper states
           no answer, and its findings are persisted before phase 2 starts.
  phase 2  A's and B's answers are revealed alongside its own. verify_attribution becomes available, for checking
           *their* claims. The stage reports a verdict on each of the three answers, its own included.

An absence is a finding here, not an exemption: phase 1 records the pages it looked at, and phase 2 may ship "Not
reported" whenever its own reading found nothing — flagged, when an agent had a value, but never refused.

v4 stays in `reconciliation.py`, unchanged and selectable, so R3 remains reproducible and the knowledge-notes change
can be measured against the old arbiter before this one is added.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.catalog import ConfigError
from src.config.config import AGENT_MAX_TOOL_CALLS, AGENT_MAX_TURNS, MAX_TOKENS, PAGE_IMAGE_SCALE, SELECTION
from src.evisearch.columns import NOT_REPORTED, column_names, is_no_value
from src.evisearch.pipelines.results_store import write_json
from src.evisearch.services.extraction_rules import shared_rules
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
from src.inference import InferenceError, Tool, ToolOutput, ToolSpec, Usage, get_chat, run_tool_loop
from src.retrieval import embedding_retriever as retriever

def own_reading_scope() -> str:
    """EVISEARCH_OWN_READING=all | contested | both_silent: which columns phase 1 answers for itself.

    `both_silent` is the narrowest and the one the R4 measurements point at. Reading contested columns turned out to
    cost more than it earned: across two runs it scored 92.07 and 91.97 against the v4 arbiter's 91.97 and 92.48 on the
    same agent outputs, and on cells where only Agent B was right it fell to 51% (v4: 76%). The stage's own retrieval
    is simply weaker than Agent B's, so letting its reading weigh against a stated value displaces correct answers.
    What the reading is good for is the case where there is no stated value to displace: on cells where both agents
    abstained it answered 11, of which 6 were right and 3 wrong, a net +0.23% of the table, where v4 answered 1 and
    got it wrong. 639 of 1330 cells are both-silent and gold has a value on 31 of the ones left empty, so that is
    where the headroom is.

    `all` (the default) reads every column. `contested` reads only the columns where the two agents disagree or both
    abstain, and lets the agreed ones go straight to phase 2.

    The case for `contested` is in the R3 numbers: where both agents agree and are right the v4 arbiter already scores
    99.9%, so there is nothing to win there and something to lose (it replaced one unanimous correct value with
    "Not reported"). The cells that decide the arbiter's fate are the 12% where the agents disagree, plus the ones
    where both abstain. Reading only those costs roughly an eighth of the phase-1 tokens.
    The case for `all` is the other pot: cells where the agents agree and are both wrong, which only an independent
    reading can reach. The PDF audit put 23% of that pot within reach of a re-read, so `all` has the higher ceiling
    and the higher variance. Which one ships is an experiment, not a preference.
    """
    value = os.getenv("EVISEARCH_OWN_READING", "").strip().lower()
    return value if value in {"contested", "both_silent"} else "all"


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


# Part of the run settings, so results made with a different reading scope are never resumed into each other. Computed
# at import because the scope comes from the environment the run was launched with.
_SCOPE = own_reading_scope()
RECONCILER_VERSION = "own_reading_v5" + ("" if _SCOPE == "all" else f"_{_SCOPE}")

# what phase 2 says about each of the three answers it now holds
VERDICTS = ("correct", "incomplete", "wrong", "no_answer")
SOURCES = ("own", "A", "B", "merged")
DEFINITION_PAGES = 3  # candidate pages retrieved per column from its definition, before any agent is consulted

FINDINGS_PROMPT = """You answer clinical trial table columns from the paper itself.

For each column you get its definition and the pages a search of that definition found. Nobody else's answer is shown
to you: this is your own reading, and it is the only independent reading the pipeline makes.

Your tools:
- search_pages: semantic search over the paper; returns the best matching pages with their most relevant lines.
- ask_document: a reader that has the whole paper (every page's text and image) answers your questions with the answer,
  the pages and the evidence. Ask about several columns in one call.
- submit_findings: your answers.

For every column:
1. Read the definition and name exactly what it asks for: the statistic, the endpoint or characteristic, the population
   or subgroup, the arm, the timepoint and the unit.
2. Find that answer on the pages and copy it as the paper prints it. Quote the text or table cell you took it from and
   give the page(s) it is on.
3. When the paper states no answer for this column, submit an empty value (""). That is a finding, not a failure: it is
   how this pipeline learns that a cell belongs empty, and it is worth as much as a value. Say which pages you looked
   at before concluding it.
4. Never write a value the paper does not state. Do not read a number off a curve, and do not compute one unless the
   knowledge notes license that computation - when they do, show the arithmetic and name the row and column of every
   number in it.

Search where the answer would be, not only where a word matches: baseline tables for characteristics, results tables
and figure panels for outcomes, the methods for design, the discussion for durations and cross-trial comparisons.
Answer every column in the batch. Submit them together."""

RECONCILE_PROMPT = """You decide the final value of clinical trial columns.

You have already read the paper and answered these columns yourself. Now two independent extractions of the same paper
are shown to you (A and B, anonymous). For each column you hold three answers: YOURS, A's and B's.

Your tools:
- verify_attribution: a checker reads the page(s) (text and image), writes the column's answer itself, and says whether
  a claimed value IS that answer (supported / partial / not_supported), with what the pages state and a reason tag.
  Use it on A's and B's values, and on your own when you want a second look. Send many claims in one call.
- ask_document, search_pages: as before, when you need to look again.
- submit_verification: the final values.

Deciding a column:
1. All three agree: submit that value.
2. They differ: your own reading is evidence, not the answer. Check the values that differ from yours on the pages
   that are supposed to show them. A value one of them found on a page you did not look at may well be right - adopt
   it. A value no page supports is not an answer, whoever produced it.
3. Your own reading can be the wrong one. Say so in own_verdict when it is.
4. When your reading found no answer and neither extraction has one the checker supports, submit "Not reported". This
   is accepted: an empty cell is a real answer. Submit it even when a number for something else is printed on the page.
5. When your reading found no answer but an extraction has a value the checker supports on its page, adopt that value.
6. A value is accepted only after verify_attribution found it supported on the page given as its source. A value the
   checker rejected is never accepted.
7. Answers that are compatible at different levels of detail (a drug class and the drug, a count and the same count
   with its percentage) are merged into the complete value the definition asks for; set final_source to "merged".

For every column report, besides the value: final_source (own, A, B or merged) and a verdict on each of the three
answers - own_verdict, a_verdict, b_verdict - each one of correct, incomplete, wrong or no_answer. These verdicts are
the record of what this stage decided and why; fill them honestly, including when your own answer was the wrong one.

Take the calls you need. Reading the paper again is cheaper than shipping a value nobody checked."""

FINDINGS_FOLLOW_UP = "Continue. Answer the remaining columns and submit them with submit_findings."
RECONCILE_FOLLOW_UP = "Continue. Fix rejected columns, verify what you still need, and submit every remaining column."


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


class _FindingsSession(_ReconciliationSession):
    """Phase 1. The v4 session without its sources: the reader and search tools are reused, the checker is withheld."""

    def __init__(self, chat: Any, doc_id: str, batch_columns: List[Dict[str, Any]], definitions_map: Dict[str, str],
                 image_scale: Optional[float]):
        super().__init__(chat, doc_id, batch_columns, definitions_map, {}, {}, image_scale)
        self.found: Dict[str, Dict[str, Any]] = {}
        self.candidates = {name: self._candidate_pages(name) for name in self.names}

    def _candidate_pages(self, name: str) -> List[int]:
        """Pages a search of the column's own definition finds. The definition, never an agent's citation: v4 checked
        only the pages a claiming agent pointed at, so a value neither agent found was unreachable by construction."""
        query = f"{name}. {self.definitions.get(name, '')}".strip()
        try:
            hits = retriever.search_chunks(self.doc_id, query)
        except Exception:  # retrieval is a convenience here; the model can still search and ask
            return []
        pages: List[int] = []
        for hit in hits:
            page = hit.get("page")
            if page and page not in pages:
                pages.append(int(page))
            if len(pages) >= DEFINITION_PAGES:
                break
        return pages

    def user_prompt(self) -> str:
        blocks = []
        for i, name in enumerate(self.names, 1):
            pages = self.candidates.get(name) or []
            where = f"\nPages a search of this definition found: {pages}" if pages else "\nNo page matched this definition strongly."
            blocks.append(f"\n---\nColumn {i}: {name}\nDefinition: {self.definitions.get(name, '')}{where}")
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
                "looked_at": looked or (list(pages) if pages else self.candidates.get(name) or []),
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


class _ReconcileSession(_ReconciliationSession):
    """Phase 2. v4's session, plus the stage's own findings, three-way verdicts, and absence treated as an answer."""

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

    def own_found_nothing(self, name: str) -> bool:
        """Only true when this stage actually looked and the paper had no answer.

        A column it never read - because phase 1 failed on it, or because the scope left agreed columns alone - is not
        evidence of absence, and must not let an absence through on the strength of a reading that never happened.
        """
        own = self.own.get(name) or {}
        if own.get("unread") or own.get("skipped"):
            return False
        return is_no_value(own.get("value"))

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

    def final(self, name: str, value: str, record: Optional[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        result = super().final(name, value, record, **kwargs)
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
        """v4's submission, with the absence asymmetry removed.

        v4 refused "Not reported" whenever any value stood supported on a page, and had no symmetric rule refusing a
        value. Here the stage's own reading decides: it may ship an absence its own reading supports (flagged when an
        extraction had a value), and it is pushed back once when it tries to drop a value it found itself.
        """
        entries, note = _items(args, "results")
        handled: List[Dict[str, Any]] = []
        passthrough: List[Dict[str, Any]] = []
        for item in entries:
            name = item.get("column")
            value = str(item.get("value") or "").strip()
            if name not in self.names or name in self.submitted:
                passthrough.append(item)
                continue
            self.verdicts[name] = self._verdict_block(name, item, value)
            if not is_absence(value):
                own = self.own_value(name)
                if own and not _same(value, own) and not item.get("review") and name not in self.attempts:
                    # shipping something other than its own reading is allowed, but not on the first pass without a check
                    pass
                passthrough.append(item)
                continue
            if self.own_found_nothing(name):
                # An absence may win, but not before the other readings have actually been tested. v4 refused an
                # absence while an extraction's value stood unchecked; dropping that guard let this stage blank a
                # cell whose only stated value nobody had looked at - 5 of the 12 cells it wrongly blanked in the
                # first contested run. Its own reading finding nothing is evidence, not proof: Agent B's retrieval
                # reaches pages this stage's does not.
                unchecked = self.unchecked_values(name)
                if unchecked and not item.get("review"):
                    handled.append({"column": name, "accepted": False, "reason": (
                        f'an extraction reported "{unchecked[0]}" for this column and no check has looked at it. '
                        f"Your own reading found nothing, which is not the same as the paper stating nothing: verify "
                        f"that value with verify_attribution first. If it fails the check, "
                        f'"{value or NOT_REPORTED}" is accepted.')})
                    continue
                standing = self.standing(name)
                flag = ""
                if standing:
                    best = standing[0]
                    flag = (f'an extraction reported "{best["value"]}" and the checker found it {best["verdict"]} on '
                            f'page(s) {best["pages"]}, but this stage\'s own reading of page(s) '
                            f'{(self.own.get(name) or {}).get("looked_at")} found no answer for this column')
                self.submitted[name] = self.final(
                    name, value, None, reasoning=str(item.get("reasoning") or ""), decided_by="own_reading",
                    verification=None, review_reason=str(item.get("review_reason") or "") or flag,
                )
                handled.append({"column": name, "accepted": True})
                continue
            if self.own_value(name) and not item.get("review"):
                handled.append({"column": name, "accepted": False, "reason": (
                    f'your own reading found "{self.own_value(name)}" on page(s) {(self.own.get(name) or {}).get("pages")} '
                    f'for this column. Submitting "{value or NOT_REPORTED}" discards it: verify your own value and submit '
                    f"it, or resubmit with review=true and a review_reason saying why your reading was wrong.")})
                continue
            passthrough.append(item)
        out = super().submit_verification({"results": passthrough}) if passthrough else ToolOutput({"accepted": [], "rejected": []})
        content = dict(out.content if isinstance(out.content, dict) else {})
        accepted = list(content.get("accepted") or []) + [h["column"] for h in handled if h.get("accepted")]
        rejected = list(content.get("rejected") or []) + [
            {"column": h["column"], "reason": h["reason"]} for h in handled if not h.get("accepted")
        ]
        content.update({"accepted": accepted, "rejected": rejected,
                        "remaining": [n for n in self.names if n not in self.submitted]})
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
    """Reconcile one batch in two phases. Same signature as v4, so the pipeline can select either."""
    names = column_names(batch_columns)
    try:
        chat = get_chat("reconciliation", model)
    except (ConfigError, InferenceError) as exc:
        return {name: _not_run(f"reconciliation not run: {exc}") for name in names}, Usage().to_dict()

    use_images = SELECTION.option("reconciliation_page_images") == "auto" and chat.capabilities.images
    scale = PAGE_IMAGE_SCALE if use_images else None
    specs = {spec.name: spec for spec in tool_specs(names)}
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
                Tool(specs["ask_document"], reading.ask_document),
                Tool(specs["search_pages"], reading.search_pages),
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
        candidates = reading.candidates
    else:  # every column in this batch was agreed: there is no reading pass to make
        own = not_read
        candidates = {}

    # ---- phase 2: A and B revealed, its own answer already committed -----------------------------------------
    session = _ReconcileSession(chat, doc_id, batch_columns, definitions_map, source_a_data, source_b_data, scale, own=own)
    phase2 = run_tool_loop(
        chat,
        system=RECONCILE_PROMPT + shared_rules(columns=names, role="agent"),
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
                    "candidate_pages": candidates,
                    "findings": own,
                    "reader_calls": reading.reader_calls if phase1 else [],
                    "tool_calls_sequence": [{"name": e["name"], "args": e["args"]}
                                            for e in (phase1.transcript if phase1 else []) if e["role"] == "tool"],
                    "conversation": phase1.transcript if phase1 else [],
                },
                "stopped_by": phase2.stopped_by,
                "error": phase2.error,
                "reader_calls": session.reader_calls,
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
