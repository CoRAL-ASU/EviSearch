#!/usr/bin/env python3
"""
Check that the pipeline executed correctly for one document: every model, tool and service is reachable, every stage
wrote complete outputs, and no call failed. It does not judge whether the extracted values are right.

Usage:
  python experiment-scripts/check_run.py "<doc_id>" --run smoke_e2e            # outputs of the three stages
  python experiment-scripts/check_run.py "<doc_id>" --run smoke_e2e --probe    # also call each model, tool and service

FAIL (exit code 1): the pipeline did not execute correctly (a call failed, an output is missing or cut off, images
fell back, a batch was lost, the reranker was bypassed). WARN: it ran, but the model did something worth a look
(submitted only when forced, left a column out, gave a value without attribution).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config import MAX_TOKENS, PAGE_IMAGE_SCALE, SELECTION
from src.evisearch.columns import MODALITIES
from src.evisearch.pipelines import results_store
from src.evisearch.pipelines.batching import build_batches, load_groups

STAGES = ("agent", "search", "reconciliation")
EXECUTION_ERRORS = {
    "agent": ("pdf_query not run", "pdf_query failed", "JSON parse error"),
    "search": ("search_agent not run", "Agent did not submit"),
    "reconciliation": ("reconciliation not run", "Agent did not resolve before limits"),
}
OMITTED = {"agent": "Not returned by the model", "search": "Not extracted"}
TOOLS = {
    "search": ("search_chunks", "get_chunks_by_page", "submit_extraction"),
    "reconciliation": ("ask_document", "search_pages", "verify_attribution", "submit_verification"),
}
VERIFICATIONS = ("A_correct_B_wrong", "B_correct_A_wrong", "both_correct", "both_wrong")


class Report:
    def __init__(self) -> None:
        self.passed: Counter = Counter()
        self.problems: List[Tuple[str, str, str]] = []

    def check(self, section: str, ok: bool, message: str, warn: bool = False) -> bool:
        if ok:
            self.passed[section] += 1
        else:
            self.problems.append(("WARN" if warn else "FAIL", section, message))
        return ok


def batch_number(path: Path) -> int:
    match = re.search(r"batch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def expected_columns(batch_count: int) -> List[str]:
    return [col["column_name"] for batch in build_batches(load_groups(), None, done=set())[:batch_count] for col in batch]


def check_columns(report: Report, section: str, method: str, columns: Dict[str, Any], expected: List[str]) -> None:
    missing = [name for name in expected if name not in columns]
    report.check(section, not missing, f"{len(missing)} expected columns missing from results, e.g. {missing[:3]}")
    for name in expected:
        entry = columns.get(name)
        if not isinstance(entry, dict):
            continue
        reasoning = str(entry.get("reasoning", ""))
        failed = next((marker for marker in EXECUTION_ERRORS[method] if reasoning.startswith(marker)), None)
        report.check(section, failed is None, f"column '{name}': {reasoning[:160]}")
        if method in OMITTED and reasoning.startswith(OMITTED[method]):
            report.check(section, False, f"column '{name}' was left out by the model", warn=True)
        report.check(section, isinstance(entry.get("value"), str) and entry.get("tried") is True, f"column '{name}': value/tried missing")
        attribution = entry.get("attribution")
        valid = isinstance(attribution, list) and all(
            isinstance(a, dict) and a.get("modality") in MODALITIES and (a.get("page") is None or (isinstance(a["page"], int) and a["page"] >= 1))
            for a in attribution
        )
        report.check(section, valid, f"column '{name}': malformed attribution {attribution}")
        if method == "reconciliation":
            report.check(section, entry.get("verification") in VERIFICATIONS, f"column '{name}': verification={entry.get('verification')}")
            report.check(section, isinstance(entry.get("source"), dict), f"column '{name}': source missing")
            has_value = entry.get("value", "Not reported") not in ("Not reported", "")
            if "verified" in entry:  # verified reconciler: every value is verified on its page or flagged for review
                has_value = has_value and str(entry.get("value")).strip().lower() not in ("no", "n")  # absence answers
                verified_attr = any(isinstance(a, dict) and a.get("verified") for a in attribution or [])
                report.check(section, not has_value or entry.get("verified") or entry.get("needs_review"),
                             f"column '{name}': value accepted without verification or review flag")
                report.check(section, not entry.get("verified") or verified_attr, f"column '{name}': verified but no verified attribution")
                if entry.get("needs_review"):
                    report.check(section, False, f"column '{name}' flagged for review: {str(entry.get('review_reason'))[:120]}", warn=True)
                if entry.get("decided_by") == "unsubmitted":
                    report.check(section, False, f"column '{name}' was never accepted by submit_verification", warn=True)
                continue
        else:
            report.check(section, isinstance(entry.get("found"), bool), f"column '{name}': found missing")
            has_value = entry.get("found") is True
        if has_value and valid and not attribution:
            report.check(section, False, f"column '{name}' has a value but no attribution", warn=True)


def check_agent_logs(report: Report, section: str, logs: List[Path]) -> None:
    for path in logs:
        log = json.loads(path.read_text(encoding="utf-8"))
        document = log.get("document") or {}
        report.check(section, "error" not in log, f"{path.name}: model call failed: {str(log.get('error'))[:200]}")
        final = log.get("follow_up") or log  # a follow-up call asked again for columns the first reply left out or lost
        if "follow_up" in log:
            report.check(section, False, f"{path.name}: follow-up for {len(final.get('columns') or [])} columns "
                         f"(first reply finish_reason={log.get('finish_reason')})", warn=True)
            report.check(section, "error" not in final, f"{path.name}: follow-up call failed: {str(final.get('error'))[:200]}")
        report.check(section, str(final.get("finish_reason")).lower() == "stop", f"{path.name}: finish_reason={final.get('finish_reason')} (answer cut off?)")
        report.check(section, (log.get("usage") or {}).get("input_tokens", 0) > 0, f"{path.name}: no token usage recorded")
        report.check(section, document.get("fallback") is None, f"{path.name}: page images fell back to {document.get('fallback')}")
        report.check(section, not document.get("warnings"), f"{path.name}: {document.get('warnings')}")
        if document.get("input_mode") == "markdown_images":
            pages = document.get("pages", 0)
            report.check(section, document.get("image_pages") == list(range(1, pages + 1)), f"{path.name}: images for {len(document.get('image_pages') or [])} of {pages} pages")
            report.check(section, document.get("page_image_scale") == PAGE_IMAGE_SCALE, f"{path.name}: page_image_scale={document.get('page_image_scale')}")


def check_loop_logs(report: Report, section: str, method: str, logs: List[Path]) -> None:
    reranker = SELECTION.model_key("reranker") is not None
    images = SELECTION.option("reconciliation_page_images") == "auto" and SELECTION.model("reconciliation").capabilities.images
    used: Counter = Counter()
    for path in logs:
        log = json.loads(path.read_text(encoding="utf-8"))
        report.check(section, log.get("error") is None, f"{path.name}: model call failed: {str(log.get('error'))[:200]}")
        stopped = log.get("stopped_by")
        if stopped == "forced_finish":
            report.check(section, False, f"{path.name}: submitted only when forced (tool budget or turns used up)", warn=True)
        else:
            report.check(section, stopped in ("finish_tool", "done"), f"{path.name}: stopped_by={stopped}")
        for call in log.get("verifier_calls", []):
            report.check(section, "error" not in call, f"{path.name}: verifier call on page {call.get('page')} failed: {str(call.get('error'))[:160]}")
            if images:
                report.check(section, call.get("image") is True, f"{path.name}: verifier checked page {call.get('page')} without its image")
        for call in log.get("reader_calls", []):
            report.check(section, "error" not in call, f"{path.name}: ask_document call failed: {str(call.get('error'))[:160]}")
            fallback = (call.get("document") or {}).get("fallback")
            report.check(section, fallback is None, f"{path.name}: the reader saw page images only for {fallback}", warn=True)
        failed_checks = [c for c in log.get("checks", []) if c.get("verdict") == "error"]
        report.check(section, not failed_checks, f"{path.name}: {len(failed_checks)} verifier checks without a verdict")
        called_here: Counter = Counter()
        for turn in log.get("conversation", []):
            if turn.get("role") != "tool":
                continue
            tool, response = turn.get("name"), turn.get("response") or {}
            called_here[tool] += 1
            report.check(section, tool in TOOLS[method], f"{path.name}: unknown tool '{tool}'")
            error = response.get("error") if isinstance(response, dict) else None
            report.check(section, not error, f"{path.name}: {tool} returned an error: {str(error)[:200]}")
            if tool == "search_chunks" and reranker and isinstance(response, dict) and response.get("matches"):
                report.check(section, response.get("retrieval") == "rerank", f"{path.name}: search_chunks retrieval={response.get('retrieval')}")
        if method == "reconciliation" and called_here["get_page"] and images:
            report.check(section, log.get("page_images", 0) > 0, f"{path.name}: get_page was called but no page images were attached")
        used.update(called_here)
    for tool in TOOLS[method]:
        report.check(section, used[tool] > 0, f"{tool} was never called, so this run did not exercise it", warn=True)


def check_stage(report: Report, doc_id: str, method: str) -> Optional[Dict[str, Any]]:
    section = results_store.METHOD_DIRS[method]
    folder = results_store.method_dir(doc_id, method)
    if not report.check(section, results_store.results_path(doc_id, method).exists(), f"no results at {results_store.results_path(doc_id, method)}"):
        return None
    report.check(section, (folder / "extraction_metadata.json").exists(), f"no extraction_metadata.json in {folder}")
    metadata = results_store.load_metadata(doc_id, method)
    report.check(section, metadata.get("run", "") == results_store.current_run(), f"metadata run={metadata.get('run')!r}, expected {results_store.current_run()!r}")
    report.check(section, (metadata.get("usage") or {}).get("api_calls", 0) > 0, "metadata records no API calls")
    if method == "agent":
        report.check(section, metadata.get("fallback_batches") == [], f"fallback_batches={metadata.get('fallback_batches')}")

    log_dir = folder / results_store.LOG_DIRS[method]
    logs = sorted(log_dir.glob("batch_*.json"), key=batch_number) if log_dir.exists() else []
    if not report.check(section, bool(logs), f"no batch logs in {log_dir}"):
        return None
    columns = results_store.load_columns(doc_id, method)
    check_columns(report, section, method, columns, expected_columns(len(logs)))
    if method == "agent":
        check_agent_logs(report, section, logs)
        report.check(section, (metadata.get("usage") or {}).get("api_calls") == len(logs), f"{len(logs)} batches but {metadata.get('usage', {}).get('api_calls')} API calls")
    else:
        check_loop_logs(report, section, method, logs)
    return columns


def probe(report: Report, doc_id: str) -> None:
    from src.evisearch.services import pdf_query
    from src.evisearch.services.highlight import resolve_pdf_path
    from src.evisearch.services.page_images import pdf_page_count, render_pages
    from src.inference import ImagePart, Message, ToolSpec, get_chat
    from src.inference.factory import get_embedder, get_reranker
    from src.retrieval import embedding_retriever as retriever

    section = "probe"

    def attempt(label: str, run: Callable[[], Tuple[bool, str]]) -> None:
        try:
            ok, detail = run()
        except Exception as exc:  # a probe that raises is a failure to report, not a crash
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        report.check(section, ok, f"{label}: {detail}")
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {detail}")

    pdf_path = resolve_pdf_path(doc_id)
    for key in sorted({SELECTION.model_key(role) for role in ("pdf_query", "search_agent", "reconciliation")}):
        chat = get_chat("pdf_query", key)
        attempt(f"{key} chat", lambda: (lambda r: (bool(r.text), repr(r.text[:30])))(chat.chat([Message.user("Reply with the word OK.")], max_tokens=16)))
        ping = ToolSpec("ping", "Call this tool.", {"type": "object", "properties": {"note": {"type": "string"}}})
        attempt(f"{key} tool calling", lambda: (lambda r: (bool(r.tool_calls), f"{len(r.tool_calls)} call(s)"))(
            chat.chat([Message.user("Call the ping tool.")], tools=[ping], tool_choice="required", max_tokens=64)))
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
        attempt(f"{key} structured output", lambda: (lambda r: ("answer" in r.json(), repr(" ".join(r.text.split())[:40])))(
            chat.chat([Message.user("Answer with JSON: what is 2+2?")], response_schema=schema, max_tokens=64)))
        if chat.capabilities.images and pdf_path:
            png = render_pages(Path(pdf_path), [1])[0].png
            attempt(f"{key} page image input", lambda: (lambda r: (bool(r.text), repr(r.text[:40])))(
                chat.chat([Message.user("What is the title on this page? One short line.", ImagePart(png))], max_tokens=48)))

    attempt("embedding server", lambda: (lambda v: (v.shape[-1] > 0, f"dimension {v.shape[-1]}"))(get_embedder().embed(["overall survival"], kind="query")))
    if SELECTION.model_key("reranker"):
        attempt("reranker server", lambda: (lambda r: (len(r) == 2, str([(i, round(s, 3)) for i, s in r])))(
            get_reranker().rerank("median overall survival", ["Median overall survival was 76.6 months.", "Adverse events were similar."])))
    attempt("search_chunks retrieval", lambda: (lambda hits: (bool(hits) and all(h["retrieval"] == ("rerank" if SELECTION.model_key("reranker") else "embedding") for h in hits),
                                                              f"pages {[h['page'] for h in hits]}, retrieval={hits[0]['retrieval'] if hits else None}"))(
        retriever.search_chunks(doc_id, "median overall survival")))
    attempt("page text", lambda: (lambda text: (bool(text.strip()), f"{len(text)} characters on page 1"))(retriever.get_page_content(doc_id, [1])[1]))
    if pdf_path:
        attempt("page count", lambda: (retriever.get_total_pages(doc_id) == pdf_page_count(Path(pdf_path)),
                                       f"markdown {retriever.get_total_pages(doc_id)} vs PDF {pdf_page_count(Path(pdf_path))} pages"))
        attempt("page rendering", lambda: (lambda image: (image.png.startswith(b"\x89PNG"), f"{image.width}x{image.height} px"))(render_pages(Path(pdf_path), [1])[0]))

    def document_fits() -> Tuple[bool, str]:
        batches = build_batches(load_groups(), None, done=set())
        longest = max((pdf_query.build_columns_prompt(batch, "") for batch in batches), key=len)
        mode = SELECTION.option("pdf_query_input")
        model = SELECTION.model("pdf_query")
        budget = pdf_query.document_token_budget(model.context_tokens, pdf_query.system_prompt_text() + longest, MAX_TOKENS["pdf_query"])
        info = pdf_query.build_document_input(doc_id, mode, budget, model.image_tokens).info
        return info["fallback"] is None and not info["warnings"], f"{mode}: {len(info['image_pages'])}/{info['pages']} page images, ~{info['estimated_tokens']} of {budget} tokens"

    attempt("Arm A document fits", document_fits)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check that a pipeline run executed correctly for one document")
    parser.add_argument("doc_id")
    parser.add_argument("--run", help="Run name the stages used (default: EVISEARCH_RUN)")
    parser.add_argument("--stages", default=",".join(STAGES), help="Comma-separated: agent,search,reconciliation")
    parser.add_argument("--probe", action="store_true", help="Also call every model, tool and service the pipeline uses")
    args = parser.parse_args(argv)
    if args.run is not None:
        results_store.use_run(args.run)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]

    report = Report()
    print(f"[check_run] doc_id={args.doc_id} run={results_store.current_run() or '-'} stages={stages}")
    if args.probe:
        probe(report, args.doc_id)
    columns = {stage: check_stage(report, args.doc_id, stage) for stage in stages}
    if columns.get("reconciliation") and columns.get("agent") and columns.get("search"):
        extra = set(columns["reconciliation"]) - (set(columns["agent"]) & set(columns["search"]))
        report.check("reconciliation_agent", not extra, f"{len(extra)} reconciled columns have no Arm A/B result, e.g. {sorted(extra)[:3]}")

    for level, section, message in report.problems:
        print(f"[{level}] {section}: {message}")
    for section, count in report.passed.items():
        print(f"[check_run] {section}: {count} checks passed")
    failures = sum(1 for level, _, _ in report.problems if level == "FAIL")
    warnings = len(report.problems) - failures
    print(f"[check_run] {'FAIL' if failures else 'PASS'}: {sum(report.passed.values())} passed, {warnings} warnings, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
