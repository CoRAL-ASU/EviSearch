#!/usr/bin/env python3
"""
Run one benchmark system over the gold documents and record what ran, how long it took and whether it ran cleanly.

Systems (results go to new_pipeline_outputs/results/<doc_id>/runs/<run>/):
  B1  parsed-markdown baseline: one call per definition group, preset's `baseline` model   -> markdown_baseline/
  B2  Arm A alone (pdf_query, markdown_images input)                                      -> agent_extractor/
  E   Arm A + Arm B (search_agent) + reconciliation       -> agent_extractor/, search_agent/, reconciliation_agent/

Usage:
  python experiment-scripts/run_benchmark.py --system B2 --docs all --run qwen_b2
  python experiment-scripts/run_benchmark.py --system E --docs all --run qwen_e --reuse-a-from qwen_b2
  python experiment-scripts/run_benchmark.py --system E --docs dev --run mistral_e --preset local_mistral --dry-run
  python experiment-scripts/run_benchmark.py --system B1 --docs "NCT00309985_Sweeney_CHAARTED_NEJM'15" --run smoke_b1

Environment (schedule only; both off by default, so every run up to R4 reproduces exactly):
  EVISEARCH_STAGE_PARALLEL=1     a document's independent stages run at the same time (E: Arm A with Arm B, then the
                                 arbiter, which reads both arms' saved results). Stage records, timings and the manifest
                                 stay as they are; only the wall clock changes.
  EVISEARCH_STAGE_CONCURRENCY=N  a stage's column batches run N at a time (src/evisearch/pipelines/batch_runner.py).
Requests in flight on the server are --parallel x stages x batches, and one vLLM instance with --max-num-seqs 8 is the
ceiling; raising all three at once only lengthens its queue.

--docs: all | dev | heldout | comma-separated doc ids from the gold table. heldout is James STAMPEDE IJC'22, Sweeney
CHAARTED NEJM'15 and Smith ARASENS NEJM'22; dev is the other 7. Running a run name again resumes it (only missing
columns are extracted); results made with other settings are refused, as in the single-stage CLIs. --reuse-a-from
(E only) copies Arm A from a finished B2 run of the same documents instead of running it again. After each B2/E
document, experiment-scripts/check_run.py checks that the run executed correctly.

Outputs:
  results/<doc_id>/runs/<run>/benchmark_manifest.json   stages, timings, usage and check result for the document
  results/<doc_id>/runs/<run>/check_run.log             check_run.py output
  new_pipeline_outputs/benchmark_runs/<run>.json        git commit, preset, models per role, docs, timings, failures
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.catalog import ConfigError, load_catalog  # noqa: E402  (src.config.config is imported after --preset)
from src.evisearch.pipelines.batch_runner import stage_concurrency, stage_parallel  # noqa: E402  (imports no configuration)

SYSTEMS = {"B1": ("baseline",), "B2": ("agent",), "E": ("agent", "search", "reconciliation")}
DESCRIPTIONS = {
    "B1": "parsed-markdown baseline (markdown_baseline), no evaluation",
    "B2": "Arm A (pdf_query) alone, markdown_images input",
    "E": "Arm A + Arm B (search_agent) + reconciliation",
}
STAGE_ROLES = {"baseline": "baseline", "agent": "pdf_query", "search": "search_agent", "reconciliation": "reconciliation"}
HELDOUT = ("NCT00268476_James_STAMPEDE_IJC'22", "NCT00309985_Sweeney_CHAARTED_NEJM'15", "NCT02799602_Smith_ARASENS_NEJM'22")
ARM_A_INPUT = "markdown_images"  # B2 and E: each page's parsed text followed by its rendered image
CHECK_SCRIPT = PROJECT_ROOT / "experiment-scripts" / "check_run.py"
CHECK_STAGES = {"B2": "agent", "E": "agent,search,reconciliation"}
SUMMARY_KEYS = ("filled", "total", "usage", "fallback_batches", "failed_groups", "reused_from", "copied")

_manifest_lock = threading.Lock()


class StageError(RuntimeError):
    """A stage could not run or finish for a document."""


# ---- documents ---------------------------------------------------------------------------------------

def gold_doc_ids(path: Optional[Path] = None) -> List[str]:
    """Document ids of the gold table in its order ("Document Name" without .pdf)."""
    from src.config.config import GOLD_TABLE_JSON_PATH

    rows = json.loads(Path(path or GOLD_TABLE_JSON_PATH).read_text(encoding="utf-8"))["data"]
    return [row["Document Name"]["value"].removesuffix(".pdf") for row in rows]


def has_parse(doc_id: str) -> bool:
    """A paper outside the gold table can run once it has a parsed markdown (uploaded through the web app); it is
    extracted like any other, just never scored."""
    from src.retrieval.embedding_retriever import parsed_markdown_path

    return parsed_markdown_path(doc_id).exists()


def select_docs(spec: str, gold: List[str], known=None) -> List[str]:
    """all | dev | heldout (in gold-table order) or comma-separated doc ids (in the order given): gold doc ids, plus any
    doc for which `known(doc_id)` is true (e.g. `has_parse`)."""
    missing = [doc for doc in HELDOUT if doc not in gold]
    if missing:
        raise ValueError(f"held-out documents missing from the gold table: {missing}")
    spec = spec.strip()
    if spec == "all":
        return list(gold)
    if spec == "heldout":
        return [doc for doc in gold if doc in HELDOUT]
    if spec == "dev":
        return [doc for doc in gold if doc not in HELDOUT]
    docs = list(dict.fromkeys(doc.strip().removesuffix(".pdf") for doc in spec.split(",") if doc.strip()))
    unknown = [doc for doc in docs if doc not in gold and not (known and known(doc))]
    if unknown or not docs:
        raise ValueError(f"unknown doc id(s) {unknown or [spec]}; use all, dev, heldout, gold doc ids or parsed uploads: {', '.join(gold)}")
    return docs


# ---- stages --------------------------------------------------------------------------------------------

def benchmark_columns() -> List[str]:
    """Every column of the definitions CSV, in batch order."""
    from src.evisearch.pipelines.batching import build_batches, load_groups

    return [col["column_name"] for batch in build_batches(load_groups(), None, done=set()) for col in batch]


def stage_settings(stage: str) -> Dict[str, Any]:
    """Settings the stage's saved results must match to be resumed (as in its pipeline)."""
    from src.inference.factory import model_key_for

    model_key = model_key_for(STAGE_ROLES[stage])
    if stage == "agent":
        from src.evisearch.pipelines.pdf_query_pipeline import run_settings

        return run_settings(model_key, ARM_A_INPUT)
    if stage == "reconciliation":
        from src.evisearch.pipelines.reconciliation_pipeline import run_settings

        return run_settings(model_key)
    if stage == "search":
        from src.evisearch.pipelines.search_pipeline import run_settings

        return run_settings(model_key)
    from src.evisearch.services.extraction_rules import rules_setting

    return {"model": model_key, **rules_setting()}  # the B1 baseline


def run_stage(stage: str, doc_id: str) -> Dict[str, Any]:
    """Run one stage for one document with the pipeline its single-stage CLI uses. Raises when the stage cannot run
    (e.g. results_store.ResumeError); model errors inside batches are left for the checker to report."""
    def progress(event: Dict[str, Any]) -> None:
        if event.get("type") in ("columns_written", "search_batch_done"):
            print(f"[benchmark] {doc_id}: {stage} batch {event['batch']}/{event['total_batches']}", flush=True)

    if stage == "baseline":
        from src.evisearch.services.markdown_baseline import run_baseline_stage

        return run_baseline_stage(doc_id)
    if stage == "agent":
        from src.evisearch.pipelines.pdf_query_pipeline import run_pdf_query_pipeline

        return run_pdf_query_pipeline(doc_id, input_mode=ARM_A_INPUT, on_event=progress)
    if stage == "search":
        from src.evisearch.pipelines.search_pipeline import run_search_agent_pipeline

        return run_search_agent_pipeline(doc_id, on_event=progress)
    if stage == "reconciliation":
        from src.evisearch.pipelines.reconciliation_pipeline import run_reconciliation_pipeline

        result = run_reconciliation_pipeline(doc_id)
        if result.get("error"):
            raise StageError(result["error"])
        return result
    raise ValueError(f"unknown stage {stage!r}")


def stage_inputs(stage: str) -> Tuple[str, ...]:
    """Which other stages' saved results this stage reads, asked of the pipeline that runs it.

    Reconciliation answers with `reconciliation_pipeline.SOURCE_METHODS`, the tuple it loops over before it builds any
    batch: it refuses to start until both arms' extraction_results.json exist. Arm A reads the parsed markdown and the
    page images, Arm B the embedded chunks, and the B1 baseline the parsed markdown - a document, never another stage -
    so they read nothing here. That is the whole dependency rule; the overlap follows from it.
    """
    if stage == "reconciliation":
        from src.evisearch.pipelines.reconciliation_pipeline import SOURCE_METHODS

        return tuple(SOURCE_METHODS)
    return ()


def stage_waves(system: str, parallel: Optional[bool] = None) -> List[List[str]]:
    """The system's stages in groups that may run at the same time, each group after the one before it.

    Serial (the default) is one stage per group, exactly the order of SYSTEMS[system]. With EVISEARCH_STAGE_PARALLEL on,
    a stage joins the current group when none of the stages it reads is still waiting, so E becomes [agent, search] then
    [reconciliation], and B1/B2 stay a single stage in a single group. Stage order inside and across groups stays the
    SYSTEMS order, so records and manifests read the same either way. A stage whose input is not part of this system
    (E's Arm A with --reuse-a-from, say) waits for nothing here, as today.
    """
    stages = list(SYSTEMS[system])
    if not (stage_parallel() if parallel is None else parallel):
        return [[stage] for stage in stages]
    waves: List[List[str]] = []
    waiting = list(stages)
    while waiting:
        ready = [stage for stage in waiting if not set(stage_inputs(stage)) & set(waiting)]
        if not ready:  # a stage waiting for itself: fall back to serial rather than hang
            ready = waiting[:1]
        waves.append(ready)
        waiting = [stage for stage in waiting if stage not in ready]
    return waves


def source_arm_a_dir(doc_id: str, source_run: str) -> Path:
    from src.evisearch.pipelines import results_store

    return results_store.RESULTS_ROOT / doc_id / "runs" / source_run / results_store.METHOD_DIRS["agent"]


def reuse_problem(doc_id: str, source_run: str) -> Optional[str]:
    """Why Arm A of `source_run` cannot be copied into the current run for this document (None: it can)."""
    from src.evisearch.pipelines import results_store

    source = source_arm_a_dir(doc_id, source_run)
    try:
        saved = json.loads((source / "extraction_metadata.json").read_text(encoding="utf-8"))
        columns = json.loads((source / results_store.RESULT_FILES["agent"]).read_text(encoding="utf-8")).get("columns", {})
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        return f"no finished Arm A results in {source} ({type(exc).__name__})"
    changed = [f"{key}: {saved.get(key)!r} there, {value!r} here" for key, value in stage_settings("agent").items() if saved.get(key) != value]
    if changed:
        return f"{source} was made with other settings ({'; '.join(changed)})"
    missing = [name for name in benchmark_columns() if name not in columns]
    if missing:
        return f"{source} is incomplete ({len(missing)} columns missing); finish run {source_run!r} first"
    target = results_store.method_dir(doc_id, "agent")
    if target.exists() and any(target.iterdir()) and results_store.load_metadata(doc_id, "agent").get("reused_from") != source_run:
        return f"{target} already holds Arm A results not copied from run {source_run!r}; use another --run"
    return None


def reuse_arm_a(doc_id: str, source_run: str) -> Dict[str, Any]:
    """Copy Arm A (results, metadata, raw responses) from `source_run` into the current run, as E's Arm A."""
    from src.evisearch.pipelines import results_store

    problem = reuse_problem(doc_id, source_run)
    if problem:
        raise results_store.ResumeError(problem)
    metadata = results_store.load_metadata(doc_id, "agent")
    if metadata.get("reused_from") == source_run:
        return {"reused_from": source_run, "copied": False, "usage": metadata.get("usage")}
    shutil.copytree(source_arm_a_dir(doc_id, source_run), results_store.method_dir(doc_id, "agent"), dirs_exist_ok=True)
    metadata = results_store.load_metadata(doc_id, "agent")
    metadata.pop("doc_id", None)
    results_store.save_metadata(doc_id, "agent", {**metadata, "run": results_store.current_run(), "reused_from": source_run})
    return {"reused_from": source_run, "copied": True, "usage": metadata.get("usage")}


def check_baseline(doc_id: str) -> Dict[str, Any]:
    """B1 has no check_run.py stage: FAIL when a column is missing or its group's call failed."""
    from src.evisearch.pipelines import results_store

    columns = results_store.load_columns(doc_id, "baseline")
    missing = [name for name in benchmark_columns() if name not in columns]
    failed = [name for name, cell in columns.items() if isinstance(cell, dict) and cell.get("value") == "Extraction error"]
    ok = not missing and not failed
    summary = "PASS: every column extracted" if ok else f"FAIL: {len(missing)} columns missing, {len(failed)} from failed calls"
    return {"result": "PASS" if ok else "FAIL", "summary": summary}


def run_check(doc_id: str, system: str, run: str) -> Dict[str, Any]:
    """check_run.py for the stages the system ran; its output is kept in runs/<run>/check_run.log."""
    from src.evisearch.pipelines import results_store

    if system == "B1":
        return check_baseline(doc_id)
    command = [sys.executable, str(CHECK_SCRIPT), doc_id, "--run", run, "--stages", CHECK_STAGES[system]]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, env=dict(os.environ))
    log = results_store.RESULTS_ROOT / doc_id / "runs" / run / "check_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    verdicts = [line for line in completed.stdout.splitlines() if line.startswith(("[check_run] PASS", "[check_run] FAIL"))]
    summary = verdicts[-1] if verdicts else (completed.stderr.strip().splitlines() or ["no output"])[-1]
    return {"result": "PASS" if completed.returncode == 0 else "FAIL", "exit_code": completed.returncode, "summary": summary, "log": str(log)}


# ---- one document ---------------------------------------------------------------------------------

def run_one_stage(stage: str, doc_id: str, reuse_a_from: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run (or copy) one stage for one document and return its record plus its error, if it failed.

    Nothing outside this document and stage is touched, so it is safe to call for two stages at the same time: each
    writes its own results, metadata and logs under its own method directory.
    """
    from src.evisearch.pipelines import results_store

    stage_start = time.time()
    try:
        outcome = reuse_arm_a(doc_id, reuse_a_from) if stage == "agent" and reuse_a_from else run_stage(stage, doc_id)
    except Exception as exc:  # a failed document must not stop the others
        error = f"{type(exc).__name__}: {exc}"
        print(f"[benchmark] {doc_id}: {stage} failed: {error}", file=sys.stderr, flush=True)
        return {"status": "failed", "duration_s": round(time.time() - stage_start, 3), "error": error}, error
    return {
        "status": "reused" if stage == "agent" and reuse_a_from else "ok",
        "duration_s": round(time.time() - stage_start, 3),
        **{key: outcome[key] for key in SUMMARY_KEYS if key in (outcome or {})},
        "timing": results_store.load_metadata(doc_id, stage).get("timing"),  # this stage's extraction_metadata.json
    }, None


def run_doc(doc_id: str, system: str, run: str, reuse_a_from: Optional[str] = None) -> Dict[str, Any]:
    """Every stage of `system` for one document, then its check; writes runs/<run>/benchmark_manifest.json.

    Stages run one after another unless EVISEARCH_STAGE_PARALLEL is on, in which case the stages of a wave (the ones
    that read none of the stages still waiting) run at the same time and the next wave starts once they are all done.
    Either way every stage that ran keeps its own record, a failure stops the waves that would have read it, and the
    check runs only after the last stage.
    """
    from src.evisearch.pipelines import results_store
    from src.inference.types import utc_timestamp

    started = time.time()
    record: Dict[str, Any] = {"doc_id": doc_id, "system": system, "run": run, "status": "ok", "error": None, "stages": {}, "check": None}
    waves = stage_waves(system)
    plan = ", ".join(SYSTEMS[system]) if all(len(wave) == 1 for wave in waves) else " then ".join(" + ".join(wave) for wave in waves)
    print(f"[benchmark] {doc_id}: start {system} ({plan})", flush=True)
    for wave in waves:
        if len(wave) == 1:
            outcomes = [run_one_stage(wave[0], doc_id, reuse_a_from)]
        else:  # one thread per stage; the work is inside the model calls, so the GIL is not in the way
            with ThreadPoolExecutor(max_workers=len(wave), thread_name_prefix="stage") as pool:
                outcomes = list(pool.map(lambda stage: run_one_stage(stage, doc_id, reuse_a_from), wave))
        first_error = None
        for stage, (stage_record, error) in zip(wave, outcomes):  # in SYSTEMS order, whatever finished first
            record["stages"][stage] = stage_record
            if error and first_error is None:
                first_error = f"{stage}: {error}"
        if first_error:  # what the later waves would read was not produced
            record.update(status="failed", error=first_error)
            break
    record["check"] = {"result": "skipped"}
    if record["status"] == "ok":
        try:
            record["check"] = run_check(doc_id, system, run)
        except Exception as exc:  # the checker itself failed to run
            record["check"] = {"result": "FAIL", "summary": f"check did not run: {type(exc).__name__}: {exc}"}
    finished = time.time()
    record.update(started_at=utc_timestamp(started), finished_at=utc_timestamp(finished), duration_s=round(finished - started, 3))
    results_store.write_json(results_store.RESULTS_ROOT / doc_id / "runs" / run / "benchmark_manifest.json", record)
    print(f"[benchmark] {doc_id}: {record['status']} in {record['duration_s']:.0f}s, check {record['check']['result']}", flush=True)
    return record


# ---- whole run ---------------------------------------------------------------------------------------

def git_state() -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def manifest_path(run: str) -> Path:
    from src.evisearch.pipelines import results_store

    return results_store.RESULTS_ROOT.parent / "benchmark_runs" / f"{run}.json"


def run_header(system: str, run: str, reuse_a_from: Optional[str], parallel: int) -> Dict[str, Any]:
    """What this run uses: preset, the model behind every role, input and git state."""
    from src.config.config import PAGE_IMAGE_SCALE, SELECTION

    catalog = SELECTION.catalog
    return {
        "run": run,
        "system": system,
        "description": DESCRIPTIONS[system],
        "preset": SELECTION.preset,
        "models": dict(SELECTION.roles),
        "model_names": {role: catalog.models[key].name for role, key in SELECTION.roles.items() if key},
        "stage_models": {stage: SELECTION.roles.get(STAGE_ROLES[stage]) for stage in SYSTEMS[system]},
        "input_mode": "parsed_markdown" if system == "B1" else ARM_A_INPUT,
        "page_image_scale": None if system == "B1" else PAGE_IMAGE_SCALE,
        "extraction_rules": SELECTION.option("extraction_rules"),
        "reuse_a_from": reuse_a_from,
        "parallel": parallel,
        "stage_parallel": stage_parallel(),  # schedule only: which stages overlapped, never what was sent or written
        "stage_concurrency": stage_concurrency(),
        "git": git_state(),
    }


def write_run_manifest(header: Dict[str, Any], invocation: Dict[str, Any], records: List[Dict[str, Any]], order: List[str]) -> Path:
    """new_pipeline_outputs/benchmark_runs/<run>.json, merged with earlier invocations of the same run by document
    (documents listed in `order`, the gold-table order)."""
    from src.evisearch.pipelines import results_store

    path = manifest_path(header["run"])
    with _manifest_lock:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        by_doc = {record["doc_id"]: record for record in previous.get("results", [])}
        by_doc.update({record["doc_id"]: record for record in records})
        invocations = [item for item in previous.get("invocations", []) if item.get("started_at") != invocation["started_at"]]
        rank = {doc_id: index for index, doc_id in enumerate(order)}
        results = sorted(by_doc.values(), key=lambda record: rank.get(record["doc_id"], len(rank)))
        manifest = {
            **header,
            "docs": [record["doc_id"] for record in results],
            "invocations": invocations + [invocation],
            "timings": {record["doc_id"]: {"duration_s": record["duration_s"], **{s: v.get("duration_s") for s, v in record["stages"].items()}} for record in results},
            "failures": [{"doc_id": record["doc_id"], "error": record["error"]} for record in results if record["status"] != "ok"],
            "check_failures": [record["doc_id"] for record in results if (record.get("check") or {}).get("result") == "FAIL"],
            "results": results,
        }
        results_store.write_json(path, manifest)
    return path


def describe_doc(doc_id: str, system: str, run: str, reuse_a_from: Optional[str]) -> List[str]:
    """Dry-run lines: what each stage would do for this document; no model is called."""
    from src.config.config import SELECTION
    from src.evisearch.pipelines import results_store
    from src.evisearch.pipelines.batching import build_batches, done_columns, load_groups

    lines = []
    for stage in SYSTEMS[system]:
        model = SELECTION.roles.get(STAGE_ROLES[stage])
        if stage == "agent" and reuse_a_from:
            problem = reuse_problem(doc_id, reuse_a_from)
            lines.append(f"agent: copy {source_arm_a_dir(doc_id, reuse_a_from)}" + (f"  REFUSED: {problem}" if problem else ""))
            continue
        try:
            results_store.check_resume(doc_id, stage, stage_settings(stage))
        except results_store.ResumeError as exc:
            lines.append(f"{stage}: REFUSED: {exc}")
            continue
        existing = results_store.load_columns(doc_id, stage)
        if stage == "baseline":
            from src.retrieval.embedding_retriever import parsed_markdown_path

            markdown = parsed_markdown_path(doc_id)
            size = f"{markdown.stat().st_size} bytes" if markdown.exists() else "MISSING"
            failed = sum(1 for cell in existing.values() if isinstance(cell, dict) and cell.get("value") == "Extraction error")
            lines.append(f"baseline: model={model} markdown={markdown} ({size}); {len(existing) - failed} columns already done")
            continue
        batches = build_batches(load_groups(), None, done=done_columns(existing))
        line = f"{stage}: model={model} batches={len(batches)} ({len(existing)} columns already done)"
        if stage == "agent" and batches:
            from src.evisearch.pipelines.pdf_query_pipeline import describe_fit

            line += f"\n      {describe_fit(doc_id, batches, model, ARM_A_INPUT)}"
        lines.append(line)
    if system in CHECK_STAGES:
        lines.append(f"check: {CHECK_SCRIPT.name} \"{doc_id}\" --run {run} --stages {CHECK_STAGES[system]}")
    return lines


def print_summary(records: List[Dict[str, Any]]) -> None:
    width = max([len(record["doc_id"]) for record in records] + [3])
    print(f"\n{'doc':<{width}}  system  status  {'duration':>9}  check")
    for record in records:
        check = record.get("check") or {}
        print(f"{record['doc_id']:<{width}}  {record['system']:<6}  {record['status']:<6}  {record['duration_s']:>8.0f}s  {check.get('result', '-')}")


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark system B1, B2 or E over the gold documents")
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS), help="B1 markdown baseline | B2 Arm A | E full pipeline")
    parser.add_argument("--docs", default="all", help="all | dev | heldout | comma-separated gold doc ids (default: all)")
    parser.add_argument("--run", required=True, help="Run name: results go to results/<doc_id>/runs/<run>/")
    parser.add_argument("--preset", choices=sorted(load_catalog().presets), help="Catalog preset (default: EVISEARCH_PRESET or config.py)")
    parser.add_argument("--parallel", type=int, default=2, help="Documents processed at the same time (default: 2)")
    parser.add_argument("--reuse-a-from", metavar="RUN", help="E only: copy Arm A from this finished B2 run instead of running it")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without calling any model or writing results")
    args = parser.parse_args(argv)
    if args.reuse_a_from and args.system != "E":
        parser.error("--reuse-a-from only applies to --system E")
    if args.reuse_a_from == args.run:
        parser.error("--reuse-a-from must name another run than --run")
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.preset:
        os.environ["EVISEARCH_PRESET"] = args.preset  # read when src.config.config is first imported; check_run inherits it
    try:
        from src.config import config
        from src.evisearch.pipelines import results_store
        from src.inference.types import utc_timestamp

        if args.preset and config.SELECTION.preset != args.preset:
            raise ConfigError(f"--preset {args.preset}: src.config was already loaded with preset '{config.SELECTION.preset}'")
        if args.reuse_a_from and not results_store.RUN_NAME_RE.match(args.reuse_a_from):
            raise ValueError(f"--reuse-a-from {args.reuse_a_from!r}: use letters, digits, '.', '_' and '-'")
        results_store.use_run(args.run)
        gold = gold_doc_ids()
        docs = select_docs(args.docs, gold, known=has_parse)
        if args.system != "B1" and not config.SELECTION.model("pdf_query").capabilities.images:
            raise ConfigError(f"{args.system} needs page images, but pdf_query model '{config.SELECTION.model_key('pdf_query')}' cannot read them")
    except (ConfigError, ValueError) as exc:
        print(f"[benchmark] {exc}", file=sys.stderr)
        return 2

    header = run_header(args.system, args.run, args.reuse_a_from, args.parallel)
    try:
        saved = json.loads(manifest_path(args.run).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved = None
    if saved:
        saved = {"extraction_rules": "none", **saved}  # manifests from before the option existed ran the E0 prompts
        clash = [f"{key}: {saved.get(key)!r} there, {header[key]!r} now" for key in ("system", "preset", "stage_models", "reuse_a_from", "extraction_rules") if saved.get(key) != header[key]]
        if clash:
            print(f"[benchmark] run {args.run!r} was started with other settings ({'; '.join(clash)}); use another --run", file=sys.stderr)
            return 2

    print(f"[benchmark] {args.system} ({DESCRIPTIONS[args.system]}) run={args.run} preset={header['preset']} docs={len(docs)} parallel={args.parallel}"
          + (f" stages={' then '.join(' + '.join(wave) for wave in stage_waves(args.system))}" if header["stage_parallel"] else "")
          + (f" batches={header['stage_concurrency']}" if header["stage_concurrency"] > 1 else ""))
    print(f"[benchmark] models: " + ", ".join(f"{stage}={model}" for stage, model in header["stage_models"].items())
          + f"; input={header['input_mode']}; git {(header['git'].get('commit') or '?')[:12]}{' (dirty)' if header['git'].get('dirty') else ''}")
    if args.dry_run:
        for doc_id in docs:
            print(f"\n{doc_id}")
            for line in describe_doc(doc_id, args.system, args.run, args.reuse_a_from):
                print(f"  {line}")
        print(f"\n[benchmark] dry run: nothing was called or written. Manifest would be {manifest_path(args.run)}")
        return 0

    started = time.time()
    invocation: Dict[str, Any] = {"started_at": utc_timestamp(started), "argv": sys.argv[1:] if argv is None else list(argv), "docs": docs, "git": header["git"]}
    records: List[Dict[str, Any]] = []

    def process(doc_id: str) -> Dict[str, Any]:
        record = run_doc(doc_id, args.system, args.run, args.reuse_a_from)
        with _manifest_lock:
            records.append(record)
            done = list(records)
        write_run_manifest(header, {**invocation, "finished_at": None}, done, gold)  # kept current while documents run
        return record

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(process, docs))
    finished = time.time()
    invocation.update(finished_at=utc_timestamp(finished), duration_s=round(finished - started, 3))
    records.sort(key=lambda record: docs.index(record["doc_id"]))
    path = write_run_manifest(header, invocation, records, gold)

    print_summary(records)
    failed = [record["doc_id"] for record in records if record["status"] != "ok" or (record.get("check") or {}).get("result") == "FAIL"]
    print(f"\n[benchmark] {len(records) - len(failed)}/{len(records)} documents ran and passed their check; manifest: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
