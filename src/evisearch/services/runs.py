"""A run's outputs on disk, read-only: which runs exist, which papers each covers, each paper's final cells and how far
an unfinished paper has got.

Layout (written by experiment-scripts/run_benchmark.py and run_schema.py):
  RESULTS_ROOT/<doc>/runs/<run>/{agent_extractor,search_agent,reconciliation_agent,markdown_baseline}/...
  RESULTS_ROOT/<doc>/runs/<run>/benchmark_manifest.json     per-paper status, stages, timings (written when the paper ends)
  <RESULTS_ROOT.parent>/benchmark_runs/<run>.json           run header: system, models, preset, docs, invocations
A document's top-level stage folders (no run) are the legacy single-run layout; `run=None` reads those.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import runtime_paths

STAGE_DIRS = {"agent": "agent_extractor", "search": "search_agent", "reconciliation": "reconciliation_agent",
              "baseline": "markdown_baseline"}
RESULT_FILES = {"agent_extractor": "extraction_results.json", "search_agent": "extraction_results.json",
                "reconciliation_agent": "reconciled_results.json", "markdown_baseline": "extraction_results.json"}
NOT_REPORTED = frozenset({"", "not reported", "not found", "n/a", "na", "not applicable", "-", "—", "none", "nr"})
_SCHEMA_RUN = re.compile(r"^schema-(?P<schema>.+?)-v(?P<version>\d+(?:draft)?)(?:-(?P<variant>.+))?$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,160}$")


def check_name(run: str) -> str:
    if not _NAME.match(run or ""):
        raise ValueError(f"bad run name {run!r}")
    return run


def check_doc(doc_id: str) -> str:
    if not doc_id or "/" in doc_id or "\\" in doc_id or doc_id.startswith(".") or "\x00" in doc_id:
        raise ValueError(f"bad document id {doc_id!r}")
    return doc_id


def is_not_reported(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in NOT_REPORTED


def base_dir(doc_id: str, run: Optional[str]) -> Path:
    root = runtime_paths.RESULTS_ROOT / check_doc(doc_id)
    return root if not run else root / "runs" / check_name(run)


def headers_dir() -> Path:
    return runtime_paths.RESULTS_ROOT.parent / "benchmark_runs"


def parse_run_name(run: str) -> Dict[str, Any]:
    """schema-<id>-v<N>[-variant] -> {schema_id, version, variant}; other names -> {}."""
    m = _SCHEMA_RUN.match(run or "")
    if not m:
        return {}
    version = m.group("version")
    return {"schema_id": m.group("schema"), "version": int(version) if version.isdigit() else version,
            "variant": m.group("variant") or ""}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def header(run: str) -> Dict[str, Any]:
    data = _read_json(headers_dir() / f"{check_name(run)}.json")
    return data if isinstance(data, dict) else {}


def list_runs(schema_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every named run found on disk, with its papers; `schema_id` keeps only that schema's runs."""
    root = runtime_paths.RESULTS_ROOT
    found: Dict[str, List[str]] = {}
    if root.exists():
        for run_dir in root.glob("*/runs/*"):
            if run_dir.is_dir() and _NAME.match(run_dir.name):
                found.setdefault(run_dir.name, []).append(run_dir.parent.parent.name)
    out = []
    for run, docs in found.items():
        info = parse_run_name(run)
        if schema_id and info.get("schema_id") != schema_id:
            continue
        head = header(run)
        started = [i.get("started_at") for i in head.get("invocations", []) if i.get("started_at")]
        out.append({"run": run, "docs": sorted(docs), **info, "system": head.get("system"), "preset": head.get("preset"),
                    "models": head.get("stage_models") or {}, "started_at": min(started) if started else None,
                    "header_docs": head.get("docs") or []})
    return sorted(out, key=lambda r: (r.get("started_at") or "", r["run"]))


def stage_of(doc_id: str, run: Optional[str]) -> Optional[str]:
    """The folder holding the paper's final values: the arbiter's if it ran, else the baseline, else one agent."""
    base = base_dir(doc_id, run)
    for folder in ("reconciliation_agent", "markdown_baseline", "agent_extractor", "search_agent"):
        if (base / folder / RESULT_FILES[folder]).exists():
            return folder
    return None


def _columns(doc_id: str, run: Optional[str], folder: str) -> Dict[str, Any]:
    data = _read_json(base_dir(doc_id, run) / folder / RESULT_FILES[folder])
    cols = (data or {}).get("columns") if isinstance(data, dict) else None
    return cols if isinstance(cols, dict) else {}


def _evidence(cell: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{page, modality, quote}] from a stage's source/attribution fields."""
    out, seen = [], set()
    items = []
    if isinstance(cell.get("source"), dict):
        items.append(cell["source"])
    items += [a for a in (cell.get("attribution") or []) if isinstance(a, dict)]
    for a in items:
        try:
            page = int(a.get("page"))
        except (TypeError, ValueError):
            continue
        quote = str(a.get("verbatim_quote") or a.get("evidence") or a.get("snippet") or "").strip()
        key = (page, quote[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append({"page": page, "modality": str(a.get("modality") or a.get("source_type") or "text"), "quote": quote,
                    "verdict": a.get("verdict")})
    return out


def _value(cell: Any) -> str:
    if isinstance(cell, dict):
        v = cell.get("value", cell.get("primary_value", ""))
    else:
        v = cell
    return "" if v is None else str(v)


def final_cells(doc_id: str, run: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """{column: {value, flagged, flag_reason, verified, verification, decided_by, evidence, a, b, a_evidence, b_evidence,
    reasoning}} for one paper of one run (empty if the run has no output for it yet)."""
    folder = stage_of(doc_id, run)
    if not folder:
        return {}
    final = _columns(doc_id, run, folder)
    arms = {"a": _columns(doc_id, run, "agent_extractor"), "b": _columns(doc_id, run, "search_agent")}
    out: Dict[str, Dict[str, Any]] = {}
    for column in list(final) + [c for arm in arms.values() for c in arm if c not in final]:
        cell = final.get(column) if isinstance(final.get(column), dict) else {"value": _value(final.get(column))}
        a, b = arms["a"].get(column), arms["b"].get(column)
        out[column] = {
            "value": _value(cell),
            "flagged": bool(cell.get("needs_review")),
            "flag_reason": str(cell.get("review_reason") or ""),
            "verified": cell.get("verified"),
            "verification": str(cell.get("verification") or ""),
            "decided_by": str(cell.get("decided_by") or ""),
            "reasoning": str(cell.get("reasoning") or ""),
            "evidence": _evidence(cell),
            "a": _value(a) if a is not None else None,
            "b": _value(b) if b is not None else None,
            "a_reasoning": str(a.get("reasoning") or "") if isinstance(a, dict) else "",
            "b_reasoning": str(b.get("reasoning") or "") if isinstance(b, dict) else "",
            "a_evidence": _evidence(a) if isinstance(a, dict) else [],
            "b_evidence": _evidence(b) if isinstance(b, dict) else [],
            "stage": folder,
        }
    return out


def disputed(cell: Dict[str, Any]) -> bool:
    """Agents A and B gave different answers (ignoring case and spacing)."""
    a, b = cell.get("a"), cell.get("b")
    if a is None or b is None:
        return False
    norm = lambda v: " ".join(str(v).lower().split())
    if is_not_reported(a) and is_not_reported(b):
        return False
    return norm(a) != norm(b)


def _count_files(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.exists() else 0


def doc_progress(doc_id: str, run: str, batches: int = 11) -> Dict[str, Any]:
    """Status of one paper in a run: finished papers report their manifest; unfinished ones report which stage outputs
    and how many column batches exist so far."""
    base = base_dir(doc_id, run)
    manifest = _read_json(base / "benchmark_manifest.json")
    if isinstance(manifest, dict) and manifest.get("status"):
        stages = {k: {"status": v.get("status"), "filled": v.get("filled"), "total": v.get("total"),
                      "duration_s": v.get("duration_s")} for k, v in (manifest.get("stages") or {}).items() if isinstance(v, dict)}
        return {"doc_id": doc_id, "status": manifest["status"], "error": manifest.get("error"), "stages": stages,
                "duration_s": manifest.get("duration_s"), "finished_at": manifest.get("finished_at")}
    stages: Dict[str, Any] = {}
    for key, folder in STAGE_DIRS.items():
        path = base / folder
        if not path.exists():
            continue
        done = (path / RESULT_FILES[folder]).exists()
        n = max(_count_files(path / "raw_llm_responses", "*.json"), _count_files(path / "verification_logs", "batch_*"),
                _count_files(path / "raw_logs", "*.json"))
        stages[key] = {"status": "ok" if done else "running", "batches_done": batches if done else min(n, batches),
                       "batches": batches}
    return {"doc_id": doc_id, "status": "running" if stages else "waiting", "stages": stages}
