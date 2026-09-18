"""Score extraction outputs against the gold table, with Claude as the judge, under the evaluator_v2 rubric.

evaluator_v2 sends each column to an LLM judge with a category prompt (exact_match, numeric_tolerance,
structured_text). Here those prompts are the rubric and Claude does the judging, column by column:

  cells(...)        every (paper, column) of a system's output, paired with its gold value
  build_queue(...)  blinded batches of (column, definition, gold, prediction) for the scorer: no system names, each
                    distinct (paper, column, prediction) once, already-scored pairs skipped
  ingest(...)       validates a scorer's batch result and appends it to the label store
  report(...)       accuracy per paper, split, category; agreement-cell accuracy and review flags for EviSearch

Only the rubric's empty/empty rule (gold empty and prediction empty-equivalent -> 1/1) is applied mechanically;
every other cell is scored by the scorer. Unlike evaluator_v2, a column missing from a system's output counts as an
empty prediction instead of being skipped, so a system cannot gain by dropping columns, and a failed model call
("Extraction error") counts as an empty prediction too.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config.config import DEFINITIONS_EVAL_CATEGORY_PATH, GOLD_TABLE_JSON_PATH, PROJECT_ROOT

RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"
SCORING_DIR = PROJECT_ROOT / "experiment-scripts" / "scoring"
CATEGORIES = ("exact_match", "numeric_tolerance", "structured_text")
SCORES = (0.0, 0.5, 1.0)

# Empty-equivalent values listed in the evaluator_v2 prompts, compared case-insensitively.
EMPTY_VALUES = {"", "not reported", "not found", "not applicable", "n/a", "na", "nan", "not present"}
# Markers a pipeline writes when a model call failed: no answer, scored like an empty prediction for every system.
FAILED_VALUES = {"extraction error"}

HELDOUT = (
    "NCT00268476_James_STAMPEDE_IJC'22",
    "NCT00309985_Sweeney_CHAARTED_NEJM'15",
    "NCT02799602_Smith_ARASENS_NEJM'22",
)

STAGE_FILES = {
    "agent_extractor": "extraction_results.json",
    "search_agent": "extraction_results.json",
    "reconciliation_agent": "reconciled_results.json",
    "markdown_baseline": "extraction_results.json",
}


def normalize(value) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def is_empty(value) -> bool:
    return normalize(value).rstrip(".").strip().lower() in EMPTY_VALUES


@dataclass(frozen=True)
class Column:
    name: str
    category: str
    definition: str


@dataclass
class Cell:
    doc: str
    column: str
    category: str
    definition: str
    gold: str
    pred: str
    verification: Optional[str] = None  # reconciler label, EviSearch outputs only
    missing: bool = False  # column absent from the system's output
    failed: bool = False  # the model call for this column failed (its value was a failure marker)

    @property
    def id(self) -> str:
        return item_id(self.doc, self.column, self.pred)

    @property
    def mechanical(self) -> bool:
        return is_empty(self.gold) and is_empty(self.pred)


def item_id(doc: str, column: str, pred: str) -> str:
    return hashlib.sha1(f"{doc}\x1f{column}\x1f{normalize(pred)}".encode()).hexdigest()[:16]


def load_columns(path: Path = DEFINITIONS_EVAL_CATEGORY_PATH) -> Dict[str, Column]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["Column Name"]: Column(row["Column Name"], row["eval_category"], row.get("Definition", "") or "")
        for row in rows
        if row.get("eval_category") in CATEGORIES
    }


def doc_id(document_name: str) -> str:
    return document_name[:-4] if document_name.lower().endswith(".pdf") else document_name


def load_gold(path: Path = GOLD_TABLE_JSON_PATH) -> Dict[str, Dict[str, str]]:
    data = json.loads(Path(path).read_text())
    gold = {}
    for row in data["data"]:
        values = {name: normalize((cell or {}).get("value", "")) for name, cell in row.items() if name != "Document Name"}
        gold[doc_id(row["Document Name"]["value"])] = values
    return gold


def select_docs(spec: str, gold: Dict[str, Dict[str, str]]) -> List[str]:
    docs = sorted(gold)
    if spec == "all":
        return docs
    if spec == "heldout":
        return [d for d in docs if d in HELDOUT]
    if spec == "dev":
        return [d for d in docs if d not in HELDOUT]
    chosen = [d.strip() for d in spec.split(",") if d.strip()]
    unknown = [d for d in chosen if d not in gold]
    if unknown:
        raise ValueError(f"Not in the gold table: {unknown}")
    return chosen


@dataclass(frozen=True)
class System:
    """A named system output: `name=run/stage`, e.g. `B1=b1_qwen/markdown_baseline`."""

    name: str
    run: str
    stage: str

    @classmethod
    def parse(cls, spec: str) -> "System":
        match = re.fullmatch(r"([^=]+)=([^/]+)/([a-z_]+)", spec.strip())
        if not match or match.group(3) not in STAGE_FILES:
            raise ValueError(f"System spec must be name=run/stage with stage in {sorted(STAGE_FILES)}: {spec!r}")
        return cls(*match.groups())

    def path(self, doc: str, results_root: Path) -> Path:
        return Path(results_root) / doc / "runs" / self.run / self.stage / STAGE_FILES[self.stage]


def load_output(path: Path) -> Dict[str, dict]:
    data = json.loads(Path(path).read_text())
    columns = data.get("columns", data)
    return {name: (entry if isinstance(entry, dict) else {"value": entry}) for name, entry in columns.items()}


def cells(system: System, docs: Sequence[str], results_root: Path = RESULTS_ROOT,
          columns: Optional[Dict[str, Column]] = None, gold: Optional[Dict[str, Dict[str, str]]] = None) -> List[Cell]:
    columns = columns or load_columns()
    gold = gold or load_gold()
    out: List[Cell] = []
    for doc in docs:
        path = system.path(doc, results_root)
        if not path.exists():
            raise FileNotFoundError(f"{system.name}: no output for {doc} at {path}")
        predicted = load_output(path)
        for name, column in columns.items():
            entry = predicted.get(name)
            pred = normalize((entry or {}).get("value", ""))
            failed = pred.lower() in FAILED_VALUES
            out.append(Cell(
                doc=doc,
                column=name,
                category=column.category,
                definition=column.definition,
                gold=gold[doc].get(name, ""),
                pred="" if failed else pred,
                verification=(entry or {}).get("verification"),
                missing=entry is None,
                failed=failed,
            ))
    return out


# ---------------------------------------------------------------- label store

def labels_path(scoring_dir: Path = SCORING_DIR) -> Path:
    return Path(scoring_dir) / "labels.jsonl"


def load_labels(path: Path) -> Dict[str, dict]:
    labels: Dict[str, dict] = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                labels[record["id"]] = record
    return labels


def build_queue(all_cells: Iterable[Cell], labels: Dict[str, dict], queue_dir: Path, batch_size: int = 40) -> List[Path]:
    """Write blinded batches (one paper and category per batch) for every unscored, non-mechanical pair."""
    pending: Dict[str, Cell] = {}
    for cell in all_cells:
        if cell.mechanical or cell.id in labels or cell.id in pending:
            continue
        pending[cell.id] = cell
    groups: Dict[Tuple[str, str], List[Cell]] = {}
    for cell in pending.values():
        groups.setdefault((cell.doc, cell.category), []).append(cell)
    queue_dir = Path(queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    written = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for (doc, category), group in sorted(groups.items()):
        group.sort(key=lambda c: (c.column, c.pred))
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            name = f"{stamp}__{_slug(doc)}__{category}__{start // batch_size:02d}"
            batch = {
                "batch": name,
                "doc": doc,
                "category": category,
                "items": [
                    {"id": c.id, "column": c.column, "definition": c.definition, "gold": c.gold, "pred": c.pred}
                    for c in chunk
                ],
            }
            path = queue_dir / f"{name}.json"
            path.write_text(json.dumps(batch, indent=1, ensure_ascii=False))
            written.append(path)
    return written


def ingest(result_path: Path, queue_dir: Path, store: Path, scorer: str) -> int:
    """Validate a scorer's result for one queued batch and append it to the label store."""
    result = json.loads(Path(result_path).read_text())
    batch_path = Path(queue_dir) / f"{result['batch']}.json"
    batch = json.loads(batch_path.read_text())
    items = {item["id"]: item for item in batch["items"]}
    scored = {entry["id"]: entry for entry in result["results"]}
    problems = [f"unknown id {i}" for i in scored if i not in items]
    problems += [f"missing id {i}" for i in items if i not in scored]
    for entry in scored.values():
        for key in ("correctness", "completeness"):
            if float(entry.get(key, -1)) not in SCORES:
                problems.append(f"{entry['id']}: {key}={entry.get(key)!r} is not 0, 0.5 or 1")
        if not str(entry.get("reason", "")).strip():
            problems.append(f"{entry['id']}: empty reason")
    if problems:
        raise ValueError(f"{result_path}: " + "; ".join(problems[:10]))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(store, "a") as handle:
        for item_id_, entry in scored.items():
            item = items[item_id_]
            handle.write(json.dumps({
                "id": item_id_, "doc": batch["doc"], "column": item["column"], "category": batch["category"],
                "gold": item["gold"], "pred": item["pred"],
                "correctness": float(entry["correctness"]), "completeness": float(entry["completeness"]),
                "reason": entry["reason"].strip(), "scorer": scorer, "batch": batch["batch"], "scored_at": now,
            }, ensure_ascii=False) + "\n")
    return len(scored)


# ---------------------------------------------------------------- metrics

@dataclass
class Scored:
    cell: Cell
    correctness: float
    completeness: float
    source: str  # "mechanical" or "label"

    @property
    def score(self) -> float:
        return (self.correctness + self.completeness) / 2


def score(all_cells: Iterable[Cell], labels: Dict[str, dict]) -> Tuple[List[Scored], List[Cell]]:
    scored, unscored = [], []
    for cell in all_cells:
        if cell.mechanical:
            scored.append(Scored(cell, 1.0, 1.0, "mechanical"))
        elif cell.id in labels:
            label = labels[cell.id]
            scored.append(Scored(cell, label["correctness"], label["completeness"], "label"))
        else:
            unscored.append(cell)
    return scored, unscored


def summarize(rows: Sequence[Scored]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": None, "correctness": None, "completeness": None}
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(100 * sum(r.score for r in rows) / n, 2),
        "correctness": round(100 * sum(r.correctness for r in rows) / n, 2),
        "completeness": round(100 * sum(r.completeness for r in rows) / n, 2),
    }


def review_stats(rows: Sequence[Scored]) -> Optional[dict]:
    """Agreement-cell accuracy (T3) and review flags (both_wrong), for outputs carrying reconciler labels."""
    labelled = [r for r in rows if r.cell.verification]
    if not labelled:
        return None
    agreed = [r for r in labelled if r.cell.verification == "both_correct"]
    agreed_value = [r for r in agreed if not is_empty(r.cell.pred)]
    agreed_absent = [r for r in agreed if is_empty(r.cell.pred)]
    flagged = [r for r in labelled if r.cell.verification == "both_wrong"]
    errors = [r for r in labelled if r.score < 1]
    flagged_errors = [r for r in flagged if r.score < 1]
    unflagged = [r for r in labelled if r.cell.verification != "both_wrong"]
    after_review = sum(1.0 if r.cell.verification == "both_wrong" else r.score for r in labelled) / len(labelled)
    counts: Dict[str, int] = {}
    for r in labelled:
        counts[r.cell.verification] = counts.get(r.cell.verification, 0) + 1
    return {
        "verification_counts": counts,
        "agreed": summarize(agreed),
        "agreed_on_value": summarize(agreed_value),
        "agreed_not_reported": summarize(agreed_absent),
        "flag_rate": round(100 * len(flagged) / len(labelled), 2),
        "flag_precision": round(100 * len(flagged_errors) / len(flagged), 2) if flagged else None,
        "flag_recall": round(100 * len(flagged_errors) / len(errors), 2) if errors else None,
        "unflagged": summarize(unflagged),
        "accuracy_after_simulated_review": round(100 * after_review, 2),
    }


def report(system: System, docs: Sequence[str], labels: Dict[str, dict], results_root: Path = RESULTS_ROOT) -> dict:
    all_cells = cells(system, docs, results_root)
    scored, unscored = score(all_cells, labels)
    by_doc = {doc: summarize([r for r in scored if r.cell.doc == doc]) for doc in docs}
    return {
        "system": system.name, "run": system.run, "stage": system.stage,
        "complete": not unscored, "unscored": len(unscored),
        "missing_columns": sum(c.missing for c in all_cells),
        "failed_cells": sum(c.failed for c in all_cells),
        "overall": summarize(scored),
        "dev": summarize([r for r in scored if r.cell.doc not in HELDOUT]),
        "heldout": summarize([r for r in scored if r.cell.doc in HELDOUT]),
        "gold_filled": summarize([r for r in scored if not is_empty(r.cell.gold)]),
        "by_category": {c: summarize([r for r in scored if r.cell.category == c]) for c in CATEGORIES},
        "by_doc": by_doc,
        "review": review_stats(scored),
    }


# ---------------------------------------------------------------- consistency re-score

def build_recheck(labels: Dict[str, dict], fraction: float, seed: int, queue_dir: Path, batch_size: int = 40) -> List[Path]:
    """Queue a random sample of already-scored pairs for an independent second pass (earlier scores not shown)."""
    ids = sorted(labels)
    sample = random.Random(seed).sample(ids, max(1, round(fraction * len(ids)))) if ids else []
    columns = load_columns()
    recheck_cells = [
        Cell(doc=labels[i]["doc"], column=labels[i]["column"], category=labels[i]["category"],
             definition=columns[labels[i]["column"]].definition if labels[i]["column"] in columns else "",
             gold=labels[i]["gold"], pred=labels[i]["pred"])
        for i in sample
    ]
    return build_queue(recheck_cells, {}, queue_dir, batch_size)


def agreement(first: Dict[str, dict], second: Dict[str, dict]) -> dict:
    """Exact agreement on (correctness, completeness) and Cohen's kappa on the cell score."""
    common = sorted(set(first) & set(second))
    if not common:
        return {"n": 0}
    a = [(first[i]["correctness"] + first[i]["completeness"]) / 2 for i in common]
    b = [(second[i]["correctness"] + second[i]["completeness"]) / 2 for i in common]
    exact = sum(1 for i in common if (first[i]["correctness"], first[i]["completeness"])
                == (second[i]["correctness"], second[i]["completeness"])) / len(common)
    observed = sum(x == y for x, y in zip(a, b)) / len(common)
    levels = sorted(set(a) | set(b))
    expected = sum((a.count(level) / len(a)) * (b.count(level) / len(b)) for level in levels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": len(common), "exact_pair_agreement": round(100 * exact, 2),
            "score_agreement": round(100 * observed, 2), "cohen_kappa": round(kappa, 3)}


def rubric_text(category: str) -> str:
    """The evaluator_v2 prompt for a category, verbatim (header and closing instructions, no columns)."""
    from src.evaluation.evaluator_v2 import EvaluatorV2

    return EvaluatorV2.build_prompt(object.__new__(EvaluatorV2), category, [])


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60]
