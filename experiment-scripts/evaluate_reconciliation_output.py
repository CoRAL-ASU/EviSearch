#!/usr/bin/env python3
"""
Evaluate reconciliation_agent output against ground truth.

Scans new_pipeline_outputs/results/*/reconciliation_agent/reconciled_results.json,
converts to extraction format, runs evaluator_v2, and prints aggregate metrics.

Usage:
  python experiment-scripts/evaluate_reconciliation_output.py
  python experiment-scripts/evaluate_reconciliation_output.py --results-dir new_pipeline_outputs/results
  python experiment-scripts/evaluate_reconciliation_output.py --doc "NCT00104715_Gravis_GETUG_EU'15"  # single doc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root for imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from baseline_utils import load_definitions_with_metadata, run_evaluation

DEFINITIONS_PATH = "src/table_definitions/Definitions_with_eval_category.csv"
GROUND_TRUTH_FILE = "dataset/Manual_Benchmark_GoldTable_cleaned.json"
RESULTS_ROOT = repo_root / "new_pipeline_outputs" / "results"


def reconciled_to_extraction(columns: dict) -> dict:
    """Convert reconciled_results columns to extraction_metadata format for evaluator."""
    definitions = load_definitions_with_metadata(DEFINITIONS_PATH)
    metadata = {}
    for col_name, col_data in columns.items():
        if not isinstance(col_data, dict):
            continue
        value = col_data.get("value")
        if value is None:
            value = ""
        value = str(value).strip() if value else ""
        reasoning = col_data.get("reasoning") or "Not applicable"
        col_def = definitions.get(col_name, {})
        metadata[col_name] = {
            "value": value or "Not applicable",
            "evidence": reasoning,
            "chunk_id": "reconciliation_agent_extraction",
            "page": "Not applicable",
            "column_index": col_def.get("index", "Not applicable"),
            "group_name": col_def.get("label", "Not applicable"),
            "plan_found_in_pdf": "Not applicable",
            "plan_page": "Not applicable",
            "plan_source_type": "Not applicable",
            "plan_confidence": "Not applicable",
            "plan_extraction_plan": "Not applicable",
        }
    return metadata


def find_reconciliation_outputs(results_dir: Path, doc_id: str | None) -> list[tuple[Path, str]]:
    """Find all reconciled_results.json paths. Returns [(path, doc_id), ...]."""
    out = []
    if doc_id:
        rec_path = results_dir / doc_id / "reconciliation_agent" / "reconciled_results.json"
        if rec_path.exists():
            out.append((rec_path, doc_id))
    else:
        for trial_dir in sorted(results_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            rec_path = trial_dir / "reconciliation_agent" / "reconciled_results.json"
            if rec_path.exists():
                out.append((rec_path, trial_dir.name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reconciliation_agent output against ground truth"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
        help="Root dir containing doc_id/reconciliation_agent/reconciled_results.json",
    )
    parser.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Evaluate only this doc_id (default: all)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluator; only aggregate existing evaluation/*/summary_metrics.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluation even if summary_metrics.json already exists",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        sys.exit(1)

    pairs = find_reconciliation_outputs(results_dir, args.doc)
    if not pairs:
        print("No reconciled_results.json found.")
        sys.exit(0)

    print(f"Found {len(pairs)} reconciliation output(s)\n")
    all_summaries = []

    for rec_path, doc_id in pairs:
        out_dir = rec_path.parent
        eval_dir = out_dir / "evaluation"
        extraction_path = out_dir / "extraction_for_eval.json"
        document_name = f"{doc_id}.pdf"

        summary_path = eval_dir / "summary_metrics.json"
        if not args.skip_eval:
            if summary_path.exists() and not args.force:
                print(f"  {doc_id}: already evaluated, skipping (use --force to re-run)")
            else:
                with open(rec_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                columns = data.get("columns") or {}
                if not columns:
                    print(f"  {doc_id}: no columns in reconciled_results, skipping")
                    continue
                extraction = reconciled_to_extraction(columns)
                with open(extraction_path, "w", encoding="utf-8") as f:
                    json.dump(extraction, f, indent=2, ensure_ascii=False)
                try:
                    run_evaluation(
                        extraction_file=str(extraction_path),
                        document_name=document_name,
                        output_dir=str(out_dir),
                        ground_truth_file=GROUND_TRUTH_FILE,
                        definitions_file=DEFINITIONS_PATH,
                    )
                except Exception as e:
                    print(f"  {doc_id}: evaluation failed — {e}")
                    continue

        # summary_path already set above
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary["doc_id"] = doc_id
            all_summaries.append(summary)
            ov = summary.get("overall", {})
            print(
                f"  {doc_id}: correctness={ov.get('avg_correctness', 0):.3f} "
                f"completeness={ov.get('avg_completeness', 0):.3f} "
                f"overall={ov.get('avg_overall', 0):.3f}"
            )
        else:
            print(f"  {doc_id}: no summary_metrics.json (eval may have failed)")

    if not all_summaries:
        print("\nNo evaluation summaries to aggregate.")
        sys.exit(0)

    # Aggregate
    n = len(all_summaries)
    avg_corr = sum(s.get("overall", {}).get("avg_correctness", 0) for s in all_summaries) / n
    avg_comp = sum(s.get("overall", {}).get("avg_completeness", 0) for s in all_summaries) / n
    avg_overall = sum(s.get("overall", {}).get("avg_overall", 0) for s in all_summaries) / n
    total_cols = sum(s.get("overall", {}).get("total_columns", 0) for s in all_summaries)

    print("\n" + "=" * 60)
    print("RECONCILIATION EVALUATION — AGGREGATE")
    print("=" * 60)
    print(f"  Documents: {n}")
    print(f"  Total columns evaluated: {total_cols}")
    print(f"  Avg Correctness:   {avg_corr:.3f}")
    print(f"  Avg Completeness:  {avg_comp:.3f}")
    print(f"  Avg Overall:       {avg_overall:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
