
# main/main_v2.py
# Pipeline V2: Chunking -> Planning -> Extraction -> Evaluation (plan-based)
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_FOLDER = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_FOLDER))

from src.chunking.chunking import process_pdf
from src.config.config import (
    DEFINITIONS_EVAL_CATEGORY_PATH,
    EXTRACTION_MODEL_V2,
    EXTRACTION_PROVIDER_V2,
    EXTRACTION_WORKERS,
    EVALUATION_MODEL_V2,
    EVALUATION_PROVIDER_V2,
    EVALUATION_WORKERS,
    GOLD_TABLE_JSON_PATH,
    PLANNING_MODEL,
    PLANNING_PROVIDER,
    PLANNING_WORKERS,
    PROJECT_ROOT,
    RESULTS_BASE_DIR,
    SKIP_STAGE_IF_EXISTS,
    STRUCTURER_BASE_URL,
    STRUCTURER_MODEL,
    USE_LLM_PAGE_CLASSIFICATION,
    VERSION_OUTPUTS,
)
from src.evaluation.evaluator_v2 import EvaluatorV2
from src.extraction.plan_executor import PlanExecutor, load_plans_from_dir
from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.planning.plan_generator import PlanGenerator
from src.table_definitions.definitions import load_definitions
from src.utils.costing import build_pipeline_cost_summary
from src.utils.logging_utils import setup_logger

logger = setup_logger("main_v2")


def save_pipeline_costs(
    run_dir: Path,
    chunk_usage: dict | None = None,
    plan_usage: dict | None = None,
    extraction_usage: dict | None = None,
    eval_usage: dict | None = None,
) -> None:
    """Write per-module cost JSON and summary to run_dir/cost/."""
    cost_dir = run_dir / "cost"
    cost_dir.mkdir(parents=True, exist_ok=True)
    if chunk_usage is not None:
        (cost_dir / "chunking.json").write_text(json.dumps(chunk_usage, indent=2), encoding="utf-8")
    if plan_usage is not None:
        (cost_dir / "planning.json").write_text(json.dumps(plan_usage, indent=2), encoding="utf-8")
    if extraction_usage is not None:
        (cost_dir / "extraction.json").write_text(json.dumps(extraction_usage, indent=2), encoding="utf-8")
    if eval_usage is not None:
        (cost_dir / "evaluation.json").write_text(json.dumps(eval_usage, indent=2), encoding="utf-8")
    summary = build_pipeline_cost_summary(
        chunking=chunk_usage,
        planning=plan_usage,
        extraction=extraction_usage,
        evaluation=eval_usage,
    )
    (cost_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Costs saved to %s", cost_dir)


def _find_existing(paths: list) -> Path | None:
    """Return the first path that exists, or None. Ignores None entries."""
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _find_extraction_in_trial_dir(trial_dir: Path) -> Path | None:
    """Find extraction_metadata.json under a trial dir. Checks extraction/, extractions/, latest/extraction/."""
    for sub in ("extraction", "extractions"):
        candidate = trial_dir / sub / "extraction_metadata.json"
        if candidate.exists():
            return candidate
    latest = trial_dir / "latest" / "extraction" / "extraction_metadata.json"
    if latest.exists():
        return latest
    return None


def _get_eval_output_dir(trial_dir: Path, extraction_file: Path) -> Path:
    """Return the evaluation output dir for this trial (same level as extraction)."""
    # If extraction is under trial/latest/extraction/, write to trial/latest/evaluation/
    if "latest" in extraction_file.parts:
        return trial_dir / "latest" / "evaluation"
    return trial_dir / "evaluation"


def run_evaluation_for_all_trials_in_dir(alt_base: Path) -> list[tuple[str, Path | None, str | None]]:
    """
    Run evaluation for every trial under alt_base that has extraction.
    alt_base can be the 'new_pipeline_outputs copy' folder (we use alt_base/results) or the results dir itself.
    Returns list of (trial_name, eval_dir or None, error_message or None).
    """
    results_dir = alt_base / "results" if (alt_base / "results").is_dir() else alt_base
    if not results_dir.is_dir():
        return [("", None, f"Not a directory: {results_dir}")]

    outcomes = []
    for trial_dir in sorted(results_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        trial_name = trial_dir.name
        extraction_file = _find_extraction_in_trial_dir(trial_dir)
        if extraction_file is None:
            outcomes.append((trial_name, None, "No extraction_metadata.json found"))
            continue
        eval_dir = _get_eval_output_dir(trial_dir, extraction_file)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if SKIP_STAGE_IF_EXISTS and (eval_dir / "evaluation_results.json").exists():
            outcomes.append((trial_name, eval_dir, None))  # skipped
            continue
        if not GOLD_TABLE_JSON_PATH.exists():
            outcomes.append((trial_name, None, "Ground truth file not found"))
            continue
        with open(GOLD_TABLE_JSON_PATH, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        doc_names = [row.get("Document Name", {}).get("value", "") for row in gt_data.get("data", [])]
        doc_name_match = f"{trial_name}.pdf"
        if doc_name_match not in doc_names:
            outcomes.append((trial_name, None, f"No ground truth for {doc_name_match}"))
            continue
        try:
            evaluator = EvaluatorV2(
                extraction_file=str(extraction_file),
                ground_truth_file=str(GOLD_TABLE_JSON_PATH),
                definitions_file=str(DEFINITIONS_EVAL_CATEGORY_PATH),
                document_name=doc_name_match,
                output_dir=str(eval_dir),
            )
            evaluator.run()
            outcomes.append((trial_name, eval_dir, None))
        except Exception as e:
            outcomes.append((trial_name, None, str(e)))
    return outcomes


def _run_evaluation_all_alternate():
    """Prompt for alternate results folder and run evaluation for every trial that has extraction."""
    default_alt = "new_pipeline_outputs copy"
    alt_input = input(f"\nEnter path to alternate results folder [{default_alt}]: ").strip() or default_alt
    alt_base = Path(alt_input)
    if not alt_base.is_absolute():
        alt_base = PROJECT_ROOT / alt_base
    if not alt_base.is_dir():
        print("Folder not found:", alt_base)
        sys.exit(1)
    print("\nRunning evaluation for all trials with extraction in:", alt_base)
    outcomes = run_evaluation_for_all_trials_in_dir(alt_base)
    print("\n" + "=" * 60)
    print("EVALUATION BATCH COMPLETE")
    print("=" * 60)
    for trial_name, eval_dir, err in outcomes:
        if err:
            print(f"  {trial_name}: FAILED - {err}")
        else:
            status = "skipped (already exists)" if eval_dir and (eval_dir / "evaluation_results.json").exists() and SKIP_STAGE_IF_EXISTS else "done"
            print(f"  {trial_name}: {status}" + (f" -> {eval_dir}" if eval_dir else ""))
    print("=" * 60)


def create_versioned_output_dir(base_dir: Path) -> tuple:
    """Create timestamped run directory and update 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_link = base_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(run_dir.name)
    except OSError:
        # On some systems symlink may fail; leave without latest
        pass
    return run_dir, latest_link


def run_chunking(
    pdf_path: Path,
    output_dir: Path,
    base_output: Path | None = None,
) -> tuple[Path, dict | None]:
    """Stage 1: Chunk PDF into text/image/table chunks. Skips if chunks already exist.
    Returns (chunk_file_path, usage_dict for costing or None if skipped)."""
    logger.info("=" * 60)
    logger.info("STAGE 1: CHUNKING")
    logger.info("=" * 60)
    chunk_dir = output_dir / "chunking"
    chunk_dir.mkdir(exist_ok=True)
    chunk_file = chunk_dir / "pdf_chunked.json"

    if SKIP_STAGE_IF_EXISTS:
        existing = _find_existing([
            chunk_file,
            (base_output / "chunking" / "pdf_chunked.json") if base_output else None,
            (base_output / "latest" / "chunking" / "pdf_chunked.json") if base_output else None,
        ])
        if existing is not None:
            logger.info("Chunks already exist, skipping chunking: %s", existing)
            return existing, None

    chunks, usage_dict = process_pdf(
        str(pdf_path),
        output_path=str(chunk_file),
        use_llm_classification=USE_LLM_PAGE_CLASSIFICATION,
    )
    logger.info("Chunks saved to %s", chunk_file)
    return chunk_file, usage_dict


def run_planning(
    pdf_path: Path,
    chunk_file: Path,
    output_dir: Path,
    base_output: Path | None = None,
) -> tuple[Path, dict | None]:
    """Stage 2: Generate extraction plans for all column groups. Skips if plans already exist.
    Returns (plan_dir, usage_dict for costing or None if skipped)."""
    logger.info("=" * 60)
    logger.info("STAGE 2: PLANNING")
    logger.info("=" * 60)
    plan_dir = output_dir / "planning"
    plan_dir.mkdir(exist_ok=True)

    if SKIP_STAGE_IF_EXISTS:
        for candidate in [plan_dir, (base_output / "planning") if base_output else None, (base_output / "latest" / "planning") if base_output else None]:
            if candidate is None:
                continue
            existing_plans = load_plans_from_dir(Path(candidate))
            if existing_plans:
                logger.info("Plans already exist (%d groups), skipping planning: %s", len(existing_plans), candidate)
                return Path(candidate), None

    provider = LLMProvider(provider=PLANNING_PROVIDER, model=PLANNING_MODEL)
    definitions = load_definitions()
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    planner = PlanGenerator(provider, definitions)
    plans, usage_dict = planner.generate_plans(
        pdf_path=pdf_path,
        chunks=chunks,
        output_dir=plan_dir,
        workers=PLANNING_WORKERS,
    )
    logger.info("Generated %d extraction plans in %s", len(plans), plan_dir)
    return plan_dir, usage_dict


def run_extraction(
    pdf_path: Path,
    chunk_file: Path,
    plan_dir: Path,
    output_dir: Path,
    base_output: Path | None = None,
) -> tuple[Path, dict | None]:
    """Stage 3: Execute extraction plans. Skips if extraction_metadata.json already exists.
    Returns (extraction_file path, usage_dict for costing or None if skipped)."""
    logger.info("=" * 60)
    logger.info("STAGE 3: EXTRACTION")
    logger.info("=" * 60)
    extraction_dir = output_dir / "extraction"
    extraction_dir.mkdir(exist_ok=True)
    extraction_file = extraction_dir / "extraction_metadata.json"

    if SKIP_STAGE_IF_EXISTS:
        existing = _find_existing([
            extraction_file,
            (base_output / "extraction" / "extraction_metadata.json") if base_output else None,
            (base_output / "latest" / "extraction" / "extraction_metadata.json") if base_output else None,
        ])
        if existing is not None:
            logger.info("Extraction already exists, skipping: %s", existing)
            return existing, None

    extraction_provider = LLMProvider(provider=EXTRACTION_PROVIDER_V2, model=EXTRACTION_MODEL_V2)
    structurer = OutputStructurer(base_url=STRUCTURER_BASE_URL, model=STRUCTURER_MODEL)
    plans = load_plans_from_dir(plan_dir)
    if not plans:
        logger.error("No plans found in %s", plan_dir)
        sys.exit(1)
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    executor = PlanExecutor(extraction_provider, structurer)
    _, usage_dict = executor.execute_plans(
        pdf_path=pdf_path,
        chunks=chunks,
        plans=plans,
        output_path=extraction_file,
        workers=EXTRACTION_WORKERS,
    )
    logger.info("Extraction complete: %s", extraction_file)
    return extraction_file, usage_dict


def run_evaluation(
    extraction_file: Path,
    pdf_name: str,
    output_dir: Path,
    base_output: Path | None = None,
):
    """Stage 4: Evaluate extraction against ground truth. Skips if evaluation results already exist."""
    logger.info("=" * 60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("=" * 60)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    if SKIP_STAGE_IF_EXISTS:
        for eval_base in [eval_dir, (base_output / "evaluation") if base_output else None, (base_output / "latest" / "evaluation") if base_output else None]:
            if eval_base is None:
                continue
            if (Path(eval_base) / "evaluation_results.json").exists() or (Path(eval_base) / "summary_metrics.json").exists():
                logger.info("Evaluation results already exist, skipping: %s", eval_base)
                return None, None

    if not GOLD_TABLE_JSON_PATH.exists():
        logger.warning("Ground truth file not found, skipping evaluation")
        return None, None
    with open(GOLD_TABLE_JSON_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    doc_names = [row.get("Document Name", {}).get("value", "") for row in gt_data.get("data", [])]
    doc_name_match = f"{pdf_name}.pdf"
    if doc_name_match not in doc_names:
        logger.warning("No ground truth for %s, skipping evaluation", pdf_name)
        return None, None
    evaluator = EvaluatorV2(
        extraction_file=str(extraction_file),
        ground_truth_file=str(GOLD_TABLE_JSON_PATH),
        definitions_file=str(DEFINITIONS_EVAL_CATEGORY_PATH),
        document_name=doc_name_match,
        output_dir=str(eval_dir),
    )
    results = evaluator.run()
    usage_dict = evaluator.get_usage()
    logger.info("Evaluation complete: %s", eval_dir)
    return results, usage_dict


def run_pipeline_from_args(pdf_path: Path, choice: str):
    """
    Run pipeline non-interactively (e.g. from web UI).
    Same logic as main() but takes pdf_path and choice as arguments.
    Returns: (run_dir, extraction_file, error_message).
    If error_message is not None, run_dir and extraction_file may be None.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return None, None, f"PDF not found: {pdf_path}"
        pdf_name = pdf_path.stem
        choice = str(choice).strip()
        if choice not in ("1", "2", "3", "4", "5", "6", "7"):
            return None, None, "Invalid choice. Use 1-7."

        base_output = RESULTS_BASE_DIR / pdf_name
        if VERSION_OUTPUTS:
            run_dir, _ = create_versioned_output_dir(base_output)
        else:
            run_dir = base_output
            run_dir.mkdir(parents=True, exist_ok=True)

        chunk_file = None
        plan_dir = None
        extraction_file = None
        chunk_usage = plan_usage = extraction_usage = eval_usage = None

        if choice in ("1", "5"):
            chunk_file, chunk_usage = run_chunking(pdf_path, run_dir, base_output=base_output)

        if choice in ("2", "5", "6"):
            if chunk_file is None:
                chunk_file = _find_existing([
                    run_dir / "chunking" / "pdf_chunked.json",
                    base_output / "chunking" / "pdf_chunked.json",
                    base_output / "latest" / "chunking" / "pdf_chunked.json",
                ])
            if chunk_file is None or not chunk_file.exists():
                return run_dir, None, "Chunks not found. Run chunking first (option 1 or 5)."
            plan_dir, plan_usage = run_planning(pdf_path, chunk_file, run_dir, base_output=base_output)

        if choice in ("3", "5", "6"):
            if chunk_file is None:
                chunk_file = _find_existing([
                    run_dir / "chunking" / "pdf_chunked.json",
                    base_output / "chunking" / "pdf_chunked.json",
                    base_output / "latest" / "chunking" / "pdf_chunked.json",
                ])
            if plan_dir is None:
                for d in [run_dir / "planning", base_output / "planning", base_output / "latest" / "planning"]:
                    if d.exists() and load_plans_from_dir(d):
                        plan_dir = d
                        break
                else:
                    plan_dir = run_dir / "planning"
            if chunk_file is None or not chunk_file.exists():
                return run_dir, None, "Chunks not found. Run chunking first."
            if not plan_dir.exists() or not load_plans_from_dir(plan_dir):
                return run_dir, None, "Plans not found. Run planning first."
            extraction_file, extraction_usage = run_extraction(pdf_path, chunk_file, plan_dir, run_dir, base_output=base_output)

        if choice in ("4", "5", "6"):
            if extraction_file is None:
                extraction_file = _find_existing([
                    run_dir / "extraction" / "extraction_metadata.json",
                    base_output / "extraction" / "extraction_metadata.json",
                    base_output / "latest" / "extraction" / "extraction_metadata.json",
                ])
            if extraction_file is None or not extraction_file.exists():
                return run_dir, None, "Extraction not found. Run extraction first (option 3, 5, or 6)."
            _, eval_usage = run_evaluation(extraction_file, pdf_name, run_dir, base_output=base_output)

        save_pipeline_costs(run_dir, chunk_usage, plan_usage, extraction_usage, eval_usage)
        return run_dir, extraction_file, None
    except Exception as e:
        logger.exception("Pipeline run failed")
        return None, None, str(e)


def main():
    print("\n" + "=" * 60)
    print("CLINICAL TRIAL EXTRACTION PIPELINE V2")
    print("=" * 60)

    print("\nSelect pipeline stage to run:")
    print("  1. Chunking only")
    print("  2. Planning only (requires chunks)")
    print("  3. Extraction only (requires chunks + plans)")
    print("  4. Evaluation only (requires extraction)")
    print("  5. Complete pipeline (all stages)")
    print("  6. Planning -> Extraction -> Evaluation (resume from chunks)")
    print("  7. Evaluation only for ALL trials (from alternate results folder, e.g. new_pipeline_outputs copy)")
    choice = input("\nEnter choice (1-7): ").strip()

    if choice == "7":
        _run_evaluation_all_alternate()
        return

    pdf_path = input("\nEnter PDF path: ").strip()
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print("PDF not found:", pdf_path)
        sys.exit(1)
    pdf_name = pdf_path.stem

    # For choice 4, allow using an alternate results folder (e.g. new_pipeline_outputs copy)
    base_output = RESULTS_BASE_DIR / pdf_name
    use_alternate_for_eval = False
    if choice == "4":
        use_alt = input("Use alternate results folder for extraction? (y/n) [n]: ").strip().lower() or "n"
        if use_alt == "y":
            alt_input = input("Enter path to alternate folder (e.g. new_pipeline_outputs copy): ").strip()
            if alt_input:
                alt_base = Path(alt_input)
                if not alt_base.is_absolute():
                    alt_base = PROJECT_ROOT / alt_base
                if alt_base.is_dir():
                    base_output = (alt_base / "results" / pdf_name) if (alt_base / "results").is_dir() else (alt_base / pdf_name)
                    use_alternate_for_eval = True
                    print("Looking for extraction and writing evaluation to:", base_output)
                else:
                    print("Folder not found, using default results dir.")
    base_output.mkdir(parents=True, exist_ok=True)

    # Option A: For evaluation-only, use existing run (latest); don't create a new versioned run.
    if use_alternate_for_eval:
        run_dir = base_output
        extraction_file_alt = _find_extraction_in_trial_dir(run_dir)
        if extraction_file_alt is None:
            print("Extraction not found under", run_dir, "(checked extraction/, extractions/, latest/extraction/)")
            sys.exit(1)
        run_dir = _get_eval_output_dir(base_output, extraction_file_alt).parent
        print("\nOutput directory (evaluation only, alternate folder):", run_dir)
        latest_link = None
    elif VERSION_OUTPUTS:
        if choice == "4":
            latest_dir = base_output / "latest"
            if latest_dir.exists() and latest_dir.is_symlink():
                run_dir = latest_dir.resolve()
                print("\nOutput directory (evaluation only, using existing run):", run_dir)
            else:
                run_dir = base_output
                print("\nOutput directory (evaluation only, no latest run):", run_dir)
            latest_link = base_output / "latest"
        else:
            run_dir, latest_link = create_versioned_output_dir(base_output)
            print("\nOutput directory:", run_dir)
            if latest_link.exists():
                print("Latest link:", latest_link, "->", run_dir.name)
    else:
        run_dir = base_output
        latest_link = None

    chunk_file = None
    plan_dir = None
    extraction_file = None
    extraction_file_alt = None  # set when use_alternate_for_eval
    chunk_usage = plan_usage = extraction_usage = eval_usage = None

    if choice in ("1", "5"):
        chunk_file, chunk_usage = run_chunking(pdf_path, run_dir, base_output=base_output)

    if choice in ("2", "5", "6"):
        if chunk_file is None:
            chunk_file = _find_existing([
                run_dir / "chunking" / "pdf_chunked.json",
                base_output / "chunking" / "pdf_chunked.json",
                base_output / "latest" / "chunking" / "pdf_chunked.json",
            ])
        if chunk_file is None or not chunk_file.exists():
            print("Chunks not found. Run chunking first.")
            sys.exit(1)
        plan_dir, plan_usage = run_planning(pdf_path, chunk_file, run_dir, base_output=base_output)

    if choice in ("3", "5", "6"):
        if chunk_file is None:
            chunk_file = _find_existing([
                run_dir / "chunking" / "pdf_chunked.json",
                base_output / "chunking" / "pdf_chunked.json",
                base_output / "latest" / "chunking" / "pdf_chunked.json",
            ])
        if plan_dir is None:
            for d in [run_dir / "planning", base_output / "planning", base_output / "latest" / "planning"]:
                if d.exists() and load_plans_from_dir(d):
                    plan_dir = d
                    break
            else:
                plan_dir = run_dir / "planning"
        if chunk_file is None or not chunk_file.exists():
            print("Chunks not found. Run chunking first.")
            sys.exit(1)
        if not plan_dir.exists() or not load_plans_from_dir(plan_dir):
            print("Plans not found. Run planning first.")
            sys.exit(1)
        extraction_file, extraction_usage = run_extraction(pdf_path, chunk_file, plan_dir, run_dir, base_output=base_output)

    if choice in ("4", "5", "6"):
        if use_alternate_for_eval:
            extraction_file = extraction_file_alt
        elif extraction_file is None:
            extraction_file = _find_existing([
                run_dir / "extraction" / "extraction_metadata.json",
                run_dir / "extractions" / "extraction_metadata.json",
                base_output / "extraction" / "extraction_metadata.json",
                base_output / "latest" / "extraction" / "extraction_metadata.json",
            ])
        if extraction_file is None or not extraction_file.exists():
            print("Extraction not found. Run extraction first.")
            sys.exit(1)
        _, eval_usage = run_evaluation(extraction_file, pdf_name, run_dir, base_output=base_output)

    save_pipeline_costs(run_dir, chunk_usage, plan_usage, extraction_usage, eval_usage)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("Results:", run_dir)
    if latest_link and latest_link.exists():
        print("Latest:", latest_link, "->", run_dir.name)
    print("=" * 60)


if __name__ == "__main__":
    main()
