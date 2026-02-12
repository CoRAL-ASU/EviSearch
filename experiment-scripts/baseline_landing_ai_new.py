import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from landingai_ade import LandingAIADE

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))  # Add experiment-scripts to path

from baseline_utils import (  # noqa: E402
    load_definitions_with_metadata,
    convert_to_extraction_metadata,
    run_evaluation,
    build_schema_from_definitions,
)


load_dotenv()

# Paths configuration
BASE_PDF_PATH = "dataset"  # Directory containing PDFs
DEFINITIONS_PATH = "src/table_definitions/Definitions_with_eval_category.csv"
OUTPUT_BASE_DIR = "experiment-scripts/baselines_landing_ai_new_results"
GROUND_TRUTH_FILE = "dataset/Manual_Benchmark_GoldTable_cleaned.json"

# Model configuration (override with env vars if needed)
PARSE_MODEL = os.getenv("LANDING_AI_PARSE_MODEL", "dpt-2-latest")
EXTRACT_MODEL = os.getenv("LANDING_AI_EXTRACT_MODEL", "extract-latest")

# Create base output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Load definitions and build schema
definitions = load_definitions_with_metadata(DEFINITIONS_PATH)
schema = build_schema_from_definitions(definitions)
schema_json = json.dumps(schema)


def _ensure_api_key() -> None:
    api_key = os.getenv("VISION_AGENT_API_KEY") or os.getenv("LANDING_AI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing API key. Set VISION_AGENT_API_KEY (preferred) or LANDING_AI_API_KEY."
        )
    if not os.getenv("VISION_AGENT_API_KEY"):
        os.environ["VISION_AGENT_API_KEY"] = api_key


def _init_client() -> LandingAIADE:
    env = os.getenv("LANDING_AI_ENV", "").strip().lower()
    if env == "eu":
        return LandingAIADE(environment="eu")
    return LandingAIADE()


def _serialize_response(resp):
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    if hasattr(resp, "__dict__"):
        return resp.__dict__
    return resp


def process_pdf(client: LandingAIADE, pdf_path: str, pdf_name: str):
    """
    Parse PDF with ADE Parse and extract fields with ADE Extract.

    Returns:
        Extracted data dict or None if failed
    """
    # Create subdirectory for this PDF
    pdf_stem = pdf_name.rsplit(".pdf", 1)[0]
    pdf_dir = os.path.join(OUTPUT_BASE_DIR, pdf_stem)
    os.makedirs(pdf_dir, exist_ok=True)

    raw_output_file = os.path.join(pdf_dir, "raw_extract_response.json")
    markdown_file = os.path.join(pdf_dir, "parsed_markdown.md")

    # Check if the extracted file already exists
    if os.path.exists(raw_output_file):
        print(f"Raw extraction for {pdf_name} already exists at {raw_output_file}. Skipping API call.")
        try:
            with open(raw_output_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict) and "extraction" in raw_data:
                return raw_data["extraction"]
            return raw_data
        except Exception as e:
            print(f"Error reading existing file for {pdf_name}: {str(e)}. Proceeding with API call.")

    print(f"Sending {pdf_name} to LandingAI ADE Parse...")
    try:
        parse_response = client.parse(
            document=Path(pdf_path),
            model=PARSE_MODEL,
        )
    except Exception as e:
        print(f"Error parsing {pdf_name}: {str(e)}")
        return None

    if not getattr(parse_response, "markdown", None):
        print(f"No markdown returned for {pdf_name}.")
        return None

    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(parse_response.markdown)

    print(f"Extracting fields from {pdf_name} with ADE Extract...")
    try:
        extract_response = client.extract(
            schema=schema_json,
            markdown=parse_response.markdown,
            model=EXTRACT_MODEL,
        )
    except Exception as e:
        print(f"Error extracting {pdf_name}: {str(e)}")
        return None

    extracted_info = getattr(extract_response, "extraction", None)
    if not isinstance(extracted_info, dict):
        print(f"Extraction output missing or invalid for {pdf_name}.")
        return None

    with open(raw_output_file, "w", encoding="utf-8") as f:
        json.dump(_serialize_response(extract_response), f, indent=2)

    print(f"Extraction saved to {raw_output_file}")
    return extracted_info


def convert_and_evaluate(pdf_name: str, extracted_data: dict, skip_eval: bool = False):
    """
    Convert extraction output to extraction_metadata.json and optionally run evaluation.
    """
    if not extracted_data:
        print(f"No extracted data for {pdf_name}")
        return

    print(f"Converting {pdf_name} to extraction_metadata format...")

    extraction_metadata = convert_to_extraction_metadata(
        extracted_data,
        definitions,
        source="landing_ai_new",
    )

    pdf_stem = pdf_name.rsplit(".pdf", 1)[0]
    pdf_dir = os.path.join(OUTPUT_BASE_DIR, pdf_stem)
    os.makedirs(pdf_dir, exist_ok=True)

    extraction_file = os.path.join(pdf_dir, "extraction_metadata.json")
    with open(extraction_file, "w", encoding="utf-8") as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
    print(f"Extraction metadata saved to {extraction_file}")

    if skip_eval:
        print("Skipping evaluation (per --skip-eval).")
        return

    print(f"Running evaluation for {pdf_name}...")
    try:
        results = run_evaluation(
            extraction_file=extraction_file,
            document_name=pdf_name,
            output_dir=pdf_dir,
            ground_truth_file=GROUND_TRUTH_FILE,
            definitions_file=DEFINITIONS_PATH,
        )
        if results and "overall" in results:
            print("\nSummary Metrics:")
            print(f"  Correctness:  {results['overall']['avg_correctness']:.3f}")
            print(f"  Completeness: {results['overall']['avg_completeness']:.3f}")
            print(f"  Overall:      {results['overall']['avg_overall']:.3f}")
    except Exception as e:
        print(f"Evaluation skipped or failed: {str(e)}")


def _normalize_pdf_name(pdf_name: str) -> str:
    return pdf_name if pdf_name.lower().endswith(".pdf") else f"{pdf_name}.pdf"


def run_eval_only(pdf_name: str) -> None:
    pdf_stem = pdf_name.rsplit(".pdf", 1)[0]
    pdf_dir = os.path.join(OUTPUT_BASE_DIR, pdf_stem)
    extraction_file = os.path.join(pdf_dir, "extraction_metadata.json")
    if not os.path.exists(extraction_file):
        print(f"Error: extraction_metadata.json not found at {extraction_file}")
        return

    print("\n" + "=" * 60)
    print(f"LANDING AI BASELINE (NEW) - EVAL ONLY - {pdf_name}")
    print("=" * 60 + "\n")

    print(f"Running evaluation for {pdf_name}...")
    try:
        results = run_evaluation(
            extraction_file=extraction_file,
            document_name=pdf_name,
            output_dir=pdf_dir,
            ground_truth_file=GROUND_TRUTH_FILE,
            definitions_file=DEFINITIONS_PATH,
        )
        if results and "overall" in results:
            print("\nSummary Metrics:")
            print(f"  Correctness:  {results['overall']['avg_correctness']:.3f}")
            print(f"  Completeness: {results['overall']['avg_completeness']:.3f}")
            print(f"  Overall:      {results['overall']['avg_overall']:.3f}")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")


def main(pdf_name: str, skip_eval: bool = False, run_eval_only_flag: bool = False):
    pdf_name = _normalize_pdf_name(pdf_name)
    if run_eval_only_flag:
        run_eval_only(pdf_name)
        return

    pdf_path = os.path.join(BASE_PDF_PATH, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file {pdf_name} not found at {pdf_path}")
        return

    print("\n" + "=" * 60)
    print(f"LANDING AI BASELINE (NEW) - {pdf_name}")
    print("=" * 60 + "\n")

    _ensure_api_key()
    client = _init_client()

    print("Step 1: Parse + Extract")
    extracted_data = process_pdf(client, pdf_path, pdf_name)

    if extracted_data:
        print("\nStep 2: Conversion & Evaluation")
        convert_and_evaluate(pdf_name, extracted_data, skip_eval=skip_eval)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Results: {OUTPUT_BASE_DIR}/{pdf_name.rsplit('.pdf', 1)[0]}/")
        print("=" * 60 + "\n")
    else:
        print(f"\nExtraction failed for {pdf_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract clinical data from PDF using LandingAI ADE Parse + Extract and evaluate."
    )
    parser.add_argument(
        "--pdf_name",
        type=str,
        required=True,
        help="Name of the PDF file to process (e.g., NCT00268476_Attard_STAMPEDE_Lancet'23.pdf)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step (extraction only)",
    )
    parser.add_argument(
        "--run-eval-only",
        action="store_true",
        help="Run evaluation only using existing extraction_metadata.json",
    )
    args = parser.parse_args()

    main(args.pdf_name, skip_eval=args.skip_eval, run_eval_only_flag=args.run_eval_only)
