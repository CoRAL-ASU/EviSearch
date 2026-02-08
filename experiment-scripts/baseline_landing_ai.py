import json
import requests
import os
from dotenv import load_dotenv

# Add repo root to path for imports
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))  # Add experiment-scripts to path

from baseline_utils import (
    load_definitions_with_metadata,
    convert_to_extraction_metadata,
    run_evaluation,
    build_schema_from_definitions
)

# Load environment variables
load_dotenv()

# API configuration
VA_API_KEY = os.getenv("LANDING_AI_API_KEY", "b2Q2ZTVyZTNkODNvc2lnc2gwbzluOmZSdE9EczQyRTNYZ2tMUUNpbXN4WmhNY1I2NllMbjQ5")
headers = {"Authorization": f"Basic {VA_API_KEY}"}
url = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"

# Paths configuration
base_pdf_path = "dataset"  # Directory containing PDFs
definitions_path = "src/table_definitions/Definitions_with_eval_category.csv"
output_base_dir = "experiment-scripts/baselines_landing_ai"
ground_truth_file = "dataset/Manual_Benchmark_GoldTable_cleaned.json"

# Create base output directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Load definitions and build schema
definitions = load_definitions_with_metadata(definitions_path)
schema = build_schema_from_definitions(definitions)

# Function to process a single PDF
def process_pdf(pdf_path, pdf_name):
    """
    Extract data from PDF using Landing AI API.
    
    Args:
        pdf_path: Full path to PDF file
        pdf_name: Name of PDF file
        
    Returns:
        Extracted data dict or None if failed
    """
    # Create subdirectory for this PDF
    pdf_stem = pdf_name.rsplit('.pdf', 1)[0]
    pdf_dir = os.path.join(output_base_dir, pdf_stem)
    os.makedirs(pdf_dir, exist_ok=True)
    
    raw_output_file = os.path.join(pdf_dir, "raw_llm_response.json")
    
    # Check if the extracted file already exists
    if os.path.exists(raw_output_file):
        print(f"Raw extraction for {pdf_name} already exists at {raw_output_file}. Skipping API call.")
        try:
            with open(raw_output_file, "r", encoding="utf-8") as f:
                extracted_info = json.load(f)
            return extracted_info
        except Exception as e:
            print(f"Error reading existing file for {pdf_name}: {str(e)}. Proceeding with API call.")
    
    print(f"📤 Sending {pdf_name} to Landing AI API...")
    files = [
        ("pdf", (pdf_name, open(pdf_path, "rb"), "application/pdf")),
    ]
    payload = {"fields_schema": json.dumps(schema)}
    
    try:
        response = requests.request("POST", url, headers=headers, files=files, data=payload)
        response_data = response.json()
        
        if "data" in response_data and "extracted_schema" in response_data["data"]:
            extracted_info = response_data["data"]["extracted_schema"]
            
            # Save raw LLM response
            with open(raw_output_file, "w", encoding="utf-8") as f:
                json.dump(extracted_info, f, indent=2)
            print(f"✅ Data extracted from {pdf_name} and saved to {raw_output_file}")
            return extracted_info
        else:
            print(f"❌ Failed to extract data from {pdf_name}. Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error processing {pdf_name}: {str(e)}")
        return None

# Function to convert Landing AI output to extraction_metadata format and evaluate
def convert_and_evaluate(pdf_name, extracted_data):
    """
    Convert Landing AI output to extraction_metadata.json and run evaluation.
    
    Args:
        pdf_name: Name of PDF file
        extracted_data: Raw extraction from Landing AI (flat dict with column: value pairs)
    """
    if not extracted_data:
        print(f"❌ No extracted data for {pdf_name}")
        return
    
    print(f"🔄 Converting {pdf_name} to extraction_metadata format...")
    
    # Landing AI now returns a flat dict of {column_name: value}
    # No need to unwrap from "clinical_trial_data"
    
    # Convert to extraction_metadata format
    extraction_metadata = convert_to_extraction_metadata(
        extracted_data,
        definitions,
        source="landing_ai"
    )
    
    # Create subdirectory for this PDF
    pdf_stem = pdf_name.rsplit('.pdf', 1)[0]
    pdf_dir = os.path.join(output_base_dir, pdf_stem)
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Save extraction_metadata.json
    extraction_file = os.path.join(pdf_dir, "extraction_metadata.json")
    with open(extraction_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Extraction metadata saved to {extraction_file}")
    
    # Run evaluation
    print(f"📊 Running evaluation for {pdf_name}...")
    try:
        results = run_evaluation(
            extraction_file=extraction_file,
            document_name=pdf_name,
            output_dir=pdf_dir,
            ground_truth_file=ground_truth_file,
            definitions_file=definitions_path
        )
        print(f"✅ Evaluation complete!")
        
        # Print summary metrics
        if results and 'overall' in results:
            print(f"\n📈 Summary Metrics:")
            print(f"   Correctness:  {results['overall']['avg_correctness']:.3f}")
            print(f"   Completeness: {results['overall']['avg_completeness']:.3f}")
            print(f"   Overall:      {results['overall']['avg_overall']:.3f}")
    except Exception as e:
        print(f"⚠️  Evaluation skipped or failed: {str(e)}")

# Main function to process a single PDF
def main(pdf_name):
    """
    Main pipeline: Extract from PDF and evaluate.
    
    Args:
        pdf_name: Name of PDF file to process
    """
    pdf_path = os.path.join(base_pdf_path, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file {pdf_name} not found at {pdf_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"LANDING AI BASELINE - {pdf_name}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract data via Landing AI API
    print("📋 Step 1: Extraction")
    extracted_data = process_pdf(pdf_path, pdf_name)
    
    if extracted_data:
        # Step 2: Convert and evaluate
        print("\n📋 Step 2: Conversion & Evaluation")
        convert_and_evaluate(pdf_name, extracted_data)
        
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE!")
        print(f"📁 Results: experiment-scripts/baselines_landing_ai/{pdf_name.rsplit('.pdf', 1)[0]}/")
        print(f"{'='*60}\n")
    else:
        print(f"\n❌ Extraction failed for {pdf_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract clinical data from PDF using Landing AI and evaluate with evaluator_v2."
    )
    parser.add_argument(
        "--pdf_name",
        type=str,
        required=True,
        help="Name of the PDF file to process (e.g., NCT00268476_Attard_STAMPEDE_Lancet'23.pdf)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step (extraction only)"
    )
    args = parser.parse_args()
    
    main(args.pdf_name)