"""
Shared utilities for baseline methods (Landing AI, File Search).
Provides functions to convert baseline outputs to extraction_metadata.json format
and run evaluator_v2 for consistent evaluation across all methods.
"""

import os
import csv
import json
from typing import Dict, Any
from pathlib import Path


def load_definitions_with_metadata(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load definitions with column_index and label info.
    
    Args:
        csv_path: Path to Definitions_with_eval_category.csv
        
    Returns:
        Dictionary mapping column names to metadata:
        {
            "Column Name": {
                "definition": "...",
                "label": "...",
                "eval_category": "...",
                "index": 0
            }
        }
    """
    definitions = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            definitions[row['Column Name'].strip()] = {
                'definition': row['Definition'].strip(),
                'label': row['Label'].strip(),
                'eval_category': row['eval_category'].strip(),
                'index': idx
            }
    return definitions


def convert_to_extraction_metadata(
    extracted_dict: Dict[str, Any],
    definitions: Dict[str, Dict[str, Any]],
    source: str = "baseline"
) -> Dict[str, Dict[str, Any]]:
    """
    Convert any baseline output to extraction_metadata.json format.
    
    Args:
        extracted_dict: {column_name: value} mapping
        definitions: Column definitions with Label info (from load_definitions_with_metadata)
        source: "landing_ai" | "file_search_openai" | "file_search_gemini"
    
    Returns:
        extraction_metadata format dict matching pipeline output structure
    """
    metadata = {}
    
    for col_name, value in extracted_dict.items():
        # Find column in definitions
        col_def = definitions.get(col_name, {})
        
        metadata[col_name] = {
            "value": value if value else "Not applicable",
            "evidence": "Not applicable",
            "chunk_id": f"{source}_extraction",
            "page": "Not applicable",
            "column_index": col_def.get("index", "Not applicable"),
            "group_name": col_def.get("label", "Not applicable"),
            "plan_found_in_pdf": "Not applicable",
            "plan_page": "Not applicable",
            "plan_source_type": "Not applicable",
            "plan_confidence": "Not applicable",
            "plan_extraction_plan": "Not applicable"
        }
    
    return metadata


def run_evaluation(
    extraction_file: str,
    document_name: str,
    output_dir: str,
    ground_truth_file: str = "dataset/Manual_Benchmark_GoldTable_cleaned.json",
    definitions_file: str = "src/table_definitions/Definitions_with_eval_category.csv"
) -> Dict[str, Any]:
    """
    Run evaluator_v2 on extraction results.
    
    Args:
        extraction_file: Path to extraction_metadata.json
        document_name: PDF name (e.g., "NCT00104715_Gravis_GETUG_EU'15.pdf")
        output_dir: Directory to save evaluation results
        ground_truth_file: Path to ground truth JSON
        definitions_file: Path to definitions CSV
        
    Returns:
        Evaluation results dictionary
    """
    from src.evaluation.evaluator_v2 import EvaluatorV2
    
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Ensure document name has .pdf extension
    if not document_name.endswith('.pdf'):
        document_name = f"{document_name}.pdf"
    
    evaluator = EvaluatorV2(
        extraction_file=extraction_file,
        ground_truth_file=ground_truth_file,
        definitions_file=definitions_file,
        document_name=document_name,
        output_dir=eval_dir
    )
    
    results = evaluator.run()
    
    print(f"✅ Evaluation complete. Results saved to {eval_dir}")
    return results


def build_schema_from_definitions(definitions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build Landing AI schema format from definitions.
    
    Schema must follow Landing AI requirements:
    - Top-level must be type "object"
    - Fields defined in "properties"
    - Each field should have type and description
    
    Args:
        definitions: Output from load_definitions_with_metadata
        
    Returns:
        Schema dict compatible with Landing AI API
    """
    properties = {}
    
    for col_name, col_info in definitions.items():
        properties[col_name] = {
            "type": "string",
            "description": col_info['definition']
        }
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": []  # Make all fields optional since not all may be present
    }
    
    return schema
