"""
New evaluation framework with category-aware scoring.
Supports: exact_match, numeric_tolerance, structured_text
Returns: correctness, completeness, overall scores per column
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.utils.logging_utils import setup_logger
from pydantic import BaseModel

logger = setup_logger("evaluator_v2")


class ColumnEvaluationResult(BaseModel):
    """Schema for individual column evaluation result."""
    column: str
    correctness: float
    completeness: float
    reason: str


class EvaluationResults(BaseModel):
    """Schema for batch evaluation results."""
    results: list[ColumnEvaluationResult]


class EvaluatorV2:
    def __init__(
        self,
        extraction_file: str,
        ground_truth_file: str,
        definitions_file: str,
        document_name: str,
        output_dir: str
    ):
        self.extraction_file = Path(extraction_file)
        self.ground_truth_file = Path(ground_truth_file)
        self.definitions_file = Path(definitions_file)
        self.document_name = document_name if document_name.endswith('.pdf') else f"{document_name}.pdf"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM providers
        self.eval_provider = LLMProvider(provider="gemini", model="gemini-2.0-flash-001")
        self.structurer = OutputStructurer(
            base_url="http://localhost:8001/v1",
            model="Qwen/Qwen3-8B"
        )
        
        # Data storage
        self.predicted_values = {}
        self.ground_truth_values = {}
        self.column_categories = {}
        self.column_definitions = {}
        self.column_labels = {}
        self.results = {}
        self.llm_logs = {"gemini": [], "structurer": []}
        
    def load_data(self):
        """Load all input data."""
        logger.info("Loading extraction data...")
        with open(self.extraction_file, 'r') as f:
            extraction_data = json.load(f)
            # Extract values from nested structure
            for col_name, col_data in extraction_data.items():
                if isinstance(col_data, dict) and 'value' in col_data:
                    # Convert None to empty string for consistency
                    self.predicted_values[col_name] = col_data['value'] if col_data['value'] is not None else ""
                else:
                    self.predicted_values[col_name] = col_data if col_data is not None else ""
        
        logger.info("Loading ground truth...")
        with open(self.ground_truth_file, 'r') as f:
            gt_data = json.load(f)
            # Find matching document
            for row in gt_data['data']:
                doc_name_cell = row.get('Document Name', {})
                if doc_name_cell.get('value') == self.document_name:
                    # Extract values from all columns
                    for col_name, cell_data in row.items():
                        self.ground_truth_values[col_name] = cell_data.get('value', '')
                    break
        
        if not self.ground_truth_values:
            raise ValueError(f"No ground truth found for document: {self.document_name}")
        
        logger.info("Loading column categories, labels, and definitions...")
        df = pd.read_csv(self.definitions_file)
        self.column_categories = dict(zip(df['Column Name'], df['eval_category']))
        self.column_labels = dict(zip(df['Column Name'], df['Label']))
        
        # Load definitions from the original definitions file
        original_defs_path = Path(self.definitions_file).parent / "Definitions_open_ended.csv"
        if original_defs_path.exists():
            defs_df = pd.read_csv(original_defs_path)
            self.column_definitions = dict(zip(defs_df['Column Name'], defs_df['Definition']))
        else:
            # Fallback: use Definition column from eval_category file if it exists
            if 'Definition' in df.columns:
                self.column_definitions = dict(zip(df['Column Name'], df['Definition']))
            else:
                self.column_definitions = {col: "" for col in df['Column Name']}
        
        logger.info(f"Loaded {len(self.predicted_values)} predicted values")
        logger.info(f"Loaded {len(self.ground_truth_values)} ground truth values")
        logger.info(f"Loaded {len(self.column_categories)} column categories")
        logger.info(f"Loaded {len(self.column_definitions)} column definitions")
    
    def group_columns_by_category(self) -> Dict[str, List[str]]:
        """Group columns by their evaluation category."""
        grouped = {
            'exact_match': [],
            'numeric_tolerance': [],
            'structured_text': []
        }
        
        # Only evaluate columns that exist in both predicted and GT
        common_columns = set(self.predicted_values.keys()) & set(self.ground_truth_values.keys())
        
        for col in common_columns:
            category = self.column_categories.get(col)
            if category in grouped:
                grouped[category].append(col)
        
        logger.info(f"Grouped columns: exact_match={len(grouped['exact_match'])}, "
                   f"numeric_tolerance={len(grouped['numeric_tolerance'])}, "
                   f"structured_text={len(grouped['structured_text'])}")
        
        return grouped
    
    def create_batches_by_label(self, columns: List[str], batch_size: int = 6) -> List[List[str]]:
        """
        Create batches keeping columns with the same label together.
        For numeric_tolerance columns, group by label first, then batch within each group.
        """
        # Group columns by their label
        label_groups = defaultdict(list)
        for col in columns:
            label = self.column_labels.get(col, col)
            label_groups[label].append(col)
        
        # Create batches
        batches = []
        for label, cols in sorted(label_groups.items()):
            # Split this label group into batches
            for i in range(0, len(cols), batch_size):
                batch = cols[i:i + batch_size]
                batches.append(batch)
        
        return batches
    
    def create_batches(self, columns: List[str], batch_size: int = 6) -> List[List[str]]:
        """Split columns into batches (simple sequential split)."""
        batches = []
        for i in range(0, len(columns), batch_size):
            batches.append(columns[i:i + batch_size])
        return batches
    
    def build_prompt(self, category: str, columns: List[str]) -> str:
        """Build category-specific evaluation prompt."""
        
        if category == 'exact_match':
            prompt = """You are evaluating clinical trial data extraction for exact match columns (identifiers, categorical values, binary fields).

Compare Ground Truth (GT) vs Predicted (Pred) values for exact/semantic equivalence.

Rules:
- Case-insensitive comparison
- Accept synonyms: "Yes"/"Y", "No"/"N", "Full Pub"/"Full Publication", "Phase 3"/"3.0", etc.
- Empty values: "Not reported", "not present", "", "NaN" are all equivalent to empty
- For identifiers (NCT, Trial Name): must match exactly (ignoring case/whitespace)
- Consider the column definition to understand acceptable variations

For each column, evaluate:
- If values are equivalent (by rules above): correctness=1.0, completeness=1.0
- If values differ: correctness=0.0, completeness=0.0

Columns to evaluate:\n"""
        
        elif category == 'numeric_tolerance':
            prompt = """You are evaluating clinical trial data extraction for numeric columns.

Compare Ground Truth (GT) vs Predicted (Pred) values by extracting and comparing ALL numbers.

Rules:
- Extract all numbers from both GT and Pred (handle formats like "250 (63.6%)", "median 12.5 months", etc.)
- Allow tolerance: ±0.1 for absolute values, ±2% relative for percentages
- Handle multi-value cells (e.g., "High-volume: 250 (63.6%) Low-volume: 150 (36.4%)")
- If both are empty: correctness=1.0, completeness=1.0
- If one is empty: correctness=0.0, completeness=0.0
- Consider the column definition to understand what numbers to extract

Scoring (use only 0.0, 0.5, or 1.0):
- correctness: What fraction of numbers in Pred match numbers in GT (within tolerance)?
  - 1.0 = all predicted numbers match GT
  - 0.5 = some predicted numbers match, some don't
  - 0.0 = no predicted numbers match GT or predicted has extra wrong numbers
  
- completeness: What fraction of numbers in GT are captured in Pred?
  - 1.0 = all GT numbers are in prediction
  - 0.5 = some GT numbers are captured
  - 0.0 = no GT numbers are captured

Columns to evaluate:\n"""
        
        else:  # structured_text
            prompt = """You are evaluating clinical trial data extraction for structured text columns (treatments, regimens, endpoints, classifications).

Compare Ground Truth (GT) vs Predicted (Pred) for semantic information content.

Rules:
- Be lenient with abbreviations (e.g., "ADT" = "Androgen Deprivation Therapy")
- Accept rephrasing if meaning is preserved
- Focus on key medical facts (drug names, doses, schedules, endpoints)
- If both empty: correctness=1.0, completeness=1.0
- If one empty: correctness=0.0, completeness=0.0
- Consider the column definition to understand expected information

Scoring (use only 0.0, 0.5, or 1.0):
- correctness: What fraction of information in Pred is factually correct according to GT?
  - 1.0 = all predicted info is in GT
  - 0.5 = some predicted info is correct, some is wrong/extra
  - 0.0 = predicted info is wrong or not in GT
  
- completeness: What fraction of information in GT is captured in Pred?
  - 1.0 = prediction captures all GT info
  - 0.5 = prediction captures some GT info
  - 0.0 = prediction misses all GT info

Columns to evaluate:\n"""
        
        # Add column comparisons with definitions
        for i, col in enumerate(columns, 1):
            gt_val = self.ground_truth_values.get(col, "")
            pred_val = self.predicted_values.get(col, "")
            definition = self.column_definitions.get(col, "")
            
            prompt += f"{i}. {col}:\n"
            prompt += f"   Definition: {definition}\n"
            prompt += f"   GT: {gt_val}\n"
            prompt += f"   Pred: {pred_val}\n\n"
        
        prompt += """\nFor each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise."""
        
        return prompt
    
    def structure_response(self, response: str, columns: List[str], max_retries: int = 5) -> List[Dict]:
        """Structure free-form response into JSON using Qwen structurer, fallback to Gemini."""
        
        # Try Qwen structurer first
        logger.info(f"Structuring with Qwen...")
        
        result = self.structurer.structure(
            text=response,
            schema=EvaluationResults,
            max_retries=max_retries,
            return_dict=True
        )
        
        # Log the attempt
        self.llm_logs['structurer'].append({
            'timestamp': datetime.now().isoformat(),
            'success': result.success,
            'attempts': result.attempts,
            'error': result.error if not result.success else None
        })
        
        if result.success:
            logger.info(f"Successfully structured with Qwen after {result.attempts} attempt(s)")
            return result.data['results']
        
        # Fallback to Gemini
        logger.warning(f"Qwen structuring failed: {result.error}")
        logger.warning("Falling back to Gemini for structuring...")
        
        try:
            structure_prompt = f"""Extract the evaluation results from the following response and format as JSON array.

Expected columns: {', '.join(columns)}

Response to parse:
{response}

Return a JSON array with this exact structure for each column:
[
  {{
    "column": "column name",
    "correctness": 0.0 or 0.5 or 1.0,
    "completeness": 0.0 or 0.5 or 1.0,
    "reason": "brief explanation"
  }}
]

Return ONLY the JSON array, no other text."""
            
            llm_response = self.eval_provider.generate(
                prompt=structure_prompt,
                system_prompt="You are a JSON extraction assistant. Extract structured evaluation data and return valid JSON only.",
                temperature=0.0,
                max_tokens=2000
            )
            
            if not llm_response.success:
                logger.error(f"Gemini fallback failed: {llm_response.error}")
                self.llm_logs['structurer'].append({
                    'timestamp': datetime.now().isoformat(),
                    'fallback': 'gemini',
                    'success': False,
                    'error': llm_response.error
                })
                return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"Structuring failed: {llm_response.error}"} for col in columns]
            
            parsed = json.loads(llm_response.text)
            
            self.llm_logs['structurer'].append({
                'timestamp': datetime.now().isoformat(),
                'fallback': 'gemini',
                'success': True,
                'input_tokens': llm_response.input_tokens,
                'output_tokens': llm_response.output_tokens
            })
            
            logger.info("Successfully structured with Gemini fallback")
            return parsed
            
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")
            # Return default results
            return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"Parsing failed: {e}"} for col in columns]
    
    def evaluate_batch(self, category: str, columns: List[str]) -> List[Dict]:
        """Evaluate a batch of columns."""
        logger.info(f"Evaluating batch of {len(columns)} columns in category '{category}'")
        
        # Build prompt
        prompt = self.build_prompt(category, columns)
        
        # Call Gemini for evaluation
        logger.info("Calling Gemini for evaluation...")
        llm_response = self.eval_provider.generate(
            prompt=prompt,
            system_prompt="You are an expert clinical trial data evaluator with deep knowledge of medical terminology, statistical measures, and data extraction accuracy.",
            temperature=0.0,
            max_tokens=4000
        )
        
        if not llm_response.success:
            logger.error(f"Gemini call failed: {llm_response.error}")
            # Return default results on failure
            return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"LLM call failed: {llm_response.error}"} for col in columns]
        
        response = llm_response.text
        
        # Log Gemini call
        self.llm_logs['gemini'].append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'columns': columns,
            'prompt': prompt,
            'response': response,
            'success': llm_response.success,
            'input_tokens': llm_response.input_tokens,
            'output_tokens': llm_response.output_tokens
        })
        
        # Structure response
        structured_results = self.structure_response(response, columns)
        
        return structured_results
    
    def evaluate_all(self, max_workers=5):
        """Run evaluation on all columns with parallel processing."""
        logger.info("Starting evaluation...")
        
        # Group columns by category
        grouped = self.group_columns_by_category()
        
        # Collect all batch tasks
        all_tasks = []
        
        for category, columns in grouped.items():
            if not columns:
                logger.info(f"No columns in category '{category}', skipping")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating category: {category} ({len(columns)} columns)")
            logger.info(f"{'='*60}")
            
            # Determine batch size and batching strategy
            if category == 'exact_match':
                batch_size = len(columns)  # Single batch
                batches = self.create_batches(columns, batch_size)
            elif category == 'numeric_tolerance':
                batch_size = 6
                # Use label-based batching to keep related columns together
                batches = self.create_batches_by_label(columns, batch_size)
            else:  # structured_text
                batch_size = 6
                batches = self.create_batches(columns, batch_size)
            
            logger.info(f"Created {len(batches)} batches")
            
            # Add tasks
            for i, batch in enumerate(batches, 1):
                all_tasks.append({
                    'category': category,
                    'batch': batch,
                    'batch_num': i,
                    'total_batches': len(batches)
                })
        
        # Process batches in parallel
        logger.info(f"\n🚀 Processing {len(all_tasks)} batches with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.evaluate_batch, task['category'], task['batch']): task
                for task in all_tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    batch_results = future.result()
                    
                    logger.info(f"✓ [{completed}/{len(all_tasks)}] Completed {task['category']} batch {task['batch_num']}/{task['total_batches']}")
                    
                    # Store results with full context (thread-safe as we're collecting sequentially)
                    for result in batch_results:
                        # Strip trailing colon from column name (added in prompt formatting)
                        col_name = result['column'].rstrip(':').strip()
                        self.results[col_name] = {
                            'correctness': result['correctness'],
                            'completeness': result['completeness'],
                            'overall': (result['correctness'] + result['completeness']) / 2,
                            'reason': result['reason'],
                            'category': task['category'],
                            'definition': self.column_definitions.get(col_name, ""),
                            'ground_truth': self.ground_truth_values.get(col_name, ""),
                            'predicted': self.predicted_values.get(col_name, "")
                        }
                
                except Exception as e:
                    logger.error(f"✗ [{completed}/{len(all_tasks)}] Failed {task['category']} batch {task['batch_num']}: {e}")
        
        logger.info(f"\n✅ Evaluation complete: {len(self.results)} columns evaluated")
    
    def aggregate_metrics(self) -> Dict:
        """Aggregate results into summary metrics."""
        if not self.results:
            return {}
        
        # Overall metrics
        all_correctness = [r['correctness'] for r in self.results.values()]
        all_completeness = [r['completeness'] for r in self.results.values()]
        all_overall = [r['overall'] for r in self.results.values()]
        
        summary = {
            'overall': {
                'avg_correctness': sum(all_correctness) / len(all_correctness),
                'avg_completeness': sum(all_completeness) / len(all_completeness),
                'avg_overall': sum(all_overall) / len(all_overall),
                'total_columns': len(self.results)
            },
            'by_category': {}
        }
        
        # Per-category metrics
        for category in ['exact_match', 'numeric_tolerance', 'structured_text']:
            cat_results = {k: v for k, v in self.results.items() if v['category'] == category}
            if cat_results:
                cat_correctness = [r['correctness'] for r in cat_results.values()]
                cat_completeness = [r['completeness'] for r in cat_results.values()]
                cat_overall = [r['overall'] for r in cat_results.values()]
                
                summary['by_category'][category] = {
                    'avg_correctness': sum(cat_correctness) / len(cat_correctness),
                    'avg_completeness': sum(cat_completeness) / len(cat_completeness),
                    'avg_overall': sum(cat_overall) / len(cat_overall),
                    'column_count': len(cat_results)
                }
        
        return summary
    
    def save_results(self):
        """Save all outputs."""
        logger.info("Saving results...")
        
        # Create llm_logs directory
        logs_dir = self.output_dir / "llm_logs"
        logs_dir.mkdir(exist_ok=True)
        
        # 1. evaluation_results.json
        results_output = {
            'document_name': self.document_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'columns': self.results
        }
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results_output, f, indent=2)
        logger.info(f"✅ Saved evaluation_results.json")
        
        # 2. summary_metrics.json
        summary = self.aggregate_metrics()
        with open(self.output_dir / 'summary_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Saved summary_metrics.json")
        
        # 3. LLM logs
        with open(logs_dir / 'gemini_calls.jsonl', 'w') as f:
            for log in self.llm_logs['gemini']:
                f.write(json.dumps(log) + '\n')
        logger.info(f"✅ Saved gemini_calls.jsonl ({len(self.llm_logs['gemini'])} calls)")
        
        with open(logs_dir / 'structurer_calls.jsonl', 'w') as f:
            for log in self.llm_logs['structurer']:
                f.write(json.dumps(log) + '\n')
        logger.info(f"✅ Saved structurer_calls.jsonl ({len(self.llm_logs['structurer'])} calls)")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*60}")
        if summary and 'overall' in summary:
            logger.info(f"Overall Correctness: {summary['overall']['avg_correctness']:.3f}")
            logger.info(f"Overall Completeness: {summary['overall']['avg_completeness']:.3f}")
            logger.info(f"Overall Score: {summary['overall']['avg_overall']:.3f}")
            logger.info(f"\nBy Category:")
            for cat, metrics in summary.get('by_category', {}).items():
                logger.info(f"  {cat}: {metrics['avg_overall']:.3f} ({metrics['column_count']} cols)")
        else:
            logger.warning("No results to summarize")
    
    def run(self):
        """Main execution pipeline."""
        try:
            self.load_data()
            self.evaluate_all()
            self.save_results()
            return self.results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Test with NCT00104715_Gravis_GETUG_EU'15
    pdf_name = "NCT02799602_Hussain_ARASENS_JCO'23"
    
    evaluator = EvaluatorV2(
        extraction_file=f"new_pipeline_outputs/results/{pdf_name}/extractions/extraction_metadata.json",
        ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
        definitions_file="src/table_definitions/Definitions_with_eval_category.csv",
        document_name=pdf_name,
        output_dir=f"new_pipeline_outputs/results/{pdf_name}/evaluation"
    )
    
    evaluator.run()
