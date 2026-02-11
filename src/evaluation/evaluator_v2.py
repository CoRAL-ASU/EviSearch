"""
New evaluation framework with category-aware scoring.
Supports: exact_match, numeric_tolerance, structured_text
Returns: correctness, completeness, overall scores per column

Uses table definitions for semantic/contextual matching:
- Single-number columns: GT "63.0 (IQR 57.0–68.2)" vs Pred "63" → correct (primary number match)
- Count-only columns (e.g. "No. of Deaths"): GT "4 deaths (2 neutropenia-related)" vs Pred "4" → correct
- N (%) columns: column requires both count and percentage; Pred "156" when GT "156 (81%)" → incomplete/wrong
"""
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.utils.costing import usage_to_cost_dict
from src.utils.logging_utils import setup_logger
from pydantic import BaseModel, Field
from typing import Literal

logger = setup_logger("evaluator_v2")


class ColumnEvaluationResult(BaseModel):
    """Schema for individual column evaluation result."""
    column: str = Field(..., description="The EXACT column name as provided, without any numbering prefix")
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
        self.column_format_hint = {}  # "n_only" | "n_and_pct" from table definitions
        self.results = {}
        self.llm_logs = {"gemini": [], "structurer": []}

    # ---------- Semantic / rule-based evaluation helpers (use table definitions) ----------

    def _requires_n_and_pct(self, column_name: str, definition: str) -> bool:
        """True if column name or definition requires both count (N) and percentage."""
        name_lower = (column_name or "").lower()
        def_lower = (definition or "").lower()
        if "n (%)" in name_lower or "n(%)" in name_lower:
            return True
        if "count and percentage" in def_lower or "number and percentage" in def_lower:
            return True
        if "include count and" in def_lower and "percentage" in def_lower:
            return True
        return False

    @staticmethod
    def _normalize_empty(val) -> str:
        """Normalize empty/not-reported to empty string for comparison."""
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() in ("", "nan", "not reported", "not present", "na", "n/a"):
            return ""
        return s

    @staticmethod
    def _extract_primary_number(text: str):
        """
        Extract the primary/leading number from text (e.g. median, count).
        Examples: "63.0 (IQR 57.0–68.2)" -> 63.0; "4 deaths (2 neutropenia-related)" -> 4;
                  "Median treatment duration was 41 months" -> 41; "655" -> 655.
        Returns float or int, or None if no number found.
        """
        if not text:
            return None
        s = str(text).strip()
        # Try to get the first number (integer or decimal)
        match = re.search(r"(\d+(?:\.\d+)?)", s)
        if match:
            num_str = match.group(1)
            try:
                return int(num_str) if "." not in num_str else float(num_str)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_all_numbers(text: str) -> List[float]:
        """Extract all numbers from text (ignoring the % symbol for numeric value)."""
        if not text:
            return []
        s = str(text)
        return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", s)]

    @staticmethod
    def _extract_n_and_pct(text: str) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Extract (n, percentage) pairs from strings like "156 (81%)", "439(67.5%)", "62(33)", "62 (33)".
        Accepts: "n (pct%)", "n(pct%)", "n (pct)", "n(pct)" - second number in parens is treated as pct.
        Returns list of (n, pct); pct is 0-100 or None.
        """
        if not text:
            return []
        s = str(text)
        # Standard: "62 (33%)" or "62(33%)"
        parts = re.findall(r"(\d+(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*%?\s*\)", s)
        if parts:
            return [(float(n), float(p)) for n, p in parts]
        # Two numbers where first could be n and second pct: "62(33)" or "128 (67)" (no %)
        two_nums = re.findall(r"(\d+(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*\)", s)
        if two_nums:
            return [(float(n), float(p)) for n, p in two_nums]
        # Single number: "156" or "81%"
        nums = re.findall(r"(\d+(?:\.\d+)?)\s*%?", s)
        if nums:
            return [(float(nums[0]), None)]
        return []

    def _rule_based_numeric_eval(
        self, column_name: str, gt_raw, pred_raw, category: str
    ) -> Optional[Dict]:
        """
        Rule-based evaluation for numeric_tolerance using column definitions.
        Returns dict with correctness, completeness, reason or None to defer to LLM.
        """
        if category != "numeric_tolerance":
            return None
        gt = self._normalize_empty(gt_raw)
        pred = self._normalize_empty(pred_raw)
        definition = self.column_definitions.get(column_name, "")
        requires_both = self._requires_n_and_pct(column_name, definition)

        # Both empty / not reported
        if not gt and not pred:
            return {"correctness": 1.0, "completeness": 1.0, "reason": "Both empty or not reported; rule-based match."}

        # One empty
        if not gt or not pred:
            return None  # Let LLM decide equivalence of "Not reported" vs ""

        # Column requires N and (%): check if GT has both n and %
        if requires_both:
            gt_pairs = self._extract_n_and_pct(gt)
            pred_pairs = self._extract_n_and_pct(pred)
            gt_has_pct = any(p is not None for _, p in gt_pairs)
            pred_has_pct = any(p is not None for _, p in pred_pairs)
            # If GT has percentage but prediction has only count (no %), mark incomplete
            if gt_has_pct and not pred_has_pct:
                gt_nums = self._extract_all_numbers(gt)
                pred_nums = self._extract_all_numbers(pred)
                if len(gt_nums) >= 1 and len(pred_nums) >= 1:
                    n_match = abs(gt_nums[0] - pred_nums[0]) < 0.2
                    # Column asks for n (%); missing percentage is wrong/incomplete
                    return {
                        "correctness": 0.5 if n_match else 0.0,
                        "completeness": 0.5 if n_match else 0.0,
                        "reason": "Column requires both count and percentage (N (%)). Ground truth has both but prediction has only count; scored as wrong/incomplete.",
                    }
            # If both have n and pct, compare and return Correct when they match
            if gt_has_pct and pred_has_pct and gt_pairs and pred_pairs:
                gn, gp = gt_pairs[0]
                pn, pp = pred_pairs[0]
                n_ok = abs(gn - pn) < 0.2
                pct_ok = gp is not None and pp is not None and abs(gp - pp) < 2.0
                if n_ok and pct_ok:
                    return {"correctness": 1.0, "completeness": 1.0, "reason": "N and percentage match (rule-based)."}
                if n_ok and not pct_ok:
                    return {"correctness": 0.5, "completeness": 0.5, "reason": "Count matches but percentage does not (rule-based)."}
            return None

        # Single-number / count-only column: primary number match
        gt_primary = self._extract_primary_number(gt)
        pred_primary = self._extract_primary_number(pred)
        if gt_primary is not None and pred_primary is not None:
            tol = 0.2 if (gt_primary >= 1 or pred_primary >= 1) else 0.1
            if abs(gt_primary - pred_primary) <= tol:
                return {
                    "correctness": 1.0,
                    "completeness": 1.0,
                    "reason": "Primary number match (contextual/semantic match); rule-based.",
                }
        return None

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
        
        # Build column format hints from definitions (for semantic evaluation)
        for col in self.column_definitions:
            self.column_format_hint[col] = (
                "n_and_pct" if self._requires_n_and_pct(col, self.column_definitions.get(col, "")) else "n_only"
            )

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

Compare Ground Truth (GT) vs Predicted (Pred) values using the column Definition and column name.

CRITICAL RULES (use table definitions):

1) Single-number / count-only columns (e.g. "Median Age", "Control Arm - N", "No. of Deaths - N"):
   - Extract the PRIMARY number only from GT and Pred. Ignore extra context in GT.
   - If the primary numbers match (within ±0.2), score correctness=1.0, completeness=1.0.
   - Examples that are CORRECT:
     - GT "63.0 (IQR 57.0–68.2)", Pred "63" → match (primary value 63)
     - GT "4 deaths (2 neutropenia-related, HVD subgroup)", Pred "4" → match (count 4)
     - GT "Median treatment duration was 41 months", Pred "41" → match

2) Columns whose name or definition requires BOTH count AND percentage (e.g. "N (%)", "include count and percentage"):
   - These columns expect format like "156 (81%)". If GT has both n and %, Pred must also include the percentage.
   - If GT is "156 (81%)" and Pred is only "156" (no percentage), score as INCOMPLETE: completeness ≤ 0.5, correctness ≤ 0.5 (missing required percentage).
   - If both GT and Pred have count and percentage, compare both within tolerance (±0.2 for n, ±2% for %).

3) General:
   - Extract numbers from formats like "250 (63.6%)", "median 12.5 months". Allow tolerance ±0.1 for absolute values, ±2% for percentages.
   - If both empty / "Not reported": correctness=1.0, completeness=1.0.
   - If one is empty: correctness=0.0, completeness=0.0.

Scoring (use only 0.0, 0.5, or 1.0):
- correctness: Fraction of predicted information that is correct (and in required format when column asks for n and %).
- completeness: Fraction of GT information captured (including percentage when column requires it).

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
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N"."""
        
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

        # Rule-based override using table definitions (semantic/contextual matching)
        for result in structured_results:
            col_name = result.get("column", "").rstrip(":").strip()
            gt_val = self.ground_truth_values.get(col_name, "")
            pred_val = self.predicted_values.get(col_name, "")
            rule_result = self._rule_based_numeric_eval(col_name, gt_val, pred_val, category)
            if rule_result is not None:
                result["correctness"] = rule_result["correctness"]
                result["completeness"] = rule_result["completeness"]
                result["reason"] = rule_result["reason"]
        
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
                        # Get the column name and strip trailing colon (added in prompt for readability)
                        col_name = result['column'].rstrip(':').strip()
                        overall = (result['correctness'] + result['completeness']) / 2
                        verdict_val = "Correct" if overall >= 1.0 else ("Partial" if overall > 0.0 else "Wrong")
                        self.results[col_name] = {
                            'correctness': result['correctness'],
                            'completeness': result['completeness'],
                            'overall': overall,
                            'verdict': verdict_val,
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
        """Aggregate results into summary metrics (verdict counts and by-category)."""
        if not self.results:
            return {}
        
        n_correct = sum(1 for r in self.results.values() if r.get('verdict') == 'Correct')
        n_partial = sum(1 for r in self.results.values() if r.get('verdict') == 'Partial')
        n_wrong = sum(1 for r in self.results.values() if r.get('verdict') == 'Wrong')
        total = len(self.results)
        avg_overall = (n_correct * 1.0 + n_partial * 0.5) / total if total else 0.0

        summary = {
            'summary': {
                'Document': self.document_name,
                'Total columns': total,
                'Correct': n_correct,
                'Partial': n_partial,
                'Wrong': n_wrong,
                'avg_overall': round(avg_overall, 4),
            },
            'by_category': []
        }

        for category in ['exact_match', 'numeric_tolerance', 'structured_text']:
            cat_results = [r for r in self.results.values() if r['category'] == category]
            if cat_results:
                c = sum(1 for r in cat_results if r.get('verdict') == 'Correct')
                p = sum(1 for r in cat_results if r.get('verdict') == 'Partial')
                w = sum(1 for r in cat_results if r.get('verdict') == 'Wrong')
                summary['by_category'].append({
                    'Category': category,
                    'Total': len(cat_results),
                    'Correct': c,
                    'Partial': p,
                    'Wrong': w,
                })

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
        if summary and 'summary' in summary:
            s = summary['summary']
            logger.info(f"Document: {s.get('Document', '')}")
            logger.info(f"Total columns: {s.get('Total columns', 0)} | Correct: {s.get('Correct', 0)} | Partial: {s.get('Partial', 0)} | Wrong: {s.get('Wrong', 0)}")
            logger.info(f"Avg overall: {s.get('avg_overall', 0):.3f}")
            logger.info(f"\nBy Category:")
            for row in summary.get('by_category', []):
                logger.info(f"  {row.get('Category', '')}: Total={row.get('Total', 0)} Correct={row.get('Correct', 0)} Partial={row.get('Partial', 0)} Wrong={row.get('Wrong', 0)}")
        else:
            logger.warning("No results to summarize")

    def get_usage(self) -> Dict:
        """Aggregate token usage and cost from LLM calls (Gemini eval + optional structurer fallback)."""
        total_in = 0
        total_out = 0
        for log in self.llm_logs.get("gemini", []):
            total_in += log.get("input_tokens", 0)
            total_out += log.get("output_tokens", 0)
        for log in self.llm_logs.get("structurer", []):
            total_in += log.get("input_tokens", 0)
            total_out += log.get("output_tokens", 0)
        provider = getattr(self.eval_provider, "provider", "gemini")
        model = getattr(self.eval_provider, "model", "gemini-2.0-flash-001")
        d = usage_to_cost_dict(provider, model, total_in, total_out)
        return d

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
    # Default: read/write under new_pipeline_outputs_10th_feb/results/
    import os
    _base = os.environ.get("EVAL_RESULTS_BASE", "new_pipeline_outputs_10th_feb")
    pdf_name = "NCT02799602_Hussain_ARASENS_JCO'23"
    evaluator = EvaluatorV2(
        extraction_file=f"{_base}/results/{pdf_name}/extractions/extraction_metadata.json",
        ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
        definitions_file="src/table_definitions/Definitions_with_eval_category.csv",
        document_name=pdf_name,
        output_dir=f"{_base}/results/{pdf_name}/evaluation",
    )
    evaluator.run()
