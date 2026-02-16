"""
New evaluation framework with category-aware scoring.
Supports: exact_match, numeric_tolerance, structured_text
Returns: correctness, completeness, overall scores per column
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
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

Compare Ground Truth (GT) vs Predicted (Pred) values using the column Definition to determine what is required.

CRITICAL RULES:

1) **What's required vs optional (check the Definition)**:
   - If definition says "include count AND percentage" or "N (%)": both N and % are REQUIRED.
   - If definition says "at X years" or "X-year rate": the timepoint is REQUIRED.
   - If definition says "median" without specifying CI/IQR: only the median number is required; CI, IQR, SD, range, p-values are OPTIONAL context.
   - Default: primary numbers are required; statistical context (CI, IQR, p-values) is optional unless definition explicitly asks for it.

2) **Tolerance**: ±0.1 for absolute values, ±2% relative for percentages.

3) **Correctness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all predicted numbers match GT numbers (within tolerance); extra optional stats (IQR, CI) in pred are fine
   - 0.5 = some predicted numbers match, some don't or contradict GT
   - 0.0 = no predicted numbers match GT, or predicted numbers contradict GT

4) **Completeness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all REQUIRED numbers from GT are present in pred (check definition to determine what's required)
   - 0.5 = some required numbers present, some missing
   - 0.0 = required numbers missing

5) **Empty handling**:
   - If both empty ("Not reported", "", etc.): correctness=1.0, completeness=1.0
   - If one is empty: correctness=0.0, completeness=0.0

EXAMPLES:
- GT="35.1 months (95% CI 29.9–43.6)", Pred="35.1 months", Def="median OS" → correctness=1.0 (35.1 matches), completeness=1.0 (CI is optional)
- GT="70% at 5 years", Pred="70%", Def="OS rate at 5 years" → correctness=1.0 (70% matches), completeness=0.5 (missing required timepoint)
- GT="250 (63.6%)", Pred="250", Def="include count and percentage" → correctness=1.0 (N matches), completeness=0.5 (missing required %)
- GT="250 (63.6%)", Pred="250 (64%)", Def="include count and percentage" → correctness=1.0 (both within tolerance), completeness=1.0 (both present)
- GT="High-volume: 92 (48%) Low-volume: 100 (52%)", Pred="92 (48%)", Def="by volume subgroup" → correctness=1.0 (high-volume matches), completeness=0.5 (missing low-volume)

Columns to evaluate:\n"""
        
        else:  # structured_text
            prompt = """You are evaluating clinical trial data extraction for structured text columns (treatments, regimens, endpoints, classifications).

Compare Ground Truth (GT) vs Predicted (Pred) for semantic information content using the column Definition to determine what is required.

CRITICAL RULES:

1) **Correctness = no contradiction** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all information in Pred matches or is compatible with GT; extra correct detail (e.g., drug mechanism, trial name, expanded abbreviations) is fine and does NOT lower the score
   - 0.5 = some predicted info correct, some contradicts GT (mixed: core fact right but extra detail wrong)
   - 0.0 = predicted info contradicts GT or core facts are wrong

2) **Completeness = required facts only** (use only 0.0, 0.5, or 1.0):
   - Use the Definition to identify what information is REQUIRED for this column
   - For "treatment regimen": drug name and combination partner (e.g., ADT) are typically required; dose and schedule are required if definition implies detail (e.g., "describe the regimen") but optional if definition just asks "what treatment"
   - For "endpoint": endpoint name is required; timepoints/thresholds are required only if definition specifies
   - Use medical judgment based on the definition to decide what's required vs optional context
   - 1.0 = all required facts from GT are present in pred
   - 0.5 = some required facts present, some missing
   - 0.0 = required facts missing
   - Do NOT penalize for missing optional context (mechanism, rationale, historical notes, expanded forms)

3) **General**:
   - Be lenient with abbreviations (e.g., "ADT" = "Androgen Deprivation Therapy")
   - Accept rephrasing if meaning is preserved
   - If both empty: correctness=1.0, completeness=1.0
   - If one empty: correctness=0.0, completeness=0.0

EXAMPLES:
- GT="ADT", Pred="ADT (LHRH agonist for testosterone suppression)", Def="control arm regimen" → correctness=1.0 (extra mechanism is fine, no contradiction), completeness=1.0 (core fact "ADT" present)
- GT="Docetaxel 75 mg/m² every 21 days + ADT", Pred="Docetaxel + ADT", Def="treatment regimen (describe)" → correctness=1.0 (no contradiction), completeness=0.5 (missing dose/schedule which are required by "describe")
- GT="ADT", Pred="ADT with docetaxel", Def="control arm" → correctness=0.5 (ADT correct but extra "docetaxel" contradicts), completeness=1.0 (ADT present)
- GT="Overall survival", Pred="Overall survival at 3 years", Def="primary endpoint" → correctness=1.0 (extra timepoint is fine), completeness=1.0 (endpoint name present)

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
    
    def _evaluate_batch_gemini_native_json(self, category: str, columns: List[str], max_retries: int = 3) -> List[Dict]:
        """Evaluate a batch using Gemini with native JSON schema (no structurer)."""
        try:
            from pydantic import ValidationError
            from google.genai import types as genai_types
        except ImportError:
            logger.warning("google.genai not available for native JSON evaluation")
            return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": "Native JSON evaluation unavailable (missing google.genai)"} for col in columns]
        client = getattr(self.eval_provider, "client", None)
        model = getattr(self.eval_provider, "model", "gemini-2.5-flash")
        if client is None:
            return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": "Gemini client not available"} for col in columns]
        prompt = self.build_prompt(category, columns)
        prompt += "\n\nRespond with a single JSON object containing a \"results\" array. Each element must have: \"column\" (exact column name), \"correctness\" (0.0, 0.5, or 1.0), \"completeness\" (0.0, 0.5, or 1.0), \"reason\" (brief explanation)."
        json_schema = EvaluationResults.model_json_schema()
        last_response_text = None
        for attempt in range(1, max_retries + 1):
            try:
                config = genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=32000,
                    response_mime_type="application/json",
                    response_schema=json_schema,
                )
                api_response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                last_response_text = (api_response.text or "").strip()
                if not last_response_text:
                    raise ValueError("Empty response from Gemini")
                validated = EvaluationResults.model_validate(json.loads(last_response_text))
                usage = getattr(api_response, "usage_metadata", None)
                in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                self.llm_logs["gemini"].append({
                    "timestamp": datetime.now().isoformat(),
                    "category": category,
                    "columns": columns,
                    "prompt": prompt,
                    "response": last_response_text,
                    "success": True,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "native_json": True,
                })
                return validated.model_dump()["results"]
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                logger.warning(f"Gemini native JSON attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    self.llm_logs["gemini"].append({
                        "timestamp": datetime.now().isoformat(),
                        "category": category,
                        "columns": columns,
                        "prompt": prompt,
                        "response": last_response_text,
                        "success": False,
                        "error": str(e),
                        "native_json": True,
                    })
                    return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"Native JSON evaluation failed: {e}"} for col in columns]
            except Exception as e:
                logger.warning(f"Gemini native JSON attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    self.llm_logs["gemini"].append({
                        "timestamp": datetime.now().isoformat(),
                        "category": category,
                        "columns": columns,
                        "prompt": prompt,
                        "response": last_response_text,
                        "success": False,
                        "error": str(e),
                        "native_json": True,
                    })
                    return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"Evaluation failed: {e}"} for col in columns]
        return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": "Evaluation failed after retries"} for col in columns]
    
    def _structure_with_gemini_schema(self, response: str, columns: List[str], max_retries: int = 3) -> Optional[List[Dict]]:
        """Structure evaluation response using Gemini with native JSON schema (no free-form parsing)."""
        if getattr(self.eval_provider, "provider", None) != "gemini":
            return None
        try:
            from pydantic import ValidationError
            from google.genai import types as genai_types
        except ImportError:
            logger.warning("google.genai not available for structurer fallback")
            return None
        client = getattr(self.eval_provider, "client", None)
        model = getattr(self.eval_provider, "model", "gemini-2.0-flash-001")
        if client is None:
            return None
        prompt = f"""You previously evaluated clinical trial data extraction accuracy. Your evaluation is shown below.

Your task: Structure your evaluation into the required JSON format.

Expected columns: {', '.join(columns)}

Required JSON structure:
{{
  "results": [
    {{
      "column": "exact column name",
      "correctness": 0.0 or 0.5 or 1.0,
      "completeness": 0.0 or 0.5 or 1.0,
      "reason": "full explanation preserving all your reasoning"
    }}
  ]
}}

Instructions:
- Extract correctness and completeness scores for each column
- If scores are explicitly stated (e.g., "correctness: 1.0"), use them directly
- If scores are implied (e.g., "values match exactly" → correctness=1.0, completeness=1.0; "no match" → 0.0)
- Preserve your FULL reasoning in the "reason" field - do not summarize or shorten it
- Handle markdown tables, prose, bullet points, or partial JSON in the response
- Use exact column names from the expected columns list above (without number prefixes)

Your previous evaluation response:
{response}"""
        json_schema = EvaluationResults.model_json_schema()
        for attempt in range(1, max_retries + 1):
            try:
                config = genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                    response_schema=json_schema,
                )
                api_response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                raw_json = (api_response.text or "").strip()
                validated = EvaluationResults.model_validate(json.loads(raw_json))
                usage = getattr(api_response, "usage_metadata", None)
                in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                self.llm_logs["structurer"].append({
                    "timestamp": datetime.now().isoformat(),
                    "fallback": "gemini",
                    "success": True,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "input": prompt,
                    "output": validated.model_dump()
                })
                return validated.model_dump()["results"]
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries:
                    logger.warning(f"Gemini schema structuring failed after {max_retries} attempts: {e}")
                    self.llm_logs["structurer"].append({
                        "timestamp": datetime.now().isoformat(),
                        "fallback": "gemini",
                        "success": False,
                        "error": str(e),
                        "input": prompt,
                        "output": None
                    })
                    return None
            except Exception as e:
                logger.warning(f"Gemini structurer attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    self.llm_logs["structurer"].append({
                        "timestamp": datetime.now().isoformat(),
                        "fallback": "gemini",
                        "success": False,
                        "error": str(e),
                        "input": prompt,
                        "output": None
                    })
                    return None
        return None
    
    def structure_response(self, response: str, columns: List[str], max_retries: int = 5) -> List[Dict]:
        """Structure free-form response into JSON using Qwen structurer, fallback to Gemini with native JSON schema."""
        
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
            'error': result.error if not result.success else None,
            'input': response,
            'output': result.data if result.success else None
        })
        
        if result.success:
            logger.info(f"Successfully structured with Qwen after {result.attempts} attempt(s)")
            return result.data['results']
        
        # Fallback to Gemini with native JSON schema
        logger.warning(f"Qwen structuring failed: {result.error}")
        logger.warning("Falling back to Gemini for structuring (native JSON schema)...")
        try:
            fallback_results = self._structure_with_gemini_schema(response, columns, max_retries=3)
            if fallback_results is not None:
                logger.info("Successfully structured with Gemini fallback (native JSON schema)")
                return fallback_results
        except Exception as e:
            logger.error(f"Gemini fallback also failed: {e}")
        
        # Both structurers failed - preserve original evaluation as reason
        logger.error("All structuring attempts failed. Preserving original evaluation text.")
        truncated_response = response[:1000] if len(response) > 1000 else response
        return [{
            "column": col,
            "correctness": 0.0,
            "completeness": 0.0,
            "reason": f"Structuring failed. Original evaluation: {truncated_response}"
        } for col in columns]
    
    def evaluate_batch(self, category: str, columns: List[str]) -> List[Dict]:
        """Evaluate a batch of columns. Uses Gemini native JSON when available; otherwise free-form + structurer."""
        logger.info(f"Evaluating batch of {len(columns)} columns in category '{category}'")
        
        # Use Gemini native JSON (no structurer) when provider is Gemini and client is available
        if getattr(self.eval_provider, "provider", None) == "gemini" and getattr(self.eval_provider, "client", None) is not None:
            logger.info("Using Gemini native JSON for evaluation (no structurer).")
            return self._evaluate_batch_gemini_native_json(category, columns)
        
        # Fallback: free-form Gemini then Qwen/Gemini structurer (e.g. non-Gemini or client missing)
        prompt = self.build_prompt(category, columns)
        logger.info("Calling Gemini for evaluation (free-form)...")
        llm_response = self.eval_provider.generate(
            prompt=prompt,
            system_prompt="You are an expert clinical trial data evaluator with deep knowledge of medical terminology, statistical measures, and data extraction accuracy.",
            temperature=0.0,
            max_tokens=4000
        )
        
        if not llm_response.success:
            logger.error(f"Gemini call failed: {llm_response.error}")
            return [{"column": col, "correctness": 0.0, "completeness": 0.0, "reason": f"LLM call failed: {llm_response.error}"} for col in columns]
        
        response = llm_response.text
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
        return self.structure_response(response, columns)
    
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
