from __future__ import annotations
"""baseline_file_search.py

File Search baseline supporting both OpenAI and Gemini:
1. **Extraction phase** – parallel label-group queries against a PDF via file upload APIs
2. **Structuring phase** – uses OutputStructurer to parse responses into clean JSON
3. **Evaluation phase** – uses evaluator_v2 for consistent evaluation

Usage example:
bash
python baseline_file_search.py \
  --pdf "dataset/NCT00104715_Gravis_GETUG_EU'15.pdf" \
  --provider openai \
  --model gpt-4o \
  --workers 10
"""

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from pathlib import Path
from openai import OpenAI
try:
    from google import genai
    GENAI_NEW = True
except ImportError:
    import google.generativeai as genai
    GENAI_NEW = False
from dotenv import load_dotenv

# Add repo root to path for imports
import sys
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))  # Add experiment-scripts to path

# Import baseline utilities
from baseline_utils import (
    load_definitions_with_metadata,
    convert_to_extraction_metadata,
    run_evaluation
)

# Import OutputStructurer
from src.LLMProvider.structurer import OutputStructurer

# Load environment variables
load_dotenv()

# ─── Pricing (USD per 1 K tokens) ──────────────────────────────────────────────
PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gemini-2.0-flash-001": {"input": 0.00015, "output": 0.0006},
}

# ─── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("File Search baseline with OpenAI/Gemini support")
parser.add_argument("--pdf", required=True, help="Path to the PDF file")
parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", 
                    help="Provider to use (openai or gemini)")
parser.add_argument("--model", default=None, 
                    help="Model name (defaults based on provider)")
parser.add_argument("--workers", type=int, default=10, 
                    help="Parallel label groups (default 10)")
parser.add_argument("--skip-eval", action="store_true",
                    help="Skip evaluation step")
parser.add_argument("--reliability-runs", type=int, default=1,
                    help="Number of extraction runs for reliability testing (default 1)")

args = parser.parse_args()

# Set default model based on provider
if args.model is None:
    args.model = "gpt-4.1" if args.provider == "openai" else "gemini-2.0-flash-001"

# ─── Helper Functions ──────────────────────────────────────────────────────────
def sanitize_filename(filename):
    return re.sub(r'[\/\\:*?"<>|]', '_', filename)
    
def get_stem_pathlib(path: str) -> str:
    return sanitize_filename(Path(path).stem)

# Setup output directory
pdf_stem = get_stem_pathlib(args.pdf)
if args.provider == "openai":
    dirname = f"experiment-scripts/baselines_openai_file_search/{args.model}/{pdf_stem}"
else:  # gemini
    dirname = f"experiment-scripts/baselines_gemini_file_search/{args.model}/{pdf_stem}"
os.makedirs(dirname, exist_ok=True)

print(f"\n{'='*60}")
print(f"FILE SEARCH BASELINE - {args.provider.upper()}")
print(f"PDF: {pdf_stem}")
print(f"Model: {args.model}")
if args.reliability_runs > 1:
    print(f"Reliability mode: {args.reliability_runs} runs")
print(f"{'='*60}\n")

# ─── Load Definitions ──────────────────────────────────────────────────────────
definitions_path = "src/table_definitions/Definitions_with_eval_category.csv"
definitions = load_definitions_with_metadata(definitions_path)

# Group by Label
label_groups = defaultdict(list)
for col_name, col_info in definitions.items():
    label_groups[col_info['label']].append({
        "column": col_name,
        "definition": col_info['definition']
    })
label_groups = OrderedDict(label_groups)

print(f"📋 Loaded {len(definitions)} columns in {len(label_groups)} label groups")

# ─── Provider Abstraction ──────────────────────────────────────────────────────
class PDFQueryProvider:
    """Abstraction for querying PDFs via OpenAI or Gemini APIs."""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.pdf_handle = None
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
            self.assistant = None
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not set")
            
            if GENAI_NEW:
                # Using new google.genai package
                self.client = genai.Client(api_key=api_key)
            else:
                # Using deprecated google.generativeai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
    
    def upload_pdf(self, pdf_path: str):
        """Upload PDF and return handle."""
        print(f"📤 Uploading PDF to {self.provider}...", end=" ")
        
        if self.provider == "openai":
            with open(pdf_path, "rb") as f:
                pdf_file = self.client.files.create(file=f, purpose="assistants")
            self.pdf_handle = {"file_id": pdf_file.id}
            
            # Create assistant with file_search
            self.assistant = self.client.beta.assistants.create(
                name="PDF Extractor (file_search)",
                instructions=(
                    "You are provided with column definitions and a clinical trial PDF. "
                    "Extract the required values for each column from the PDF. "
                    "If information is not present, return 'not found'. "
                    "For numeric values, extract exactly as shown. "
                    "Provide step-by-step reasoning and final answers in JSON format: "
                    "{\"Column Name\": \"Value\"}"
                ),
                model=self.model,
                tools=[{"type": "file_search"}],
            )
            print(f"✅ File ID: {pdf_file.id}, Assistant ID: {self.assistant.id}")
        
        elif self.provider == "gemini":
            if GENAI_NEW:
                # New API: Create PDF Part from bytes (no file upload needed)
                from google.genai import types as genai_types
                pdf_bytes = Path(pdf_path).read_bytes()
                pdf_part = genai_types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )
                self.pdf_handle = {"file_part": pdf_part}
                print(f"✅ PDF loaded as Part ({len(pdf_bytes)} bytes)")
            else:
                # Deprecated API
                uploaded_file = genai.upload_file(pdf_path)
                self.pdf_handle = {"file_part": uploaded_file}
                print(f"✅ File uploaded: {uploaded_file.name}")
    
    def query_pdf(self, prompt: str) -> Tuple[str, int, int]:
        """
        Query the PDF with a prompt.
        Returns: (response_text, input_tokens, output_tokens)
        """
        if self.provider == "openai":
            # Create thread with file_search
            thread = self.client.beta.threads.create(
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "attachments": [{
                        "file_id": self.pdf_handle["file_id"],
                        "tools": [{"type": "file_search"}]
                    }]
                }]
            )
            
            # Run assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            # Poll for completion
            while True:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run.status == "completed":
                    break
                if run.status in {"failed", "cancelled", "expired"}:
                    raise RuntimeError(f"Run failed: {run.status}")
                time.sleep(2)
            
            # Extract usage
            usage = run.usage
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            
            # Get response
            msgs = self.client.beta.threads.messages.list(thread_id=thread.id)
            response_text = next(
                (m.content[0].text.value for m in msgs.data if m.role == "assistant"),
                ""
            ).strip()
            
            return response_text, in_tok, out_tok
        
        elif self.provider == "gemini":
            file_part = self.pdf_handle["file_part"]
            
            if GENAI_NEW:
                # New API
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, file_part]
                )
                usage = getattr(response, 'usage_metadata', None)
                in_tok = getattr(usage, 'prompt_token_count', 0) if usage else 0
                out_tok = getattr(usage, 'candidates_token_count', 0) if usage else 0
                return response.text.strip(), in_tok, out_tok
            else:
                # Deprecated API
                model = genai.GenerativeModel(self.model)
                response = model.generate_content([prompt, file_part])
                in_tok = response.usage_metadata.prompt_token_count
                out_tok = response.usage_metadata.candidates_token_count
                return response.text.strip(), in_tok, out_tok
    
    def cleanup_pdf(self):
        """Clean up uploaded PDF if needed."""
        if self.provider == "openai" and self.pdf_handle:
            try:
                self.client.files.delete(self.pdf_handle["file_id"])
                print("🗑️  Cleaned up OpenAI file")
            except:
                pass

# ─── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(label: str, items: List[Dict[str, str]]) -> str:
    """Build prompt for a label group."""
    lines = [f"Extract values for the following columns (Label: {label}):\n"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {item['column']}: {item['definition']}\n"
            f"   If not present, return 'not found'."
        )
    lines.append("\n" + "="*60)
    lines.append("IMPORTANT: Output ONLY a valid JSON object in this exact format:")
    lines.append('{"Column Name 1": "Value 1", "Column Name 2": "Value 2", ...}')
    lines.append("\nDo NOT use markdown code blocks (no ```json or ```).")
    lines.append("Do NOT add any explanations or extra text.")
    lines.append("Output ONLY the raw JSON object, nothing else.")
    lines.append("="*60)
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
#  Extraction function (Phase 1 + 2 + 3: query, structure, convert)
# ──────────────────────────────────────────────────────────────────────────────
def extract_once(
    provider: PDFQueryProvider,
    label_groups: OrderedDict,
    definitions: Dict,
    structurer: OutputStructurer,
    output_dir: str,
    workers: int
) -> Tuple[Dict, int, int]:
    """
    Run one full extraction pass: query all label groups, structure, convert to extraction_metadata.
    Returns: (extraction_metadata, total_input_tokens, total_output_tokens)
    """
    # Token counters
    lock = threading.Lock()
    total_in = 0
    total_out = 0
    raw_responses = OrderedDict()

    def process_label(label: str, items: List[Dict[str, str]]) -> Tuple[str, str, int, int]:
        """Process a single label group."""
        prompt = build_prompt(label, items)
        response_text, in_tok, out_tok = provider.query_pdf(prompt)
        return label, response_text, in_tok, out_tok

    # Process all label groups in parallel
    max_workers = min(workers, len(label_groups)) or 1
    print(f"🚀 Processing {len(label_groups)} label groups with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(process_label, lbl, items): lbl 
            for lbl, items in label_groups.items()
        }
        
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                label, response, in_tok, out_tok = fut.result()
                raw_responses[label] = response
                with lock:
                    total_in += in_tok
                    total_out += out_tok
                print(f"✅ {label} (in={in_tok}, out={out_tok})")
            except Exception as exc:
                raw_responses[lbl] = f"ERROR: {exc}"
                print(f"❌ {lbl}: {exc}")

    # Save raw responses
    raw_file = os.path.join(output_dir, "raw_llm_responses.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, ensure_ascii=False, indent=2)
    print(f"💾 Raw responses saved to {raw_file}")

    # Phase 2: Structure responses
    print("\n🔧 Phase 2: Structuring responses with Qwen")
    extracted_dict = {}
    for label, raw_response in raw_responses.items():
        if raw_response.startswith("ERROR:"):
            for item in label_groups[label]:
                extracted_dict[item['column']] = "Extraction error"
            continue
        
        try:
            columns = [item['column'] for item in label_groups[label]]
            from pydantic import BaseModel, Field, create_model
            field_definitions = {
                col: (str, Field(default="", description=f"Value for {col}"))
                for col in columns
            }
            DynamicSchema = create_model(
                f'{label.replace(" ", "_").replace("-", "")}_Schema',
                **field_definitions
            )
            
            structured_response = structurer.structure(
                raw_response,
                DynamicSchema,
                return_dict=True
            )
            
            if structured_response.success:
                for col_name, value in structured_response.data.items():
                    extracted_dict[col_name] = value
            else:
                print(f"⚠️  Structuring failed for {label}: {structured_response.error}")
                for col in columns:
                    extracted_dict[col] = "Structuring error"
        except Exception as e:
            print(f"⚠️  Structuring exception for {label}: {e}")
            for col in columns:
                extracted_dict[col] = "Structuring error"

    print(f"✅ Extracted {len(extracted_dict)} column values")

    # Phase 3: Convert to extraction_metadata format
    print("\n📦 Phase 3: Converting to extraction_metadata format")
    extraction_metadata = convert_to_extraction_metadata(
        extracted_dict,
        definitions,
        source=f"file_search_{provider.provider}"
    )

    extraction_file = os.path.join(output_dir, "extraction_metadata.json")
    with open(extraction_file, 'w', encoding='utf-8') as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Extraction metadata saved to {extraction_file}")

    return extraction_metadata, total_in, total_out


def run_reliability_test(
    provider: PDFQueryProvider,
    label_groups: OrderedDict,
    definitions: Dict,
    structurer: OutputStructurer,
    base_dir: str,
    pdf_name: str,
    n_runs: int,
    workers: int,
    definitions_path: str
) -> Dict:
    """
    Run N extraction passes, evaluate each, and aggregate reliability metrics.
    Returns: aggregated reliability summary (mean, std, consistency per column).
    """
    print(f"\n{'='*60}")
    print(f"RELIABILITY TEST: {n_runs} runs")
    print(f"{'='*60}\n")
    
    all_eval_results = []  # List of evaluation_results.json dicts
    all_summaries = []     # List of summary_metrics.json dicts
    total_tokens = {"input": 0, "output": 0}
    
    for run_id in range(1, n_runs + 1):
        print(f"\n--- Run {run_id}/{n_runs} ---")
        run_dir = os.path.join(base_dir, f"reliability_run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Extract
        _, in_tok, out_tok = extract_once(
            provider, label_groups, definitions, structurer, run_dir, workers
        )
        total_tokens["input"] += in_tok
        total_tokens["output"] += out_tok
        
        # Evaluate
        extraction_file = os.path.join(run_dir, "extraction_metadata.json")
        try:
            eval_results = run_evaluation(
                extraction_file=extraction_file,
                document_name=pdf_name,
                output_dir=run_dir,
                ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
                definitions_file=definitions_path
            )
            
            # Load evaluation results and summary
            eval_res_path = Path(run_dir) / "evaluation" / "evaluation_results.json"
            summary_path = Path(run_dir) / "evaluation" / "summary_metrics.json"
            
            if eval_res_path.exists():
                with open(eval_res_path, "r", encoding="utf-8") as f:
                    all_eval_results.append(json.load(f))
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    all_summaries.append(json.load(f))
        except Exception as e:
            print(f"⚠️  Evaluation failed for run {run_id}: {e}")
    
    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATING RELIABILITY METRICS")
    print(f"{'='*60}\n")
    
    import numpy as np
    
    # Overall metrics
    overall_corr = [s["overall"]["avg_correctness"] for s in all_summaries if "overall" in s]
    overall_comp = [s["overall"]["avg_completeness"] for s in all_summaries if "overall" in s]
    overall_ov = [s["overall"]["avg_overall"] for s in all_summaries if "overall" in s]
    
    # Per-column metrics: collect scores across runs
    column_scores = defaultdict(lambda: {"correctness": [], "completeness": [], "overall": []})
    for eval_res in all_eval_results:
        for col, metrics in eval_res.get("columns", {}).items():
            column_scores[col]["correctness"].append(metrics.get("correctness", 0))
            column_scores[col]["completeness"].append(metrics.get("completeness", 0))
            column_scores[col]["overall"].append(metrics.get("overall", 0))
    
    # Aggregate per column
    per_column = {}
    for col, scores in column_scores.items():
        corr_arr = np.array(scores["correctness"])
        comp_arr = np.array(scores["completeness"])
        ov_arr = np.array(scores["overall"])
        
        # Consistency: fraction of runs with overall >= 0.99
        consistency = (ov_arr >= 0.99).sum() / len(ov_arr) if len(ov_arr) > 0 else 0
        
        per_column[col] = {
            "mean_correctness": float(np.mean(corr_arr)) if len(corr_arr) > 0 else 0,
            "std_correctness": float(np.std(corr_arr)) if len(corr_arr) > 0 else 0,
            "mean_completeness": float(np.mean(comp_arr)) if len(comp_arr) > 0 else 0,
            "std_completeness": float(np.std(comp_arr)) if len(comp_arr) > 0 else 0,
            "mean_overall": float(np.mean(ov_arr)) if len(ov_arr) > 0 else 0,
            "std_overall": float(np.std(ov_arr)) if len(ov_arr) > 0 else 0,
            "consistency": float(consistency),
            "n_runs": len(ov_arr)
        }
    
    # Build summary
    reliability_summary = {
        "n_runs": n_runs,
        "provider": provider.provider,
        "model": provider.model,
        "overall": {
            "mean_correctness": float(np.mean(overall_corr)) if overall_corr else 0,
            "std_correctness": float(np.std(overall_corr)) if overall_corr else 0,
            "mean_completeness": float(np.mean(overall_comp)) if overall_comp else 0,
            "std_completeness": float(np.std(overall_comp)) if overall_comp else 0,
            "mean_overall": float(np.mean(overall_ov)) if overall_ov else 0,
            "std_overall": float(np.std(overall_ov)) if overall_ov else 0,
        },
        "per_column": per_column,
        "total_tokens": total_tokens
    }
    
    # Write reliability summary
    summary_path = os.path.join(base_dir, "reliability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reliability_summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Reliability summary saved to {summary_path}")
    
    # Print quick stats
    low_consistency_cols = [col for col, m in per_column.items() if m["consistency"] < 0.8]
    print(f"\n📊 Reliability Stats:")
    print(f"   Mean overall: {reliability_summary['overall']['mean_overall']:.3f} ± {reliability_summary['overall']['std_overall']:.3f}")
    print(f"   Mean correctness: {reliability_summary['overall']['mean_correctness']:.3f} ± {reliability_summary['overall']['std_correctness']:.3f}")
    print(f"   Mean completeness: {reliability_summary['overall']['mean_completeness']:.3f} ± {reliability_summary['overall']['std_completeness']:.3f}")
    print(f"   Columns with consistency < 0.8: {len(low_consistency_cols)}/{len(per_column)}")
    if low_consistency_cols and len(low_consistency_cols) <= 10:
        print(f"   Low consistency columns: {low_consistency_cols[:10]}")
    
    return reliability_summary

# ──────────────────────────────────────────────────────────────────────────────
#  Main Execution
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
#  Main Execution
# ──────────────────────────────────────────────────────────────────────────────
print("\n📑 Phase 1: Extraction")

# Initialize provider
provider = PDFQueryProvider(args.provider, args.model)
provider.upload_pdf(args.pdf)

# Initialize structurer
from src.config.config import STRUCTURER_BASE_URL, STRUCTURER_MODEL
structurer = OutputStructurer(base_url=STRUCTURER_BASE_URL, model=STRUCTURER_MODEL)

# Decide single vs reliability mode
if args.reliability_runs > 1:
    # Reliability mode: N runs, evaluate each, aggregate
    reliability_summary = run_reliability_test(
        provider=provider,
        label_groups=label_groups,
        definitions=definitions,
        structurer=structurer,
        base_dir=dirname,
        pdf_name=Path(args.pdf).name,
        n_runs=args.reliability_runs,
        workers=args.workers,
        definitions_path=definitions_path
    )
    total_in = reliability_summary["total_tokens"]["input"]
    total_out = reliability_summary["total_tokens"]["output"]
else:
    # Single run mode
    extraction_metadata, total_in, total_out = extract_once(
        provider=provider,
        label_groups=label_groups,
        definitions=definitions,
        structurer=structurer,
        output_dir=dirname,
        workers=args.workers
    )
    
    # Phase 4: Evaluation
    if not args.skip_eval:
        print("\n📊 Phase 4: Evaluation")
        
        pdf_name = Path(args.pdf).name
        extraction_file = os.path.join(dirname, "extraction_metadata.json")
        try:
            results = run_evaluation(
                extraction_file=extraction_file,
                document_name=pdf_name,
                output_dir=dirname,
                ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
                definitions_file=definitions_path
            )
            
            # Print summary metrics
            if results and 'overall' in results:
                print(f"\n📈 Summary Metrics:")
                print(f"   Correctness:  {results['overall']['avg_correctness']:.3f}")
                print(f"   Completeness: {results['overall']['avg_completeness']:.3f}")
                print(f"   Overall:      {results['overall']['avg_overall']:.3f}")
        except Exception as e:
            print(f"⚠️  Evaluation skipped or failed: {str(e)}")

# Clean up
provider.cleanup_pdf()


# ─── Cost Summary ──────────────────────────────────────────────────────────────
pricing = PRICING.get(args.model, {"input": 0, "output": 0})
input_cost = (total_in / 1000) * pricing["input"]
output_cost = (total_out / 1000) * pricing["output"]
total_cost = input_cost + output_cost

# Save cost metrics
cost_metrics = {
    "provider": args.provider,
    "model": args.model,
    "tokens": {
        "input": total_in,
        "output": total_out,
        "total": total_in + total_out
    },
    "cost_usd": {
        "input": round(input_cost, 4),
        "output": round(output_cost, 4),
        "total": round(total_cost, 4)
    }
}

cost_file = os.path.join(dirname, "cost_metrics.json")
with open(cost_file, 'w') as f:
    json.dump(cost_metrics, f, indent=2)

print(f"\n💰 Cost Summary ({args.model}):")
print(f"   Input tokens:  {total_in:,} (${input_cost:.4f})")
print(f"   Output tokens: {total_out:,} (${output_cost:.4f})")
print(f"   Total cost:    ${total_cost:.4f}")

print(f"\n{'='*60}")
print(f"✅ PIPELINE COMPLETE!")
print(f"📁 Results: {dirname}/")
print(f"{'='*60}\n")
