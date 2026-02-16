from __future__ import annotations
"""baseline_file_search_gemini.py

Gemini-only file-search baseline that uses native JSON schema (no Qwen structurer).
For each column the model returns:
  - value: the extracted value (or 'not found')
  - reasoning: short reasoning on where it found the value and how it derived it, or 'not found'

Usage:
  python baseline_file_search_gemini.py --pdf "dataset/NCT00104715_Gravis_GETUG_EU'15.pdf" --model gemini-2.0-flash-001 --workers 10
"""

import argparse
import json
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Repo root and experiment-scripts for imports
repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_utils import (
    load_definitions_with_metadata,
    convert_to_extraction_metadata,
    run_evaluation,
)

load_dotenv()

# ─── Pricing (USD per 1K tokens) ─────────────────────────────────────────────
PRICING = {
    "gemini-2.0-flash-001": {"input": 0.00015, "output": 0.0025},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0025},
}

# ─── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("File Search baseline (Gemini only, native JSON)")
parser.add_argument("--pdf", required=True, help="Path to the PDF file")
parser.add_argument("--model", default="gemini-2.0-flash-001", help="Gemini model name")
parser.add_argument("--workers", type=int, default=10, help="Parallel label groups")
parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
parser.add_argument("--run-eval-only", action="store_true", help="Skip extraction; use existing extraction_metadata.json and run evaluation only")
parser.add_argument("--reliability-runs", type=int, default=1, help="Number of runs for reliability (default 1)")
args = parser.parse_args()

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/\\:*?"<>|]', '_', filename)

pdf_stem = sanitize_filename(Path(args.pdf).stem)
dirname = f"experiment-scripts/baselines_file_search_results/gemini_native/{args.model}/{pdf_stem}"
os.makedirs(dirname, exist_ok=True)

print(f"\n{'='*60}")
print("FILE SEARCH BASELINE (GEMINI – native JSON with reasoning)")
print(f"PDF: {pdf_stem}")
print(f"Model: {args.model}")
if args.reliability_runs > 1:
    print(f"Reliability: {args.reliability_runs} runs")
print(f"{'='*60}\n")

if not GENAI_AVAILABLE:
    raise RuntimeError("google.genai is required. Install with: pip install google-genai")

# ─── Definitions ───────────────────────────────────────────────────────────
definitions_path = "src/table_definitions/Definitions_with_eval_category.csv"
definitions = load_definitions_with_metadata(definitions_path)
label_groups = defaultdict(list)
for col_name, col_info in definitions.items():
    label_groups[col_info["label"]].append({
        "column": col_name,
        "definition": col_info["definition"],
    })
label_groups = OrderedDict(label_groups)
print(f"Loaded {len(definitions)} columns in {len(label_groups)} label groups")

# ─── Run-only-eval path: skip extraction, run evaluation on existing extraction_metadata.json ──
if args.run_eval_only:
    extraction_file = os.path.join(dirname, "extraction_metadata.json")
    if not os.path.isfile(extraction_file):
        sys.exit(f"run-eval-only: extraction_metadata.json not found at {extraction_file}. Run extraction first.")
    print("\nRun-eval-only: skipping extraction, running evaluation on existing extraction_metadata.json")
    pdf_name = Path(args.pdf).name
    try:
        results = run_evaluation(
            extraction_file=extraction_file,
            document_name=pdf_name,
            output_dir=dirname,
            ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
            definitions_file=definitions_path,
        )
        if results and "overall" in results:
            print("\nSummary: Correctness = {:.3f}, Completeness = {:.3f}, Overall = {:.3f}".format(
                results["overall"]["avg_correctness"],
                results["overall"]["avg_completeness"],
                results["overall"]["avg_overall"],
            ))
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)
    print(f"\nDone. Results: {dirname}/")
    sys.exit(0)

# ─── Schema description for reasoning (user-requested) ───────────────────────
REASONING_DESCRIPTION = (
    "Brief reasoning on where in the document you found the value and how you derived it; "
    "or 'not found' if not reported."
)
VALUE_DESCRIPTION = (
    "The extracted value exactly as in the document (e.g. number, percentage, text); "
    "use 'not found' if not reported."
)

def build_json_schema_for_group(columns: List[str]) -> Dict[str, Any]:
    """Build a JSON schema for one label group: one property per column, each with value + reasoning."""
    properties = {}
    for col in columns:
        properties[col] = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": VALUE_DESCRIPTION,
                },
                "reasoning": {
                    "type": "string",
                    "description": REASONING_DESCRIPTION,
                },
            },
            "required": ["value", "reasoning"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(columns),
    }

def build_prompt(label: str, items: List[Dict[str, str]]) -> str:
    """Build prompt for a label group (same style as original baseline)."""
    lines = [f"Extract values for the following columns (Label: {label}):\n"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {item['column']}: {item['definition']}\n"
            "   If not present, use value: 'not found' and reasoning: 'not found'."
        )
    lines.append("\n" + "=" * 60)
    lines.append(
        "Output a single JSON object. For each column provide "
        "'value' (the extracted value or 'not found') and "
        "'reasoning' (where you found it and how you derived it, or 'not found')."
    )
    lines.append("=" * 60)
    return "\n".join(lines)


# ─── Gemini provider (PDF as Part, JSON schema) ──────────────────────────────
class GeminiPDFProvider:
    def __init__(self, model: str):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self._pdf_part = None

    def upload_pdf(self, pdf_path: str) -> None:
        pdf_bytes = Path(pdf_path).read_bytes()
        self._pdf_part = genai_types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
        print(f"PDF loaded as Part ({len(pdf_bytes)} bytes)")

    def query_pdf_with_schema(
        self, prompt: str, json_schema: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        """Call Gemini with native JSON schema. Returns (response_text, input_tokens, output_tokens)."""
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=json_schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, self._pdf_part],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return (response.text or "").strip(), in_tok, out_tok

    def cleanup_pdf(self) -> None:
        self._pdf_part = None


# ─── Extraction (one pass) ───────────────────────────────────────────────────
def extract_once(
    provider: GeminiPDFProvider,
    label_groups: OrderedDict,
    definitions: Dict,
    output_dir: str,
    workers: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]], int, int]:
    """
    One extraction pass: query each label group with JSON schema, merge results.
    Returns: (extraction_metadata, raw_group_responses, total_in_tok, total_out_tok).
    raw_group_responses[label] = parsed JSON (with value + reasoning per column).
    """
    lock = threading.Lock()
    total_in, total_out = 0, 0
    raw_parsed = OrderedDict()  # label -> full parsed JSON (with value/reasoning)

    def process(label: str, items: List[Dict[str, str]]) -> Tuple[str, Any, int, int]:
        columns = [it["column"] for it in items]
        prompt = build_prompt(label, items)
        schema = build_json_schema_for_group(columns)
        text, in_tok, out_tok = provider.query_pdf_with_schema(prompt, schema)
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = {"_raw": text, "_error": "JSON decode failed"}
        return label, parsed, in_tok, out_tok

    max_workers = min(workers, len(label_groups)) or 1
    print(f"Processing {len(label_groups)} label groups with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(process, lbl, items): lbl
            for lbl, items in label_groups.items()
        }
        for fut in as_completed(futures):
            lbl = futures[fut]
            try:
                label, parsed, in_tok, out_tok = fut.result()
                raw_parsed[label] = parsed
                with lock:
                    total_in += in_tok
                    total_out += out_tok
                print(f"  {label} (in={in_tok}, out={out_tok})")
            except Exception as e:
                raw_parsed[lbl] = {"_error": str(e)}
                print(f"  {lbl}: ERROR {e}")

    # Save raw responses (with reasoning)
    raw_file = os.path.join(output_dir, "raw_llm_responses.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_parsed, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved to {raw_file}")

    # Build extracted_dict: one value per column (for evaluation)
    extracted_dict = {}
    for label, parsed in raw_parsed.items():
        items = label_groups[label]
        columns = [it["column"] for it in items]
        if "_error" in parsed:
            for col in columns:
                extracted_dict[col] = "Extraction error"
            continue
        for col in columns:
            cell = parsed.get(col)
            if isinstance(cell, dict):
                val = cell.get("value")
                extracted_dict[col] = val if val is not None and str(val).strip() else "not found"
            else:
                extracted_dict[col] = "not found"

    # Convert to extraction_metadata; optionally fill evidence from reasoning
    extraction_metadata = convert_to_extraction_metadata(
        extracted_dict,
        definitions,
        source="file_search_gemini",
    )
    for label, parsed in raw_parsed.items():
        if "_error" in parsed:
            continue
        for col, cell in (parsed or {}).items():
            if col.startswith("_"):
                continue
            if isinstance(cell, dict) and col in extraction_metadata:
                reasoning = cell.get("reasoning")
                if reasoning is not None and str(reasoning).strip():
                    extraction_metadata[col]["evidence"] = str(reasoning).strip()

    extraction_file = os.path.join(output_dir, "extraction_metadata.json")
    with open(extraction_file, "w", encoding="utf-8") as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
    print(f"Extraction metadata saved to {extraction_file}")

    return extraction_metadata, raw_parsed, total_in, total_out


# ─── Reliability test (N runs + aggregate) ──────────────────────────────────
def run_reliability_test(
    provider: GeminiPDFProvider,
    label_groups: OrderedDict,
    definitions: Dict,
    base_dir: str,
    pdf_name: str,
    n_runs: int,
    workers: int,
) -> Dict:
    all_eval_results = []
    all_summaries = []
    total_tokens = {"input": 0, "output": 0}

    for run_id in range(1, n_runs + 1):
        run_dir = os.path.join(base_dir, f"reliability_run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        _, _, in_tok, out_tok = extract_once(
            provider, label_groups, definitions, run_dir, workers
        )
        total_tokens["input"] += in_tok
        total_tokens["output"] += out_tok
        extraction_file = os.path.join(run_dir, "extraction_metadata.json")
        try:
            run_evaluation(
                extraction_file=extraction_file,
                document_name=pdf_name,
                output_dir=run_dir,
                ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
                definitions_file=definitions_path,
            )
            eval_path = Path(run_dir) / "evaluation" / "evaluation_results.json"
            summary_path = Path(run_dir) / "evaluation" / "summary_metrics.json"
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    all_eval_results.append(json.load(f))
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    all_summaries.append(json.load(f))
        except Exception as e:
            print(f"Evaluation failed for run {run_id}: {e}")

    import numpy as np
    overall_corr = [s["overall"]["avg_correctness"] for s in all_summaries if "overall" in s]
    overall_comp = [s["overall"]["avg_completeness"] for s in all_summaries if "overall" in s]
    overall_ov = [s["overall"]["avg_overall"] for s in all_summaries if "overall" in s]
    column_scores = defaultdict(lambda: {"correctness": [], "completeness": [], "overall": []})
    for ev in all_eval_results:
        for col, m in ev.get("columns", {}).items():
            column_scores[col]["correctness"].append(m.get("correctness", 0))
            column_scores[col]["completeness"].append(m.get("completeness", 0))
            column_scores[col]["overall"].append(m.get("overall", 0))
    per_column = {}
    for col, scores in column_scores.items():
        co = np.array(scores["correctness"])
        cm = np.array(scores["completeness"])
        ov = np.array(scores["overall"])
        consistency = (ov >= 0.99).sum() / len(ov) if len(ov) else 0
        per_column[col] = {
            "mean_correctness": float(np.mean(co)) if len(co) else 0,
            "std_correctness": float(np.std(co)) if len(co) else 0,
            "mean_completeness": float(np.mean(cm)) if len(cm) else 0,
            "std_completeness": float(np.std(cm)) if len(cm) else 0,
            "mean_overall": float(np.mean(ov)) if len(ov) else 0,
            "std_overall": float(np.std(ov)) if len(ov) else 0,
            "consistency": float(consistency),
            "n_runs": len(ov),
        }
    reliability_summary = {
        "n_runs": n_runs,
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
        "total_tokens": total_tokens,
    }
    summary_path = os.path.join(base_dir, "reliability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reliability_summary, f, indent=2, ensure_ascii=False)
    print(f"Reliability summary saved to {summary_path}")
    return reliability_summary


# ─── Main ───────────────────────────────────────────────────────────────────
print("\nPhase 1: Extraction (Gemini native JSON with value + reasoning)")

provider = GeminiPDFProvider(args.model)
provider.upload_pdf(args.pdf)

if args.reliability_runs > 1:
    reliability_summary = run_reliability_test(
        provider=provider,
        label_groups=label_groups,
        definitions=definitions,
        base_dir=dirname,
        pdf_name=Path(args.pdf).name,
        n_runs=args.reliability_runs,
        workers=args.workers,
    )
    total_in = reliability_summary["total_tokens"]["input"]
    total_out = reliability_summary["total_tokens"]["output"]
else:
    extraction_metadata, raw_parsed, total_in, total_out = extract_once(
        provider=provider,
        label_groups=label_groups,
        definitions=definitions,
        output_dir=dirname,
        workers=args.workers,
    )
    if not args.skip_eval:
        print("\nPhase 2: Evaluation")
        pdf_name = Path(args.pdf).name
        extraction_file = os.path.join(dirname, "extraction_metadata.json")
        try:
            results = run_evaluation(
                extraction_file=extraction_file,
                document_name=pdf_name,
                output_dir=dirname,
                ground_truth_file="dataset/Manual_Benchmark_GoldTable_cleaned.json",
                definitions_file=definitions_path,
            )
            if results and "overall" in results:
                print("\nSummary: Correctness = {:.3f}, Completeness = {:.3f}, Overall = {:.3f}".format(
                    results["overall"]["avg_correctness"],
                    results["overall"]["avg_completeness"],
                    results["overall"]["avg_overall"],
                ))
        except Exception as e:
            print(f"Evaluation failed: {e}")

provider.cleanup_pdf()

# Cost
pricing = PRICING.get(args.model, {"input": 0, "output": 0})
input_cost = (total_in / 1000) * pricing["input"]
output_cost = (total_out / 1000) * pricing["output"]
total_cost = input_cost + output_cost
cost_metrics = {
    "provider": "gemini",
    "model": args.model,
    "tokens": {"input": total_in, "output": total_out, "total": total_in + total_out},
    "cost_usd": {
        "input": round(input_cost, 4),
        "output": round(output_cost, 4),
        "total": round(total_cost, 4),
    },
}
cost_file = os.path.join(dirname, "cost_metrics.json")
with open(cost_file, "w") as f:
    json.dump(cost_metrics, f, indent=2)
print(f"\nCost ({args.model}): input={total_in}, output={total_out}, total=${total_cost:.4f}")
print(f"\nDone. Results: {dirname}/")
