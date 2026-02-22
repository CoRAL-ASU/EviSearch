from __future__ import annotations

"""
baseline_landing_ai_w_gemini.py

Gemini baseline using Landing-AI parsed markdown as the document context.
For each label-group query, the script sends:
  - prompt for group columns + definitions
  - full parsed_markdown.md text for the target trial

Output format mirrors baseline_file_search_gemini_native.py:
  - raw_llm_responses.json
  - extraction_metadata.json
  - evaluation/*
  - cost_metrics.json
"""

import argparse
import json
import os
import statistics
import threading
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
    convert_to_extraction_metadata,
    load_definitions_with_metadata,
    run_evaluation,
)

load_dotenv()

# USD per 1K tokens
PRICING = {
    "gemini-2.0-flash-001": {"input": 0.00015, "output": 0.0025},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0025},
}

DEFINITIONS_PATH = "src/table_definitions/Definitions_with_eval_category.csv"
PARSED_MARKDOWN_ROOT = (
    repo_root / "experiment-scripts" / "baselines_landing_ai_new_results"
)
RESULTS_ROOT = (
    repo_root / "experiment-scripts" / "baseline_landing_ai_w_gemini" / "results"
)
GROUND_TRUTH_FILE = "dataset/Manual_Benchmark_GoldTable_cleaned.json"

REASONING_DESCRIPTION = (
    "Brief reasoning on where in the document you found the value and how you derived it; "
    "or 'not found' if not reported."
)
VALUE_DESCRIPTION = (
    "The extracted value exactly as in the document (e.g. number, percentage, text); "
    "use 'not found' if not reported."
)


def normalize_trial(trial: str) -> str:
    value = trial.strip()
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    if not value:
        raise ValueError("--trial cannot be empty")
    if "/" in value or "\\" in value:
        raise ValueError("--trial must be a trial id/folder name, not a path")
    return value


def build_json_schema_for_group(columns: List[str]) -> Dict[str, Any]:
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


class GeminiMarkdownProvider:
    def __init__(self, model: str):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        # 30s timeout per API call to avoid hanging
        self.client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=30_000),
        )

    def query_markdown_with_schema(
        self, prompt: str, markdown_text: str, json_schema: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=json_schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, markdown_text],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return (response.text or "").strip(), in_tok, out_tok


def safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if values else 0.0


def extract_once(
    provider: GeminiMarkdownProvider,
    markdown_text: str,
    label_groups: OrderedDict,
    definitions: Dict[str, Dict[str, Any]],
    output_dir: Path,
    workers: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, int]:
    lock = threading.Lock()
    total_in, total_out = 0, 0
    raw_parsed = OrderedDict()

    def process(label: str, items: List[Dict[str, str]]) -> Tuple[str, Any, int, int]:
        columns = [it["column"] for it in items]
        prompt = build_prompt(label, items)
        schema = build_json_schema_for_group(columns)
        text, in_tok, out_tok = provider.query_markdown_with_schema(
            prompt=prompt,
            markdown_text=markdown_text,
            json_schema=schema,
        )
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = {"_raw": text, "_error": "JSON decode failed"}
        return label, parsed, in_tok, out_tok

    max_workers = min(workers, len(label_groups)) or 1
    print(f"Processing {len(label_groups)} label groups with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(process, label, items): label
            for label, items in label_groups.items()
        }
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                label, parsed, in_tok, out_tok = fut.result()
                raw_parsed[label] = parsed
                with lock:
                    total_in += in_tok
                    total_out += out_tok
                print(f"  {label} (in={in_tok}, out={out_tok})")
            except Exception as e:
                raw_parsed[label] = {"_error": str(e)}
                print(f"  {label}: ERROR {e}")

    raw_file = output_dir / "raw_llm_responses.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_parsed, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved to {raw_file}")

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

    extraction_metadata = convert_to_extraction_metadata(
        extracted_dict=extracted_dict,
        definitions=definitions,
        source="landing_ai_w_gemini",
    )

    for parsed in raw_parsed.values():
        if "_error" in parsed:
            continue
        for col, cell in (parsed or {}).items():
            if col.startswith("_"):
                continue
            if isinstance(cell, dict) and col in extraction_metadata:
                reasoning = cell.get("reasoning")
                if reasoning is not None and str(reasoning).strip():
                    extraction_metadata[col]["evidence"] = str(reasoning).strip()

    extraction_file = output_dir / "extraction_metadata.json"
    with open(extraction_file, "w", encoding="utf-8") as f:
        json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
    print(f"Extraction metadata saved to {extraction_file}")

    return extraction_metadata, raw_parsed, total_in, total_out


def run_reliability_test(
    provider: GeminiMarkdownProvider,
    markdown_text: str,
    label_groups: OrderedDict,
    definitions: Dict[str, Dict[str, Any]],
    base_dir: Path,
    trial_name: str,
    n_runs: int,
    workers: int,
) -> Dict[str, Any]:
    all_eval_results = []
    all_summaries = []
    total_tokens = {"input": 0, "output": 0}

    for run_id in range(1, n_runs + 1):
        run_dir = base_dir / f"reliability_run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _, _, in_tok, out_tok = extract_once(
            provider=provider,
            markdown_text=markdown_text,
            label_groups=label_groups,
            definitions=definitions,
            output_dir=run_dir,
            workers=workers,
        )
        total_tokens["input"] += in_tok
        total_tokens["output"] += out_tok

        extraction_file = run_dir / "extraction_metadata.json"
        try:
            run_evaluation(
                extraction_file=str(extraction_file),
                document_name=f"{trial_name}.pdf",
                output_dir=str(run_dir),
                ground_truth_file=GROUND_TRUTH_FILE,
                definitions_file=DEFINITIONS_PATH,
            )
            eval_path = run_dir / "evaluation" / "evaluation_results.json"
            summary_path = run_dir / "evaluation" / "summary_metrics.json"
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    all_eval_results.append(json.load(f))
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    all_summaries.append(json.load(f))
        except Exception as e:
            print(f"Evaluation failed for run {run_id}: {e}")

    overall_corr = [s["overall"]["avg_correctness"] for s in all_summaries if "overall" in s]
    overall_comp = [s["overall"]["avg_completeness"] for s in all_summaries if "overall" in s]
    overall_ov = [s["overall"]["avg_overall"] for s in all_summaries if "overall" in s]

    column_scores = defaultdict(
        lambda: {"correctness": [], "completeness": [], "overall": []}
    )
    for ev in all_eval_results:
        for col, metrics in ev.get("columns", {}).items():
            column_scores[col]["correctness"].append(metrics.get("correctness", 0))
            column_scores[col]["completeness"].append(metrics.get("completeness", 0))
            column_scores[col]["overall"].append(metrics.get("overall", 0))

    per_column = {}
    for col, scores in column_scores.items():
        ov = scores["overall"]
        consistency = (sum(x >= 0.99 for x in ov) / len(ov)) if ov else 0.0
        per_column[col] = {
            "mean_correctness": safe_mean(scores["correctness"]),
            "std_correctness": safe_std(scores["correctness"]),
            "mean_completeness": safe_mean(scores["completeness"]),
            "std_completeness": safe_std(scores["completeness"]),
            "mean_overall": safe_mean(scores["overall"]),
            "std_overall": safe_std(scores["overall"]),
            "consistency": float(consistency),
            "n_runs": len(ov),
        }

    reliability_summary = {
        "n_runs": n_runs,
        "model": provider.model,
        "overall": {
            "mean_correctness": safe_mean(overall_corr),
            "std_correctness": safe_std(overall_corr),
            "mean_completeness": safe_mean(overall_comp),
            "std_completeness": safe_std(overall_comp),
            "mean_overall": safe_mean(overall_ov),
            "std_overall": safe_std(overall_ov),
        },
        "per_column": per_column,
        "total_tokens": total_tokens,
    }

    summary_path = base_dir / "reliability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reliability_summary, f, indent=2, ensure_ascii=False)
    print(f"Reliability summary saved to {summary_path}")
    return reliability_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        "Landing-AI markdown baseline with Gemini (native JSON)"
    )
    parser.add_argument(
        "--trial",
        required=True,
        help="Trial id/folder, e.g. NCT02799602_Hussain_ARASENS_JCO'23",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel label-group workers",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--run-eval-only",
        action="store_true",
        help="Skip extraction and run evaluation only on existing extraction_metadata.json",
    )
    parser.add_argument(
        "--reliability-runs",
        type=int,
        default=1,
        help="Number of extraction runs for reliability (default 1)",
    )
    args = parser.parse_args()

    if not GENAI_AVAILABLE:
        raise RuntimeError("google.genai is required. Install with: pip install google-genai")

    trial_name = normalize_trial(args.trial)
    parsed_md_path = PARSED_MARKDOWN_ROOT / trial_name / "parsed_markdown.md"
    if not parsed_md_path.exists():
        raise FileNotFoundError(f"Parsed markdown not found: {parsed_md_path}")

    output_dir = RESULTS_ROOT / args.model / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = load_definitions_with_metadata(DEFINITIONS_PATH)
    label_groups = defaultdict(list)
    for col_name, col_info in definitions.items():
        label_groups[col_info["label"]].append(
            {
                "column": col_name,
                "definition": col_info["definition"],
            }
        )
    label_groups = OrderedDict(label_groups)

    print(f"\n{'=' * 70}")
    print("BASELINE: Landing-AI parsed markdown + Gemini (native JSON)")
    print(f"Trial: {trial_name}")
    print(f"Model: {args.model}")
    print(f"Parsed markdown: {parsed_md_path}")
    print(f"Output: {output_dir}")
    print(f"Loaded {len(definitions)} columns in {len(label_groups)} label groups")
    if args.reliability_runs > 1:
        print(f"Reliability runs: {args.reliability_runs}")
    print(f"{'=' * 70}\n")

    if args.run_eval_only:
        extraction_file = output_dir / "extraction_metadata.json"
        if not extraction_file.exists():
            sys.exit(
                "run-eval-only: extraction_metadata.json not found at "
                f"{extraction_file}. Run extraction first."
            )
        print("Run-eval-only: skipping extraction, running evaluation...")
        try:
            results = run_evaluation(
                extraction_file=str(extraction_file),
                document_name=f"{trial_name}.pdf",
                output_dir=str(output_dir),
                ground_truth_file=GROUND_TRUTH_FILE,
                definitions_file=DEFINITIONS_PATH,
            )
            if results and "overall" in results:
                print(
                    "Summary: Correctness = {:.3f}, Completeness = {:.3f}, Overall = {:.3f}".format(
                        results["overall"]["avg_correctness"],
                        results["overall"]["avg_completeness"],
                        results["overall"]["avg_overall"],
                    )
                )
        except Exception as e:
            print(f"Evaluation failed: {e}")
            sys.exit(1)
        print(f"\nDone. Results: {output_dir}/")
        return

    markdown_text = parsed_md_path.read_text(encoding="utf-8")
    if not markdown_text.strip():
        raise ValueError(f"Parsed markdown is empty: {parsed_md_path}")

    provider = GeminiMarkdownProvider(args.model)

    if args.reliability_runs > 1:
        reliability_summary = run_reliability_test(
            provider=provider,
            markdown_text=markdown_text,
            label_groups=label_groups,
            definitions=definitions,
            base_dir=output_dir,
            trial_name=trial_name,
            n_runs=args.reliability_runs,
            workers=args.workers,
        )
        total_in = reliability_summary["total_tokens"]["input"]
        total_out = reliability_summary["total_tokens"]["output"]
    else:
        _, _, total_in, total_out = extract_once(
            provider=provider,
            markdown_text=markdown_text,
            label_groups=label_groups,
            definitions=definitions,
            output_dir=output_dir,
            workers=args.workers,
        )
        if not args.skip_eval:
            print("\nPhase 2: Evaluation")
            extraction_file = output_dir / "extraction_metadata.json"
            try:
                results = run_evaluation(
                    extraction_file=str(extraction_file),
                    document_name=f"{trial_name}.pdf",
                    output_dir=str(output_dir),
                    ground_truth_file=GROUND_TRUTH_FILE,
                    definitions_file=DEFINITIONS_PATH,
                )
                if results and "overall" in results:
                    print(
                        "\nSummary: Correctness = {:.3f}, Completeness = {:.3f}, Overall = {:.3f}".format(
                            results["overall"]["avg_correctness"],
                            results["overall"]["avg_completeness"],
                            results["overall"]["avg_overall"],
                        )
                    )
            except Exception as e:
                print(f"Evaluation failed: {e}")

    pricing = PRICING.get(args.model, {"input": 0, "output": 0})
    input_cost = (total_in / 1000) * pricing["input"]
    output_cost = (total_out / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    cost_metrics = {
        "provider": "gemini",
        "method": "baseline_landing_ai_w_gemini",
        "model": args.model,
        "trial": trial_name,
        "tokens": {"input": total_in, "output": total_out, "total": total_in + total_out},
        "cost_usd": {
            "input": round(input_cost, 4),
            "output": round(output_cost, 4),
            "total": round(total_cost, 4),
        },
    }
    cost_file = output_dir / "cost_metrics.json"
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(cost_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nCost ({args.model}): input={total_in}, output={total_out}, total=${total_cost:.4f}")
    print(f"Done. Results: {output_dir}/")


if __name__ == "__main__":
    main()
