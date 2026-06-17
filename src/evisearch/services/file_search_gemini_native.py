from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from src.evisearch.services.markdown_baseline import (
    GEMINI_PRICING,
    build_json_schema_for_group,
    build_label_groups,
    convert_to_extraction_metadata,
    load_definitions_with_metadata,
    run_evaluation,
    safe_mean,
    safe_std,
)

try:
    from src.LLMProvider.google_genai_client import (
        create_vertex_genai_client,
        get_genai_types,
    )

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFINITIONS_PATH = "src/table_definitions/Definitions_with_eval_category.csv"
GROUND_TRUTH_FILE = "dataset/Manual_Benchmark_GoldTable_cleaned.json"
RESULTS_ROOT = PROJECT_ROOT / "experiment-scripts" / "baselines_file_search_results" / "gemini_native"


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[\/\\:*?\"<>|]", "_", filename)


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


class GeminiPDFProvider:
    def __init__(self, model: str):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google.genai is required. Install with: pip install google-genai")
        self.model = model
        self.types = get_genai_types()
        self.client = create_vertex_genai_client(timeout_ms=30_000)
        self._pdf_part = None

    def upload_pdf(self, pdf_path: str) -> None:
        pdf_bytes = Path(pdf_path).read_bytes()
        self._pdf_part = self.types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
        print(f"PDF loaded as Part ({len(pdf_bytes)} bytes)")

    def query_pdf_with_schema(
        self, prompt: str, json_schema: Dict[str, Any]
    ) -> Tuple[str, int, int]:
        config = self.types.GenerateContentConfig(
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


def extract_once(
    provider: GeminiPDFProvider,
    label_groups: OrderedDict,
    definitions: Dict[str, Dict[str, Any]],
    output_dir: Path,
    workers: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]], int, int]:
    lock = threading.Lock()
    total_in, total_out = 0, 0
    raw_parsed = OrderedDict()

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
        extracted_dict,
        definitions,
        source="file_search_gemini",
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
    provider: GeminiPDFProvider,
    label_groups: OrderedDict,
    definitions: Dict[str, Dict[str, Any]],
    base_dir: Path,
    pdf_name: str,
    n_runs: int,
    workers: int,
    definitions_path: str = DEFINITIONS_PATH,
) -> Dict[str, Any]:
    all_eval_results = []
    all_summaries = []
    total_tokens = {"input": 0, "output": 0}

    for run_id in range(1, n_runs + 1):
        run_dir = base_dir / f"reliability_run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _, _, in_tok, out_tok = extract_once(
            provider, label_groups, definitions, run_dir, workers
        )
        total_tokens["input"] += in_tok
        total_tokens["output"] += out_tok
        extraction_file = run_dir / "extraction_metadata.json"
        try:
            run_evaluation(
                extraction_file=str(extraction_file),
                document_name=pdf_name,
                output_dir=str(run_dir),
                ground_truth_file=GROUND_TRUTH_FILE,
                definitions_file=definitions_path,
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
    column_scores = defaultdict(lambda: {"correctness": [], "completeness": [], "overall": []})
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


def run_file_search_gemini_native_baseline(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("File Search baseline (Gemini only, native JSON)")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="gemini-2.0-flash-001", help="Gemini model name")
    parser.add_argument("--workers", type=int, default=10, help="Parallel label groups")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--run-eval-only", action="store_true", help="Skip extraction; use existing extraction_metadata.json and run evaluation only")
    parser.add_argument("--reliability-runs", type=int, default=1, help="Number of runs for reliability (default 1)")
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf)
    pdf_stem = sanitize_filename(pdf_path.stem)
    output_dir = RESULTS_ROOT / args.model / pdf_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("FILE SEARCH BASELINE (GEMINI - native JSON with reasoning)")
    print(f"PDF: {pdf_stem}")
    print(f"Model: {args.model}")
    if args.reliability_runs > 1:
        print(f"Reliability: {args.reliability_runs} runs")
    print(f"{'=' * 60}\n")

    definitions = load_definitions_with_metadata(DEFINITIONS_PATH)
    label_groups = build_label_groups(definitions)
    print(f"Loaded {len(definitions)} columns in {len(label_groups)} label groups")

    if args.run_eval_only:
        extraction_file = output_dir / "extraction_metadata.json"
        if not extraction_file.is_file():
            sys.exit(f"run-eval-only: extraction_metadata.json not found at {extraction_file}. Run extraction first.")
        print("\nRun-eval-only: skipping extraction, running evaluation on existing extraction_metadata.json")
        try:
            results = run_evaluation(
                extraction_file=str(extraction_file),
                document_name=pdf_path.name,
                output_dir=str(output_dir),
                ground_truth_file=GROUND_TRUTH_FILE,
                definitions_file=DEFINITIONS_PATH,
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
        print(f"\nDone. Results: {output_dir}/")
        return

    print("\nPhase 1: Extraction (Gemini native JSON with value + reasoning)")
    provider = GeminiPDFProvider(args.model)
    provider.upload_pdf(args.pdf)

    if args.reliability_runs > 1:
        reliability_summary = run_reliability_test(
            provider=provider,
            label_groups=label_groups,
            definitions=definitions,
            base_dir=output_dir,
            pdf_name=pdf_path.name,
            n_runs=args.reliability_runs,
            workers=args.workers,
        )
        total_in = reliability_summary["total_tokens"]["input"]
        total_out = reliability_summary["total_tokens"]["output"]
    else:
        _, _, total_in, total_out = extract_once(
            provider=provider,
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
                    document_name=pdf_path.name,
                    output_dir=str(output_dir),
                    ground_truth_file=GROUND_TRUTH_FILE,
                    definitions_file=DEFINITIONS_PATH,
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

    pricing = GEMINI_PRICING.get(args.model, {"input": 0, "output": 0})
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
    cost_file = output_dir / "cost_metrics.json"
    with open(cost_file, "w", encoding="utf-8") as f:
        json.dump(cost_metrics, f, indent=2)
    print(f"\nCost ({args.model}): input={total_in}, output={total_out}, total=${total_cost:.4f}")
    print(f"\nDone. Results: {output_dir}/")
