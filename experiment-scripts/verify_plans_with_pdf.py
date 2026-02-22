#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from typing_extensions import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.planning.plan_generator import safe_stem
from src.table_definitions.definitions import load_definitions


# Easy-to-edit globals (per your request)
VERIFIER_PROVIDER = "gemini"
VERIFIER_MODEL = "gemini-2.5-flash"
STRUCTURER_BASE_URL = "http://localhost:8001/v1"
STRUCTURER_MODEL = "Qwen/Qwen3-8B"
MAX_WORKERS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 60000
TEXT_CHUNK_CHAR_LIMIT = 32000
TABLE_CHUNK_CHAR_LIMIT = 32000
FALLBACK_CHUNK_COUNT = 8

VALID_SOURCE_TYPES = {"table", "text", "figure", "not_applicable"}
VALID_CONFIDENCE = {"high", "medium", "low"}


SourceType = Literal["table", "text", "figure", "not_applicable"]
Confidence = Literal["high", "medium", "low"]
Verdict = Literal["verified", "wrong", "VERIFIED", "WRONG"]


class CorrectedPlan(BaseModel):
    found_in_pdf: bool = Field(description="True if reported in PDF; False if not reported")
    page: int = Field(description="1-based page if found; -1 if not found")
    source_type: SourceType
    confidence: Confidence
    extraction_plan: str


class VerifiedColumn(BaseModel):
    column_index: int
    column_name: str
    verdict: Verdict
    reasoning: str = Field(description="Must cite Chunk IDs and include verbatim quotes from chunks/PDF.")
    issues: List[str] = Field(default_factory=list)
    corrected_plan: CorrectedPlan


class GroupVerification(BaseModel):
    group_name: str
    columns: List[VerifiedColumn]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and correct extraction plans using PDF + chunks.")
    parser.add_argument("--pdf-name", required=True, help="PDF stem only (no .pdf).")
    parser.add_argument(
        "--results-root",
        default="new_pipeline_outputs/results",
        help="Root path containing per-document pipeline outputs.",
    )
    parser.add_argument(
        "--chunk-source",
        choices=("pipeline", "landing-ai"),
        default="pipeline",
        help="Chunk source: 'pipeline' uses pdf_chunked.json; 'landing-ai' uses Landing AI ADE Parse.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Directory containing PDFs (used when --chunk-source=landing-ai).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run verification without writing output files.")
    return parser.parse_args()


def resolve_paths(pdf_name: str, results_root: Path, require_chunks: bool = True) -> Dict[str, Path]:
    base_dir = results_root / pdf_name
    planning_dir = base_dir / "planning"
    chunk_file = base_dir / "chunking" / "pdf_chunked.json"
    verifier_dir = planning_dir / "verifier"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base results directory not found: {base_dir}")
    if not planning_dir.exists():
        raise FileNotFoundError(f"Planning directory not found: {planning_dir}")
    if require_chunks and not chunk_file.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

    return {
        "base_dir": base_dir,
        "planning_dir": planning_dir,
        "chunk_file": chunk_file,
        "verifier_dir": verifier_dir,
    }

def load_chunks(chunk_file: Path) -> List[Dict[str, Any]]:
    data = json.loads(chunk_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        return data["chunks"]
    raise ValueError(f"Unexpected chunk file shape in {chunk_file}")


def load_plans(planning_dir: Path) -> Dict[str, Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    compiled = planning_dir / "plans_all_columns.json"
    if compiled.exists():
        data = json.loads(compiled.read_text(encoding="utf-8"))
        for entry in data.get("plans", []):
            if isinstance(entry, dict) and entry.get("group_name"):
                plans[entry["group_name"]] = entry
        if plans:
            return plans

    logs_dir = planning_dir / "logs"
    if logs_dir.exists():
        for plan_file in logs_dir.glob("*_plan.json"):
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("group_name"):
                plans[data["group_name"]] = data
    if not plans:
        for plan_file in planning_dir.glob("*_plan.json"):
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("group_name"):
                plans[data["group_name"]] = data
    return plans


def load_column_definitions() -> Dict[str, str]:
    grouped = load_definitions()
    out: Dict[str, str] = {}
    for _group, cols in grouped.items():
        for col in cols:
            out[col["Column Name"]] = col["Definition"]
    return out


def chunk_page_matches(chunk_page: Any, target_page: int) -> bool:
    if isinstance(chunk_page, int):
        return chunk_page == target_page
    if isinstance(chunk_page, str):
        if "-" in chunk_page:
            try:
                start_s, end_s = chunk_page.split("-", 1)
                return int(start_s) <= target_page <= int(end_s)
            except Exception:
                return False
        try:
            return int(chunk_page) == target_page
        except Exception:
            return False
    return False


def find_relevant_chunks_for_group(group_plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for col in group_plan.get("columns", []):
        source_type = str(col.get("source_type", "")).lower()
        page = col.get("page", -1)
        if source_type not in {"table", "text", "figure"} or not isinstance(page, int) or page < 1:
            continue
        for idx, chunk in enumerate(chunks):
            chunk_type = str(chunk.get("type", "text")).lower()
            if chunk_type != source_type:
                continue
            if not chunk_page_matches(chunk.get("page"), page):
                continue
            if idx in seen:
                continue
            selected.append(
                {
                    "chunk_id": idx,
                    "type": chunk_type,
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                }
            )
            seen.add(idx)

    if not selected:
        for idx, chunk in enumerate(chunks[:FALLBACK_CHUNK_COUNT]):
            selected.append(
                {
                    "chunk_id": idx,
                    "type": str(chunk.get("type", "text")).lower(),
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                }
            )
    return selected


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No chunks available."
    parts: List[str] = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        chunk_type = str(chunk.get("type", "text")).upper()
        page = chunk.get("page")
        if chunk_type == "TABLE":
            summary = str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT].replace("\n", " ")
            table_body = str(chunk.get("table_content", ""))[:TABLE_CHUNK_CHAR_LIMIT]
            body = f"Summary: {summary}\n\nStructured Table:\n{table_body}" if table_body else summary
        else:
            body = str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT]
        parts.append(f"--- Chunk {chunk_id} ({chunk_type} on page {page}) ---\n{body}")
    return "\n\n".join(parts)


def build_group_prompt(
    group_plan: Dict[str, Any],
    definitions_map: Dict[str, str],
    formatted_chunks: str,
) -> str:
    column_blocks: List[str] = []
    for column in group_plan.get("columns", []):
        name = column.get("column_name", "")
        definition = definitions_map.get(name, "")
        column_blocks.append(
            "\n".join(
                [
                    f"- column_index: {column.get('column_index')}",
                    f"  column_name: {name}",
                    f"  definition: {definition}",
                    f"  original_extraction_plan: {column.get('extraction_plan', '')}",
                ]
            )
        )

    columns_text = "\n\n".join(column_blocks)

    return f"""You are verifying extraction plans for a clinical trial paper using ONLY the provided chunks.

You have:
1) Relevant chunks selected from the chunking output.

Group: {group_plan.get('group_name')}

Columns to verify:
{columns_text}

Relevant chunks:
{formatted_chunks}

Task:
- Evaluate EACH column independently. Do not assume the original plan is correct.
- Decide if the original_extraction_plan is VALID or WRONG given ONLY the provided chunks.
- Your reasoning MUST cite Chunk IDs and include verbatim quotes that justify the verdict.
- If you cannot quote support from chunks, verdict must be WRONG.
- If WRONG, provide a corrected plan (plan only, not extracted value). If VERIFIED, corrected plan may restate the original.

Output a free-text report that follows this exact template for EVERY column (one block per column):

COLUMN <column_index>: <column_name>
VERDICT: VERIFIED|WRONG
REASONING:
- Chunk citations: Chunk <id> ... (include 1-3 short direct quotes)
- Why the original plan is right/wrong
ISSUES:
- (bullet list; empty if none)
CORRECTED_PLAN_JSON:
{{"found_in_pdf": true/false, "page": -1 or int, "source_type": "table|text|figure|not_applicable", "confidence": "high|medium|low", "extraction_plan": "..."}}

Rules:
- Keep column_index and column_name exactly as provided.
- CORRECTED_PLAN_JSON must always be present (even if VERIFIED, you can restate the original).
- Do not wrap anything in code fences.
"""

def normalize_corrected_plan(raw: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    found_in_pdf = raw.get("found_in_pdf", original.get("found_in_pdf", False))
    page = raw.get("page", original.get("page", -1))
    source_type = str(raw.get("source_type", original.get("source_type", "not_applicable"))).lower()
    confidence = str(raw.get("confidence", original.get("confidence", "low"))).lower()
    extraction_plan = raw.get("extraction_plan", original.get("extraction_plan", ""))

    if source_type not in VALID_SOURCE_TYPES:
        source_type = "not_applicable" if not found_in_pdf else str(original.get("source_type", "text")).lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = str(original.get("confidence", "low")).lower()
        if confidence not in VALID_CONFIDENCE:
            confidence = "low"

    if not found_in_pdf:
        page = -1
        source_type = "not_applicable"
    else:
        try:
            page = int(page)
        except Exception:
            page = int(original.get("page", 1) if int(original.get("page", -1)) > 0 else 1)
        if page < 1:
            page = int(original.get("page", 1) if int(original.get("page", -1)) > 0 else 1)
        if source_type not in {"table", "text", "figure"}:
            source_type = str(original.get("source_type", "text")).lower()
            if source_type not in {"table", "text", "figure"}:
                source_type = "text"

    return {
        "found_in_pdf": bool(found_in_pdf),
        "page": int(page),
        "source_type": source_type,
        "confidence": confidence,
        "extraction_plan": str(extraction_plan),
    }


def verify_group(
    group_plan: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    definitions_map: Dict[str, str],
) -> Dict[str, Any]:
    group_name = group_plan.get("group_name", "unknown_group")
    relevant_chunks = find_relevant_chunks_for_group(group_plan, chunks)
    formatted_chunks = format_chunks(relevant_chunks)
    prompt = build_group_prompt(group_plan, definitions_map, formatted_chunks)

    provider = LLMProvider(provider=VERIFIER_PROVIDER, model=VERIFIER_MODEL)
    response = provider.generate(
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    result: Dict[str, Any] = {
        "group_name": group_name,
        "model_provider": VERIFIER_PROVIDER,
        "model_name": VERIFIER_MODEL,
        "prompt_token_count": response.input_tokens,
        "output_token_count": response.output_tokens,
        "success": response.success,
        "error": response.error,
        "relevant_chunk_ids": [c["chunk_id"] for c in relevant_chunks],
        "columns": [],
        "raw_prompt": prompt,
        "raw_response": response.text or "",
    }

    verified_plan = copy.deepcopy(group_plan)
    output_by_index: Dict[int, Dict[str, Any]] = {}

    if response.success:
        structurer = OutputStructurer(
            base_url=STRUCTURER_BASE_URL,
            model=STRUCTURER_MODEL,
            enable_thinking=False,
        )
        structured = structurer.structure(
            text=response.text or "",
            schema=GroupVerification,
            max_retries=3,
            return_dict=True,
        )
        if structured.success and isinstance(structured.data, dict):
            result["structured"] = structured.data
            for item in (structured.data.get("columns") or []):
                if not isinstance(item, dict):
                    continue
                idx = item.get("column_index")
                if isinstance(idx, int):
                    output_by_index[idx] = item
        else:
            result["success"] = False
            result["error"] = f"Structuring failed: {structured.error}"

    for original_col in group_plan.get("columns", []):
        idx = original_col.get("column_index")
        raw_item = output_by_index.get(idx, {})
        verdict_raw = str(raw_item.get("verdict", "")).strip().lower()
        issues = raw_item.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        issues = [str(x) for x in issues]

        verified = verdict_raw in {"verified"}
        if not raw_item:
            verified = False
            issues.append("Missing column result from verifier response.")

        corrected = normalize_corrected_plan(raw_item.get("corrected_plan", {}), original_col)
        status = "verified" if verified else "corrected"
        if not verified and not raw_item:
            status = "failed"

        if status == "corrected":
            for field in ["found_in_pdf", "page", "source_type", "confidence", "extraction_plan"]:
                original_val = corrected[field]
                for col in verified_plan.get("columns", []):
                    if col.get("column_index") == idx:
                        col[field] = original_val
                        break

        result["columns"].append(
            {
                "column_index": idx,
                "column_name": original_col.get("column_name"),
                "status": status,
                "verdict": "verified" if verified else "wrong",
                "reasoning": str(raw_item.get("reasoning", "")),
                "issues": issues,
                "original_plan": {
                    "found_in_pdf": original_col.get("found_in_pdf"),
                    "page": original_col.get("page"),
                    "source_type": original_col.get("source_type"),
                    "confidence": original_col.get("confidence"),
                    "extraction_plan": original_col.get("extraction_plan"),
                },
                "corrected_plan": corrected,
            }
        )

    result["verified_group_plan"] = verified_plan
    return result


def write_outputs(verifier_dir: Path, results: List[Dict[str, Any]]) -> None:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = verifier_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_columns = 0
    verified_columns = 0
    corrected_columns = 0
    failed_columns = 0

    for item in results:
        group_name = item["group_name"]
        stem = safe_stem(group_name)

        # LLM log per group
        (logs_dir / f"{stem}_llm_call.txt").write_text(
            (
                f"GROUP: {group_name}\n"
                f"PROVIDER: {item.get('model_provider', VERIFIER_PROVIDER)}\n"
                f"MODEL: {item.get('model_name', VERIFIER_MODEL)}\n"
                f"SUCCESS: {item.get('success')}\n"
                f"ERROR: {item.get('error')}\n"
                f"PROMPT_TOKENS: {item.get('prompt_token_count')}\n"
                f"OUTPUT_TOKENS: {item.get('output_token_count')}\n"
                "============================================================\n"
                "INPUT PROMPT\n"
                "============================================================\n"
                f"{str(item.get('raw_prompt', ''))}\n\n"
                "============================================================\n"
                "RAW OUTPUT\n"
                "============================================================\n"
                f"{str(item.get('raw_response', ''))}\n"
            ),
            encoding="utf-8",
        )

        for col in item.get("columns", []):
            total_columns += 1
            status = col.get("status")
            if status == "verified":
                verified_columns += 1
            elif status == "corrected":
                corrected_columns += 1
            else:
                failed_columns += 1

    # Single compiled JSON with full verification for all groups
    compiled = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provider": VERIFIER_PROVIDER,
        "model": VERIFIER_MODEL,
        "max_workers": MAX_WORKERS,
        "total_groups": len(results),
        "total_columns": total_columns,
        "verified_columns": verified_columns,
        "corrected_columns": corrected_columns,
        "failed_columns": failed_columns,
        "groups_with_failures": [
            r["group_name"] for r in results if any(c.get("status") == "failed" for c in r.get("columns", []))
        ],
        "results": results,
    }
    (verifier_dir / "verification.json").write_text(
        json.dumps(compiled, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    use_landing_ai = args.chunk_source == "landing-ai"

    paths = resolve_paths(args.pdf_name, results_root, require_chunks=not use_landing_ai)

    if use_landing_ai:
        from landing_ai_chunks import load_landing_ai_chunks

        pdf_path = PROJECT_ROOT / args.dataset_dir / f"{args.pdf_name}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path} (required for --chunk-source=landing-ai)")
        cache_dir = paths["base_dir"] / "chunking"
        print(f"Loading chunks from Landing AI ADE Parse (cache: {cache_dir})...")
        chunks = load_landing_ai_chunks(pdf_path, cache_dir=cache_dir, use_cache=True)
        print(f"Loaded {len(chunks)} chunks from Landing AI")
    else:
        chunks = load_chunks(paths["chunk_file"])
    plans_by_group = load_plans(paths["planning_dir"])
    definitions_map = load_column_definitions()

    if not plans_by_group:
        raise RuntimeError(f"No plans found in {paths['planning_dir']}")

    ordered_groups = [plans_by_group[name] for name in sorted(plans_by_group.keys())]
    all_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(verify_group, group_plan, chunks, definitions_map): group_plan.get(
                "group_name", "unknown_group"
            )
            for group_plan in ordered_groups
        }
        for future in as_completed(future_map):
            group_name = future_map[future]
            try:
                all_results.append(future.result())
                print(f"Verified group: {group_name}")
            except Exception as e:
                print(f"Verification failed for group {group_name}: {e}")
                all_results.append(
                    {
                        "group_name": group_name,
                        "success": False,
                        "error": str(e),
                        "columns": [],
                        "verified_group_plan": copy.deepcopy(plans_by_group[group_name]),
                    }
                )

    all_results = sorted(all_results, key=lambda x: x.get("group_name", ""))

    if args.dry_run:
        corrected = sum(
            1 for result in all_results for col in result.get("columns", []) if col.get("status") == "corrected"
        )
        failed = sum(1 for result in all_results for col in result.get("columns", []) if col.get("status") == "failed")
        total = sum(len(result.get("columns", [])) for result in all_results)
        print(f"Dry run complete. Total columns: {total}, corrected: {corrected}, failed: {failed}")
        return 0

    write_outputs(paths["verifier_dir"], all_results)
    print(f"Verifier outputs written to: {paths['verifier_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
