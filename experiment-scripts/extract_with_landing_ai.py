#!/usr/bin/env python3
"""
extract_with_landing_ai.py

Extract column values using Landing AI chunks + LLM, with optional retry on alternative_plan.

Flow:
1. Load plans and Landing AI chunks
2. For each column: select chunks by group + page, ask LLM to extract value per plan
3. LLM returns: value, found, confidence, suggestion?, alternative_plan?
4. If alternative_plan is given: retry with that plan, compare values

Output: extraction results with value, confidence, suggestion, retry comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.planning.plan_generator import safe_stem
from src.table_definitions.definitions import load_definitions


def load_column_definitions() -> Dict[str, str]:
    grouped = load_definitions()
    out: Dict[str, str] = {}
    for _group, cols in grouped.items():
        for col in cols:
            out[col["Column Name"]] = col["Definition"]
    return out

# Config
PROVIDER = "gemini"
MODEL = "gemini-2.5-flash"
STRUCTURER_BASE_URL = "http://localhost:8001/v1"
STRUCTURER_MODEL = "Qwen/Qwen3-8B"
MAX_WORKERS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 8000
TEXT_CHUNK_CHAR_LIMIT = 32000
TABLE_CHUNK_CHAR_LIMIT = 32000
FALLBACK_CHUNK_COUNT = 8
FALLBACK_MAX_CHUNKS = 80  # When plan has no page, include more chunks to cover tables


def _parse_pages_from_plan(extraction_plan: str) -> list[int]:
    """Extract page numbers mentioned in extraction plan (e.g. 'page 5', 'Table 1 (page 5)')."""
    if not extraction_plan:
        return []
    pages: list[int] = []
    # Match "page 5", "page 5)", "(page 5", "pages 5-7", "page 5 and 6"
    for m in re.finditer(r"page[s]?\s+(\d+)(?:\s*[-–]\s*(\d+))?", extraction_plan, re.I):
        start = int(m.group(1))
        pages.append(start)
        if m.group(2):
            end = int(m.group(2))
            pages.extend(range(start + 1, end + 1))
    return sorted(set(p for p in pages if 1 <= p <= 500))


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


def find_chunks_for_column(
    column: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Select chunks relevant to a single column (page + source_type, or pages from plan)."""
    source_type = str(column.get("source_type", "")).lower()
    page = column.get("page", -1)
    extraction_plan = column.get("extraction_plan", "") or ""
    selected: List[Dict[str, Any]] = []
    seen = set()

    # Primary: use page + source_type when available
    if source_type in {"table", "text", "figure"} and isinstance(page, int) and page >= 1:
        for idx, chunk in enumerate(chunks):
            chunk_type = str(chunk.get("type", "text")).lower()
            if chunk_type != source_type:
                continue
            if not chunk_page_matches(chunk.get("page"), page):
                continue
            if idx in seen:
                continue
            selected.append({
                "chunk_id": idx,
                "type": chunk_type,
                "page": chunk.get("page"),
                "content": chunk.get("content", "") or "",
                "table_content": chunk.get("table_content", "") or "",
            })
            seen.add(idx)

    # Fallback when page=-1 / not_applicable: parse extraction_plan for page refs (e.g. "Table 1 (page 5)")
    # Only add chunks whose page is in plan_pages (page + modality together - no blanket "all tables")
    if not selected and fallback:
        plan_pages = _parse_pages_from_plan(extraction_plan)
        for idx, chunk in enumerate(chunks):
            if idx in seen:
                continue
            chunk_page = chunk.get("page")
            chunk_type = str(chunk.get("type", "text")).lower()
            # Get page as int (chunk may have int or range str like "1-4")
            page_val: int | None = None
            if isinstance(chunk_page, int):
                page_val = chunk_page
            elif isinstance(chunk_page, str) and "-" in chunk_page:
                try:
                    page_val = int(chunk_page.split("-", 1)[0].strip())
                except Exception:
                    pass
            if plan_pages and page_val is not None and page_val in plan_pages:
                selected.append({
                    "chunk_id": idx,
                    "type": chunk_type,
                    "page": chunk.get("page"),
                    "content": chunk.get("content", "") or "",
                    "table_content": chunk.get("table_content", "") or "",
                })
                seen.add(idx)

    # Last resort: include more chunks so we don't miss tables (was first 8, now first N)
    if not selected and fallback:
        for idx, chunk in enumerate(chunks[:FALLBACK_MAX_CHUNKS]):
            selected.append({
                "chunk_id": idx,
                "type": str(chunk.get("type", "text")).lower(),
                "page": chunk.get("page"),
                "content": chunk.get("content", "") or "",
                "table_content": chunk.get("table_content", "") or "",
            })
    return selected


def _clean_chunk_content(text: str) -> str:
    """Strip Landing AI markup: anchor IDs, table/cell IDs, merge to readable content."""
    if not text:
        return ""
    # Remove <a id='...'></a> anchors (often at start of chunks)
    text = re.sub(r"<a\s+id=['\"][^'\"]*['\"]\s*></a>\s*", "", text)
    # Remove id="4-1", id='4-2' etc from table/td/tr elements
    text = re.sub(r'\s+id=["\'][^"\']*["\']', "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_sort_key(chunk: Dict[str, Any]) -> tuple:
    """Sort key: (page_num, 0). Handles int, str like '5-7', etc."""
    p = chunk.get("page")
    if isinstance(p, int):
        return (p, 0)
    if isinstance(p, str) and "-" in p:
        try:
            return (int(p.split("-", 1)[0].strip()), 0)
        except Exception:
            return (0, 0)
    try:
        return (int(p) if p else 0, 0)
    except Exception:
        return (0, 0)


def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks by type: TEXT (chronological), TABLES (chronological), FIGURES (chronological)."""
    if not chunks:
        return "No chunks available."

    by_type: Dict[str, List[Dict[str, Any]]] = {"text": [], "table": [], "figure": []}
    for chunk in chunks:
        ctype = str(chunk.get("type", "text")).lower()
        if ctype not in by_type:
            by_type["text"].append(chunk)  # marginalia etc. -> text
        else:
            by_type[ctype].append(chunk)

    sections: List[str] = []

    # TEXT: all text chunks concatenated together in chronological order (at the top)
    text_group = sorted(by_type["text"], key=_page_sort_key)
    if text_group:
        text_parts: List[str] = []
        for chunk in text_group:
            body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                text_parts.append(body)
        if text_parts:
            sections.append("--- TEXT ---\n" + "\n\n".join(text_parts))

    # TABLES: each table with [Page N] label
    table_group = sorted(by_type["table"], key=_page_sort_key)
    if table_group:
        table_parts: List[str] = []
        for chunk in table_group:
            page = chunk.get("page")
            body = str(chunk.get("table_content", "") or chunk.get("content", ""))[:TABLE_CHUNK_CHAR_LIMIT]
            body = _clean_chunk_content(body)
            if not body:
                body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                table_parts.append(f"[Page {page}]\n{body}")
        if table_parts:
            sections.append("--- TABLES ---\n" + "\n\n".join(table_parts))

    # FIGURES: each figure with [Page N] label
    figure_group = sorted(by_type["figure"], key=_page_sort_key)
    if figure_group:
        figure_parts: List[str] = []
        for chunk in figure_group:
            page = chunk.get("page")
            body = _clean_chunk_content(str(chunk.get("content", ""))[:TEXT_CHUNK_CHAR_LIMIT])
            if body:
                figure_parts.append(f"[Page {page}]\n{body}")
        if figure_parts:
            sections.append("--- FIGURES ---\n" + "\n\n".join(figure_parts))

    return "\n\n".join(sections) if sections else "No content available."


def build_extraction_prompt(
    column: Dict[str, Any],
    definition: str,
    formatted_chunks: str,
    extraction_plan: str,
) -> str:
    name = column.get("column_name", "")
    return f"""You are extracting a value for a clinical trial data column from the provided document content.

Column: {name}
Definition: {definition}

Extraction plan to follow:
{extraction_plan}

Document content (grouped by type, in chronological page order):
{formatted_chunks}

Task:
1. Extract the value for this column following the plan. Return your best-effort value even if uncertain.
2. If the plan seems wrong, incomplete, or ambiguous, still return the value you found and add a suggestion and/or alternative_plan.
3. Use "not found" or "not reported" only if the value is genuinely absent from the content.

Output a JSON object with:
- value: the extracted value (string)
- found: true if you found a value, false if not reported/absent
- confidence: "high" | "medium" | "low"
- suggestion: optional string with concerns or notes (null if none)
- alternative_plan: optional string describing a different extraction approach to try (null if plan seems fine)
"""


# Pydantic schema for extraction response
try:
    from pydantic import BaseModel, Field
    from typing import Optional

    class ExtractionResult(BaseModel):
        value: str = Field(description="The extracted value")
        found: bool = Field(description="True if value was found in chunks")
        confidence: str = Field(description="high, medium, or low")
        suggestion: Optional[str] = Field(default=None, description="Optional concern or note")
        alternative_plan: Optional[str] = Field(default=None, description="Optional alternative extraction approach")
except ImportError:
    ExtractionResult = None


def _write_llm_log(
    logs_dir: Path,
    group_name: str,
    column_name: str,
    prompt: str,
    output: str,
    prompt_retry: str | None = None,
    output_retry: str | None = None,
) -> None:
    """Write clean model input/output to a txt file."""
    stem = safe_stem(f"{group_name}_{column_name}")
    parts = [
        "============================================================\nINPUT\n============================================================\n",
        prompt,
        "\n\n============================================================\nOUTPUT\n============================================================\n",
        output or "",
    ]
    if prompt_retry is not None and output_retry is not None:
        parts.extend([
            "\n\n============================================================\nINPUT (RETRY)\n============================================================\n",
            prompt_retry,
            "\n\n============================================================\nOUTPUT (RETRY)\n============================================================\n",
            output_retry or "",
        ])
    (logs_dir / f"{stem}_llm_call.txt").write_text("".join(parts), encoding="utf-8")


def extract_column(
    column: Dict[str, Any],
    group_name: str,
    definition: str,
    chunks: List[Dict[str, Any]],
    do_retry: bool = True,
    logs_dir: Path | None = None,
) -> Dict[str, Any]:
    """Extract value for one column. Optionally retry with alternative_plan."""
    plan = column.get("extraction_plan", "")
    relevant = find_chunks_for_column(column, chunks)
    formatted = format_chunks(relevant)

    prompt = build_extraction_prompt(column, definition, formatted, plan)

    provider = LLMProvider(provider=PROVIDER, model=MODEL)
    response = provider.generate(prompt=prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    result: Dict[str, Any] = {
        "column_index": column.get("column_index"),
        "column_name": column.get("column_name"),
        "group_name": group_name,
        "extraction_plan": column.get("extraction_plan", ""),
        "page": column.get("page"),
        "source_type": column.get("source_type"),
        "success": response.success,
        "error": response.error,
        "value": None,
        "found": False,
        "confidence": "low",
        "suggestion": None,
        "alternative_plan": None,
        "retry_value": None,
        "values_match": None,
        "prompt_token_count": response.input_tokens,
        "output_token_count": response.output_tokens,
        "raw_response": response.text or "",
    }

    if not response.success:
        if logs_dir:
            _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "")
        return result

    # Parse structured output
    if ExtractionResult and OutputStructurer:
        structurer = OutputStructurer(
            base_url=STRUCTURER_BASE_URL,
            model=STRUCTURER_MODEL,
            enable_thinking=False,
        )
        structured = structurer.structure(
            text=response.text or "",
            schema=ExtractionResult,
            max_retries=2,
            return_dict=True,
        )
        if structured.success and isinstance(structured.data, dict):
            d = structured.data
            result["value"] = d.get("value") or ""
            result["found"] = bool(d.get("found", False))
            result["confidence"] = str(d.get("confidence", "low")).lower()
            result["suggestion"] = d.get("suggestion")
            result["alternative_plan"] = d.get("alternative_plan")
        else:
            # Structurer failed; fall back to JSON parse from raw response
            try:
                text = (response.text or "").strip()
                if "```" in text:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                d = json.loads(text)
                result["value"] = d.get("value") or ""
                result["found"] = bool(d.get("found", False))
                result["confidence"] = str(d.get("confidence", "low")).lower()
                result["suggestion"] = d.get("suggestion")
                result["alternative_plan"] = d.get("alternative_plan")
            except Exception as e:
                result["error"] = f"Structuring failed: {getattr(structured, 'error', 'unknown')}; JSON fallback: {e}"
                if logs_dir:
                    _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "")
                return result
    else:
        # Fallback: try to parse JSON from response
        try:
            text = (response.text or "").strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            d = json.loads(text)
            result["value"] = d.get("value") or ""
            result["found"] = bool(d.get("found", False))
            result["confidence"] = str(d.get("confidence", "low")).lower()
            result["suggestion"] = d.get("suggestion")
            result["alternative_plan"] = d.get("alternative_plan")
        except Exception as e:
            result["error"] = f"Parse failed: {e}"
            if logs_dir:
                _write_llm_log(logs_dir, group_name, column.get("column_name", ""), prompt, response.text or "")
            return result

    # Option B: Retry with alternative_plan if provided
    prompt_retry: str | None = None
    output_retry: str | None = None
    if do_retry and result.get("alternative_plan"):
        alt_plan = result["alternative_plan"]
        prompt2 = build_extraction_prompt(column, definition, formatted, alt_plan)
        response2 = provider.generate(prompt=prompt2, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        prompt_retry = prompt2
        output_retry = response2.text or ""
        if response2.success and ExtractionResult:
            structurer = OutputStructurer(
                base_url=STRUCTURER_BASE_URL,
                model=STRUCTURER_MODEL,
                enable_thinking=False,
            )
            structured2 = structurer.structure(
                text=response2.text or "",
                schema=ExtractionResult,
                max_retries=2,
                return_dict=True,
            )
            if structured2.success and isinstance(structured2.data, dict):
                retry_val = structured2.data.get("value", "")
                result["retry_value"] = retry_val
                v1 = str(result.get("value", "")).strip()
                v2 = str(retry_val).strip()
                result["values_match"] = v1 == v2

    if logs_dir:
        _write_llm_log(
            logs_dir,
            group_name,
            column.get("column_name", ""),
            prompt,
            response.text or "",
            prompt_retry=prompt_retry,
            output_retry=output_retry,
        )

    return result


def load_plans(planning_dir: Path) -> Dict[str, Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}
    compiled = planning_dir / "plans_all_columns.json"
    if compiled.exists():
        data = json.loads(compiled.read_text(encoding="utf-8"))
        for entry in data.get("plans", []):
            if isinstance(entry, dict) and entry.get("group_name"):
                plans[entry["group_name"]] = entry
    return plans


def flatten_columns(plans_by_group: Dict[str, Dict[str, Any]]) -> List[tuple]:
    """(group_name, column) for each column across all groups."""
    out: List[tuple] = []
    for group_name in sorted(plans_by_group.keys()):
        group = plans_by_group[group_name]
        for col in group.get("columns", []):
            out.append((group_name, col))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract column values using Landing AI chunks + LLM with optional retry.")
    p.add_argument("--pdf-name", required=True, help="PDF stem (e.g. NCT02799602_Hussain_ARASENS_JCO'23)")
    p.add_argument("--results-root", default="new_pipeline_outputs/results")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--no-retry", action="store_true", help="Skip retry with alternative_plan")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Limit columns to process (0 = all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_root = (PROJECT_ROOT / args.results_root).resolve()
    base_dir = results_root / args.pdf_name
    planning_dir = base_dir / "planning"
    output_dir = base_dir / "planning" / "extract_landing_ai"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    if not planning_dir.exists():
        raise FileNotFoundError(f"Planning dir not found: {planning_dir}")

    # Load Landing AI chunks
    from landing_ai_chunks import load_landing_ai_chunks
    pdf_path = PROJECT_ROOT / args.dataset_dir / f"{args.pdf_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    cache_dir = base_dir / "chunking"
    print(f"Loading Landing AI chunks...")
    chunks = load_landing_ai_chunks(pdf_path, cache_dir=cache_dir, use_cache=True)
    print(f"Loaded {len(chunks)} chunks")

    plans_by_group = load_plans(planning_dir)
    if not plans_by_group:
        raise RuntimeError(f"No plans in {planning_dir}")

    definitions_map = load_column_definitions()
    flat = flatten_columns(plans_by_group)
    if args.limit > 0:
        flat = flat[: args.limit]
        print(f"Processing first {len(flat)} columns (--limit={args.limit})")

    logs_dir: Path | None = None
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for group_name, col in flat:
            defn = definitions_map.get(col.get("column_name", ""), "")
            fut = ex.submit(
                extract_column,
                col,
                group_name,
                defn,
                chunks,
                do_retry=not args.no_retry,
                logs_dir=logs_dir,
            )
            futures[fut] = (group_name, col.get("column_name"))

        for fut in as_completed(futures):
            group_name, col_name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                retry = " [retry]" if r.get("retry_value") is not None else ""
                match = f" match={r.get('values_match')}" if r.get("values_match") is not None else ""
                print(f"  {group_name} / {col_name}: {r.get('value', 'N/A')}{retry}{match}")
            except Exception as e:
                results.append({
                    "group_name": group_name,
                    "column_name": col_name,
                    "success": False,
                    "error": str(e),
                })
                print(f"  {group_name} / {col_name}: ERROR {e}")

    if args.dry_run:
        print(f"Dry run: {len(results)} columns processed")
        return 0

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pdf_name": args.pdf_name,
        "provider": PROVIDER,
        "model": MODEL,
        "do_retry": not args.no_retry,
        "total_columns": len(results),
        "with_alternative_plan": sum(1 for r in results if r.get("alternative_plan")),
        "retries_done": sum(1 for r in results if r.get("retry_value") is not None),
        "values_matched": sum(1 for r in results if r.get("values_match") is True),
        "values_differed": sum(1 for r in results if r.get("values_match") is False),
        "results": results,
    }
    out_file = output_dir / "extraction_results.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_file}")
    print(f"  With alternative_plan: {summary['with_alternative_plan']}")
    print(f"  Retries done: {summary['retries_done']}")
    print(f"  Values matched: {summary['values_matched']}, differed: {summary['values_differed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
