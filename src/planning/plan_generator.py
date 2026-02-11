# src/planning/plan_generator.py
"""
Plan-based extraction (V2): generate extraction plans for each column group.
Uses multimodal LLM (PDF + chunks) for free-form planning, then local structurer for JSON.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.config.config import STRUCTURER_BASE_URL, STRUCTURER_MODEL
from src.utils.costing import usage_to_cost_dict
from src.utils.logging_utils import setup_logger

logger = setup_logger("planning")


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Column:
    name: str
    definition: str


@dataclass
class ColumnGroup:
    name: str
    columns: List[Column]


SourceType = Literal["table", "text", "figure", "not_applicable"]
Confidence = Literal["high", "medium", "low"]


class ColumnExtractionPlanV2(BaseModel):
    """Extraction plan for one canonical column."""

    column_index: int = Field(description="1-based index into the EXPECTED_COLUMNS list")
    column_name: str = Field(description="Must exactly match the canonical column name at column_index")
    found_in_pdf: bool = Field(
        description="True if the value exists in the PDF; False if not reported/available"
    )
    page: int = Field(description="Page number if found; -1 if not found")
    source_type: SourceType = Field(description="table/text/figure if found, else not_applicable")
    confidence: Confidence = Field(description="high/medium/low")
    extraction_plan: str = Field(description="How to extract, or why not reported")
    column_name_raw: Optional[str] = None


class GroupExtractionPlanV2(BaseModel):
    """Plans for all columns in a group."""

    group_name: str
    columns: List[ColumnExtractionPlanV2]


# -----------------------------
# Helpers
# -----------------------------


def safe_stem(name: str) -> str:
    """Stable filename stem for group names."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("|", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def definitions_to_column_groups(definitions: Dict[str, List[Dict[str, str]]]) -> List[ColumnGroup]:
    """Convert load_definitions() output to list of ColumnGroup."""
    out: List[ColumnGroup] = []
    for group_name, cols in definitions.items():
        out.append(
            ColumnGroup(
                name=group_name,
                columns=[Column(name=c["Column Name"], definition=c["Definition"]) for c in cols],
            )
        )
    return out


def format_chunk_summaries(chunks: list) -> str:
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("type", "unknown")
        page = chunk.get("page", "?")
        content_preview = (chunk.get("content", "") or "")[:120].replace("\n", " ")
        if chunk_type == "table":
            summary = f"Chunk {i}: TABLE on page {page} - {content_preview}..."
        elif chunk_type == "figure":
            summary = f"Chunk {i}: FIGURE on page {page} - {content_preview}..."
        else:
            summary = f"Chunk {i}: TEXT on pages {page} - {content_preview}..."
        summaries.append(summary)
    return "\n".join(summaries)


def build_expected_columns_block(group: ColumnGroup) -> str:
    lines = []
    for i, col in enumerate(group.columns, 1):
        lines.append(f"{i}. {col.name}\n   Definition: {col.definition}")
    return "\n".join(lines)


def _normalize_not_found(plan: ColumnExtractionPlanV2) -> ColumnExtractionPlanV2:
    if not plan.found_in_pdf:
        plan.page = -1
        plan.source_type = "not_applicable"
    return plan


def validate_and_normalize_group_plan(
    *,
    group_name: str,
    plan: GroupExtractionPlanV2,
    expected_columns: List[str],
    name_policy: Literal["strict", "override"] = "strict",
) -> GroupExtractionPlanV2:
    """Enforce canonical identity: indices + exact names."""
    n = len(expected_columns)
    by_idx = {}
    dupes = []
    out_of_range = []
    for item in plan.columns:
        idx = item.column_index
        if idx in by_idx:
            dupes.append(idx)
            continue
        if idx < 1 or idx > n:
            out_of_range.append(idx)
            continue
        by_idx[idx] = item

    missing = [i for i in range(1, n + 1) if i not in by_idx]
    if dupes or out_of_range or missing:
        raise ValueError(
            f"Invalid column_index set for group '{group_name}'. "
            f"dupes={sorted(set(dupes))}, out_of_range={sorted(set(out_of_range))}, missing_count={len(missing)}"
        )

    normalized: List[ColumnExtractionPlanV2] = []
    name_mismatches = []
    for idx in range(1, n + 1):
        item = by_idx[idx]
        canonical = expected_columns[idx - 1]
        if item.column_name != canonical:
            name_mismatches.append((idx, item.column_name, canonical))
            if name_policy == "override":
                item.column_name_raw = item.column_name
                item.column_name = canonical
        item = _normalize_not_found(item)
        normalized.append(item)

    if name_mismatches and name_policy == "strict":
        preview = "\n".join(
            [f"  • Index {i}: Got '{got}'\n             Expected '{want}'" for i, got, want in name_mismatches[:5]]
        )
        total = len(name_mismatches)
        more = f" (and {total - 5} more)" if total > 5 else ""
        raise ValueError(
            f"\n❌ Column name mismatch(es) for group '{group_name}':\n"
            f"The structurer output incorrect column names.\n"
            f"Mismatches{more}:\n{preview}"
        )

    plan.group_name = group_name
    plan.columns = normalized
    return plan


# -----------------------------
# PlanGenerator
# -----------------------------


class PlanGenerator:
    """
    Generate extraction plans for all column groups.
    Uses provider (e.g. Gemini) with PDF for free-form planning, then structurer for JSON.
    """

    def __init__(
        self,
        provider: LLMProvider,
        definitions: Dict[str, List[Dict[str, str]]],
        structurer: Optional[OutputStructurer] = None,
        name_policy: Literal["strict", "override"] = "strict",
    ):
        self.provider = provider
        self.definitions = definitions
        self.groups = definitions_to_column_groups(definitions)
        self.structurer = structurer or OutputStructurer(
            base_url=STRUCTURER_BASE_URL,
            model=STRUCTURER_MODEL,
            enable_thinking=False,
        )
        self.name_policy = name_policy

    def generate_plan_for_group(
        self,
        group: ColumnGroup,
        pdf_handle: Any,
        chunks: list,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generate extraction plan for one column group."""
        logger.info("Planning extraction for group: %s (%d columns)", group.name, len(group.columns))

        chunk_summaries = format_chunk_summaries(chunks)
        expected_block = build_expected_columns_block(group)

        prompt = f"""You are creating an extraction plan for a clinical trial data extraction task.

You have:
- The FULL PDF loaded (for structure + precise reference)
- A list of pre-extracted chunks (to orient you)

AVAILABLE CHUNKS:
{chunk_summaries}

TASK:
For EACH of the following canonical columns, decide whether the value is reported in this PDF.
If reported, identify WHERE and HOW to extract it.
If not reported, say it is not reported.

⚠️ CRITICAL INSTRUCTION:
When you refer to these columns in your response, you MUST use their EXACT names as listed below.
Do NOT paraphrase, abbreviate, or modify column names in ANY way.
Character-for-character match is REQUIRED (including spaces, punctuation, pipes |, parentheses).

CANONICAL COLUMNS (ORDERED; use these EXACT names in your response):
{expected_block}

Rules:
- Be honest: many columns will NOT be reported.
- If found_in_pdf=false, state why in extraction_plan.
- If found_in_pdf=true, include page number, source type (table/text/figure), and concrete instructions.
- ALWAYS refer to columns using their exact canonical names (copy-paste from the list above).
"""

        response = self.provider.generate_with_pdf(
            prompt=prompt,
            pdf_handle=pdf_handle,
            temperature=0.1,
            max_tokens=8000,
        )
        if not response.success:
            raise ValueError(f"Planning LLM failed: {response.error}")

        free_form = (response.text or "").strip()

        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(group.name)
        raw_file = logs_dir / f"{stem}_raw.txt"
        raw_file.write_text(free_form, encoding="utf-8")

        expected_columns = [c.name for c in group.columns]
        expected_names_list = "\n".join([f"{i}. {name}" for i, name in enumerate(expected_columns, 1)])

        structuring_prompt = f"""Convert the following free-form extraction plan into STRICT JSON.

Group name (use exactly this, do not rename): "{group.name}"

EXPECTED_COLUMNS (ordered; index is canonical):
{expected_names_list}

⚠️ For each item, column_name MUST be an EXACT CHARACTER-FOR-CHARACTER match with EXPECTED_COLUMNS[column_index].

Output rules:
- Return JSON with fields: group_name, columns
- columns must be a list with EXACTLY {len(expected_columns)} items
- Each item: column_index (1..{len(expected_columns)}), column_name (exact match), found_in_pdf, page, source_type (table/text/figure/not_applicable), confidence (high/medium/low), extraction_plan

Free-form plan text:
{free_form}

Return ONLY valid JSON.
"""

        structured = self.structurer.structure(
            text=structuring_prompt,
            schema=GroupExtractionPlanV2,
            max_retries=5,
            return_dict=False,
        )
        if not structured.success:
            raise ValueError(f"Structuring failed for group '{group.name}': {structured.error}")

        plan: GroupExtractionPlanV2 = structured.data
        plan = validate_and_normalize_group_plan(
            group_name=group.name,
            plan=plan,
            expected_columns=expected_columns,
            name_policy=self.name_policy,
        )

        plan_path = logs_dir / f"{stem}_plan.json"
        plan_path.write_text(json.dumps(plan.model_dump(), indent=2), encoding="utf-8")

        found = sum(1 for c in plan.columns if c.found_in_pdf)
        logger.info("Plan summary for %s: found_in_pdf=true for %d/%d", group.name, found, len(plan.columns))

        usage = usage_to_cost_dict(
            self.provider.provider,
            self.provider.model,
            response.input_tokens,
            response.output_tokens,
        )
        return plan.model_dump(), usage

    def generate_plans(
        self,
        pdf_path: Path,
        chunks: Any,
        output_dir: Path,
        workers: int = 10,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Generate extraction plans for all column groups.
        chunks: list of chunk dicts, or dict with "chunks" key.
        Returns: (plans {group_name: plan_data}, usage_dict for costing)
        """
        from src.utils.costing import aggregate_usage

        chunks_list = chunks if isinstance(chunks, list) else (chunks.get("chunks", []) if isinstance(chunks, dict) else [])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_handle = self.provider.upload_pdf(pdf_path)
        usage_list: List[Dict[str, Any]] = []
        try:
            plans = {}
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_to_group = {
                    executor.submit(
                        self.generate_plan_for_group,
                        group,
                        pdf_handle,
                        chunks_list,
                        output_dir,
                    ): group
                    for group in self.groups
                }
                for future in as_completed(future_to_group):
                    group = future_to_group[future]
                    try:
                        plan_data, usage = future.result()
                        plans[plan_data["group_name"]] = plan_data
                        usage_list.append(usage)
                        logger.info("Planned: %s", group.name)
                    except Exception as e:
                        logger.error("Failed group '%s': %s", group.name, e)
                        # Continue with other groups (same as new_pipeline_outputs/generate_extraction_plan_v2.py)
            if plans:
                compiled_path = output_dir / "plans_all_columns.json"
                compiled_path.write_text(
                    json.dumps(
                        {
                            "total_groups": len(plans),
                            "total_columns": sum(len(p.get("columns", [])) for p in plans.values()),
                            "plans": list(plans.values()),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.info("Saved compiled plans to %s", compiled_path.name)
            usage_dict = aggregate_usage(usage_list) if usage_list else {}
            return plans, usage_dict
        finally:
            self.provider.cleanup_pdf(pdf_handle)
