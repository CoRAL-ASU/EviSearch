# src/extraction/plan_executor.py
"""
Execute extraction plans (V2): run extraction per group using plans and structurer.
Uses LLMProvider.generate_with_pdf for extraction, OutputStructurer for JSON.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer, StructurerResponse
from src.table_definitions.definitions import load_definitions
from src.utils.logging_utils import setup_logger

from src.planning.plan_generator import (
    ColumnExtractionPlanV2,
    GroupExtractionPlanV2,
    safe_stem,
)

logger = setup_logger("extraction")


# -----------------------------
# Extraction schema (V2)
# -----------------------------


class ColumnExtractionV2(BaseModel):
    column_index: int
    column_name: str
    value: Optional[str] = None  # Always string to preserve formatting (e.g., "517 (79.4%)", "Not reported")
    evidence: Optional[str] = None
    page: str = ""
    confidence: str = "low"
    column_name_raw: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"high", "h", "certain", "very_high", "veryhigh"}:
                return "high"
            if s in {"medium", "med", "m", "moderate"}:
                return "medium"
            if s in {"low", "l", "uncertain", "unsure"}:
                return "low"
            if "high" in s:
                return "high"
            if "medium" in s:
                return "medium"
            if "low" in s:
                return "low"
            return "low"
        return v


class GroupExtractionV2(BaseModel):
    group_name: str
    extractions: List[ColumnExtractionV2]


# -----------------------------
# Helpers
# -----------------------------


def expected_columns_for_group(groups_dict: Dict[str, list], group_name: str) -> List[str]:
    if group_name not in groups_dict:
        raise KeyError(f"Group not found in definitions: {group_name}")
    return [c["Column Name"] for c in groups_dict[group_name]]


def validate_and_normalize_plan(
    *,
    plan: GroupExtractionPlanV2,
    expected_columns: List[str],
    name_policy: Literal["strict", "override"] = "strict",
) -> GroupExtractionPlanV2:
    n = len(expected_columns)
    by_idx: Dict[int, ColumnExtractionPlanV2] = {}
    dupes, out_of_range = [], []
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
            f"Invalid plan indices for group '{plan.group_name}': "
            f"dupes={sorted(set(dupes))}, out_of_range={sorted(set(out_of_range))}, missing_count={len(missing)}"
        )
    normalized = []
    mismatches = []
    for idx in range(1, n + 1):
        item = by_idx[idx]
        canonical = expected_columns[idx - 1]
        if item.column_name != canonical:
            mismatches.append((idx, item.column_name, canonical))
            if name_policy == "override":
                item.column_name_raw = item.column_name
                item.column_name = canonical
        if not item.found_in_pdf:
            item.page = -1
            item.source_type = "not_applicable"
        normalized.append(item)
    if mismatches and name_policy == "strict":
        preview = "; ".join([f"#{i}: '{got}' != '{want}'" for i, got, want in mismatches[:5]])
        raise ValueError(f"Plan column_name mismatch(es) for group '{plan.group_name}': {preview}")
    plan.columns = normalized
    return plan


def validate_and_normalize_extraction(
    *,
    extraction: GroupExtractionV2,
    expected_columns: List[str],
    expected_indices: List[int],
    name_policy: Literal["strict", "override"] = "strict",
) -> GroupExtractionV2:
    expected_indices_set = set(expected_indices)
    by_idx: Dict[int, ColumnExtractionV2] = {}
    dupes, extras = [], []
    for item in extraction.extractions:
        idx = item.column_index
        if idx in by_idx:
            dupes.append(idx)
            continue
        if idx not in expected_indices_set:
            extras.append(idx)
            continue
        by_idx[idx] = item
    missing = [i for i in expected_indices if i not in by_idx]
    if dupes or extras or missing:
        logger.warning(
            f"[Soft-validate] Group '{extraction.group_name}': "
            f"dupes={sorted(set(dupes))}, extras={sorted(set(extras))}, missing_count={len(missing)}"
        )
    normalized = []
    mismatches = []
    for idx in expected_indices:
        canonical = expected_columns[idx - 1]
        item = by_idx.get(idx)
        if item is None:
            normalized.append(
                ColumnExtractionV2(
                    column_index=idx,
                    column_name=canonical,
                    value=None,
                    evidence=None,
                    page="",
                    confidence="low",
                )
            )
            continue
        if item.column_name != canonical:
            mismatches.append((idx, item.column_name, canonical))
            if name_policy == "override":
                item.column_name_raw = item.column_name
                item.column_name = canonical
        normalized.append(item)
    if mismatches and name_policy == "strict":
        preview = "; ".join([f"#{i}: '{got}' != '{want}'" for i, got, want in mismatches[:5]])
        logger.warning(f"[Soft-validate] Extraction column_name mismatch(es) for group '{extraction.group_name}': {preview}")
    extraction.extractions = normalized
    return extraction


def find_relevant_chunks(col_plans: List[ColumnExtractionPlanV2], chunks: list) -> list:
    """Find chunks relevant to the plan by source_type and page."""
    relevant = []
    seen_chunk_ids = set()
    for col_plan in col_plans:
        if not col_plan.found_in_pdf or col_plan.page == -1:
            continue
        source_type = (col_plan.source_type or "").lower()
        target_page = col_plan.page
        for i, chunk in enumerate(chunks):
            chunk_id = f"{i}_{chunk.get('type', 'text')}"
            if chunk_id in seen_chunk_ids:
                continue
            chunk_type = (chunk.get("type", "text") or "text").lower()
            chunk_page = chunk.get("page")
            if chunk_type != source_type:
                continue
            page_match = False
            if isinstance(chunk_page, int):
                page_match = chunk_page == target_page
            elif isinstance(chunk_page, str):
                if "-" in chunk_page:
                    try:
                        start_s, end_s = chunk_page.split("-", 1)
                        page_match = int(start_s) <= target_page <= int(end_s)
                    except Exception:
                        pass
                else:
                    try:
                        page_match = int(chunk_page) == target_page
                    except Exception:
                        pass
            if page_match:
                chunk_data = {
                    "chunk_id": i,
                    "type": chunk_type,
                    "page": chunk_page,
                    "content": (chunk.get("content", "") or "")[:1000],
                }
                if chunk_type == "table" and "table_content" in chunk:
                    chunk_data["table_content"] = chunk.get("table_content", "")
                relevant.append(chunk_data)
                seen_chunk_ids.add(chunk_id)
    return relevant


def format_chunks(chunks: list) -> str:
    if not chunks:
        return "No relevant chunks found."
    parts = []
    for c in chunks:
        chunk_id, chunk_type, page = c["chunk_id"], c["type"].upper(), c["page"]
        if chunk_type == "TABLE" and c.get("table_content"):
            content_summary = (c.get("content", "") or "").replace("\n", " ")
            parts.append(
                f"--- Chunk {chunk_id} ({chunk_type} on page {page}) ---\n"
                f"Summary: {content_summary}\n\nStructured Table:\n{c['table_content']}"
            )
        else:
            content = (c.get("content", "") or "")
            parts.append(f"--- Chunk {chunk_id} ({chunk_type} on page {page}) ---\n{content}")
    return "\n\n".join(parts)


def format_columns_for_prompt(
    found_cols: List[ColumnExtractionPlanV2],
    groups_dict: Dict[str, list],
    group_name: str,
) -> str:
    column_definitions = {}
    if group_name in groups_dict:
        for col_info in groups_dict[group_name]:
            column_definitions[col_info["Column Name"]] = col_info["Definition"]
    out = []
    for plan in found_cols:
        definition = column_definitions.get(plan.column_name, "No definition available")
        out.append(
            "\n".join(
                [
                    f'- Column index {plan.column_index}: "{plan.column_name}"',
                    f"  Definition: {definition}",
                    "  Extraction Plan:",
                    f"  {plan.extraction_plan}",
                ]
            )
        )
    return "\n\n".join(out)


# -----------------------------
# Fallback structurers
# -----------------------------


def _structure_with_openai(
    client: Any,
    prompt: str,
    schema: Type[BaseModel],
    model: str = "gpt-4.1",
    max_retries: int = 3,
) -> StructurerResponse:
    from pydantic import ValidationError
    json_schema = schema.model_json_schema()
    system_prompt = f"""You are a JSON formatter. Convert the provided text into valid JSON matching this exact schema:
{json.dumps(json_schema, indent=2)}
Rules: Output ONLY valid JSON, no markdown. Match the schema exactly. Use null for missing values."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content.strip()
            validated_data = schema.model_validate(json.loads(raw_json))
            return StructurerResponse(data=validated_data, success=True, attempts=attempt, error=None)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                return StructurerResponse(data={}, success=False, attempts=attempt, error=str(e))
        except Exception as e:
            return StructurerResponse(data={}, success=False, attempts=attempt, error=str(e))
    return StructurerResponse(data={}, success=False, attempts=max_retries, error="Max retries reached")


def _structure_with_gemini(
    client: Any,
    prompt: str,
    schema: Type[BaseModel],
    model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> StructurerResponse:
    from pydantic import ValidationError
    from google.genai import types as genai_types
    json_schema = schema.model_json_schema()
    system_instruction = f"""You are a JSON formatter. Convert the provided text into valid JSON matching this schema:
{json.dumps(json_schema, indent=2)}
Rules: Output ONLY valid JSON. Match the schema exactly. Use null for missing values."""
    for attempt in range(1, max_retries + 1):
        try:
            config = genai_types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=json_schema,
            )
            response = client.models.generate_content(
                model=model,
                contents=[system_instruction, prompt],
                config=config,
            )
            raw_json = (response.text or "").strip()
            validated_data = schema.model_validate(json.loads(raw_json))
            return StructurerResponse(data=validated_data, success=True, attempts=attempt, error=None)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                return StructurerResponse(data={}, success=False, attempts=attempt, error=str(e))
        except Exception as e:
            return StructurerResponse(data={}, success=False, attempts=attempt, error=str(e))
    return StructurerResponse(data={}, success=False, attempts=max_retries, error="Max retries reached")


# -----------------------------
# Group extraction
# -----------------------------


def _extract_group(
    *,
    group_name: str,
    plan: GroupExtractionPlanV2,
    expected_columns: List[str],
    pdf_handle: Any,
    chunks: list,
    provider: LLMProvider,
    structurer: OutputStructurer,
    output_dir: Path,
    name_policy: Literal["strict", "override"],
    groups_dict: Dict[str, list],
) -> GroupExtractionV2:
    found_cols = [c for c in plan.columns if c.found_in_pdf]
    if not found_cols:
        return GroupExtractionV2(group_name=group_name, extractions=[])

    relevant_chunks = find_relevant_chunks(found_cols, chunks)
    columns_block = format_columns_for_prompt(found_cols, groups_dict, group_name)
    chunks_block = format_chunks(relevant_chunks)

    prompt = f"""You are extracting clinical trial data from a research paper.

TASK: Extract values for the following columns using the provided extraction plans.

COLUMNS TO EXTRACT:
{columns_block}

RELEVANT CHUNKS:
{chunks_block}

GUIDELINES:
- Follow the extraction plan for each column precisely
- If the value cannot be extracted as described in the plan, set value=null
- Provide evidence as an exact quote from the PDF
- Include page number(s) and confidence level (high/medium/low)
- Use the EXACT column names from the list above

Output format (for each column):
Column index <N>: "<Exact Column Name>"
Value: <extracted value or null>
Evidence: "<exact quote>"
Page: <page number>
Confidence: <high/medium/low>
"""

    response = provider.generate_with_pdf(
        prompt=prompt,
        pdf_handle=pdf_handle,
        temperature=0.0,
        max_tokens=8000,
    )
    if not response.success:
        raise ValueError(f"Extraction LLM failed: {response.error}")
    free_form = (response.text or "").strip()

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(group_name)
    raw_path = logs_dir / f"{stem}_{provider.provider}_raw.txt"
    raw_path.write_text(f"INPUT PROMPT:\n{prompt}\n\nOUTPUT:\n{free_form}", encoding="utf-8")

    expected = [(c.column_index, c.column_name) for c in found_cols]
    expected_block = "\n".join([f"{idx}. {name}" for idx, name in expected])
    expected_indices = [idx for idx, _ in expected]

    structuring_prompt = f"""Convert the following free-form extraction into STRICT JSON.

Group name: "{group_name}"

EXPECTED columns (ONLY these; exact names):
{expected_block}

Rules: Output JSON with group_name, extractions (list). Each item: column_index, column_name (EXACT match above), value, evidence, page, confidence.

Free-form extraction:
{free_form}

Return ONLY valid JSON.
"""

    structured = structurer.structure(
        text=structuring_prompt,
        schema=GroupExtractionV2,
        max_retries=5,
        return_dict=False,
    )

    if not structured.success:
        logger.warning("Local structurer failed for '%s', trying fallback", group_name)
        if provider.provider == "openai":
            structured = _structure_with_openai(
                provider.client, structuring_prompt, GroupExtractionV2, model=provider.model, max_retries=3
            )
        elif provider.provider == "gemini":
            structured = _structure_with_gemini(
                provider.client, structuring_prompt, GroupExtractionV2, model=provider.model, max_retries=3
            )
        if not structured.success:
            logger.error("Fallback structurer failed for '%s'", group_name)
            return GroupExtractionV2(
                group_name=group_name,
                extractions=[
                    ColumnExtractionV2(
                        column_index=idx,
                        column_name=expected_columns[idx - 1],
                        value=None,
                        evidence=None,
                        page="",
                        confidence="low",
                    )
                    for idx in expected_indices
                ],
            )

    extraction: GroupExtractionV2 = structured.data
    extraction.group_name = group_name
    extraction = validate_and_normalize_extraction(
        extraction=extraction,
        expected_columns=expected_columns,
        expected_indices=expected_indices,
        name_policy=name_policy,
    )
    return extraction


def _generate_outputs(
    *,
    groups_dict: Dict[str, list],
    plans_by_group: Dict[str, GroupExtractionPlanV2],
    extractions_by_group: Dict[str, GroupExtractionV2],
    output_path: Path,
) -> None:
    """Write extraction_metadata.json and extracted_table.csv to output_path's directory."""
    output_dir = output_path.parent
    all_columns = []
    for group_cols in groups_dict.values():
        for c in group_cols:
            all_columns.append(c["Column Name"])

    csv_row = {col: "" for col in all_columns}
    metadata = {col: {"value": None, "evidence": None, "chunk_id": "not_extracted", "page": None} for col in all_columns}

    for group_name, extraction in extractions_by_group.items():
        expected_cols = expected_columns_for_group(groups_dict, group_name)
        for item in extraction.extractions:
            canonical_name = expected_cols[item.column_index - 1]
            csv_row[canonical_name] = item.value if item.value is not None else ""
            metadata[canonical_name] = {
                "value": item.value,
                "evidence": item.evidence,
                "chunk_id": f"{group_name}::{item.column_index}",
                "page": item.page,
                "column_index": item.column_index,
                "group_name": group_name,
            }
            if item.column_name_raw:
                metadata[canonical_name]["extraction_column_name_raw"] = item.column_name_raw

    for group_name, plan in plans_by_group.items():
        expected_cols = expected_columns_for_group(groups_dict, group_name)
        for p in plan.columns:
            canonical_name = expected_cols[p.column_index - 1]
            meta = metadata.get(canonical_name, {})
            meta.setdefault("plan_found_in_pdf", p.found_in_pdf)
            meta.setdefault("plan_page", p.page)
            meta.setdefault("plan_source_type", p.source_type)
            meta.setdefault("plan_confidence", p.confidence)
            meta.setdefault("plan_extraction_plan", p.extraction_plan)
            meta.setdefault("column_index", p.column_index)
            meta.setdefault("group_name", group_name)
            
            # If column was not found in PDF and value is still None, set to "Not reported"
            if not p.found_in_pdf and meta.get("value") is None:
                meta["value"] = "Not reported"
                csv_row[canonical_name] = "Not reported"
            
            metadata[canonical_name] = meta

    meta_path = output_dir / "extraction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    csv_path = output_dir / "extracted_table.csv"
    pd.DataFrame([csv_row], columns=all_columns).to_csv(csv_path, index=False)


# -----------------------------
# PlanExecutor
# -----------------------------


class PlanExecutor:
    """
    Execute extraction plans: run extraction per group and write extraction_metadata.json.
    """

    def __init__(
        self,
        provider: LLMProvider,
        structurer: OutputStructurer,
        name_policy: Literal["strict", "override"] = "strict",
    ):
        self.provider = provider
        self.structurer = structurer
        self.name_policy = name_policy

    def execute_plans(
        self,
        pdf_path: Path,
        chunks: Any,
        plans: Dict[str, Dict[str, Any]],
        output_path: Path,
        workers: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute all plans in parallel and save extraction_metadata.json (+ CSV).
        plans: {group_name: plan_data} (from PlanGenerator.generate_plans or loaded from planning dir).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        groups_dict = load_definitions()

        chunks_list = chunks if isinstance(chunks, list) else (chunks.get("chunks", []) if isinstance(chunks, dict) else [])
        validated_plans = {}
        for group_name, plan_data in plans.items():
            plan = GroupExtractionPlanV2.model_validate(plan_data)
            expected_cols = expected_columns_for_group(groups_dict, group_name)
            validated_plans[group_name] = validate_and_normalize_plan(
                plan=plan, expected_columns=expected_cols, name_policy=self.name_policy
            )

        pdf_handle = self.provider.upload_pdf(pdf_path)
        extractions_by_group = {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_to_name = {
                    executor.submit(
                        _extract_group,
                        group_name=group_name,
                        plan=plan,
                        expected_columns=expected_columns_for_group(groups_dict, group_name),
                        pdf_handle=pdf_handle,
                        chunks=chunks_list,
                        provider=self.provider,
                        structurer=self.structurer,
                        output_dir=output_path.parent,
                        name_policy=self.name_policy,
                        groups_dict=groups_dict,
                    ): group_name
                    for group_name, plan in validated_plans.items()
                }
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        extractions_by_group[name] = future.result()
                        logger.info("Extracted: %s", name)
                    except Exception as e:
                        logger.error("Group failed %s: %s", name, e)
                        expected_cols = expected_columns_for_group(groups_dict, name)
                        extractions_by_group[name] = GroupExtractionV2(
                            group_name=name,
                            extractions=[
                                ColumnExtractionV2(
                                    column_index=i,
                                    column_name=expected_cols[i - 1],
                                    value=None,
                                    evidence=None,
                                    page="",
                                    confidence="low",
                                )
                                for i in range(1, len(expected_cols) + 1)
                            ],
                        )
            _generate_outputs(
                groups_dict=groups_dict,
                plans_by_group=validated_plans,
                extractions_by_group=extractions_by_group,
                output_path=output_path,
            )
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        finally:
            self.provider.cleanup_pdf(pdf_handle)


def load_plans_from_dir(plans_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load plans from planning directory. plans_all_columns.json at root; *_plan.json in logs/ (or root for backward compat)."""
    plans_dir = Path(plans_dir)
    plans = {}
    compiled = plans_dir / "plans_all_columns.json"
    if compiled.exists():
        data = json.loads(compiled.read_text(encoding="utf-8"))
        for p in data.get("plans", []):
            plans[p["group_name"]] = p
        return plans
    logs_dir = plans_dir / "logs"
    if logs_dir.exists():
        for plan_file in logs_dir.glob("*_plan.json"):
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            plans[data["group_name"]] = data
    if not plans:
        for plan_file in plans_dir.glob("*_plan.json"):
            data = json.loads(plan_file.read_text(encoding="utf-8"))
            plans[data["group_name"]] = data
    return plans
