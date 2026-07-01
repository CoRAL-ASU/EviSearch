from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.runtime_paths import RESULTS_ROOT
from src.LLMProvider.provider import LLMProvider
from src.evisearch.knowledge.preferences import load_extraction_preferences

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARSED_MARKDOWN_BASELINES = PROJECT_ROOT / "experiment-scripts" / "baselines_landing_ai_new_results"
VALID_MODALITIES = frozenset({"text", "table", "figure"})
NO_VALUE_PLACEHOLDERS = frozenset({"", "not reported", "not found", "not applicable", "n/a", "na", "-", "--"})


def resolve_parsed_markdown_path(doc_id: str) -> Path:
    """Resolve parsed markdown for a document, preferring current pipeline outputs."""
    for base in (RESULTS_ROOT / doc_id / "chunking", PARSED_MARKDOWN_BASELINES / doc_id):
        path = base / "parsed_markdown.md"
        if path.exists():
            return path
    return RESULTS_ROOT / doc_id / "chunking" / "parsed_markdown.md"


def _is_no_value(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NO_VALUE_PLACEHOLDERS


def _normalize_attribution(raw: Any, found: bool) -> List[Dict[str, Any]]:
    if not found or not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            page = item.get("page")
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        if page is None or page < 1:
            continue
        modality = str(item.get("modality") or item.get("source_type") or "text").lower()
        if modality not in VALID_MODALITIES:
            modality = "text"
        out.append({"page": page, "modality": modality})
    return out


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        parsed = json.loads(raw[start:end])
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_prompt(batch_columns: List[Dict[str, Any]], markdown_text: str, preferences: str) -> str:
    col_blocks = []
    for i, col in enumerate(batch_columns, 1):
        name = col.get("column_name", "")
        definition = col.get("definition", "")
        col_blocks.append(f"\n---\nColumn {i}: {name}\nDefinition: {definition}")

    preferences_block = ""
    if preferences:
        preferences_block = f"\n\nHUMAN EXTRACTION PREFERENCES:\n{preferences}\n"

    return f"""You are extracting clinical trial data from parsed markdown of a research paper.

DOCUMENT MARKDOWN:
{markdown_text}

END DOCUMENT MARKDOWN.

Use ONLY the provided document markdown. Return values for every requested column.
{preferences_block}
COLUMNS TO EXTRACT:
{"".join(col_blocks)}

OUTPUT FORMAT:
Return ONLY valid JSON with this exact shape:
{{
  "columns": {{
    "Column Name": {{
      "value": "extracted value or Not reported",
      "reasoning": "brief evidence-based explanation",
      "found": true,
      "attribution": [{{"page": 1, "modality": "text"}}]
    }}
  }}
}}

Rules:
- Include every requested column exactly by name.
- Use "Not reported" and found=false when the document does not report the value.
- Attribution must use 1-based page numbers from markdown page markers when available.
- Use modality "table" for values taken from tables, "figure" for figures, and "text" for prose.
- Do not include chain-of-thought, markdown fences, prose, or commentary outside JSON.
- Return a single JSON object and nothing else. /no_think
"""


def _normalize_results(parsed: Dict[str, Any], batch_columns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    columns_obj = parsed.get("columns") if isinstance(parsed.get("columns"), dict) else parsed
    if not isinstance(columns_obj, dict):
        columns_obj = {}

    results: Dict[str, Dict[str, Any]] = {}
    for col in batch_columns:
        name = col.get("column_name", "")
        if not name:
            continue
        raw = columns_obj.get(name)
        if not isinstance(raw, dict):
            value = "Not reported" if _is_no_value(raw) else str(raw)
            found = not _is_no_value(value)
            results[name] = {
                "value": value,
                "reasoning": "" if found else "Not reported in provided markdown",
                "found": found,
                "attribution": [],
                "tried": True,
            }
            continue

        value = raw.get("value")
        found = bool(raw.get("found", not _is_no_value(value)))
        if _is_no_value(value):
            value = "Not reported"
            found = False
        results[name] = {
            "value": str(value),
            "reasoning": str(raw.get("reasoning") or ""),
            "found": found,
            "attribution": _normalize_attribution(raw.get("attribution"), found),
            "tried": True,
        }
    return results


def run_markdown_pdf_query_agent(
    doc_id: str,
    batch_columns: List[Dict[str, Any]],
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    preferences: Optional[str] = None,
    raw_response_path: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Run a full-markdown PDF query agent with the agent_extractor output contract."""
    markdown_path = resolve_parsed_markdown_path(doc_id)
    if not markdown_path.exists():
        results = {
            c.get("column_name", ""): {
                "value": "Not reported",
                "reasoning": f"Parsed markdown not found: {markdown_path}",
                "found": False,
                "attribution": [],
                "tried": True,
            }
            for c in batch_columns
            if c.get("column_name")
        }
        return results, {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}

    markdown_text = markdown_path.read_text(encoding="utf-8")
    max_chars = int(os.getenv("MARKDOWN_PDF_QUERY_MAX_CHARS", "0") or "0")
    if max_chars > 0 and len(markdown_text) > max_chars:
        markdown_text = markdown_text[:max_chars] + "\n\n[... markdown truncated by MARKDOWN_PDF_QUERY_MAX_CHARS ...]"

    pref_text = load_extraction_preferences() if preferences is None else preferences
    prompt = _build_prompt(batch_columns, markdown_text, pref_text)

    provider = LLMProvider(
        provider=provider_name or os.getenv("PDF_QUERY_PROVIDER") or "local",
        model=model or os.getenv("PDF_QUERY_MODEL") or None,
    )
    response = provider.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=int(os.getenv("PDF_QUERY_MAX_TOKENS", "16000")),
        response_mime_type="application/json",
    )
    usage = {
        "input_tokens": getattr(response, "input_tokens", 0) or 0,
        "output_tokens": getattr(response, "output_tokens", 0) or 0,
        "api_calls": 1,
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    if raw_response_path is not None:
        try:
            raw_response_path.parent.mkdir(parents=True, exist_ok=True)
            raw_response_path.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response_text": getattr(response, "text", "") or "",
                        "success": bool(getattr(response, "success", False)),
                        "error": getattr(response, "error", None),
                        "usage": usage,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    if not response.success:
        results = {
            c.get("column_name", ""): {
                "value": "Not reported",
                "reasoning": response.error or "Markdown PDF query failed",
                "found": False,
                "attribution": [],
                "tried": True,
            }
            for c in batch_columns
            if c.get("column_name")
        }
        return results, usage

    try:
        parsed = _extract_json_object(response.text)
    except Exception as exc:
        results = {
            c.get("column_name", ""): {
                "value": "Not reported",
                "reasoning": f"JSON parse error: {exc}",
                "found": False,
                "attribution": [],
                "tried": True,
            }
            for c in batch_columns
            if c.get("column_name")
        }
        return results, usage

    return _normalize_results(parsed, batch_columns), usage
