"""Shape of a per-column result shared by every method: {value, reasoning, found, attribution, tried}.

attribution is a list of {page, modality, evidence}: the 1-based page the value came from and, when the model gave it,
the text on that page that supports the value (a quoted sentence, a table row with its column header, or a figure
label with what was read from it). The reconciler has a verifier check each (value, page, evidence) claim."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

NOT_REPORTED = "Not reported"
MODALITIES = ("text", "table", "figure")
NO_VALUE_PLACEHOLDERS = frozenset({"", "not reported", "not found", "not applicable", "n/a", "na", "-", "--", "—"})
EVIDENCE_FORMAT = "page_evidence_v1"  # arm outputs carry evidence text per attribution; part of the run settings
EVIDENCE_MAX_CHARS = 400
EVIDENCE_DESCRIPTION = (
    "The text on this page that supports the value, copied as printed: the sentence, or the table row label with the "
    "column header and cell, or the figure label and what you read from it"
)


def is_no_value(value: Any) -> bool:
    return value is None or str(value).strip().lower() in NO_VALUE_PLACEHOLDERS


def normalize_attribution(raw: Any, found: bool = True) -> List[Dict[str, Any]]:
    """[{page, modality[, evidence]}] with 1-based pages; accepts `source_type` for modality and drops invalid entries."""
    if not found or not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        if page < 1:
            continue
        modality = str(item.get("modality") or item.get("source_type") or "text").lower()
        entry: Dict[str, Any] = {"page": page, "modality": modality if modality in MODALITIES else "text"}
        evidence = str(item.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence[:EVIDENCE_MAX_CHARS]
        out.append(entry)
    return out


def column_result(value: Any, reasoning: Any = "", found: Any = None, attribution: Any = None) -> Dict[str, Any]:
    no_value = is_no_value(value)
    found = (not no_value) if found is None else (bool(found) and not no_value)
    return {
        "value": NOT_REPORTED if no_value else str(value),
        "reasoning": str(reasoning or ""),
        "found": found,
        "attribution": normalize_attribution(attribution, found),
        "tried": True,
    }


def not_reported(reason: str) -> Dict[str, Any]:
    return column_result(NOT_REPORTED, reason, found=False)


def column_names(batch_columns: Iterable[Dict[str, Any]]) -> List[str]:
    return [c.get("column_name", "") for c in batch_columns if c.get("column_name")]


def fill_missing(results: Dict[str, Dict[str, Any]], names: Iterable[str], reason: str) -> Dict[str, Dict[str, Any]]:
    for name in names:
        results.setdefault(name, not_reported(reason))
    return results


def parse_column_entries(data: Any, names: Iterable[str], list_key: str = "columns") -> Dict[str, Dict[str, Any]]:
    """Read extraction output in any of the shapes models produce:
    {"columns": [{"column": ..., "value": ...}]}, {"columns": {name: {...}}} or {name: {...}}."""
    wanted = set(names)
    entries = data.get(list_key, data) if isinstance(data, dict) else data
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and item.get("column") in wanted:
                out[item["column"]] = column_result(item.get("value"), item.get("reasoning"), item.get("found"), item.get("attribution"))
    elif isinstance(entries, dict):
        for name in wanted:
            raw = entries.get(name)
            if isinstance(raw, dict):
                out[name] = column_result(raw.get("value"), raw.get("reasoning"), raw.get("found"), raw.get("attribution"))
            elif raw is not None:
                out[name] = column_result(raw)
    return out


def extraction_items_schema(names: List[str]) -> Dict[str, Any]:
    """JSON schema for a list of column extractions, restricted to the requested column names."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": list(names)},
                "value": {"type": "string", "description": "Extracted value, or 'Not reported'"},
                "reasoning": {"type": "string", "description": "Where the value was found and how it was derived"},
                "found": {"type": "boolean"},
                "attribution": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer", "description": "1-based page number"},
                            "modality": {"type": "string", "enum": list(MODALITIES)},
                            "evidence": {"type": "string", "description": EVIDENCE_DESCRIPTION},
                        },
                        "required": ["page", "modality", "evidence"],
                    },
                },
            },
            "required": ["column", "value", "reasoning", "found", "attribution"],
        },
    }


def count_found(columns: Dict[str, Any]) -> int:
    return sum(1 for v in columns.values() if isinstance(v, dict) and v.get("found"))


def optional_text(value: Optional[str]) -> str:
    return (value or "").strip()
