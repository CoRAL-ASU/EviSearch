"""Column batching shared by every pipeline: groups from the definitions CSV, at most BATCH_MAX_COLUMNS per call."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from src.config.config import BATCH_MAX_COLUMNS


def load_groups() -> Dict[str, List[Dict[str, Any]]]:
    from src.table_definitions.definitions import load_definitions

    return load_definitions()


def definitions_map(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    return {c["Column Name"]: c.get("Definition", "") for cols in groups.values() for c in cols}


def parse_group_names(value: Optional[str]) -> Optional[List[str]]:
    """Comma-separated CLI value -> list of group names (None means all groups)."""
    if not value:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or None


def unknown_groups(groups: Dict[str, List[Dict[str, Any]]], group_names: Optional[Iterable[str]]) -> List[str]:
    """Requested group names that are not Labels in the definitions CSV."""
    return [name for name in (group_names or []) if name not in groups]


def done_columns(columns: Dict[str, Any]) -> Set[str]:
    """Columns that already have a result (entries explicitly marked tried=False are redone)."""
    return {name for name, value in (columns or {}).items() if value is not None and (not isinstance(value, dict) or value.get("tried", True))}


def build_batches(
    groups: Dict[str, List[Dict[str, Any]]],
    group_names: Optional[Iterable[str]] = None,
    done: Optional[Set[str]] = None,
    max_per_batch: int = BATCH_MAX_COLUMNS,
) -> List[List[Dict[str, str]]]:
    """Split groups larger than max_per_batch; pack smaller groups (smallest first) into shared batches."""
    if group_names:
        wanted = set(group_names)
        groups = {name: cols for name, cols in groups.items() if name in wanted}
    done = done or set()

    remaining = []
    for cols in groups.values():
        specs = [
            {"column_name": c["Column Name"], "definition": c.get("Definition", "")}
            for c in cols
            if c["Column Name"] not in done
        ]
        if specs:
            remaining.append(specs)

    batches: List[List[Dict[str, str]]] = []
    for specs in (s for s in remaining if len(s) > max_per_batch):
        batches.extend(specs[i : i + max_per_batch] for i in range(0, len(specs), max_per_batch))

    current: List[Dict[str, str]] = []
    for specs in sorted((s for s in remaining if len(s) <= max_per_batch), key=len):
        if len(current) + len(specs) > max_per_batch and current:
            batches.append(current)
            current = []
        current = current + specs
    if current:
        batches.append(current)
    return batches


def add_usage(total: Dict[str, int], usage: Dict[str, Any]) -> Dict[str, int]:
    for key in ("input_tokens", "output_tokens", "api_calls"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)
    total["total_tokens"] = total.get("input_tokens", 0) + total.get("output_tokens", 0)
    return total


def empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "total_tokens": 0}
