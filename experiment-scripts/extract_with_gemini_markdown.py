#!/usr/bin/env python3
"""
extract_with_gemini_markdown.py

Extract column values from a trial document using:
  1. Markdown built from landing_ai_parse_output.json (tables converted to markdown)
  2. Gemini with JSON schema

Output: agent-extractor format {doc_id, columns: {col: {value, reasoning, found, tried}}}

Usage:
  python experiment-scripts/extract_with_gemini_markdown.py "NCT02799602_Hussain_ARASENS_JCO'23" --groups "Region - N (%),Race - N (%)"
  python experiment-scripts/extract_with_gemini_markdown.py "NCT02799602_Hussain_ARASENS_JCO'23" --groups "Region - N (%)"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(script_dir))

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from src.table_definitions.definitions import load_definitions
from web.highlight_service import load_landing_ai_parse, _chunk_text, _landing_type_to_pipeline
from web.table_utils import html_table_to_markdown

# Reuse schema/provider from baseline
from baseline_landing_ai_w_gemini import (
    build_json_schema_for_group,
    GeminiMarkdownProvider,
)

load_dotenv()

PIPELINE_RESULTS = repo_root / "new_pipeline_outputs" / "results"
MAX_MARKDOWN_CHARS = 100_000
NO_VALUE_PLACEHOLDERS = frozenset({"", "not reported", "not found", "not applicable", "n/a", "na", "—", "-"})


def _is_no_value(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in NO_VALUE_PLACEHOLDERS:
        return True
    return False


SCOPE_AGGREGATION_INSTRUCTIONS = """
SCOPE: Match the query exactly. If the column asks for "Treatment" arm, report only treatment-arm values. If it asks for "Control" arm, report only control-arm values. Do not mix arms or substitute subgroup values when the query specifies an arm.

AGGREGATION: If the paper reports values split by subgroups (e.g., high-volume vs low-volume, synchronous vs metachronous) that together make up the whole population for that arm, sum the subgroup values to obtain the whole-arm value when the query asks for the whole arm. Check table and figure captions to confirm whether reported values are for the whole population or subgroups.

REASONING: In your reasoning for each column, explicitly state where you looked (e.g., Table 2, Figure 1, Methods section on page 3). If you use a table, explicitly explain whether the table's structure matches the scope of the query: e.g., "Table 2 reports demographics by Treatment and Control arms—matches query scope" or "Table 3 reports by high-volume vs low-volume; I summed the Treatment-arm subgroups to get the whole-arm value." If the table does not match the query scope, explain why you used it or why you could not extract. Be detailed in your reasoning.
"""


def build_prompt(label: str, items: list) -> str:
    """Build extraction prompt with scope/aggregation instructions."""
    lines = [f"Extract values for the following columns (Label: {label}):\n"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. {item['column']}: {item['definition']}\n"
            "   If not present, use value: 'not found' and reasoning: 'not found'."
        )
    lines.append("\n" + "=" * 60)
    lines.append(SCOPE_AGGREGATION_INSTRUCTIONS.strip())
    lines.append(
        "Output a single JSON object. For each column provide "
        "'value' (the extracted value or 'not found') and "
        "'reasoning' (where you looked, whether the source matches the query scope, and how you derived the value—or 'not found')."
    )
    lines.append("=" * 60)
    return "\n".join(lines)


def build_markdown_from_landing_ai(doc_id: str, strip_figure_placeholders: bool = True) -> str:
    """
    Build full markdown from landing_ai_parse_output.json.
    - Sort chunks by (page, box.top)
    - Text/figure: use markdown as-is (optionally strip <::...::>)
    - Table: convert HTML to markdown via html_table_to_markdown
    """
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        raise FileNotFoundError(
            f"landing_ai_parse_output.json not found for {doc_id}. "
            f"Expected: {PIPELINE_RESULTS / doc_id / 'chunking' / 'landing_ai_parse_output.json'}"
        )

    chunks = parse_data.get("chunks") or []
    if not chunks:
        raise ValueError(f"No chunks in landing_ai_parse_output.json for {doc_id}")

    def sort_key(c):
        g = c.get("grounding") or {}
        page = int(g.get("page", 0))
        box = g.get("box") or {}
        top = float(box.get("top", 0))
        return (page, top)

    sorted_chunks = sorted(chunks, key=sort_key)
    parts = []

    for chunk in sorted_chunks:
        raw = _chunk_text(chunk)
        chunk_type = _landing_type_to_pipeline(chunk.get("type", "text"))

        if chunk_type == "table" and "<table" in raw.lower():
            md = html_table_to_markdown(raw)
        else:
            md = raw
            if strip_figure_placeholders:
                md = re.sub(r"<::[^>]*::>", "", md, flags=re.DOTALL)

        if md.strip():
            parts.append(md.strip())

    full = "\n\n".join(parts)
    if len(full) > MAX_MARKDOWN_CHARS:
        full = full[:MAX_MARKDOWN_CHARS] + "\n\n[... truncated ...]"
    return full


def extract_groups(
    provider: GeminiMarkdownProvider,
    markdown_text: str,
    label_groups: OrderedDict,
) -> dict:
    """Extract one group per API call. Returns {col: {value, reasoning, found, tried}}."""
    out = {}
    total = len(label_groups)
    for idx, (label, items) in enumerate(label_groups.items(), 1):
        columns = [it["Column Name"] for it in items]
        print(f"  [{idx}/{total}] {label} ({len(columns)} columns)...")
        prompt = build_prompt(label, [{"column": it["Column Name"], "definition": it["Definition"]} for it in items])
        schema = build_json_schema_for_group(columns)
        text, _, _ = provider.query_markdown_with_schema(
            prompt=prompt,
            markdown_text=markdown_text,
            json_schema=schema,
        )
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = {"_error": text or "empty response"}

        for col in columns:
            cell = parsed.get(col) if "_error" not in parsed else None
            if isinstance(cell, dict):
                val = cell.get("value")
                reasoning = (cell.get("reasoning") or "").strip()
                found = not _is_no_value(val)
                out[col] = {
                    "value": val if val is not None and str(val).strip() else "Not reported",
                    "reasoning": reasoning,
                    "found": found,
                    "tried": True,
                }
            else:
                out[col] = {
                    "value": "Not reported",
                    "reasoning": parsed.get("_error", "Extraction error") if "_error" in parsed else "",
                    "found": False,
                    "tried": True,
                }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract column values from trial document using markdown built from landing_ai_parse_output + Gemini"
    )
    parser.add_argument("doc_id", help="Document id, e.g. NCT02799602_Hussain_ARASENS_JCO'23")
    parser.add_argument(
        "--groups",
        type=str,
        required=True,
        help="Comma-separated group Labels to extract (e.g. \"Region - N (%%),Race - N (%%)\")",
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: temp/extract_<doc_id>_<timestamp>.json)",
    )
    args = parser.parse_args()

    if not GENAI_AVAILABLE:
        raise RuntimeError("google.genai required. pip install google-genai")

    doc_id = args.doc_id.strip()
    if not doc_id:
        sys.exit("doc_id cannot be empty")

    all_groups = load_definitions()
    labels = [g.strip() for g in args.groups.split(",") if g.strip()]
    label_groups = OrderedDict((k, v) for k, v in all_groups.items() if k in labels)
    missing = set(labels) - set(label_groups)
    if missing:
        print(f"Warning: unknown groups {missing}; available: {list(all_groups.keys())[:10]}...")
    if not label_groups:
        sys.exit("No groups to extract")

    print(f"Building markdown from landing_ai_parse_output.json for {doc_id}...")
    markdown_text = build_markdown_from_landing_ai(doc_id)
    print(f"Markdown length: {len(markdown_text)} chars")

    provider = GeminiMarkdownProvider(args.model)
    print(f"Extracting {len(label_groups)} groups with {args.model}...")
    columns = extract_groups(provider, markdown_text, label_groups)

    result = {"doc_id": doc_id, "columns": columns}

    out_path = args.output
    if out_path is None:
        import time
        temp_dir = repo_root / "temp"
        temp_dir.mkdir(exist_ok=True)
        out_path = temp_dir / f"extract_{doc_id.replace('/', '_')}_{int(time.time())}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
