#!/usr/bin/env python3
"""
Minimal agent extractor: database + 2 tools (get_status, extract_and_write).

Tools:
  1. get_status() — see which columns are filled, which remaining, by group
  2. extract_and_write(group_names) — send PDF to Gemini for those groups, write results to DB

Usage:
  python experiment-scripts/agent_extractor.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
  python experiment-scripts/agent_extractor.py "NCT00268476_Attard_STAMPEDE_Lancet'23" --max-turns 5 --groups-only "Add-on Treatment"
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)
sys.path.insert(0, str(PROJECT_ROOT))

from src.table_definitions.definitions import load_definitions
from src.LLMProvider.provider import LLMProvider

RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"
DATASET_DIR = PROJECT_ROOT / "dataset"


def resolve_pdf_path(doc_id: str) -> Optional[Path]:
    """Resolve PDF path for doc_id."""
    for base in (RESULTS_ROOT / doc_id, DATASET_DIR):
        if not base.exists():
            continue
        for p in base.glob("**/*.pdf"):
            if p.stem.replace("'", "'") == doc_id.replace("'", "'"):
                return p
        for p in base.glob("*.pdf"):
            if p.stem in doc_id or doc_id in p.stem:
                return p
    return None


# -----------------------------------------------------------------------------
# Database (column_name -> value dict or None)
# -----------------------------------------------------------------------------

# Values that mean "tried but no value" — never retry these
NO_VALUE_PLACEHOLDERS = frozenset(
    {"", "not reported", "not found", "not applicable", "n/a", "na", "—", "-"}
)


def _is_no_value(val: Any) -> bool:
    """True if value indicates we tried but found nothing."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in NO_VALUE_PLACEHOLDERS:
        return True
    return False


def _expand_attribution_map_to_per_column(
    attribution_list: List[Dict[str, Any]], column_names: set
) -> Dict[str, List[Dict[str, Any]]]:
    """Inverted map (source -> columns) -> per-column attribution. Returns {col: [sources]}."""
    col_to_sources: Dict[str, List[Dict[str, Any]]] = {c: [] for c in column_names}
    for src in attribution_list or []:
        if not isinstance(src, dict):
            continue
        cols = src.get("columns")
        if not isinstance(cols, list):
            continue
        # Build source without "columns" key
        src_copy = {k: v for k, v in src.items() if k != "columns"}
        for col in cols:
            if col in col_to_sources:
                col_to_sources[col].append(src_copy)
    return col_to_sources


def _build_inverted_attribution(db: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-column db -> inverted attribution map (source -> columns). Deduplicates sources."""
    seen: Dict[tuple, int] = {}
    out: List[Dict[str, Any]] = []
    for col_name, val in (db or {}).items():
        if not isinstance(val, dict):
            continue
        for src in val.get("attribution", []):
            if not isinstance(src, dict):
                continue
            # Canonical key for dedup
            key = (
                src.get("page"),
                src.get("source_type"),
                src.get("snippet") or src.get("table_number") or src.get("figure_number") or "",
            )
            if key not in seen:
                seen[key] = len(out)
                out.append({**src, "columns": []})
            out[seen[key]]["columns"].append(col_name)
    return out


def _normalize_attribution(raw: List[Any], found: bool) -> List[Dict[str, Any]]:
    """Validate and normalize attribution array. When found=false, return []."""
    if not found:
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        st = str(item.get("source_type") or "").lower()
        if st not in ("text", "table", "figure"):
            continue
        page = item.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        if page is None or page < 1:
            continue
        src: Dict[str, Any] = {"source_type": st, "page": page}
        if st == "text":
            snippet = (item.get("snippet") or "").strip()
            if len(snippet) >= 10:
                src["snippet"] = snippet
            else:
                continue
        elif st == "table":
            src["table_number"] = (item.get("table_number") or "").strip()
            src["caption"] = (item.get("caption") or "").strip()
            if not src["table_number"]:
                continue
        elif st == "figure":
            src["figure_number"] = (item.get("figure_number") or "").strip()
            src["caption"] = (item.get("caption") or "").strip()
            if not src["figure_number"]:
                continue
        out.append(src)
    return out


def load_previous_extraction(doc_id: str) -> Optional[Dict[str, Any]]:
    """Load extraction_results.json. Expects {columns, attribution} format. Expands attribution map to per-column."""
    path = RESULTS_ROOT / doc_id / "agent_extractor" / "extraction_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cols = data.get("columns", {})
        if not isinstance(cols, dict):
            return None
        attribution_map = data.get("attribution")
        if isinstance(attribution_map, list):
            col_names = set(cols)
            col_to_attr = _expand_attribution_map_to_per_column(attribution_map, col_names)
            for col_name, col_data in cols.items():
                if isinstance(col_data, dict):
                    found = bool(col_data.get("found", True))
                    raw_attr = col_to_attr.get(col_name, [])
                    cols[col_name] = {
                        **col_data,
                        "attribution": _normalize_attribution(raw_attr, found),
                    }
        return cols
    except Exception as e:
        log.warning("Could not load previous extraction from %s: %s", path, e)
        return None


def init_database(
    groups: Dict[str, List[Dict]],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict]]:
    """Initialize database. If previous extraction exists, seed with those values."""
    db = {}
    for _group, cols in groups.items():
        for c in cols:
            name = c["Column Name"]
            db[name] = None  # None = not yet tried

    if previous:
        for col_name, val in previous.items():
            if col_name not in db:
                continue
            if val is None:
                continue
            if isinstance(val, dict):
                v = val.get("value")
                reasoning = (val.get("reasoning") or "").strip()
                found = bool(val.get("found", True))
                raw_attr = val.get("attribution", [])
                if not isinstance(raw_attr, list):
                    raw_attr = []
                tried = val.get("tried")
                if tried is None:
                    tried = bool(reasoning) or found
                if _is_no_value(v):
                    db[col_name] = {"value": "Not reported", "reasoning": reasoning, "found": False, "attribution": [], "tried": tried}
                else:
                    attribution = _normalize_attribution(raw_attr, found)
                    db[col_name] = {"value": str(v), "reasoning": reasoning, "found": found, "attribution": attribution, "tried": tried}
            else:
                if _is_no_value(val):
                    db[col_name] = {"value": "Not reported", "reasoning": "", "found": False, "attribution": [], "tried": False}
                else:
                    db[col_name] = {"value": str(val), "reasoning": "", "found": True, "attribution": [], "tried": True}

    return db


MAX_COLUMNS_BATCH = 15


def get_status(db: Dict[str, Optional[Dict]], groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Tool 1: Return minimal status. suggested_groups has sum(columns) <= 15.
    If a group has > 15 columns, return only that group first.
    Only returns suggested_groups + counts — no full by_group/remaining_groups to avoid overwhelming the model."""
    # Filled = has value from extract_and_write (tried=True). Never-tried (tried=False) count as remaining.
    filled = [k for k, v in db.items() if v is not None and v.get("tried", True)]
    total = len(db)
    # (group_name, remaining_count, columns_with_definitions)
    remaining_with_counts: List[tuple] = []

    for group_name, cols in groups.items():
        col_names = [c["Column Name"] for c in cols]
        remaining_in_g = [n for n in col_names if db.get(n) is None or db.get(n, {}).get("tried") is False]
        if remaining_in_g:
            col_specs = [
                {"name": c["Column Name"], "definition": c.get("Definition", "")}
                for c in cols
                if db.get(c["Column Name"]) is None or db.get(c["Column Name"], {}).get("tried") is False
            ]
            remaining_with_counts.append((group_name, len(remaining_in_g), col_specs))

    # Build suggested_groups: sum(columns) <= 15. If any group > 15 cols, return only that first.
    suggested = []
    over_limit = [(g, n, c) for g, n, c in remaining_with_counts if n > MAX_COLUMNS_BATCH]
    if over_limit:
        g, n, col_specs = over_limit[0]
        suggested.append({"name": g, "remaining_columns": n, "columns": col_specs})
    else:
        remaining_with_counts.sort(key=lambda x: x[1])
        col_sum = 0
        for g, n, col_specs in remaining_with_counts:
            if col_sum + n <= MAX_COLUMNS_BATCH:
                suggested.append({"name": g, "remaining_columns": n, "columns": col_specs})
                col_sum += n
            else:
                break

    return {
        "filled_count": len(filled),
        "total_count": total,
        "remaining_count": total - len(filled),
        "suggested_groups": suggested,
    }


def extract_and_write(
    db: Dict[str, Optional[Dict]],
    groups: Dict[str, List[Dict]],
    group_names: List[str],
    pdf_handle: Any,
    provider: LLMProvider,
) -> Dict[str, Any]:
    """Tool 2: Extract values for given groups from PDF, write to DB."""
    col_specs = []
    definitions_map = {}
    for gname in group_names:
        if gname not in groups:
            continue
        for i, c in enumerate(groups[gname], 1):
            name = c["Column Name"]
            defn = c.get("Definition", "")
            definitions_map[name] = defn
            col_specs.append(f"""
Column {i}: {name}
  Definition: {defn}
""")

    if not col_specs:
        log.warning("extract_and_write: no valid groups in %s", group_names)
        return {"extracted": {}, "written": [], "errors": ["No valid groups"]}

    log.debug("Extracting %d columns from groups %s", len(col_specs), group_names)
    prompt = f"""You are extracting clinical trial data from this research paper PDF.

TASK: Extract values for the following columns. Use the Definition to know what to extract.

COLUMNS TO EXTRACT:
{"".join(col_specs)}

GUIDELINES:
- For each column: extract the value as defined, or "Not reported" if not in the document
- Prefer exact quotes/numbers from tables and text
- For N (%) columns: include both count and percentage when reported
- reasoning: Brief explanation of what was extracted and where, or (when not found) why the value is absent.

OUTPUT FORMAT (two parts — values and attribution map):
1. "columns": object with column names as keys. Each value: {{"value": "...", "reasoning": "...", "found": true|false}}
2. "attribution": array of sources. Each source lists ALL columns that cite it (no repetition).
   - text: {{"page": N, "source_type": "text", "snippet": "exact phrase 20–150 chars", "columns": ["Col1", "Col2"]}}
   - table: {{"page": N, "source_type": "table", "table_number": "Table 2", "caption": "optional", "columns": ["Col1"]}}
   - figure: {{"page": N, "source_type": "figure", "figure_number": "Figure 1", "caption": "optional", "columns": ["Col1"]}}
   - When found=false for a column, do NOT include it in any attribution source.
   - When multiple columns cite the same source (e.g. same table), list that source ONCE with all columns in "columns".

Return: {{"columns": {{"ColName": {{"value": "...", "reasoning": "...", "found": true|false}}, ...}}, "attribution": [{{"page": ..., "source_type": "...", "columns": [...]}}]}}

Example:
{{"columns": {{"Add-on Treatment": {{"value": "abiraterone acetate", "reasoning": "Methods", "found": true}}, "Median OS (mo) | Overall | Treatment": {{"value": "76.6", "reasoning": "Table 2", "found": true}}, "Quality of Life Scale": {{"value": "Not reported", "reasoning": "Not in document", "found": false}}}}, "attribution": [{{"page": 1, "source_type": "text", "snippet": "abiraterone acetate plus prednisolone", "columns": ["Add-on Treatment"]}}, {{"page": 6, "source_type": "table", "table_number": "Table 2", "caption": "Overall survival", "columns": ["Median OS (mo) | Overall | Treatment", "PFS Rate (%)"]}}]}}

Output ONLY valid JSON. No markdown, no explanation."""

    max_retries = 2
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = provider.generate_with_pdf(
                prompt=prompt,
                pdf_handle=pdf_handle,
                temperature=0.0,
                max_tokens=16000,
                response_mime_type="application/json",
            )
            break
        except Exception as e:
            err_str = str(e).lower()
            if "504" in err_str or "deadline_exceeded" in err_str:
                if attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    log.warning("extract_and_write 504 timeout (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_retries + 1, wait, e)
                    time.sleep(wait)
                else:
                    log.warning("extract_and_write 504 after %d attempts, skipping batch: %s", max_retries + 1, e)
                    return {"extracted": {}, "written": [], "errors": [f"504 DEADLINE_EXCEEDED - skipped (try again with --resume)"]}
            else:
                raise

    written = []
    errors = []
    if not response.success:
        log.error("extract_and_write LLM failed: %s", response.error)
        return {"extracted": {}, "written": [], "errors": [response.error or "LLM failed"]}

    raw = (response.text or "").strip()
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("extract_and_write JSON parse error: %s", e)
        return {"extracted": {}, "written": [], "errors": [f"JSON parse: {e}"]}

    # Required format: {columns: {...}, attribution: [...]} — modality->columns map
    columns_data = parsed.get("columns") if isinstance(parsed.get("columns"), dict) else None
    attribution_map = parsed.get("attribution") if isinstance(parsed.get("attribution"), list) else None

    if not columns_data:
        log.warning("extract_and_write: missing 'columns' in response, expected {columns, attribution} format")
        return {"extracted": {}, "written": [], "errors": ["Invalid format: missing 'columns'"]}

    col_names = {c for c in columns_data if c in db}
    col_to_attr = _expand_attribution_map_to_per_column(attribution_map or [], col_names)

    for col_name, data in columns_data.items():
        if col_name not in db:
            continue
        if isinstance(data, dict):
            val = data.get("value")
            reasoning = (data.get("reasoning") or "").strip()
            found = bool(data.get("found", True))
            raw_attr = col_to_attr.get(col_name, [])
            if _is_no_value(val):
                db[col_name] = {"value": "Not reported", "reasoning": reasoning, "found": False, "attribution": [], "tried": True}
            else:
                attribution = _normalize_attribution(raw_attr, found)
                db[col_name] = {"value": str(val), "reasoning": reasoning, "found": found, "attribution": attribution, "tried": True}
            written.append(col_name)
        elif isinstance(data, str):
            if _is_no_value(data):
                db[col_name] = {"value": "Not reported", "reasoning": "", "found": False, "attribution": [], "tried": True}
            else:
                db[col_name] = {"value": data, "reasoning": "", "found": True, "attribution": [], "tried": True}
            written.append(col_name)

    return {
        "extracted": {g: len([c for c in groups.get(g, []) if c["Column Name"] in written]) for g in group_names},
        "written": written,
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------

TOOL_DESCRIPTIONS = """You have two tools. Respond with ONLY a JSON object, no other text.

1. get_status — see what's filled and what remains. Returns suggested_groups (sum of columns <= 15).
   {"action": "get_status"}

2. extract_and_write — extract values for group(s) from the PDF, write to database.
   Use ALL suggested_groups from get_status in one call. Do many groups at a time.
   {"action": "extract_and_write", "group_names": ["Add-on Treatment", "Control Arm", "Primary Endpoint(s)"]}

3. done — when all columns are filled or you've finished
   {"action": "done"}
"""


def parse_tool_call(text: str) -> Optional[Dict]:
    """Parse LLM output for tool call JSON."""
    text = (text or "").strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
    match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text)
    if match:
        text = match.group(0)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("action"):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _save_partial_extraction(doc_id: str, db: Dict[str, Any], messages: List[dict], turns: int) -> None:
    """Save partial extraction_results.json (inverted attribution format)."""
    out_dir = RESULTS_ROOT / doc_id / "agent_extractor"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_out = {}
    for k, v in (db or {}).items():
        if isinstance(v, dict):
            db_out[k] = {"value": v.get("value"), "reasoning": v.get("reasoning", ""), "found": v.get("found", True), "tried": v.get("tried", True)}
        else:
            db_out[k] = {"value": v, "reasoning": "", "found": v is not None, "tried": False}
    attribution_inverted = _build_inverted_attribution(db)
    extraction_data = {"doc_id": doc_id, "columns": db_out, "attribution": attribution_inverted, "turns": turns}
    extraction_path = out_dir / "extraction_results.json"
    extraction_path.write_text(json.dumps(extraction_data, indent=2), encoding="utf-8")
    log.debug("Saved partial: %d columns, %s", sum(1 for v in db.values() if v is not None), extraction_path)


def run_agent(
    doc_id: str,
    max_turns: int = 50,
    groups_filter: Optional[List[str]] = None,
    resume: bool = True,
    skip_if_done: bool = False,
) -> Dict[str, Any]:
    """Run agent loop: init DB (optionally from previous run), loop until done or max_turns."""
    log.info("Starting agent for doc_id=%s max_turns=%d resume=%s", doc_id, max_turns, resume)
    pdf_path = resolve_pdf_path(doc_id)
    if not pdf_path or not pdf_path.exists():
        log.error("PDF not found: %s", doc_id)
        return {"error": f"PDF not found for {doc_id}", "messages": []}

    log.info("PDF resolved: %s", pdf_path)
    groups = load_definitions()
    if groups_filter:
        groups = {k: v for k, v in groups.items() if k in groups_filter}
        log.info("Filtered to groups: %s", groups_filter)

    previous = load_previous_extraction(doc_id) if resume else None
    if previous:
        log.info("Loaded previous extraction: %d columns with values", sum(1 for v in previous.values() if v is not None and (not isinstance(v, str) or v.strip())))

    db = init_database(groups, previous=previous)
    filled_init = sum(1 for v in db.values() if v is not None)
    log.info("DB initialized: %d columns (%d already filled)", len(db), filled_init)

    if skip_if_done and filled_init == len(db):
        log.info("All columns already filled (skip-if-done), exiting")
        return {"database": db, "turns": 0, "messages": [], "skipped": True}
    provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
    pdf_handle = provider.upload_pdf(pdf_path)
    log.info("PDF uploaded to provider")

    system = f"""You are extracting clinical trial data from a research paper PDF into a structured schema.

OBJECTIVE: Fill all columns (or mark "Not reported" when absent). The database starts empty.

{TOOL_DESCRIPTIONS}

STRATEGY: Call get_status first. Then extract_and_write for ALL suggested_groups in one call — do many groups at a time. Repeat until done."""

    messages = [
        {"role": "user", "content": f"Extract all clinical trial data from the PDF for doc {doc_id}. Start by checking status."}
    ]

    for turn in range(max_turns):
        conv_text = "CONVERSATION:\n" + "\n".join(
            f"{m['role']}: {m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])}"
            for m in messages[-6:]
        )
        prompt = f"{system}\n\n{conv_text}\n\nassistant: Output ONLY a JSON object with action and (if extract_and_write) group_names. No other text."

        # Agent LLM: Use JSON mode + 4096 tokens. Truncation happened when model
        # hallucinated 80+ group names → output exceeded max_tokens → JSON cut mid-stream → parse fail.
        response = provider.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=4096,
            response_mime_type="application/json",
        )

        if not response.success:
            log.error("LLM failed turn %d: %s", turn + 1, response.error)
            return {"error": response.error, "database": db, "messages": messages}

        tc = parse_tool_call(response.text or "")
        if not tc:
            log.warning("Turn %d: could not parse tool call, raw: %.200s", turn + 1, response.text)
            messages.append({"role": "assistant", "content": response.text or ""})
            continue

        action = tc.get("action", "")
        log.info("Turn %d: action=%s", turn + 1, action)
        if action == "done":
            log.info("Agent signalled done")
            _save_partial_extraction(doc_id, db, messages, turn + 1)
            break
        if action == "get_status":
            result = get_status(db, groups)
            log.debug("get_status: filled=%d/%d suggested_groups=%s", result["filled_count"], result["total_count"], [g["name"] for g in result.get("suggested_groups", [])])
            messages.append({"role": "assistant", "content": json.dumps(tc)})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
            if result.get("remaining_count", 1) == 0:
                log.info("All columns filled (remaining_count=0), stopping")
                _save_partial_extraction(doc_id, db, messages, turn + 1)
                break
        elif action == "extract_and_write":
            gnames = tc.get("group_names", [])
            if isinstance(gnames, str):
                gnames = [gnames]
            log.info("extract_and_write groups: %s", gnames)
            result = extract_and_write(db, groups, gnames, pdf_handle, provider)
            written = result.get("written", [])
            n_written = len(written)
            print(f"  → Written: {n_written} columns (total filled: {sum(1 for v in db.values() if v is not None)}/{len(db)})", file=sys.stderr)
            log.info("extract_and_write wrote %d columns: %s", n_written, written[:10] if len(written) > 10 else written)
            if result.get("errors"):
                log.warning("extract_and_write errors: %s", result["errors"])
            messages.append({"role": "assistant", "content": json.dumps(tc)})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}"})
        else:
            log.warning("Unknown action: %s", action)
            messages.append({"role": "user", "content": f"Unknown action: {action}"})

        _save_partial_extraction(doc_id, db, messages, turn + 1)

    filled = sum(1 for v in db.values() if v is not None)
    log.info("Finished: %d turns, %d/%d columns filled", turn + 1, filled, len(db))
    return {"database": db, "turns": turn + 1, "messages": messages}


def _make_static_html(data: Dict[str, Any]) -> str:
    """Generate self-contained HTML with conversation data inlined."""
    json_escaped = json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e")
    doc_id = (data.get("doc_id") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Conversation — {doc_id}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: ui-monospace, monospace; background: #0f172a; color: #e2e8f0; margin: 0; padding: 16px; line-height: 1.5; }}
        pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 13px; }}
        .msg {{ margin-bottom: 12px; padding: 12px; border-radius: 8px; border: 1px solid; }}
        .user {{ background: #33415566; border-color: #475569; }}
        .assistant {{ background: #0e4a5e33; border-color: #0891b266; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; background: #0891b255; margin-bottom: 6px; }}
        .tool-call {{ font-size: 14px; font-weight: 600; color: #22d3ee; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #334155; }}
        .role {{ font-size: 11px; color: #94a3b8; margin-bottom: 4px; }}
        h1 {{ font-size: 18px; margin-bottom: 8px; color: #22d3ee; }}
        .meta {{ font-size: 12px; color: #64748b; margin-bottom: 16px; }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
        .data-table th, .data-table td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #334155; }}
        .data-table th {{ color: #64748b; font-weight: 500; }}
        .data-table td {{ color: #e2e8f0; max-width: 400px; overflow: hidden; text-overflow: ellipsis; }}
    </style>
</head>
<body>
    <h1>Agent Extractor — {doc_id}</h1>
    <div class="meta">{data.get("turns", 0)} turns · {len([k for k, v in (data.get("database") or {}).items() if v])}/{len(data.get("database") or {})} columns filled</div>
    <details open class="meta" style="margin-bottom: 20px;">
        <summary style="cursor: pointer; font-weight: 600; color: #22d3ee;">▼ Extracted Data</summary>
        <table class="data-table" id="data-table"><tbody id="data-tbody"></tbody></table>
    </details>
    <h3 style="font-size: 14px; color: #94a3b8; margin-bottom: 8px;">Conversation</h3>
    <div id="messages"></div>
    <script>
        const data = {json_escaped};
        const db = data.database || {{}};
        const tbody = document.getElementById("data-tbody");
        Object.entries(db).forEach(([col, val]) => {{
            const tr = document.createElement("tr");
            tr.innerHTML = '<td>' + col.replace(/</g, '&lt;') + '</td><td>' + (val != null ? String(val).replace(/</g, '&lt;').slice(0, 200) + (String(val).length > 200 ? '…' : '') : '<em>—</em>') + '</td>';
            tbody.appendChild(tr);
        }});
        const msgs = data.messages || [];
        const el = document.getElementById("messages");
        let lastAction = null;
        msgs.forEach((m, idx) => {{
            const div = document.createElement("div");
            div.className = "msg " + (m.role === "user" ? "user" : "assistant");
            let content = m.content || "";
            const isToolResult = content.startsWith("Tool result: ");
            const body = isToolResult ? content.slice(13) : content;
            let formatted = body;
            let action = null;
            try {{ const p = JSON.parse(body); formatted = JSON.stringify(p, null, 2); action = p.action; if (action) lastAction = action; }} catch(e) {{}}
            let header = '<div class="role">' + m.role + '</div>';
            if (action) {{
                header += '<div class="tool-call">Tool called: ' + action + '</div>';
            }} else if (isToolResult && lastAction) {{
                header += '<div class="tool-call">Result from: ' + lastAction + '</div>';
            }}
            div.innerHTML = header;
            const pre = document.createElement("pre");
            pre.textContent = formatted;
            div.appendChild(pre);
            el.appendChild(div);
        }});
    </script>
</body>
</html>"""


def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_id", nargs="?", help="Document ID")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns (stops early when remaining_count=0)")
    parser.add_argument("--groups-only", nargs="+", help="Limit to these groups")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore previous extraction_results.json")
    parser.add_argument("--skip-if-done", action="store_true", help="Exit immediately if all columns already filled")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    doc_id = args.doc_id or "NCT00268476_Attard_STAMPEDE_Lancet'23"
    result = run_agent(
        doc_id,
        max_turns=args.max_turns,
        groups_filter=args.groups_only,
        resume=not args.no_resume,
        skip_if_done=args.skip_if_done,
    )

    if "error" in result and "database" not in result:
        log.error("%s", result["error"])
        sys.exit(1)

    out_dir = RESULTS_ROOT / doc_id / "agent_extractor"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialize for output
    db_raw = result.get("database") or {}
    db_out = {}
    for k, v in db_raw.items():
        if isinstance(v, dict):
            db_out[k] = {"value": v.get("value"), "reasoning": v.get("reasoning", ""), "found": v.get("found", True), "tried": v.get("tried", True)}
        else:
            db_out[k] = {"value": v, "reasoning": "", "found": v is not None, "tried": False}

    output = {
        "doc_id": doc_id,
        "turns": result.get("turns", 0),
        "database": {k: v.get("value") if isinstance(v, dict) else v for k, v in db_out.items()},
        "messages": result.get("messages", []),
    }
    pretty = json.dumps(output, indent=2)

    # Save JSON
    json_path = out_dir / "conversation.json"
    json_path.write_text(pretty, encoding="utf-8")

    # Save extracted data: inverted attribution format (columns + attribution map)
    extraction_path = out_dir / "extraction_results.json"
    extraction_data = {
        "doc_id": doc_id,
        "columns": db_out,
        "attribution": _build_inverted_attribution(db_raw),
    }
    extraction_path.write_text(json.dumps(extraction_data, indent=2), encoding="utf-8")

    print(pretty)
    log.info("Wrote %s", json_path)
    log.info("Wrote %s (extracted data)", extraction_path)
    log.info("Open %s in browser", out_dir / "conversation_viewer.html")

    # Write static HTML with data inlined (open in browser, no server needed)
    html_path = out_dir / "conversation_viewer.html"
    html_path.write_text(_make_static_html(output), encoding="utf-8")


if __name__ == "__main__":
    main()
