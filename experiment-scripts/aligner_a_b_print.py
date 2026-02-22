#!/usr/bin/env python3
"""aligner_a_b_print.py

Two-pass structural alignment experiment.

Aligner A (Schema → Chunk):
  For each column group, send the PDF + chunk summaries to Gemini.
  Ask: for each column, where in the document does it live?
  Returns: column_name, found_in_pdf, page, modality, extraction_plan.

Aligner B (Chunk → Columns):
  Collect chunks referenced by Aligner A (page, modality).
  One Gemini call per chunk (text only, no PDF).
  Ask: which column names have extractable values in this chunk?
  Returns: chunk_index, page, modality → [column_names].

Prints both maps to stdout and saves JSON outputs.

Usage:
  python aligner_a_b_print.py \\
      --doc-id "NCT02799602_Hussain_ARASENS_JCO'23" \\
      --results-root new_pipeline_outputs/results \\
      --pdf-dir dataset \\
      [--model gemini-2.5-flash] \\
      [--workers 8] \\
      [--skip-aligner-a]   # reload saved aligner_a.json instead of re-running
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# ── Repo root on path ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise RuntimeError("google-genai not installed. Run: pip install google-genai")

from baseline_utils import load_definitions_with_metadata
from src.planning.plan_generator import format_chunk_summaries

load_dotenv()

# ── Globals ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"
DEFINITIONS_CSV = str(REPO_ROOT / "src" / "table_definitions" / "Definitions_with_eval_category_sectioned.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Gemini provider
# ══════════════════════════════════════════════════════════════════════════════

class GeminiProvider:
    """Thin wrapper around google-genai for PDF + JSON schema calls."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=120_000),
        )
        self._pdf_part = None

    def load_pdf(self, pdf_path: Path) -> None:
        pdf_bytes = pdf_path.read_bytes()
        self._pdf_part = genai_types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
        print(f"  PDF loaded: {pdf_path.name} ({len(pdf_bytes) / 1024:.0f} KB)")

    def call_with_pdf(self, prompt: str, schema: Dict) -> Tuple[str, int, int]:
        """Call Gemini with PDF in context. Returns (text, in_tok, out_tok)."""
        assert self._pdf_part is not None, "Call load_pdf() first"
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self._pdf_part, prompt],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return (response.text or "").strip(), in_tok, out_tok

    def call_text_only(self, prompt: str, schema: Dict) -> Tuple[str, int, int]:
        """Call Gemini with text only (no PDF). Returns (text, in_tok, out_tok)."""
        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return (response.text or "").strip(), in_tok, out_tok


# ══════════════════════════════════════════════════════════════════════════════
# Chunk utilities
# ══════════════════════════════════════════════════════════════════════════════

def safe_stem(name: str) -> str:
    """Filesystem-safe stem for group names."""
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("|", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def load_chunks(chunk_file: Path) -> List[Dict]:
    data = json.loads(chunk_file.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]
    raise ValueError(f"Unexpected chunk file shape: {chunk_file}")


def format_chunk_for_b(chunk: Dict, index: int) -> str:
    """Format one chunk as text block for Aligner B prompt. No truncation."""
    ctype = chunk.get("type", "?").upper()
    page = chunk.get("page", "?")
    header = f"--- Chunk {index} ({ctype}, page {page}) ---"
    if ctype == "TABLE" and chunk.get("table_content"):
        body = chunk["table_content"]
    else:
        body = chunk.get("content", "") or ""
    return f"{header}\n{body}"


# ══════════════════════════════════════════════════════════════════════════════
# Aligner A — Schema → Chunk
# ══════════════════════════════════════════════════════════════════════════════

ALIGNER_A_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column_name":     {"type": "string"},
                    "found_in_pdf":    {"type": "boolean"},
                    "page":            {"type": "integer"},
                    "modality":        {
                        "type": "string",
                        "enum": ["table", "figure", "text", "not_applicable"]
                    },
                    "extraction_plan": {"type": "string"},
                },
                "required": ["column_name", "found_in_pdf", "page", "modality", "extraction_plan"],
            }
        }
    },
    "required": ["columns"],
}


def build_aligner_a_prompt(group_name: str, columns: List[Dict], chunk_summaries: str) -> str:
    """Uses same prompt structure as plan_generator.generate_plan_for_group."""
    expected_block = "\n".join(
        f"{i}. {c['column_name']}\n   Definition: {c['definition']}"
        for i, c in enumerate(columns, 1)
    )
    return f"""You are creating an extraction plan for a clinical trial data extraction task.

You have:
- The FULL PDF loaded (for structure + precise reference)
- A list of pre-extracted chunks (to orient you)

AVAILABLE CHUNKS:
{chunk_summaries}

TASK:
For EACH of the following canonical columns, decide whether the value is reported in this PDF.
If reported, identify WHERE and HOW to extract it. Note that some values might need to be inferred from different parts of the document.
If not reported, say it is not reported.

⚠️ CRITICAL INSTRUCTION:
When you refer to these columns in your response, you MUST use their EXACT names as listed below.
Do NOT paraphrase, abbreviate, or modify column names in ANY way.
Character-for-character match is REQUIRED (including spaces, punctuation, pipes |, parentheses).

CANONICAL COLUMNS (ORDERED; use these EXACT names in your response):
{expected_block}

Rules:
- Be honest: many columns will NOT be reported.
- If found_in_pdf=true, include page number, source type (table/text/figure), and concrete instructions.
- If found_in_pdf=false, you MUST still cite WHERE you looked: which pages, tables, figures you examined to reach that conclusion (e.g. "Table 1 (page 5) provides X but not Y").
- ALWAYS refer to columns using their exact canonical names (copy-paste from the list above).
"""


def run_aligner_a(
    provider: GeminiProvider,
    label_groups: Dict[str, List[Dict]],
    chunks: List[Dict],
    workers: int,
    a_logs_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run Aligner A for all groups in parallel.
    Returns: {columns: [...], tokens: {...}} merged across all groups.
    """
    chunk_summaries = format_chunk_summaries(chunks)
    lock = threading.Lock()
    all_columns: List[Dict] = []
    total_in, total_out = 0, 0

    def process_group(group_name: str, columns: List[Dict]) -> Tuple[str, List[Dict], int, int]:
        prompt = build_aligner_a_prompt(group_name, columns, chunk_summaries)
        text, in_tok, out_tok = provider.call_with_pdf(prompt, ALIGNER_A_SCHEMA)
        try:
            parsed = json.loads(text) if text else {}
            result_cols = parsed.get("columns", [])
        except json.JSONDecodeError:
            print(f"    [Aligner A] JSON decode failed for group '{group_name}'")
            result_cols = []
            parsed = {}
        if a_logs_dir:
            log_path = a_logs_dir / f"{safe_stem(group_name)}.json"
            log_path.write_text(
                json.dumps(
                    {"prompt": prompt, "response": text, "parsed": parsed},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return group_name, result_cols, in_tok, out_tok

    max_w = min(workers, len(label_groups)) or 1
    print(f"\n  Running Aligner A on {len(label_groups)} groups ({max_w} workers)...")

    with ThreadPoolExecutor(max_workers=max_w) as exe:
        futures = {
            exe.submit(process_group, gname, gcols): gname
            for gname, gcols in label_groups.items()
        }
        for future in as_completed(futures):
            gname = futures[future]
            try:
                _, cols, in_tok, out_tok = future.result()
                with lock:
                    all_columns.extend(cols)
                    total_in += in_tok
                    total_out += out_tok
                print(f"    [A] Done: {gname} ({len(cols)} columns)")
            except Exception as e:
                print(f"    [A] FAILED: {gname} — {e}")

    found = sum(1 for c in all_columns if c.get("found_in_pdf"))
    print(f"  Aligner A complete: {found}/{len(all_columns)} columns found in PDF")
    print(f"  Tokens — in: {total_in:,}  out: {total_out:,}")

    return {"columns": all_columns, "tokens": {"input": total_in, "output": total_out}}


# ══════════════════════════════════════════════════════════════════════════════
# Bridge — collect unified chunk set from Aligner A output
# ══════════════════════════════════════════════════════════════════════════════

def collect_unified_chunks(
    aligner_a: Dict,
    chunks: List[Dict],
    definitions: Dict[str, Dict],
) -> List[Tuple[Dict, int, List[Dict]]]:
    """
    From Aligner A output, find chunks that match any (page, modality) pairs where found_in_pdf=true.
    For each chunk, merge candidate columns from ALL matching (page, modality) keys — so a text chunk
    spanning pages 1-4 can serve both (1, text) and (3, text). Returns list of (chunk, index,
    candidate_columns) where candidate_columns is [{column_name, definition}, ...].
    """
    # Build map: (page, modality) → [{column_name, definition}]
    source_to_cols: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    for col in aligner_a.get("columns", []):
        if col.get("found_in_pdf") and col.get("page", -1) > 0:
            key = (col["page"], col["modality"])
            col_name = col.get("column_name", "")
            defn = definitions.get(col_name, {}).get("definition", "")
            source_to_cols[key].append({"column_name": col_name, "definition": defn})

    def page_matches(chunk_page: Any, target: int) -> bool:
        if isinstance(chunk_page, int):
            return chunk_page == target
        if isinstance(chunk_page, str) and "-" in chunk_page:
            try:
                s, e = chunk_page.split("-", 1)
                return int(s) <= target <= int(e)
            except ValueError:
                return False
        try:
            return int(chunk_page) == target
        except (ValueError, TypeError):
            return False

    # Iterate chunks: for each chunk, find ALL (page, modality) keys that match it,
    # merge their candidate columns. This lets one chunk (e.g. 1-4 text) serve both
    # (1, text) and (3, text) without being claimed only once.
    selected: List[Tuple[Dict, int, List[Dict]]] = []
    matched_keys: set = set()
    for idx, chunk in enumerate(chunks):
        ctype = chunk.get("type", "")
        cpage = chunk.get("page")
        merged: List[Dict] = []
        seen_names: set = set()
        for (tgt_page, tgt_modality) in source_to_cols.keys():
            if tgt_modality != ctype:
                continue
            if not isinstance(tgt_page, int) or not page_matches(cpage, tgt_page):
                continue
            matched_keys.add((tgt_page, tgt_modality))
            for col in source_to_cols[(tgt_page, tgt_modality)]:
                if col["column_name"] not in seen_names:
                    seen_names.add(col["column_name"])
                    merged.append(col)
        if merged:
            selected.append((chunk, idx, merged))

    unmatched = set(source_to_cols.keys()) - matched_keys
    if unmatched:
        for k in sorted(unmatched):
            n = len(source_to_cols[k])
            print(f"    [Bridge] WARNING: No chunk matched {k} — {n} columns will not reach Aligner B")
    print(f"\n  Bridge: {len(source_to_cols)} unique (page, modality) pairs → {len(selected)} chunks retrieved")
    for chunk, idx, cands in selected:
        print(f"    Chunk {idx}  type={chunk.get('type')}  page={chunk.get('page')}  ({len(cands)} candidate columns)")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Aligner B — Chunks → Columns (one call per chunk)
# ══════════════════════════════════════════════════════════════════════════════

ALIGNER_B_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["columns"],
}


def build_aligner_b_prompt(chunk_text: str, candidate_cols: List[Dict]) -> str:
    """
    Prompt for a single chunk.
    candidate_cols: [{column_name, definition}] — only the columns A attributed to this chunk.
    """
    col_block = "\n".join(
        f"  {i+1}. {c['column_name']}\n     Definition: {c['definition']}"
        for i, c in enumerate(candidate_cols)
    )
    col_names = [c["column_name"] for c in candidate_cols]
    names_list = "\n".join(f"  - {n}" for n in col_names)
    return f"""You are performing STRUCTURAL ATTRIBUTION for clinical trial data extraction.
Do NOT extract values. Only verify which of the following columns have explicit extractable content in this chunk.

CHUNK:
{chunk_text}

CANDIDATE COLUMNS (these were attributed to this chunk by a prior schema-first pass):
{col_block}

INSTRUCTIONS:
- From the candidate columns above, list only those that have EXPLICIT extractable values
  present in this chunk (numbers, structured text, clearly labelled rows).
- Only include a column if its value is directly visible — not inferred or implied.
- Use the exact column names as listed below.
- If none are confirmed, return an empty array.

EXACT COLUMN NAMES TO USE IN YOUR RESPONSE:
{names_list}
"""


def run_aligner_b(
    provider: GeminiProvider,
    unified_chunks: List[Tuple[Dict, int, List[Dict]]],
    workers: int,
    b_logs_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run Aligner B — one text-only call per chunk, using only A's candidate columns."""
    if not unified_chunks:
        print("  [Aligner B] No unified chunks — skipping.")
        return {"chunk_mappings": [], "tokens": {"input": 0, "output": 0}}

    def process_one(item: Tuple[Dict, int, List[Dict]]) -> Tuple[int, int, int, List[str]]:
        chunk, idx, candidate_cols = item
        chunk_text = format_chunk_for_b(chunk, idx)
        prompt = build_aligner_b_prompt(chunk_text, candidate_cols)
        text, in_tok, out_tok, parsed, cols = None, 0, 0, {}, []
        try:
            text, in_tok, out_tok = provider.call_text_only(prompt, ALIGNER_B_SCHEMA)
            try:
                parsed = json.loads(text) if text else {}
                cols = parsed.get("columns", [])
            except json.JSONDecodeError:
                parsed = {}
                cols = []
        except Exception as e:
            parsed = {}
            text = None
            if b_logs_dir:
                log_path = b_logs_dir / f"chunk_{idx}.json"
                log_path.write_text(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "response": None,
                            "parsed": {},
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            raise
        if b_logs_dir:
            log_path = b_logs_dir / f"chunk_{idx}.json"
            log_path.write_text(
                json.dumps(
                    {"prompt": prompt, "response": text, "parsed": parsed},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return idx, in_tok, out_tok, cols

    max_w = min(workers, len(unified_chunks)) or 1
    print(f"\n  Running Aligner B on {len(unified_chunks)} chunks (one call per chunk, {max_w} workers)...")

    mappings: List[Dict] = []
    total_in, total_out = 0, 0
    with ThreadPoolExecutor(max_workers=max_w) as exe:
        futures = {exe.submit(process_one, item): item for item in unified_chunks}
        for future in as_completed(futures):
            chunk, idx, candidate_cols = futures[future]
            try:
                _, in_tok, out_tok, cols = future.result()
                total_in += in_tok
                total_out += out_tok
                mappings.append({
                    "chunk_index": idx,
                    "page": chunk.get("page"),
                    "modality": chunk.get("type"),
                    "columns": cols,
                })
                print(f"    [B] Chunk {idx}: {len(cols)} columns")
            except Exception as e:
                print(f"    [B] Chunk {idx} FAILED: {e}")

    total_cols = sum(len(m.get("columns", [])) for m in mappings)
    print(f"  Aligner B complete: {len(mappings)} chunks, {total_cols} total column claims")
    print(f"  Tokens — in: {total_in:,}  out: {total_out:,}")

    return {"chunk_mappings": mappings, "tokens": {"input": total_in, "output": total_out}}


# ══════════════════════════════════════════════════════════════════════════════
# Print maps
# ══════════════════════════════════════════════════════════════════════════════

def print_aligner_a_map(aligner_a: Dict) -> None:
    print("\n" + "=" * 70)
    print("MAP A  —  Column → Source (Schema-first)")
    print("=" * 70)
    columns = aligner_a.get("columns", [])
    # Group by found / not found
    found_cols = [c for c in columns if c.get("found_in_pdf")]
    not_found  = [c for c in columns if not c.get("found_in_pdf")]

    print(f"\nFOUND ({len(found_cols)} columns):")
    for c in sorted(found_cols, key=lambda x: (x.get("page", 999), x.get("column_name", ""))):
        name  = c.get("column_name", "?")
        page  = c.get("page", "?")
        mod   = c.get("modality", "?")
        plan  = c.get("extraction_plan", "")[:100]
        print(f"  p{page:>3} [{mod:^12}]  {name}")
        print(f"           ↳ {plan}")

    print(f"\nNOT FOUND ({len(not_found)} columns):")
    for c in not_found:
        print(f"  ✗  {c.get('column_name', '?')}")


def print_aligner_b_map(aligner_b: Dict) -> None:
    print("\n" + "=" * 70)
    print("MAP B  —  Chunk → Columns (Chunk-first)")
    print("=" * 70)
    for mapping in aligner_b.get("chunk_mappings", []):
        idx = mapping.get("chunk_index", "?")
        page = mapping.get("page", "?")
        mod = mapping.get("modality", "?")
        cols = mapping.get("columns", [])
        print(f"\n  Chunk {idx} (p{page} {mod})  ({len(cols)} columns)")
        for col in cols:
            print(f"    → {col}")


def _page_matches_for_overlap(b_page: Any, a_page: int) -> bool:
    """Check if A's page (int) falls within B's page (int or range like '1-4')."""
    if isinstance(b_page, int):
        return b_page == a_page
    if isinstance(b_page, str) and "-" in b_page:
        try:
            s, e = b_page.split("-", 1)
            return int(s) <= a_page <= int(e)
        except ValueError:
            return False
    try:
        return int(b_page) == a_page
    except (ValueError, TypeError):
        return False


def print_overlap(aligner_a: Dict, aligner_b: Dict) -> List[str]:
    """
    For each column found by A, check if B also attributes it to the chunk at (page, modality).
    Returns lines for writing to file.
    B's page can be int (e.g. 8) or range (e.g. "1-4"); A's page is int. We match when A's page
    falls within B's page/range.
    """
    # Build B list: [(page, modality, set of columns)] for each mapping
    b_mappings: List[Tuple[Any, str, set]] = []
    for mapping in aligner_b.get("chunk_mappings", []):
        b_page = mapping.get("page")
        b_mod = mapping.get("modality", "")
        cols = set(mapping.get("columns", []))
        b_mappings.append((b_page, b_mod, cols))

    print("\n" + "=" * 70)
    print("OVERLAP  —  A claims vs B confirmation")
    print("=" * 70)

    lines: List[str] = []
    confirmed, missing = 0, 0
    for col in aligner_a.get("columns", []):
        if not col.get("found_in_pdf"):
            continue
        name = col.get("column_name", "?")
        a_page = col.get("page")
        a_mod = col.get("modality", "")
        # Find B chunks where modality matches and A's page falls within B's page/range
        b_cols: set = set()
        for b_page, b_mod, cols in b_mappings:
            if b_mod == a_mod and isinstance(a_page, int) and _page_matches_for_overlap(b_page, a_page):
                b_cols |= cols
        if name in b_cols:
            status = "✓  confirmed"
            confirmed += 1
        else:
            status = "✗  NOT confirmed by B"
            missing += 1
        line = f"  {name}  [A: p{a_page} {a_mod}]  →  {status}"
        print(line)
        lines.append(line)

    summary = f"\nConfirmed: {confirmed} / {confirmed + missing}  |  Missing from B: {missing}"
    print(summary)
    lines.append(summary)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aligner A+B structural mapping experiment")
    p.add_argument("--doc-id",       required=True,  help="Document stem, e.g. \"NCT02799602_Hussain_ARASENS_JCO'23\"")
    p.add_argument("--results-root", default="new_pipeline_outputs/results",
                   help="Root directory containing per-document pipeline outputs")
    p.add_argument("--pdf-dir",      default="dataset",
                   help="Directory containing source PDFs")
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--workers",      type=int, default=8)
    p.add_argument("--skip-aligner-a", action="store_true",
                   help="Load existing aligner_a.json instead of re-running LLM calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    doc_id       = args.doc_id
    results_root = REPO_ROOT / args.results_root
    pdf_dir      = REPO_ROOT / args.pdf_dir

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / "experiment-scripts" / "aligner_results" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    a_logs_dir = out_dir / "a_logs"
    b_logs_dir = out_dir / "b_logs"
    a_logs_dir.mkdir(exist_ok=True)
    b_logs_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("ALIGNER A + B  —  Structural Mapping Experiment")
    print(f"Document : {doc_id}")
    print(f"Model    : {args.model}")
    print(f"Output   : {out_dir}")
    print(f"API logs : {a_logs_dir}  |  {b_logs_dir}")
    print("=" * 70)

    # ── Load chunks ─────────────────────────────────────────────────────────
    chunk_file = results_root / doc_id / "chunking" / "pdf_chunked.json"
    if not chunk_file.exists():
        sys.exit(f"Chunk file not found: {chunk_file}")
    chunks = load_chunks(chunk_file)
    print(f"\nChunks loaded: {len(chunks)}  ({sum(1 for c in chunks if c['type']=='table')} tables, "
          f"{sum(1 for c in chunks if c['type']=='figure')} figures, "
          f"{sum(1 for c in chunks if c['type']=='text')} text)")

    # ── Load definitions ────────────────────────────────────────────────────
    if not Path(DEFINITIONS_CSV).exists():
        sys.exit(f"Definitions CSV not found: {DEFINITIONS_CSV}")
    definitions = load_definitions_with_metadata(DEFINITIONS_CSV)

    # Build label_groups: {label: [{column_name, definition}]}
    label_groups: Dict[str, List[Dict]] = defaultdict(list)
    for col_name, meta in definitions.items():
        label_groups[meta["label"]].append({
            "column_name": col_name,
            "definition":  meta["definition"],
        })
    print(f"Definitions loaded: {len(definitions)} columns in {len(label_groups)} groups")

    # ── Init provider ────────────────────────────────────────────────────────
    provider = GeminiProvider(model=args.model)

    # ── Aligner A ────────────────────────────────────────────────────────────
    aligner_a_file = out_dir / "aligner_a.json"
    if args.skip_aligner_a and aligner_a_file.exists():
        print(f"\n[skip-aligner-a] Loading saved: {aligner_a_file}")
        aligner_a = json.loads(aligner_a_file.read_text(encoding="utf-8"))
    else:
        # Find PDF
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            # try sanitised name
            sanitised = re.sub(r"[\/\\:*?\"<>|']", "_", doc_id) + ".pdf"
            pdf_path = pdf_dir / sanitised
        if not pdf_path.exists():
            sys.exit(f"PDF not found in {pdf_dir}. Tried: {doc_id}.pdf and {sanitised}")
        provider.load_pdf(pdf_path)
        aligner_a = run_aligner_a(provider, label_groups, chunks, args.workers, a_logs_dir)
        aligner_a_file.write_text(json.dumps(aligner_a, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved: {aligner_a_file}")

    # ── Bridge ────────────────────────────────────────────────────────────────
    unified_chunks = collect_unified_chunks(aligner_a, chunks, definitions)

    # ── Aligner B ─────────────────────────────────────────────────────────────
    aligner_b = run_aligner_b(provider, unified_chunks, args.workers, b_logs_dir)
    aligner_b_file = out_dir / "aligner_b.json"
    aligner_b_file.write_text(json.dumps(aligner_b, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {aligner_b_file}")

    # ── Print maps ────────────────────────────────────────────────────────────
    print_aligner_a_map(aligner_a)
    print_aligner_b_map(aligner_b)
    overlap_lines = print_overlap(aligner_a, aligner_b)

    # ── Save overlap report ──────────────────────────────────────────────────
    report_file = out_dir / "overlap_report.txt"
    report_file.write_text("\n".join(overlap_lines), encoding="utf-8")
    print(f"\nOverlap report saved: {report_file}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
