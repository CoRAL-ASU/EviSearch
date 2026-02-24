#!/usr/bin/env python3
"""
Reconcile extraction outputs from Pipeline, Landing AI, and Gemini.

Simplified flow (no document/chunks to LLM):
- Load comparison data via web.comparison_service
- Per column: gather method outputs (value, evidence, page, source_type)
- LLM: compare/merge, detect contradiction, output final_value + contributing_methods
- Citation: derive from page+source_type (chunk_ids from highlight_service)

Run:
  python experiment-scripts/reconcile_extractions.py DOC_ID
  python experiment-scripts/reconcile_extractions.py DOC_ID --dry-run   # no LLM, just show inputs
  python experiment-scripts/reconcile_extractions.py --all              # run for all docs (from comparison service)
  python experiment-scripts/reconcile_extractions.py DOC_ID -o out.json # custom output path

Then rebuild the comparison report to see Reconciled column and LLM log links:
  python experiment-analysis/build_comparison_report.py

Output (when -o not given):
  new_pipeline_outputs/results/{DOC_ID}/reconciliation/reconciled_results.json
  new_pipeline_outputs/results/{DOC_ID}/reconciliation/logs/{group}_llm.txt  # raw prompt+response per batch
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_ROOT = PROJECT_ROOT / "new_pipeline_outputs" / "results"

from web.comparison_service import load_comparison_data, list_documents
from web.highlight_service import load_landing_ai_parse
from web.attribution_service import enrich_reconciled_with_attribution

# Optional: LLM for reconciliation
try:
    from src.LLMProvider.provider import LLMProvider
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Anonymous labels so the LLM does not favor any method by name
METHOD_NAMES = ("pipeline", "landing_ai_baseline", "gemini_native")
ANONYMOUS = ("Method A", "Method B", "Method C")
METHOD_TO_ANON = dict(zip(METHOD_NAMES, ANONYMOUS))
ANON_TO_METHOD = dict(zip(ANONYMOUS, METHOD_NAMES))


def _anon_to_method(contrib: list) -> list:
    """Map anonymous labels back to real method names."""
    out = []
    for x in contrib:
        key = str(x).strip()
        real = ANON_TO_METHOD.get(key)
        if not real:
            # Accept "method a", "methodA", etc.
            key_norm = key.replace(" ", "").lower()
            for anon, meth in ANON_TO_METHOD.items():
                if anon.replace(" ", "").lower() == key_norm:
                    real = meth
                    break
        if real:
            out.append(real)
    return out


def _landing_type_to_pipeline(landing_type: str) -> str:
    t = str(landing_type or "").lower()
    if "table" in t or t == "table":
        return "table"
    if "figure" in t or "logo" in t or "scancode" in t:
        return "figure"
    return "text"


def get_chunk_ids_by_page_type(doc_id: str, page: int, source_type: str) -> list[str]:
    """Return chunk IDs on given page matching source_type, in document order."""
    parse_data = load_landing_ai_parse(doc_id)
    if not parse_data:
        return []
    chunks = parse_data.get("chunks") or []
    target_type = str(source_type or "text").lower()
    if target_type not in ("text", "table", "figure"):
        target_type = "text"
    ids = []
    for c in chunks:
        g = c.get("grounding") or {}
        if not isinstance(g, dict):
            continue
        page_0 = g.get("page", 0)
        if int(page_0) + 1 != page:
            continue
        ct = _landing_type_to_pipeline(c.get("type", "text"))
        if ct != target_type:
            continue
        cid = c.get("id")
        if cid:
            ids.append(cid)
    return ids


def gather_method_outputs(row: dict) -> dict:
    """Extract value, evidence, page, source_type per method."""
    out = {}
    for method in ("pipeline", "landing_ai_baseline", "gemini_native"):
        col = (row.get("methods") or {}).get(method)
        if not col:
            out[method] = None
            continue
        v = col.get("value") or col.get("primary_value")
        ev = col.get("evidence") or (col.get("attribution") or {}).get("evidence", "")
        p = col.get("page")
        st = col.get("source_type") or "text"
        out[method] = {
            "value": str(v).strip() if v else "",
            "evidence": str(ev)[:500] if ev else "",  # truncate for prompt
            "page": p if p is not None and p != "Not applicable" and str(p).lower() not in ("n/a", "na") else None,
            "source_type": st,
        }
    return out


def build_reconciliation_prompt(columns_data: list[dict]) -> str:
    """Build prompt for batched reconciliation (method outputs only, no chunks).
    Uses anonymous labels (Method A, B, C) to avoid favoring any extraction source."""
    lines = [
        "You are reconciling clinical trial extractions from three extractors (Method A, Method B, Method C).",
        "For each column below, produce: final_value, contradiction (true if methods disagree on atomic facts), contributing_methods (list of Method A/B/C that agree or contributed).",
        "Prefer more complete answers. If one method has partial info (e.g. one trial) and another has both, merge them.",
        "Treat all three methods equally. Output valid JSON only.",
        "",
        "=== COLUMNS ===",
    ]
    for i, item in enumerate(columns_data):
        col_name = item["column_name"]
        methods = item["methods"]
        lines.append(f"\nColumn {i + 1}: {col_name}")
        for m in METHOD_NAMES:
            anon = METHOD_TO_ANON.get(m, m)
            d = methods.get(m)
            if d is None:
                lines.append(f"  {anon}: (missing)")
            else:
                v = (d.get("value") or "")[:200]
                ev = (d.get("evidence") or "")[:300]
                p = d.get("page")
                lines.append(f"  {anon}: value=\"{v}\" | page={p} | evidence=\"{ev}\"")
    lines.extend([
        "",
        "=== OUTPUT (JSON) ===",
        '{"reconciled": [{"column_name": "...", "final_value": "...", "contradiction": false, "contributing_methods": ["Method A", "Method B"]}, ...]}',
    ])
    return "\n".join(lines)


def parse_reconciliation_response(text: str) -> list[dict]:
    """Parse LLM JSON response. Fallback to empty on error."""
    try:
        s = text.strip()
        if "```" in s:
            start = s.find("```")
            end = s.find("```", start + 3)
            if start >= 0:
                s = s[start + 3 : end if end > 0 else None].strip()
            if s.startswith("json"):
                s = s[4:].strip()
        data = json.loads(s)
        items = data.get("reconciled") or []
        out = []
        for item in items:
            if not item.get("column_name"):
                continue
            contrib = item.get("contributing_methods") or []
            out.append({
                "column_name": item.get("column_name", ""),
                "final_value": item.get("final_value", ""),
                "contradiction": bool(item.get("contradiction")),
                "contributing_methods": _anon_to_method(contrib),
            })
        return out
    except Exception:
        return []


def _safe_log_name(s: str) -> str:
    """Safe filename from group/column name."""
    out = []
    for c in str(s or ""):
        if c.isalnum() or c in "_- ":
            out.append(c if c != " " else "_")
    return "".join(out).strip("_")[:120] or "batch"


def reconcile_without_llm(row: dict) -> dict:
    """Rule-based fallback: pick most complete value, no contradiction check."""
    methods = gather_method_outputs(row)
    best_value = ""
    best_methods = []
    for m, d in methods.items():
        if not d or not d.get("value"):
            continue
        v = d["value"]
        if v.lower() in ("not found", "not reported", "not applicable"):
            continue
        if len(v) > len(best_value):
            best_value = v
            best_methods = [m]
        elif v == best_value:
            best_methods.append(m)
    return {
        "column_name": row.get("column_name", ""),
        "final_value": best_value or "Not reported",
        "contradiction": False,
        "contributing_methods": best_methods or list(methods.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="Reconcile extractions for a trial")
    parser.add_argument("doc_id", nargs="?", help="Document ID")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--dry-run", action="store_true", help="No LLM; output gathered inputs only")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based only (no API)")
    parser.add_argument("--no-attribution", action="store_true", help="Skip attribution scoring (keep page+type chunk_ids only)")
    parser.add_argument("--no-semantic", action="store_true", help="Use deterministic scoring only (no sentence-transformers)")
    parser.add_argument("--list", action="store_true", help="List docs and exit")
    parser.add_argument("--all", action="store_true", help="Run for all documents")
    parser.add_argument("--random", type=int, metavar="N", help="With --all: run on N random docs only (for quick attribution runs)")
    args = parser.parse_args()

    if args.list:
        docs = list_documents()
        for d in docs:
            print(d.get("doc_id", "?"))
        return

    if args.all:
        if args.output:
            print("Note: -o is ignored when using --all; each doc written to reconciliation/reconciled_results.json", file=sys.stderr)
        run_args = argparse.Namespace(
            output=None, dry_run=args.dry_run, no_llm=args.no_llm,
            no_attribution=args.no_attribution, no_semantic=args.no_semantic,
        )
        docs = list_documents()
        n_random = getattr(args, "random", None)
        if n_random and n_random > 0:
            docs = random.sample(docs, min(n_random, len(docs)))
            print(f"Running on {len(docs)} random docs: {[d.get('doc_id') for d in docs]}", file=sys.stderr)
        for d in docs:
            doc_id = d.get("doc_id", "")
            if doc_id:
                try:
                    print(f"\n--- {doc_id} ---", file=sys.stderr)
                    _run_reconciliation(doc_id, run_args)
                except Exception as e:
                    print(f"Error for {doc_id}: {e}", file=sys.stderr)
        return

    doc_id = args.doc_id
    if not doc_id:
        docs = list_documents()
        doc_id = docs[0]["doc_id"] if docs else ""
    if not doc_id:
        print("No documents. Use --list to see available.", file=sys.stderr)
        sys.exit(1)

    try:
        _run_reconciliation(doc_id, args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _run_reconciliation(doc_id: str, args) -> None:
    """Run reconciliation for one document."""
    print(f"Loading comparison data for {doc_id}…", file=sys.stderr)
    comparison = load_comparison_data(doc_id)
    rows = comparison.get("comparison") or []
    print(f"Found {len(rows)} columns.", file=sys.stderr)
    by_group = comparison.get("by_group") or {}

    if args.dry_run:
        # Output gathered method outputs only (no LLM)
        out = []
        for row in rows:
            methods = gather_method_outputs(row)
            out.append({
                "column_name": row.get("column_name"),
                "group_name": row.get("group_name"),
                "methods": methods,
            })
        result = {"doc_id": doc_id, "dry_run": True, "columns": out}
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    # Output and log paths (logs always under results/doc_id/reconciliation/logs/)
    out_path = Path(args.output) if args.output else RESULTS_ROOT / doc_id / "reconciliation" / "reconciled_results.json"
    logs_dir = RESULTS_ROOT / doc_id / "reconciliation" / "logs"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Reconcile
    if args.no_llm or not HAS_LLM:
        reconciled = [reconcile_without_llm(r) for r in rows]
    else:
        # Batched by group
        groups = [(k, v) for k, v in by_group.items() if v]
        print(f"Reconciling {len(groups)} groups (LLM)…", file=sys.stderr)
        provider = LLMProvider(provider="gemini", model="gemini-2.5-flash")
        reconciled = []
        for i, (group_name, group_rows) in enumerate(groups, 1):
            print(f"  [{i}/{len(groups)}] {group_name}…", file=sys.stderr)
            cols_data = [
                {"column_name": r["column_name"], "methods": gather_method_outputs(r)}
                for r in group_rows
            ]
            prompt = build_reconciliation_prompt(cols_data)
            try:
                resp = provider.generate(prompt, temperature=0.0, max_tokens=8000)
                if resp.success:
                    parsed = parse_reconciliation_response(resp.text)
                    parsed_names = {p["column_name"] for p in parsed}
                    reconciled.extend(parsed)
                    for r in group_rows:
                        if r["column_name"] not in parsed_names:
                            reconciled.append(reconcile_without_llm(r))
                else:
                    for r in group_rows:
                        reconciled.append(reconcile_without_llm(r))
                # Save raw LLM logs
                if logs_dir.exists():
                    log_name = _safe_log_name(group_name) or "batch"
                    log_path = logs_dir / f"{log_name}_llm.txt"
                    log_path.write_text(
                        f"=== PROMPT ===\n{prompt}\n\n=== RESPONSE ===\n{resp.text}",
                        encoding="utf-8",
                    )
            except Exception as e:
                print(f"LLM error for {group_name}: {e}", file=sys.stderr)
                for r in group_rows:
                    reconciled.append(reconcile_without_llm(r))

    # Add page, source_type, chunk_ids from first contributing method
    col_to_row = {r["column_name"]: r for r in rows}
    for rec in reconciled:
        row = col_to_row.get(rec["column_name"], {})
        contrib = rec.get("contributing_methods") or []
        page, source_type, chunk_ids = None, None, []
        for m in contrib:
            col = (row.get("methods") or {}).get(m)
            if col and col.get("page") is not None and col["page"] != "Not applicable":
                page = col.get("page")
                source_type = col.get("source_type") or "text"
                if page and page >= 1:
                    chunk_ids = get_chunk_ids_by_page_type(doc_id, int(page), source_type or "text")
                break
        rec["page"] = page
        rec["source_type"] = source_type
        rec["chunk_ids"] = chunk_ids[:5]  # cap

    # Attribution: score chunks by value match, evidence overlap, semantic similarity; top 3
    if not getattr(args, "no_attribution", False):
        print(f"Running attribution (top 3 chunks per column)…", file=sys.stderr)
        reconciled = enrich_reconciled_with_attribution(
            doc_id,
            reconciled,
            comparison_rows=rows,
            top_k=3,
            use_semantic=not getattr(args, "no_semantic", False),
        )

    result = {
        "doc_id": doc_id,
        "columns": reconciled,
    }
    print(json.dumps(result, indent=2))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    if logs_dir.exists() and not args.no_llm and HAS_LLM:
        log_files = sorted(logs_dir.glob("*_llm.txt"))
        if log_files:
            print(f"LLM logs: {logs_dir}/ ({len(log_files)} files)", file=sys.stderr)


if __name__ == "__main__":
    main()
