#!/usr/bin/env python3
"""
Comparative visualizer: Gemini 2.0 Flash, Gemini 2.5 Flash, and Our Pipeline.
For each document, shows 3 extraction results side-by-side with evaluation scores
(correctness/completeness/overall) and highlights right/wrong per column.

Usage:
    python visualizer/visualize_comparison.py
    python visualizer/visualize_comparison.py --doc "NCT00104715_Gravis_GETUG_EU'15"

Output: experiment-analysis/comparison_report.html (or --output path)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from collections import defaultdict

# Project root (parent of visualizer/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_BASE = PROJECT_ROOT / "experiment-scripts"
RESULTS_BASE = PROJECT_ROOT / "new_pipeline_outputs" / "results"

SOURCES = [
    ("Gemini 2.0 Flash", SCRIPT_BASE / "baselines_gemini_file_search" / "Gemini-2.0"),
    ("Gemini 2.5 Flash", SCRIPT_BASE / "baselines_gemini_file_search" / "Gemini-2.5"),
    ("Our Pipeline", RESULTS_BASE),
]

# For pipeline, extraction is in extraction/ and evaluation in evaluation/
# For baselines, extraction_metadata.json and evaluation/ are in trial dir directly


def load_json(path: Optional[Path]) -> Optional[dict]:
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def get_extraction_path(base: Path, doc: str, is_pipeline: bool) -> Optional[Path]:
    if is_pipeline:
        p = base / doc / "extraction" / "extraction_metadata.json"
    else:
        p = base / doc / "extraction_metadata.json"
    return p if p.exists() else None


def get_evaluation_paths(base: Path, doc: str, is_pipeline: bool) -> tuple:
    if is_pipeline:
        eval_dir = base / doc / "evaluation"
    else:
        eval_dir = base / doc / "evaluation"
    summary = eval_dir / "summary_metrics.json" if eval_dir.exists() else None
    results = eval_dir / "evaluation_results.json" if eval_dir.exists() else None
    return (
        summary if summary and summary.exists() else None,
        results if results and results.exists() else None,
    )


def load_doc_data(doc: str) -> dict:
    """Load extraction + evaluation for all 3 sources for one document."""
    out = {
        "doc": doc,
        "sources": {},
    }
    for label, base in SOURCES:
        is_pipeline = "Pipeline" in label
        ext_path = get_extraction_path(base, doc, is_pipeline)
        summary_path, results_path = get_evaluation_paths(base, doc, is_pipeline)

        extraction = load_json(ext_path) if ext_path else None
        summary = load_json(summary_path) if summary_path else None
        results = load_json(results_path) if results_path else None

        # extraction_metadata is dict of col -> {value, evidence, ...}; drop non-column keys if any
        if extraction and isinstance(extraction, dict):
            extraction = {k: v for k, v in extraction.items() if isinstance(v, dict) and ("value" in v or "evidence" in v)}

        out["sources"][label] = {
            "extraction": extraction,
            "summary": summary,
            "evaluation_results": results,
        }
    return out


def get_all_columns(data: dict) -> list[str]:
    """Union of columns across sources and evaluation results."""
    cols = set()
    for src in data["sources"].values():
        if src["extraction"]:
            cols.update(src["extraction"].keys())
        if src["evaluation_results"] and "columns" in src["evaluation_results"]:
            cols.update(src["evaluation_results"]["columns"].keys())
    return sorted(cols)


def score_class(overall: float) -> str:
    if overall is None:
        return "na"
    if overall >= 0.99:
        return "correct"
    if overall >= 0.5:
        return "partial"
    return "wrong"


def generate_html(all_data: list[dict], output_path: Path) -> None:
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison: Gemini 2.0, Gemini 2.5, Our Pipeline</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            min-height: 100vh;
            color: #eee;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        h1 {
            text-align: center;
            padding: 24px;
            color: white;
            font-size: 28px;
        }
        .doc-section {
            background: #0f3460;
            border-radius: 12px;
            margin-bottom: 32px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .doc-header {
            background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
            color: white;
            padding: 20px 24px;
            font-size: 20px;
            font-weight: 600;
        }
        .score-cards {
            display: flex;
            gap: 20px;
            padding: 24px;
            flex-wrap: wrap;
        }
        .score-card {
            flex: 1;
            min-width: 200px;
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #e94560;
        }
        .score-card h3 {
            font-size: 14px;
            color: #e94560;
            margin-bottom: 12px;
            text-transform: uppercase;
        }
        .score-card .correctness, .score-card .completeness, .score-card .overall {
            font-size: 13px;
            margin-bottom: 4px;
        }
        .score-card .overall { font-size: 18px; font-weight: bold; color: #4ecca3; margin-top: 8px; }
        .score-card.na { border-color: #555; opacity: 0.7; }
        .score-card.na h3 { color: #888; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #333;
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background: #16213e;
            color: #4ecca3;
            position: sticky;
            top: 0;
        }
        .col-name { max-width: 220px; word-break: break-word; }
        .gt-cell { background: #1a1a2e; max-width: 180px; word-break: break-word; }
        .source-cell {
            max-width: 200px;
            word-break: break-word;
        }
        .value { font-family: 'Consolas', monospace; font-size: 12px; }
        .score-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 4px;
        }
        .score-badge.correct { background: #28a745; color: white; }
        .score-badge.partial { background: #ffc107; color: #000; }
        .score-badge.wrong { background: #dc3545; color: white; }
        .score-badge.na { background: #6c757d; color: white; }
        .source-cell.correct { background: rgba(40, 167, 69, 0.2); }
        .source-cell.partial { background: rgba(255, 193, 7, 0.2); }
        .source-cell.wrong { background: rgba(220, 53, 69, 0.2); }
        .table-wrap { overflow-x: auto; padding: 0 24px 24px; }
        .filter-bar {
            padding: 16px 24px;
            background: #16213e;
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filter-bar button {
            padding: 8px 16px;
            border: 2px solid #e94560;
            background: transparent;
            color: #e94560;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
        }
        .filter-bar button:hover, .filter-bar button.active {
            background: #e94560;
            color: white;
        }
        .nav-docs {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            justify-content: center;
        }
        .nav-docs a {
            color: #4ecca3;
            padding: 8px 14px;
            background: #16213e;
            border-radius: 8px;
            text-decoration: none;
            font-size: 13px;
        }
        .nav-docs a:hover { background: #e94560; color: white; }
        .null-val { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparison: Gemini 2.0 Flash, Gemini 2.5 Flash, Our Pipeline</h1>
        <div class="nav-docs">
""")
    for d in all_data:
        doc_id = d["doc"].replace("'", "").replace(" ", "_")
        html_parts.append(f'            <a href="#doc-{doc_id}">{d["doc"]}</a>\n')

    html_parts.append("        </div>\n")

    for data in all_data:
        doc = data["doc"]
        doc_id = doc.replace("'", "").replace(" ", "_")
        columns = get_all_columns(data)

        html_parts.append(f'        <div class="doc-section" id="doc-{doc_id}">\n')
        html_parts.append(f'            <div class="doc-header">{doc}</div>\n')
        html_parts.append('            <div class="score-cards">\n')

        for label in [s[0] for s in SOURCES]:
            src = data["sources"].get(label, {})
            summary = src.get("summary") or {}
            overall = summary.get("overall", {})
            if overall:
                corr = overall.get("avg_correctness")
                comp = overall.get("avg_completeness")
                ov = overall.get("avg_overall")
                corr_p = f"{corr*100:.1f}%" if corr is not None else "—"
                comp_p = f"{comp*100:.1f}%" if comp is not None else "—"
                ov_p = f"{ov*100:.1f}%" if ov is not None else "—"
                n_cols = overall.get("total_columns", "—")
                html_parts.append(f"""                <div class="score-card">
                    <h3>{label}</h3>
                    <div class="correctness">Correctness: {corr_p}</div>
                    <div class="completeness">Completeness: {comp_p}</div>
                    <div class="overall">Overall: {ov_p}</div>
                    <div class="correctness">Columns: {n_cols}</div>
                </div>
""")
            else:
                html_parts.append(f"""                <div class="score-card na">
                    <h3>{label}</h3>
                    <div>No evaluation data</div>
                </div>
""")

        html_parts.append("            </div>\n")
        html_parts.append('            <div class="filter-bar">\n')
        html_parts.append('                <strong>Filter:</strong>\n')
        html_parts.append('                <button class="active" onclick="filterRows(this, \'all\')">All</button>\n')
        html_parts.append('                <button onclick="filterRows(this, \'correct\')">Correct</button>\n')
        html_parts.append('                <button onclick="filterRows(this, \'partial\')">Partial</button>\n')
        html_parts.append('                <button onclick="filterRows(this, \'wrong\')">Wrong</button>\n')
        html_parts.append("            </div>\n")
        html_parts.append('            <div class="table-wrap">\n')
        html_parts.append("                <table>\n")
        html_parts.append("                    <thead><tr>")
        html_parts.append("<th>Column</th><th>Ground truth</th>")
        for label in [s[0] for s in SOURCES]:
            html_parts.append(f"<th>{label}<br/><small>value & score</small></th>")
        html_parts.append("</tr></thead>\n<tbody>\n")

        for col in columns:
            gt_val = None
            row_classes = []
            cells = []

            for label in [s[0] for s in SOURCES]:
                src = data["sources"].get(label, {})
                ext = src.get("extraction") or {}
                ev_cols = (src.get("evaluation_results") or {}).get("columns") or {}
                ev = ev_cols.get(col) or {}

                val = None
                if ext and col in ext:
                    v = ext[col].get("value")
                    val = v if v is not None and (str(v).strip() not in ("", "NA", "N/A")) else None
                pred = ev.get("predicted")
                if pred is not None and str(pred).strip():
                    val = val or pred
                if gt_val is None and ev.get("ground_truth") is not None:
                    gt_val = ev.get("ground_truth")

                overall = ev.get("overall")
                if overall is not None:
                    sc = score_class(overall)
                    row_classes.append(sc)
                else:
                    sc = "na"

                val_str = (str(val)[:200] + "…") if val is not None and len(str(val)) > 200 else (str(val) if val is not None else "—")
                score_str = f"{overall*100:.0f}%" if overall is not None else "—"
                cells.append((val_str, score_str, sc))

            if not row_classes:
                row_class = "all-na"
            else:
                row_class = " ".join(set(row_classes))
            gt_str = (str(gt_val)[:200] + "…") if gt_val is not None and len(str(gt_val)) > 200 else (str(gt_val) if gt_val else "—")

            html_parts.append(f'                    <tr data-row-class="{row_class}">\n')
            html_parts.append(f'                        <td class="col-name">{col}</td>\n')
            html_parts.append(f'                        <td class="gt-cell value">{gt_str}</td>\n')
            for val_str, score_str, sc in cells:
                html_parts.append(f'                        <td class="source-cell {sc}"><div class="value">{val_str}</div><span class="score-badge {sc}">{score_str}</span></td>\n')
            html_parts.append("                    </tr>\n")

        html_parts.append("                </tbody></table>\n")
        html_parts.append("            </div>\n")
        html_parts.append("        </div>\n")

    html_parts.append("""
        <script>
        function filterRows(btn, kind) {
            document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.doc-section tbody tr').forEach(tr => {
                const c = tr.getAttribute('data-row-class') || '';
                const show = kind === 'all' || c.includes(kind);
                tr.style.display = show ? '' : 'none';
            });
        }
        </script>
    </div>
</body>
</html>
""")

    output_path.write_text("".join(html_parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report: Gemini 2.0, Gemini 2.5, Our Pipeline")
    parser.add_argument("--doc", type=str, help="Single document (stem) to report; default: all found")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Discover documents: from pipeline results or from first baseline
    docs = []
    if args.doc:
        docs = [args.doc]
    else:
        if RESULTS_BASE.exists():
            docs = sorted([d.name for d in RESULTS_BASE.iterdir() if d.is_dir()])
        if not docs and (SCRIPT_BASE / "baselines_gemini_file_search" / "Gemini-2.0").exists():
            docs = sorted([d.name for d in (SCRIPT_BASE / "baselines_gemini_file_search" / "Gemini-2.0").iterdir() if d.is_dir()])

    if not docs:
        print("No documents found. Ensure new_pipeline_outputs/results/ or baselines_gemini_file_search/Gemini-2.0/ exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data for {len(docs)} document(s)...")
    all_data = [load_doc_data(doc) for doc in docs]

    out_path = Path(args.output) if args.output else (PROJECT_ROOT / "experiment-analysis" / "comparison_report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(all_data, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
