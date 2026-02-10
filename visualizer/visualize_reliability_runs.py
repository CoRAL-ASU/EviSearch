#!/usr/bin/env python3
"""
Visualize reliability runs: for each column, show extraction value and eval score
for Run 1, Run 2, ... in a single table.

Usage:
    # From repo root; point at doc dir that contains reliability_run_1, reliability_run_2, ...
    python visualizer/visualize_reliability_runs.py --base "experiment-scripts/baselines_gemini_file_search/gemini-2.5-flash/NCT02799602_Hussain_ARASENS_JCO'23"

    # Or use --doc + --model (and optional --provider) to build path under experiment-scripts
    python visualizer/visualize_reliability_runs.py --doc "NCT02799602_Hussain_ARASENS_JCO'23" --model gemini-2.5-flash --provider gemini

Output: HTML file (default: experiment-analysis/reliability_report_<doc>.html)
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_BASE = PROJECT_ROOT / "experiment-scripts"


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_reliability_runs(base_dir: Path) -> List[Tuple[int, Path]]:
    """Find reliability_run_1, reliability_run_2, ... and return (run_id, path) sorted by id."""
    runs = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"reliability_run_(\d+)", d.name)
        if m:
            runs.append((int(m.group(1)), d))
    return sorted(runs, key=lambda x: x[0])


def score_class(overall: Optional[float]) -> str:
    if overall is None:
        return "na"
    if overall >= 0.99:
        return "correct"
    if overall >= 0.5:
        return "partial"
    return "wrong"


def format_score(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}"


def truncate(s: str, max_len: int = 80) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "…"


def generate_html(
    base_dir: Path,
    doc_name: str,
    runs: List[Tuple[int, Path]],
    run_data: List[dict],
    output_path: Path,
) -> None:
    """Build HTML table: one row per column, columns = GT, then for each run (value, score)."""
    # Collect all column names from extraction and evaluation across runs
    all_columns_set = set()
    ground_truth = {}
    for r in run_data:
        if r.get("extraction"):
            for k, v in r["extraction"].items():
                if isinstance(v, dict) and "value" in v:
                    all_columns_set.add(k)
        if r.get("evaluation_results") and "columns" in r["evaluation_results"]:
            for col, meta in r["evaluation_results"]["columns"].items():
                all_columns_set.add(col)
                if meta.get("ground_truth") is not None and col not in ground_truth:
                    ground_truth[col] = meta.get("ground_truth", "")
                elif col not in ground_truth:
                    ground_truth[col] = meta.get("ground_truth", "")
    all_columns = sorted(all_columns_set)
    for col in all_columns:
        ground_truth.setdefault(col, "")

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reliability Runs: """ + doc_name.replace("'", "&#39;") + """</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            min-height: 100vh;
            color: #eee;
        }
        .container { max-width: 100%; margin: 0 auto; overflow-x: auto; }
        h1 {
            text-align: center;
            padding: 24px;
            color: white;
            font-size: 26px;
        }
        .meta { text-align: center; color: #888; margin-bottom: 16px; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            background: #0f3460;
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            border: 1px solid #333;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background: #16213e;
            color: #4ecca3;
            position: sticky;
            top: 0;
        }
        .col-name { max-width: 200px; word-break: break-word; font-weight: 500; }
        .gt-cell { background: #1a1a2e; max-width: 160px; word-break: break-word; }
        .value-cell { max-width: 180px; word-break: break-word; }
        .score-cell { width: 70px; text-align: center; }
        .score-cell.correct { background: rgba(78, 204, 163, 0.25); color: #4ecca3; }
        .score-cell.partial { background: rgba(233, 69, 96, 0.2); color: #f0ad4e; }
        .score-cell.wrong { background: rgba(233, 69, 96, 0.35); color: #e94560; }
        .score-cell.na { color: #666; }
        tr:nth-child(even) { background: rgba(15, 52, 96, 0.5); }
        tr:hover { background: rgba(30, 60, 110, 0.8); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Reliability runs: """ + doc_name.replace("'", "&#39;") + """</h1>
        <p class="meta">Base: """ + str(base_dir).replace("<", "&lt;").replace(">", "&gt;") + """ &middot; """ + str(len(runs)) + """ runs</p>
        <table>
            <thead>
                <tr>
                    <th class="col-name">Column</th>
                    <th class="gt-cell">Ground Truth</th>
""")
    for run_id, _ in runs:
        html_parts.append(f'                    <th colspan="2">Run {run_id}</th>\n')
    html_parts.append("                </tr>\n                <tr>\n                    <th></th>\n                    <th></th>\n")
    for _ in runs:
        html_parts.append('                    <th>Value</th>\n                    <th>Score</th>\n')
    html_parts.append("                </tr>\n            </thead>\n            <tbody>\n")

    for col in all_columns:
        gt = ground_truth.get(col, "")
        gt_display = truncate(gt, 100).replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(f'                <tr>\n                    <td class="col-name">{col.replace("<", "&lt;").replace(">", "&gt;")}</td>\n')
        html_parts.append(f'                    <td class="gt-cell">{gt_display}</td>\n')
        for r in run_data:
            # Value from extraction
            ext = r.get("extraction") or {}
            if isinstance(ext.get(col), dict):
                val = ext[col].get("value", "")
            else:
                val = ""
            val_display = truncate(str(val), 80).replace("<", "&lt;").replace(">", "&gt;")
            # Score from evaluation
            ev = r.get("evaluation_results") or {}
            col_ev = (ev.get("columns") or {}).get(col) or {}
            overall = col_ev.get("overall")
            sc = score_class(overall)
            score_str = format_score(overall)
            html_parts.append(f'                    <td class="value-cell">{val_display}</td>\n')
            html_parts.append(f'                    <td class="score-cell {sc}">{score_str}</td>\n')
        html_parts.append("                </tr>\n")

    html_parts.append("            </tbody>\n        </table>\n    </div>\n</body>\n</html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"Wrote {output_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize reliability runs as a table (column × run: value + score)")
    ap.add_argument("--base", type=str, default=None, help="Path to doc dir containing reliability_run_1, ...")
    ap.add_argument("--doc", type=str, default=None, help="Document stem (e.g. NCT02799602_Hussain_ARASENS_JCO'23)")
    ap.add_argument("--model", type=str, default=None, help="Model folder name (e.g. gemini-2.5-flash)")
    ap.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai"])
    ap.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = ap.parse_args()

    base_dir = None
    if args.base:
        base_dir = Path(args.base)
        if not base_dir.is_absolute():
            base_dir = (PROJECT_ROOT / base_dir).resolve()
    elif args.doc and args.model:
        base_dir = SCRIPT_BASE / f"baselines_{args.provider}_file_search" / args.model / args.doc
        base_dir = base_dir.resolve()
    else:
        print("Provide either --base <path> or both --doc and --model")
        return 1

    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return 1

    runs = discover_reliability_runs(base_dir)
    if not runs:
        print(f"No reliability_run_* directories found under {base_dir}")
        return 1

    doc_name = base_dir.name
    run_data = []
    for run_id, run_path in runs:
        ext_path = run_path / "extraction_metadata.json"
        eval_path = run_path / "evaluation" / "evaluation_results.json"
        extraction = load_json(ext_path)
        evaluation_results = load_json(eval_path)
        run_data.append({
            "run_id": run_id,
            "extraction": extraction,
            "evaluation_results": evaluation_results,
        })

    out_path = args.output
    if not out_path:
        out_path = PROJECT_ROOT / "experiment-analysis" / f"reliability_report_{doc_name}.html"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()

    generate_html(base_dir, doc_name, runs, run_data, out_path)
    return 0


if __name__ == "__main__":
    exit(main())
