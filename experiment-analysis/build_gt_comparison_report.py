"""
Build GT comparison report: column-wise outputs of Gemini vs Landing AI vs GT.
Run from repo root: python3 experiment-analysis/build_gt_comparison_report.py

Data source: experiment-scripts 2/baseline_results_26thfeb/
  - landing_ai/{indication}/{study}/extraction_metadata.json
  - gemini/{indication}/{study}/extraction_metadata.json
  - GT (in order of precedence):
      1. evaluation/evaluation_results.json (ground_truth per column)
      2. dataset/Manual_Benchmark_GoldTable_cleaned.json (for matching doc names)
      3. Optional: GT_JSON_PER_INDICATION for custom GT files per indication

Output: experiment-analysis/gt_comparison_report.html
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_BASE = REPO_ROOT / "experiment-scripts 2" / "baseline_results_26thfeb"
GT_JSON = REPO_ROOT / "dataset" / "Manual_Benchmark_GoldTable_cleaned.json"
OUT_HTML = REPO_ROOT / "experiment-analysis" / "gt_comparison_report.html"

# Optional: custom GT JSON per indication (study_id or doc_name.pdf -> columns)
# Example: {"mRCC_1L": REPO_ROOT / "dataset" / "mRCC_1L_gold.json", "PARP_mCRPC": ...}
GT_JSON_PER_INDICATION: dict[str, Path] = {}

INDICATIONS = ["mCSPC", "mRCC_1L", "PARP_mCRPC"]


def discover_studies() -> dict[str, list[str]]:
    """Return {indication: [study_id, ...]} for studies in either landing_ai or gemini."""
    out: dict[str, list[str]] = {ind: [] for ind in INDICATIONS}
    for ind in INDICATIONS:
        seen = set()
        for sub in ("landing_ai", "gemini"):
            p = RESULTS_BASE / sub / ind
            if not p.exists():
                continue
            for d in sorted(p.iterdir()):
                if d.is_dir():
                    seen.add(d.name)
        out[ind] = sorted(seen)
    return out


def load_extraction_values(study_path: Path) -> dict[str, str]:
    """Load column -> value from extraction_metadata.json."""
    meta = study_path / "extraction_metadata.json"
    if not meta.exists():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        out = {}
        for col, cell in data.items():
            if isinstance(cell, dict) and "value" in cell:
                v = cell["value"]
                out[col] = str(v) if v is not None else ""
            else:
                out[col] = ""
        return out
    except Exception:
        return {}


def load_gt_from_eval(study_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load (ground_truth, definitions) from evaluation_results.json. Returns ({col: gt}, {col: def})."""
    for sub in ("evaluation", "latest/evaluation"):
        ev = study_path / sub / "evaluation_results.json"
        if ev.exists():
            try:
                data = json.loads(ev.read_text(encoding="utf-8"))
                cols = data.get("columns", {})
                gt_out = {}
                def_out = {}
                for col, cell in cols.items():
                    if isinstance(cell, dict):
                        v = cell.get("ground_truth")
                        gt_out[col] = str(v) if v is not None else ""
                        d = cell.get("definition", "")
                        def_out[col] = str(d) if d else ""
                    else:
                        gt_out[col] = ""
                        def_out[col] = ""
                return gt_out, def_out
            except Exception:
                pass
    return {}, {}


def load_gt_from_json_file(gt_path: Path, pdf_ids: list[str], all_columns: set[str]) -> dict[str, dict[str, str]]:
    """Load GT from a JSON file with 'data' array and Document Name column. Returns {pdf_id: {col: value}}."""
    if not gt_path.exists():
        return {pid: {} for pid in pdf_ids}
    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
        rows = raw.get("data", [])
        out: dict[str, dict[str, str]] = {pid: {} for pid in pdf_ids}
        for rec in rows:
            doc_cell = rec.get("Document Name")
            doc_val = doc_cell.get("value", "") if isinstance(doc_cell, dict) else (doc_cell or "")
            pdf_id = doc_val[:-4] if doc_val.endswith(".pdf") else doc_val
            if pdf_id not in out:
                continue
            for col in all_columns:
                cell = rec.get(col)
                if isinstance(cell, dict) and "value" in cell:
                    v = cell["value"]
                    out[pdf_id][col] = str(v) if v is not None else ""
                else:
                    out[pdf_id][col] = ""
        return out
    except Exception:
        return {pid: {} for pid in pdf_ids}


def get_column_definition(col_name: str) -> str:
    """Placeholder - could load from Definitions CSV if column exists there."""
    return ""


def main() -> None:
    studies_by_ind = discover_studies()

    # Build pdf list for dropdown: (indication, study_id) -> full label
    pdfs_out: list[dict] = []
    for ind in INDICATIONS:
        for study_id in studies_by_ind[ind]:
            pdfs_out.append({
                "id": f"{ind}|{study_id}",
                "indication": ind,
                "study_id": study_id,
                "name": f"[{ind}] {study_id}",
            })

    # Collect all columns across studies (per indication)
    all_columns_by_ind: dict[str, set[str]] = {ind: set() for ind in INDICATIONS}
    for ind in INDICATIONS:
        for study_id in studies_by_ind[ind]:
            for sub in ("landing_ai", "gemini"):
                p = RESULTS_BASE / sub / ind / study_id
                meta = p / "extraction_metadata.json"
                if meta.exists():
                    try:
                        data = json.loads(meta.read_text(encoding="utf-8"))
                        for col in data.keys():
                            if isinstance(col, str):
                                all_columns_by_ind[ind].add(col)
                    except Exception:
                        pass
            # Also from eval
            for sub in ("landing_ai", "gemini"):
                ev = RESULTS_BASE / sub / ind / study_id / "evaluation" / "evaluation_results.json"
                if ev.exists():
                    try:
                        data = json.loads(ev.read_text(encoding="utf-8"))
                        for col in data.get("columns", {}).keys():
                            all_columns_by_ind[ind].add(col)
                    except Exception:
                        pass

    # Pre-load Manual_Benchmark GT for all study IDs (keys in Manual_Benchmark are doc names without .pdf)
    all_study_ids = []
    for ind in INDICATIONS:
        all_study_ids.extend(studies_by_ind[ind])
    all_cols = set()
    for ind in INDICATIONS:
        all_cols |= all_columns_by_ind[ind]
    manual_gt = load_gt_from_json_file(GT_JSON, all_study_ids, all_cols)

    # Merge in per-indication GT if configured
    for ind in INDICATIONS:
        custom_path = GT_JSON_PER_INDICATION.get(ind)
        if custom_path:
            ind_ids = studies_by_ind[ind]
            custom_gt = load_gt_from_json_file(custom_path, ind_ids, all_cols)
            for pid, row in custom_gt.items():
                if row:
                    manual_gt[pid] = {**manual_gt.get(pid, {}), **row}

    # Build data payload
    data: dict[str, dict] = {}
    for item in pdfs_out:
        key = item["id"]
        ind = item["indication"]
        study_id = item["study_id"]

        land_path = RESULTS_BASE / "landing_ai" / ind / study_id
        gem_path = RESULTS_BASE / "gemini" / ind / study_id

        landing_ai_vals = load_extraction_values(land_path)
        gemini_vals = load_extraction_values(gem_path)

        # GT: prefer evaluation_results, else manual benchmark
        gt_vals, def_vals = load_gt_from_eval(land_path)
        if not gt_vals:
            gt_vals, def_vals = load_gt_from_eval(gem_path)
        if not gt_vals:
            gt_vals = manual_gt.get(study_id, {})

        # Columns: union of all three
        all_cols = set(landing_ai_vals.keys()) | set(gemini_vals.keys()) | set(gt_vals.keys())
        if not all_cols:
            all_cols = all_columns_by_ind.get(ind, set())

        columns_list = sorted(all_cols)

        data[key] = {
            "indication": ind,
            "study_id": study_id,
            "columns": columns_list,
            "landing_ai": landing_ai_vals,
            "gemini": gemini_vals,
            "ground_truth": gt_vals,
            "definitions": def_vals,
        }

    payload = {
        "pdfs": pdfs_out,
        "data": data,
    }

    html = build_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print(f"  PDFs: {len(pdfs_out)}")
    print(f"  Indications: {INDICATIONS}")


def build_html(payload: dict) -> str:
    data_js = json.dumps(payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GT Comparison: Gemini vs Landing AI</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; padding: 20px; color: #eee; min-height: 100vh; }}
        .container {{ max-width: 1920px; margin: 0 auto; }}
        h1 {{ text-align: center; padding: 16px; font-size: 22px; }}
        .controls {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: center; padding: 16px; background: #16213e; border-radius: 10px; margin-bottom: 16px; }}
        .controls label {{ display: flex; align-items: center; gap: 8px; }}
        .controls select {{ padding: 8px 12px; border-radius: 6px; background: #1a1a2e; color: #eee; border: 1px solid #4ecca3; min-width: 280px; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ border: 1px solid #333; padding: 8px 10px; text-align: left; vertical-align: top; }}
        th {{ background: #16213e; color: #4ecca3; position: sticky; top: 0; }}
        .col-name {{ max-width: 220px; word-break: break-word; font-weight: 500; }}
        .def-cell {{ max-width: 200px; word-break: break-word; font-size: 11px; color: #aaa; }}
        .gt-cell {{ max-width: 200px; word-break: break-word; background: rgba(78, 204, 163, 0.12); }}
        .landing-cell {{ max-width: 200px; word-break: break-word; }}
        .gemini-cell {{ max-width: 200px; word-break: break-word; }}
        .value {{ font-family: Consolas, 'Monaco', monospace; font-size: 12px; line-height: 1.4; }}
        .value.empty {{ color: #666; font-style: italic; }}
        .match {{ color: #4ecca3; }}
        .mismatch {{ color: #e94560; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GT Comparison: Gemini vs Landing AI vs Ground Truth</h1>
        <div class="controls">
            <label>Indication <select id="ind-select"></select></label>
            <label>Study / PDF <select id="pdf-select"></select></label>
            <label><input type="text" id="search-col" placeholder="Filter column..."> </label>
        </div>
        <div class="table-wrap">
            <table id="comparison-table">
                <thead id="table-head"></thead>
                <tbody id="table-body"></tbody>
            </table>
        </div>
    </div>
    <script type="application/json" id="comparison-data">{data_js}</script>
    <script>
        const PAYLOAD = JSON.parse(document.getElementById('comparison-data').textContent);
        const {{ pdfs, data }} = PAYLOAD;

        const indSelect = document.getElementById('ind-select');
        const pdfSelect = document.getElementById('pdf-select');
        const searchCol = document.getElementById('search-col');
        const thead = document.getElementById('table-head');
        const tbody = document.getElementById('table-body');

        const indications = [...new Set(pdfs.map(p => p.indication))];

        indSelect.innerHTML = '';
        indications.forEach(ind => {{
            const opt = document.createElement('option');
            opt.value = ind;
            opt.textContent = ind;
            indSelect.appendChild(opt);
        }});

        function updatePdfSelect() {{
            const ind = indSelect.value;
            pdfSelect.innerHTML = '';
            pdfs.filter(p => p.indication === ind).forEach(p => {{
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = p.study_id;
                pdfSelect.appendChild(opt);
            }});
            if (pdfSelect.options.length) pdfSelect.selectedIndex = 0;
            buildTable();
        }}

        indSelect.addEventListener('change', updatePdfSelect);
        pdfSelect.addEventListener('change', buildTable);
        searchCol.addEventListener('input', buildTable);

        function buildTable() {{
            const pdfId = pdfSelect.value;
            const search = (searchCol.value || '').trim().toLowerCase();
            const pdfData = data[pdfId];
            if (!pdfData) {{
                tbody.innerHTML = '<tr><td colspan="5">No data for this study.</td></tr>';
                return;
            }}

            let cols = pdfData.columns || [];
            if (search) cols = cols.filter(c => c.toLowerCase().includes(search));

            thead.innerHTML = '';
            const headRow = document.createElement('tr');
            headRow.innerHTML = '<th class="col-name">Column</th><th class="def-cell">Definition</th><th class="gt-cell">Ground Truth</th><th class="landing-cell">Landing AI</th><th class="gemini-cell">Gemini</th>';
            thead.appendChild(headRow);

            tbody.innerHTML = '';
            cols.forEach(col => {{
                const gt = (pdfData.ground_truth || {{}})[col] ?? '';
                const landing = (pdfData.landing_ai || {{}})[col] ?? '';
                const gemini = (pdfData.gemini || {{}})[col] ?? '';
                const def = (pdfData.definitions || {{}})[col] ?? '';

                const tr = document.createElement('tr');
                const gtMatch = _norm(landing) === _norm(gt);
                const gemMatch = _norm(gemini) === _norm(gt);
                tr.innerHTML =
                    '<td class="col-name">' + escapeHtml(col) + '</td>' +
                    '<td class="def-cell">' + escapeHtml(def || '') + '</td>' +
                    '<td class="gt-cell"><div class="value' + (gt ? '' : ' empty') + '">' + escapeHtml(gt || '—') + '</div></td>' +
                    '<td class="landing-cell"><div class="value' + (landing ? '' : ' empty') + (gtMatch && gt ? ' match' : (landing && gt && !gtMatch ? ' mismatch' : '')) + '">' + escapeHtml(landing || '—') + '</div></td>' +
                    '<td class="gemini-cell"><div class="value' + (gemini ? '' : ' empty') + (gemMatch && gt ? ' match' : (gemini && gt && !gemMatch ? ' mismatch' : '')) + '">' + escapeHtml(gemini || '—') + '</div></td>';
                tbody.appendChild(tr);
            }});
        }}

        function _norm(s) {{
            if (s == null) return '';
            return String(s).trim().toLowerCase().replace(/\\s+/g, ' ');
        }}

        function escapeHtml(s) {{
            const div = document.createElement('div');
            div.textContent = s;
            return div.innerHTML;
        }}

        updatePdfSelect();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
