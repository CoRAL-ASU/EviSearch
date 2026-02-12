"""
Build comparison_report.html with embedded data for PDF selector and filters.
Run from repo root: python experiment-analysis/build_comparison_report.py

Data: 10 PDFs, extensible model list (native + landing_ai for now).
Output: experiment-analysis/comparison_report.html
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BASE = REPO_ROOT / "experiment-scripts"
DEFINITIONS_CSV = REPO_ROOT / "src/table_definitions/Definitions_with_eval_category.csv"
OUT_HTML = REPO_ROOT / "experiment-analysis/comparison_report.html"

# PDFs common to gemini_native and landing_ai_new (10)
PDF_LIST = [
    "NCT00104715_Gravis_GETUG_EU'15",
    "NCT00268476_Attard_STAMPEDE_Lancet'23",
    "NCT00268476_James_STAMPEDE_IJC'22",
    "NCT00309985_Kriayako_CHAARTED_JCO'18",
    "NCT00309985_Sweeney_CHAARTED_NEJM'15",
    "NCT01809691_Aggarwal_SWOG1216_JCO'22",
    "NCT01957436_Fizazi_PEACE1_Lancet'22",
    "NCT02446405_Sweeney_ENZAMET_Lancet Onc'23",
    "NCT02799602_Hussain_ARASENS_JCO'23",
    "NCT02799602_Smith_ARASENS_NEJM'22",
]

# Extensible: (display_name, path relative to experiment-scripts)
MODELS = [
    ("Gemini 2.5 Flash (native)", SCRIPT_BASE / "baselines_file_search_results/gemini_native/gemini-2.5-flash"),
    ("LandingAI (ADE)", SCRIPT_BASE / "baselines_landing_ai_new_results"),
    # ("Gemini 2.5 Flash (free-form)", SCRIPT_BASE / "baselines_file_search_results/free_form/gemini-2.5-flash"),
    # ("Our Pipeline", ...),
]


def load_definitions() -> list[dict]:
    rows = []
    with open(DEFINITIONS_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "column": r["Column Name"].strip(),
                "label": r["Label"].strip(),
                "eval_category": r["eval_category"].strip(),
                "definition": r.get("Definition", "").strip(),
            })
    return rows


def _avg(pairs: list[tuple[float, float]]) -> dict[str, float] | None:
    if not pairs:
        return None
    return {"correctness": sum(p[0] for p in pairs) / len(pairs), "completeness": sum(p[1] for p in pairs) / len(pairs)}


def load_eval(pdf_id: str, model_path: Path) -> dict | None:
    path = model_path / pdf_id / "evaluation" / "evaluation_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    definitions = load_definitions()
    col_to_label = {d["column"]: d["label"] for d in definitions}
    col_to_eval = {d["column"]: d["eval_category"] for d in definitions}

    models_out = [{"id": f"m{i}", "name": name} for i, (name, _) in enumerate(MODELS)]
    pdfs_out = [{"id": pdf, "name": pdf} for pdf in PDF_LIST]

    # data[pdf_id] = { ground_truth: { col: str }, models: { model_id: { col: { value, correctness, completeness } } } }
    data: dict = {}
    for pdf_id in PDF_LIST:
        data[pdf_id] = {"ground_truth": {}, "models": {m["id"]: {} for m in models_out}}
        for mi, (_, model_path) in enumerate(MODELS):
            mid = models_out[mi]["id"]
            ev = load_eval(pdf_id, model_path)
            if ev is None:
                continue
            for col_name, col_data in ev.get("columns", {}).items():
                data[pdf_id]["ground_truth"][col_name] = col_data.get("ground_truth", "")
                data[pdf_id]["models"][mid][col_name] = {
                    "value": col_data.get("predicted", ""),
                    "correctness": float(col_data.get("correctness", 0)),
                    "completeness": float(col_data.get("completeness", 0)),
                }

    # Unique labels for filter dropdown (preserve order)
    labels_seen = []
    for d in definitions:
        if d["label"] not in labels_seen:
            labels_seen.append(d["label"])

    # Per-PDF, per-model: average correctness/completeness overall and by eval_category
    for pdf_id in PDF_LIST:
        data[pdf_id]["summary"] = {}
        for m in models_out:
            mid = m["id"]
            model_cols = data[pdf_id]["models"].get(mid, {})
            if not model_cols:
                data[pdf_id]["summary"][mid] = None
                continue
            by_cat: dict[str, list[tuple[float, float]]] = {"exact_match": [], "structured_text": [], "numeric_tolerance": []}
            all_c, all_k = [], []
            for col_name, cell in model_cols.items():
                cat = col_to_eval.get(col_name, "")
                if cat in by_cat:
                    by_cat[cat].append((cell["correctness"], cell["completeness"]))
                all_c.append(cell["correctness"])
                all_k.append(cell["completeness"])
            data[pdf_id]["summary"][mid] = {
                "overall": {"correctness": sum(all_c) / len(all_c) if all_c else 0, "completeness": sum(all_k) / len(all_k) if all_k else 0},
                "exact_match": _avg(by_cat["exact_match"]),
                "structured_text": _avg(by_cat["structured_text"]),
                "numeric_tolerance": _avg(by_cat["numeric_tolerance"]),
            }

    payload = {
        "pdfs": pdfs_out,
        "models": models_out,
        "definitions": definitions,
        "labels": labels_seen,
        "data": data,
    }

    html = build_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")


def build_html(payload: dict) -> str:
    data_js = json.dumps(payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baseline comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; padding: 20px; color: #eee; min-height: 100vh; }}
        .container {{ max-width: 1920px; margin: 0 auto; }}
        h1 {{ text-align: center; padding: 16px; font-size: 22px; }}
        .controls {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: center; padding: 16px; background: #16213e; border-radius: 10px; margin-bottom: 16px; }}
        .controls label {{ display: flex; align-items: center; gap: 8px; }}
        .controls select {{ padding: 8px 12px; border-radius: 6px; background: #1a1a2e; color: #eee; border: 1px solid #4ecca3; min-width: 200px; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ border: 1px solid #333; padding: 8px 10px; text-align: left; vertical-align: top; }}
        th {{ background: #16213e; color: #4ecca3; position: sticky; top: 0; }}
        .col-name {{ max-width: 220px; word-break: break-word; }}
        .def-cell {{ max-width: 280px; word-break: break-word; font-size: 12px; color: #bbb; }}
        .cat-cell {{ white-space: nowrap; font-size: 12px; }}
        .gt-cell {{ max-width: 180px; word-break: break-word; }}
        .model-cell {{ max-width: 200px; word-break: break-word; }}
        .model-cell .value {{ font-family: Consolas, monospace; font-size: 12px; }}
        .model-cell .scores {{ font-size: 11px; color: #aaa; margin-top: 4px; }}
        .model-cell.na {{ color: #666; font-style: italic; }}
        .header-band {{ display: flex; flex-wrap: wrap; gap: 20px; padding: 16px 20px; background: #16213e; border-radius: 10px; margin-bottom: 16px; }}
        .header-band .model-summary {{ flex: 1; min-width: 280px; background: #1a1a2e; border-radius: 8px; padding: 14px; border: 1px solid #4ecca3; }}
        .header-band .model-summary h3 {{ font-size: 14px; color: #4ecca3; margin-bottom: 10px; }}
        .header-band .model-summary .overall {{ font-size: 14px; margin-bottom: 8px; }}
        .header-band .model-summary .by-cat {{ font-size: 12px; color: #bbb; }}
        .header-band .model-summary .by-cat span {{ display: inline-block; margin-right: 12px; margin-top: 4px; }}
        .header-band .model-summary.na {{ border-color: #555; opacity: 0.7; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Baseline comparison: by PDF</h1>
        <div id="header-band" class="header-band" style="display: none;"></div>
        <div class="controls">
            <label>PDF <select id="pdf-select"></select></label>
            <label>Group (Label) <select id="group-select"><option value="">All</option></select></label>
            <label>Correctness <select id="correctness-select"><option value="">Any</option><option value="0">0</option><option value="0.5">0.5</option><option value="1">1</option></select></label>
            <label>Completeness <select id="completeness-select"><option value="">Any</option><option value="0">0</option><option value="0.5">0.5</option><option value="1">1</option></select></label>
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
        const {{ pdfs, models, definitions, labels, data }} = PAYLOAD;

        const pdfSelect = document.getElementById('pdf-select');
        const groupSelect = document.getElementById('group-select');
        const correctnessSelect = document.getElementById('correctness-select');
        const completenessSelect = document.getElementById('completeness-select');
        const thead = document.getElementById('table-head');
        const tbody = document.getElementById('table-body');
        const headerBand = document.getElementById('header-band');

        function renderHeaderBand(pdfId) {{
            const pdfData = data[pdfId];
            if (!pdfData || !pdfData.summary) {{ headerBand.style.display = 'none'; return; }}
            headerBand.style.display = 'flex';
            headerBand.innerHTML = '';
            models.forEach(m => {{
                const sum = pdfData.summary[m.id];
                const div = document.createElement('div');
                div.className = 'model-summary' + (sum ? '' : ' na');
                let html = '<h3>' + escapeHtml(m.name) + '</h3>';
                if (!sum) {{
                    html += '<div>N/A</div>';
                }} else {{
                    const o = sum.overall;
                    html += '<div class="overall">Overall: C ' + (o.correctness * 100).toFixed(1) + '%, K ' + (o.completeness * 100).toFixed(1) + '%</div>';
                    html += '<div class="by-cat">';
                    ['exact_match', 'structured_text', 'numeric_tolerance'].forEach(cat => {{
                        const c = sum[cat];
                        if (c) html += '<span>' + cat + ': C ' + (c.correctness * 100).toFixed(1) + '%, K ' + (c.completeness * 100).toFixed(1) + '%</span>';
                    }});
                    html += '</div>';
                }}
                div.innerHTML = html;
                headerBand.appendChild(div);
            }});
        }}

        pdfs.forEach(p => {{
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            pdfSelect.appendChild(opt);
        }});
        labels.forEach(l => {{
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = l;
            groupSelect.appendChild(opt);
        }});

        function buildTable() {{
            const pdfId = pdfSelect.value;
            const group = groupSelect.value;
            const corrFilter = correctnessSelect.value === '' ? null : parseFloat(correctnessSelect.value);
            const compFilter = completenessSelect.value === '' ? null : parseFloat(completenessSelect.value);

            let rows = definitions.map(d => ({{...d}}));
            if (group) rows = rows.filter(r => r.label === group);

            const pdfData = data[pdfId];
            renderHeaderBand(pdfId);
            if (!pdfData) {{ tbody.innerHTML = '<tr><td colspan="' + (5 + models.length) + '">No data for this PDF.</td></tr>'; return; }}

            const gt = pdfData.ground_truth || {{}};
            const modelCols = pdfData.models || {{}};

            rows = rows.filter(row => {{
                let hasCorr = corrFilter === null;
                let hasComp = compFilter === null;
                for (const m of models) {{
                    const cell = (modelCols[m.id] || {{}})[row.column];
                    if (!cell) continue;
                    if (corrFilter !== null && cell.correctness === corrFilter) hasCorr = true;
                    if (compFilter !== null && cell.completeness === compFilter) hasComp = true;
                }}
                return hasCorr && hasComp;
            }});

            thead.innerHTML = '';
            const headRow = document.createElement('tr');
            headRow.innerHTML = '<th>Column</th><th>Label</th><th>Category</th><th>Definition</th><th>Ground truth</th>' +
                models.map(m => '<th>' + m.name + '</th>').join('');
            thead.appendChild(headRow);

            tbody.innerHTML = '';
            rows.forEach(row => {{
                const tr = document.createElement('tr');
                const gtVal = gt[row.column] ?? '';
                tr.innerHTML = '<td class="col-name">' + escapeHtml(row.column) + '</td><td>' + escapeHtml(row.label) + '</td><td class="cat-cell">' + escapeHtml(row.eval_category || '') + '</td><td class="def-cell">' + escapeHtml(row.definition || '') + '</td><td class="gt-cell">' + escapeHtml(gtVal) + '</td>';
                models.forEach(m => {{
                    const cell = (modelCols[m.id] || {{}})[row.column];
                    const td = document.createElement('td');
                    td.className = 'model-cell';
                    if (!cell) {{
                        td.classList.add('na');
                        td.textContent = 'N/A';
                    }} else {{
                        td.innerHTML = '<div class="value">' + escapeHtml(String(cell.value)) + '</div><div class="scores">C: ' + cell.correctness + ', K: ' + cell.completeness + '</div>';
                    }}
                    tr.appendChild(td);
                }});
                tbody.appendChild(tr);
            }});
        }}

        function escapeHtml(s) {{
            const div = document.createElement('div');
            div.textContent = s;
            return div.innerHTML;
        }}

        pdfSelect.addEventListener('change', buildTable);
        groupSelect.addEventListener('change', buildTable);
        correctnessSelect.addEventListener('change', buildTable);
        completenessSelect.addEventListener('change', buildTable);
        buildTable();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
