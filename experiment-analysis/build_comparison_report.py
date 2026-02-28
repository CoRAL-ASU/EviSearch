"""
Build comparison_report.html with embedded data for PDF selector and filters.
Run from repo root: python experiment-analysis/build_comparison_report.py

Data: 10 PDFs, 5 models: landing_ai_new, gemini_native, agent_extractor, search_agent, reconciliation_agent.
Output: experiment-analysis/comparison_report.html
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_BASE = REPO_ROOT / "experiment-scripts"
RESULTS_ROOT = REPO_ROOT / "new_pipeline_outputs" / "results"
DEFINITIONS_CSV = REPO_ROOT / "src/table_definitions/Definitions_with_eval_category.csv"
GOLD_TABLE_JSON = REPO_ROOT / "dataset/Manual_Benchmark_GoldTable_cleaned.json"
OUT_HTML = REPO_ROOT / "experiment-analysis/comparison_report.html"

# PDFs common across baselines (10)
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

# 5 models: (id, display_name, path_or_key)
# For eval models: path to dir with {pdf_id}/evaluation/evaluation_results.json
# For agent models: path key ("agent_extractor", "search_agent", "reconciliation_agent")
MODELS = [
    ("landing_ai_new", "Landing AI (new results)", SCRIPT_BASE / "baselines_landing_ai_new_results"),
    ("gemini_native", "Gemini Native 2.5 Flash", SCRIPT_BASE / "baselines_file_search_results/gemini_native/gemini-2.5-flash"),
    ("agent_extractor", "Agent Extractor", "agent_extractor"),
    ("search_agent", "Search Agent", "search_agent"),
    ("reconciliation_agent", "Reconciliation Agent", "reconciliation_agent"),
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


def load_updated_gold_table(pdf_list: list[str], definition_columns: list[str]) -> dict[str, dict[str, str]]:
    """
    Load Manual_Benchmark_GoldTable_cleaned.json and return
    pdf_id -> { column_name: value } for each PDF and each definition column.
    """
    if not GOLD_TABLE_JSON.exists():
        return {pdf_id: {} for pdf_id in pdf_list}
    raw = json.loads(GOLD_TABLE_JSON.read_text(encoding="utf-8"))
    records = raw.get("data", [])
    # Build pdf_id -> record (record is dict of key -> { "value": ..., "location": ... })
    pdf_to_record: dict[str, dict] = {}
    for rec in records:
        doc_name = (rec.get("Document Name") or {}).get("value", "")
        if not doc_name:
            continue
        # Match without .pdf: "NCT00104715_Gravis_GETUG_EU'15.pdf" -> "NCT00104715_Gravis_GETUG_EU'15"
        pdf_id = doc_name.replace(".pdf", "").strip()
        if pdf_id in pdf_list:
            pdf_to_record[pdf_id] = rec
    # For each pdf_id, extract values for each definition column
    out: dict[str, dict[str, str]] = {}
    for pdf_id in pdf_list:
        out[pdf_id] = {}
        rec = pdf_to_record.get(pdf_id, {})
        for col in definition_columns:
            cell = rec.get(col)
            if isinstance(cell, dict) and "value" in cell:
                out[pdf_id][col] = cell["value"] if cell["value"] is not None else ""
            else:
                out[pdf_id][col] = ""
    return out


def _avg(pairs: list[tuple[float, float]]) -> dict[str, float] | None:
    if not pairs:
        return None
    return {"correctness": sum(p[0] for p in pairs) / len(pairs), "completeness": sum(p[1] for p in pairs) / len(pairs)}


def load_ground_truth_from_gold_table(gold_path: Path) -> dict[str, dict[str, str]]:
    """Load gold table JSON; return dict[pdf_id, dict[column_name, value]].
    pdf_id = document name without .pdf (e.g. NCT00104715_Gravis_GETUG_EU'15).
    """
    if not gold_path.exists():
        return {}
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = raw.get("data") or []
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        doc_cell = row.get("Document Name")
        doc_value = doc_cell.get("value", "") if isinstance(doc_cell, dict) else (doc_cell or "")
        pdf_id = doc_value[:-4] if doc_value.endswith(".pdf") else doc_value
        if not pdf_id:
            continue
        out[pdf_id] = {}
        for col, cell in row.items():
            if isinstance(cell, dict):
                out[pdf_id][col] = cell.get("value") if cell.get("value") is not None else ""
            else:
                out[pdf_id][col] = str(cell) if cell is not None else ""
    return out


def load_eval(pdf_id: str, model_path: Path) -> dict | None:
    for sub in ("evaluation", "latest/evaluation"):
        path = model_path / pdf_id / sub / "evaluation_results.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_agent_columns(pdf_id: str) -> dict[str, dict] | None:
    """Load agent extractor results from agent_extractor/extraction_results.json."""
    path = RESULTS_ROOT / pdf_id / "agent_extractor" / "extraction_results.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        columns = d.get("columns", {})
        if not isinstance(columns, dict):
            return None
        out = {}
        for col, v in columns.items():
            if isinstance(v, dict):
                val = v.get("value")
            else:
                val = v
            out[col] = {"value": str(val) if val is not None else ""}
        return out
    except Exception:
        return None


def load_search_agent_columns(pdf_id: str) -> dict[str, dict] | None:
    """Load search agent results from search_agent/extraction_results.json."""
    path = RESULTS_ROOT / pdf_id / "search_agent" / "extraction_results.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        columns = d.get("columns", {})
        if not isinstance(columns, dict):
            return None
        out = {}
        for col, v in columns.items():
            if isinstance(v, dict):
                val = v.get("value")
            else:
                val = v
            out[col] = {"value": str(val) if val is not None else ""}
        return out
    except Exception:
        return None


def load_reconciliation_agent_columns(pdf_id: str) -> dict[str, dict] | None:
    """Load reconciliation agent results from reconciliation_agent/reconciled_results.json."""
    path = RESULTS_ROOT / pdf_id / "reconciliation_agent" / "reconciled_results.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        columns = d.get("columns", {})
        if not isinstance(columns, dict):
            return None
        out = {}
        for col, v in columns.items():
            if isinstance(v, dict):
                val = v.get("value")
            else:
                val = v
            out[col] = {"value": str(val) if val is not None else "", "reason": str(v.get("reasoning", "")) if isinstance(v, dict) else ""}
        return out if out else None
    except Exception:
        return None


def list_reconciliation_agent_logs(pdf_id: str) -> list[str]:
    """List conversation log paths for reconciliation agent (relative to repo root)."""
    logs_dir = RESULTS_ROOT / pdf_id / "reconciliation_agent" / "verification_logs"
    if not logs_dir.exists():
        return []
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in logs_dir.glob("*_conversation.json"))


def main() -> None:
    definitions = load_definitions()
    col_to_label = {d["column"]: d["label"] for d in definitions}
    col_to_eval = {d["column"]: d["eval_category"] for d in definitions}

    models_out = [{"id": mid, "name": name} for mid, name, _ in MODELS]
    pdfs_out = [{"id": pdf, "name": pdf} for pdf in PDF_LIST]

    definition_columns = [d["column"] for d in definitions]
    ground_truth_updated = load_updated_gold_table(PDF_LIST, definition_columns)

    data: dict = {}
    for pdf_id in PDF_LIST:
        agent_cols = load_agent_columns(pdf_id)
        search_agent_cols = load_search_agent_columns(pdf_id)
        rec_cols = load_reconciliation_agent_columns(pdf_id)
        data[pdf_id] = {
            "ground_truth": {},
            "ground_truth_updated": ground_truth_updated.get(pdf_id, {}),
            "models": {m["id"]: {} for m in models_out},
            "agent_stats": {"columns_filled": len(agent_cols)} if agent_cols else None,
            "search_agent_stats": {"columns_filled": len(search_agent_cols)} if search_agent_cols else None,
            "reconciliation_agent_stats": {"columns_filled": len(rec_cols)} if rec_cols else None,
            "reconciliation_logs": list_reconciliation_agent_logs(pdf_id),
        }
        for mid, _name, path_or_key in MODELS:
            if isinstance(path_or_key, Path):
                ev = load_eval(pdf_id, path_or_key)
                if ev is None:
                    continue
                for col_name, col_data in ev.get("columns", {}).items():
                    data[pdf_id]["models"][mid][col_name] = {
                        "value": col_data.get("predicted", ""),
                        "correctness": float(col_data.get("correctness", 0)),
                        "completeness": float(col_data.get("completeness", 0)),
                        "reason": col_data.get("reason", ""),
                    }
            elif path_or_key == "agent_extractor" and agent_cols:
                for col_name, col_data in agent_cols.items():
                    data[pdf_id]["models"][mid][col_name] = col_data
            elif path_or_key == "search_agent" and search_agent_cols:
                for col_name, col_data in search_agent_cols.items():
                    data[pdf_id]["models"][mid][col_name] = col_data
            elif path_or_key == "reconciliation_agent" and rec_cols:
                for col_name, col_data in rec_cols.items():
                    data[pdf_id]["models"][mid][col_name] = col_data

    # Unique labels for filter dropdown (preserve order)
    labels_seen = []
    for d in definitions:
        if d["label"] not in labels_seen:
            labels_seen.append(d["label"])

    # Per-PDF, per-model: average correctness/completeness for eval models (landing_ai_new, gemini_native)
    eval_model_ids = ("landing_ai_new", "gemini_native")
    for pdf_id in PDF_LIST:
        data[pdf_id]["summary"] = {}
        for m in models_out:
            mid = m["id"]
            model_cols = data[pdf_id]["models"].get(mid, {})
            if not model_cols:
                data[pdf_id]["summary"][mid] = None
                continue
            if mid not in eval_model_ids:
                data[pdf_id]["summary"][mid] = None
                continue
            by_cat: dict[str, list[tuple[float, float]]] = {"exact_match": [], "structured_text": [], "numeric_tolerance": []}
            all_c, all_k = [], []
            for col_name, cell in model_cols.items():
                c, k = cell.get("correctness"), cell.get("completeness")
                if c is None or k is None:
                    continue
                cat = col_to_eval.get(col_name, "")
                if cat in by_cat:
                    by_cat[cat].append((float(c), float(k)))
                all_c.append(float(c))
                all_k.append(float(k))
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
        .gt-updated-cell {{ max-width: 180px; word-break: break-word; background: rgba(78, 204, 163, 0.08); }}
        .model-cell {{ max-width: 200px; word-break: break-word; }}
        .model-cell .value {{ font-family: Consolas, monospace; font-size: 12px; }}
        .model-cell .reason {{ font-size: 11px; color: #888; margin-top: 4px; margin-bottom: 4px; line-height: 1.3; white-space: pre-wrap; }}
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
            if (!pdfData) {{ headerBand.style.display = 'none'; return; }}
            const hasAny = pdfData.summary && Object.values(pdfData.summary).some(s => s)
                || pdfData.agent_stats || pdfData.search_agent_stats || pdfData.reconciliation_agent_stats;
            if (!hasAny) {{ headerBand.style.display = 'none'; return; }}
            headerBand.style.display = 'flex';
            headerBand.innerHTML = '';
            models.forEach(m => {{
                const div = document.createElement('div');
                div.className = 'model-summary';
                let html = '<h3>' + escapeHtml(m.name) + '</h3>';
                const sum = pdfData.summary && pdfData.summary[m.id];
                if (m.id === 'landing_ai_new' || m.id === 'gemini_native') {{
                    if (!sum) {{ div.classList.add('na'); html += '<div>N/A</div>'; }}
                    else {{
                        const o = sum.overall;
                        html += '<div class="overall">Overall: C ' + (o.correctness * 100).toFixed(1) + '%, K ' + (o.completeness * 100).toFixed(1) + '%</div>';
                        html += '<div class="by-cat">';
                        ['exact_match', 'structured_text', 'numeric_tolerance'].forEach(cat => {{
                            const c = sum[cat];
                            if (c) html += '<span>' + cat + ': C ' + (c.correctness * 100).toFixed(1) + '%, K ' + (c.completeness * 100).toFixed(1) + '%</span>';
                        }});
                        html += '</div>';
                    }}
                }} else {{
                    const stats = m.id === 'agent_extractor' ? pdfData.agent_stats
                        : m.id === 'search_agent' ? pdfData.search_agent_stats
                        : pdfData.reconciliation_agent_stats;
                    if (stats) html += '<div class="overall">Columns filled: ' + stats.columns_filled + '</div>';
                    else {{ div.classList.add('na'); html += '<div>N/A</div>'; }}
                    if (m.id === 'reconciliation_agent' && (pdfData.reconciliation_logs || []).length > 0) {{
                        html += '<div class="by-cat"><a href="../' + pdfData.reconciliation_logs[0] + '" target="_blank" style="color:#4ecca3;">View logs</a></div>';
                    }}
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
            if (!pdfData) {{ tbody.innerHTML = '<tr><td colspan="' + (6 + models.length) + '">No data for this PDF.</td></tr>'; return; }}

            const gt = pdfData.ground_truth || {{}};
            const gtUpdated = pdfData.ground_truth_updated || {{}};
            const modelCols = pdfData.models || {{}};

            rows = rows.filter(row => {{
                let hasCorr = corrFilter === null;
                let hasComp = compFilter === null;
                for (const m of models) {{
                    const cell = (modelCols[m.id] || {{}})[row.column];
                    if (!cell) continue;
                    if (cell.correctness != null && corrFilter !== null && cell.correctness === corrFilter) hasCorr = true;
                    if (cell.completeness != null && compFilter !== null && cell.completeness === compFilter) hasComp = true;
                }}
                return hasCorr && hasComp;
            }});

            thead.innerHTML = '';
            const headRow = document.createElement('tr');
            headRow.innerHTML = '<th>Column</th><th>Label</th><th>Category</th><th>Definition</th><th>Ground truth</th><th>Ground truth (updated)</th>' +
                models.map(m => '<th>' + m.name + '</th>').join('');
            thead.appendChild(headRow);

            tbody.innerHTML = '';
            const gtNotFoundMsg = pdfData.gt_not_found ? 'Column not found for PDF' : '';
            rows.forEach(row => {{
                const tr = document.createElement('tr');
                const gtVal = gt[row.column] ?? '';
                const gtUpdatedVal = gtUpdated[row.column] ?? '';
                tr.innerHTML = '<td class="col-name">' + escapeHtml(row.column) + '</td><td>' + escapeHtml(row.label) + '</td><td class="cat-cell">' + escapeHtml(row.eval_category || '') + '</td><td class="def-cell">' + escapeHtml(row.definition || '') + '</td><td class="gt-cell">' + escapeHtml(gtVal) + '</td><td class="gt-cell gt-updated-cell">' + escapeHtml(gtUpdatedVal) + '</td>';
                models.forEach(m => {{
                    const cell = (modelCols[m.id] || {{}})[row.column];
                    const td = document.createElement('td');
                    td.className = 'model-cell';
                    if (!cell) {{
                        td.classList.add('na');
                        td.textContent = 'N/A';
                    }} else {{
                        const reasonHtml = cell.reason ? '<div class="reason">' + escapeHtml(cell.reason) + '</div>' : '';
                        let scoresHtml = '';
                        if (cell.correctness != null && cell.completeness != null) {{
                            scoresHtml = '<div class="scores">C: ' + cell.correctness + ', K: ' + cell.completeness + '</div>';
                        }} else if (cell.confidence) {{
                            let s = 'conf: ' + cell.confidence;
                            if (cell.retry_value != null) s += ' | retry: ' + (cell.values_match ? 'match' : 'diff');
                            scoresHtml = '<div class="scores">' + s + '</div>';
                        }}
                        td.innerHTML = '<div class="value">' + escapeHtml(String(cell.value)) + '</div>' + reasonHtml + scoresHtml;
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
