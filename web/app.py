#!/usr/bin/env python3
"""
Simple web interface for the Clinical Trial Extraction Pipeline V2.
Run from project root: python web/app.py
Then open http://127.0.0.1:5000
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "visualizer"))

from flask import Flask, request, redirect, url_for, render_template_string, send_file
from src.main.main_v2 import run_pipeline_from_args
from src.config.config import RESULTS_BASE_DIR

app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pipeline V2 – Test Interface</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        label { display: block; margin-top: 12px; font-weight: 600; }
        input[type="text"] { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
        .option { margin: 8px 0; }
        .option input { margin-right: 8px; }
        button { margin-top: 20px; padding: 10px 24px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
        button:hover { background: #5568d3; }
        .error { color: #c00; margin-top: 12px; }
        .success { color: #0a0; margin-top: 12px; }
        .hint { font-size: 13px; color: #666; margin-top: 4px; }
    </style>
</head>
<body>
    <h1>Clinical Trial Extraction Pipeline V2</h1>
    <p>Enter the path to a PDF on the server and choose a pipeline option.</p>
    <form method="post" action="{{ url_for('run') }}">
        <label for="pdf_path">PDF path (server path)</label>
        <input type="text" id="pdf_path" name="pdf_path" value="{{ pdf_path or '' }}" placeholder="e.g. dataset/NCT00309985_Kriayako_CHAARTED_JCO'18.pdf" required>
        <p class="hint">Relative to project root or absolute path.</p>

        <label style="margin-top: 20px;">Pipeline option</label>
        <div class="option"><input type="radio" name="choice" value="1" {{ 'checked' if choice == '1' else '' }}> 1. Chunking only</div>
        <div class="option"><input type="radio" name="choice" value="2" {{ 'checked' if choice == '2' else '' }}> 2. Planning only (requires chunks)</div>
        <div class="option"><input type="radio" name="choice" value="3" {{ 'checked' if choice == '3' else '' }}> 3. Extraction only (requires chunks + plans)</div>
        <div class="option"><input type="radio" name="choice" value="4" {{ 'checked' if choice == '4' else '' }}> 4. Evaluation only (requires extraction)</div>
        <div class="option"><input type="radio" name="choice" value="5" {{ 'checked' if choice == '5' else '' }}> 5. Complete pipeline (all stages)</div>
        <div class="option"><input type="radio" name="choice" value="6" {{ 'checked' if choice == '6' else '' }}> 6. Planning → Extraction → Evaluation (resume from chunks)</div>

        <button type="submit">Run pipeline</button>
    </form>
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
    {% if message %}
    <p class="success">{{ message }}</p>
    {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, pdf_path="", choice="5", error=None, message=None)


@app.route("/run", methods=["POST"])
def run():
    pdf_path_raw = (request.form.get("pdf_path") or "").strip()
    choice = (request.form.get("choice") or "5").strip()
    if not pdf_path_raw:
        return render_template_string(
            INDEX_HTML, pdf_path=pdf_path_raw, choice=choice,
            error="Please enter a PDF path.", message=None
        )
    # Resolve path: if not absolute, relative to project root
    pdf_path = Path(pdf_path_raw)
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    if not pdf_path.exists():
        return render_template_string(
            INDEX_HTML, pdf_path=pdf_path_raw, choice=choice,
            error=f"PDF not found: {pdf_path}", message=None
        )
    run_dir, extraction_file, error = run_pipeline_from_args(pdf_path, choice)
    if error:
        return render_template_string(
            INDEX_HTML, pdf_path=pdf_path_raw, choice=choice,
            error=error, message=None
        )
    pdf_name = pdf_path.stem
    # If we have extraction output, generate visualization and redirect to view
    if extraction_file is not None:
        csv_path = extraction_file.parent / "extracted_table.csv"
        if csv_path.exists():
            try:
                from visualize_extraction import generate_html_report
                report_path = generate_html_report(
                    csv_path,
                    trial_name=pdf_name,
                    output_path=extraction_file.parent / f"{pdf_name}_visualization.html"
                )
                if report_path and Path(report_path).exists():
                    return redirect(url_for("view", rel=pdf_name))
            except Exception as e:
                return render_template_string(
                    INDEX_HTML, pdf_path=pdf_path_raw, choice=choice,
                    error=f"Pipeline finished but report failed: {e}", message=None
                )
    return render_template_string(
        INDEX_HTML, pdf_path=pdf_path_raw, choice=choice,
        error=None,
        message=f"Pipeline complete. Results: {run_dir}"
    )


@app.route("/view")
def view():
    rel = request.args.get("rel", "").strip()
    if not rel or ".." in rel or "/" in rel:
        return "Invalid report", 400
    report_path = RESULTS_BASE_DIR / rel / "latest" / "extraction" / f"{rel}_visualization.html"
    try:
        report_path = report_path.resolve()
        base = RESULTS_BASE_DIR.resolve()
        if not str(report_path).startswith(str(base)) or not report_path.exists():
            return "Report not found", 404
    except Exception:
        return "Report not found", 404
    return send_file(report_path, mimetype="text/html")


if __name__ == "__main__":
    print("Pipeline V2 web interface: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
