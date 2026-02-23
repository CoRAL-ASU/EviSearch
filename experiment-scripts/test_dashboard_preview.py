#!/usr/bin/env python3
"""
Test script: generate dashboard payload and serve a minimal preview.

Usage:
  cd /mnt/data1/nahuja11/Mayo/CoRal-Map-Make
  python experiment-scripts/test_dashboard_preview.py [doc_id]

  Then open http://127.0.0.1:8765 in your browser.

  The script:
  1. Loads real comparison data for the document
  2. Runs the explainability service (get_document_dashboard)
  3. Saves JSON to experiment-scripts/test_dashboard_output.json
  4. Starts a minimal HTTP server with a preview page
"""
from __future__ import annotations

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default doc with good data
DEFAULT_DOC = "NCT02799602_Hussain_ARASENS_JCO'23"
OUTPUT_JSON = PROJECT_ROOT / "experiment-scripts" / "test_dashboard_output.json"
PORT = 8765


def get_dashboard_data(doc_id: str) -> dict:
    from web.explainability_service import get_document_dashboard
    return get_document_dashboard(doc_id)


def build_preview_html(data: dict) -> str:
    """Minimal HTML to preview the dashboard structure."""
    h = data.get("highlights", [])
    cr = data.get("core_reasoning", [])
    nf = data.get("not_found_with_reasoning", [])
    ag = data.get("agreement_summary", [])

    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    highlights_html = ""
    for x in h:
        t = x.get("type", "")
        cn = x.get("column_name", "")
        sm = x.get("summary", "")
        if t == "disagreement":
            vals = x.get("values_by_method", {})
            highlights_html += f'<div class="card highlight disagreement"><strong>Disagreement:</strong> {esc(cn)} — {esc(sm)}<pre>{json.dumps(vals, indent=2)}</pre></div>'
        elif t == "multi_candidate":
            highlights_html += f'<div class="card highlight multi"><strong>Multi-candidate:</strong> {esc(cn)} — {esc(sm)}</div>'
        else:
            highlights_html += f'<div class="card highlight"><strong>{esc(t)}:</strong> {esc(cn)} — {esc(sm)}</div>'

    core_html = ""
    for r in cr[:15]:  # Show first 15
        col = r.get("column_name", "")
        val = r.get("primary_value", "")
        reason = r.get("reasoning", {})
        ev = reason.get("evidence", "")[:500]
        src = reason.get("source", "")
        conf = reason.get("confidence", "")
        why = reason.get("why_not_found", "")
        by_m = r.get("by_method", {})
        by_str = ", ".join(f"{k}: {v.get('value', '')[:40]}" for k, v in by_m.items())
        core_html += f"""
        <div class="card reasoning">
            <div class="col-name">{esc(col)}</div>
            <div class="value">{esc(val)}</div>
            <div class="evidence">{esc(ev)}</div>
            <div class="meta">Source: {esc(src)} | Confidence: {esc(conf)}</div>
            {f'<div class="why-not">Why not found: {esc(why)[:300]}</div>' if why else ''}
            <div class="by-method"><small>{esc(by_str)}</small></div>
        </div>
        """

    nf_html = ""
    for x in nf:
        cn = x.get("column_name", "")
        where = x.get("where_we_looked", "")
        why = x.get("why", "")
        nf_html += f'<div class="card not-found"><strong>{esc(cn)}</strong><br>Where: {esc(where)}<br>Why: {esc(why)[:300]}</div>'

    ag_html = ""
    for x in ag:
        cn = x.get("column_name", "")
        v = x.get("agreed_value", "")
        methods = x.get("methods_agreeing", [])
        ag_html += f'<div class="card agreement"><strong>{esc(cn)}</strong> = {esc(v)} (all agree: {", ".join(methods)})</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dashboard Preview — {esc(data.get("doc_id", ""))}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #22d3ee; font-size: 1.5rem; }}
        h2 {{ color: #94a3b8; font-size: 1.1rem; margin-top: 32px; margin-bottom: 12px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
        .card.reasoning {{ border-left: 4px solid #06b6d4; }}
        .card.highlight {{ border-left: 4px solid #f59e0b; }}
        .card.highlight.disagreement {{ border-left-color: #ef4444; }}
        .card.not-found {{ border-left: 4px solid #64748b; }}
        .card.agreement {{ border-left: 4px solid #10b981; }}
        .col-name {{ font-weight: 600; color: #38bdf8; margin-bottom: 4px; }}
        .value {{ font-size: 1.1rem; margin: 8px 0; }}
        .evidence {{ color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin: 8px 0; }}
        .meta {{ font-size: 0.8rem; color: #64748b; }}
        .why-not {{ color: #fbbf24; font-size: 0.9rem; margin-top: 8px; }}
        .by-method {{ margin-top: 8px; color: #64748b; font-size: 0.85rem; }}
        pre {{ font-size: 0.8rem; overflow-x: auto; margin: 8px 0; }}
        a {{ color: #22d3ee; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-right: 4px; }}
        .badge-high {{ background: #065f46; color: #6ee7b7; }}
        .badge-med {{ background: #78350f; color: #fcd34d; }}
        .badge-low {{ background: #7f1d1d; color: #fca5a5; }}
    </style>
</head>
<body>
    <h1>Dashboard Preview — {esc(data.get("doc_id", ""))}</h1>
    <p style="color: #64748b;">Methods: {", ".join(data.get("methods_available", []))}</p>
    <p><a href="/data">View raw JSON</a></p>

    <h2>Highlights ({len(h)})</h2>
    {highlights_html or '<p class="card">None</p>'}

    <h2>Core Reasoning ({len(cr)})</h2>
    {core_html or '<p class="card">None</p>'}

    <h2>Not Found With Reasoning ({len(nf)})</h2>
    {nf_html or '<p class="card">None</p>'}

    <h2>Agreement Summary ({len(ag)})</h2>
    {ag_html or '<p class="card">None</p>'}

    <p style="margin-top: 40px; color: #64748b; font-size: 0.9rem;">
        This is a minimal preview. Run: python experiment-scripts/test_dashboard_preview.py
    </p>
</body>
</html>
"""


class PreviewHandler(BaseHTTPRequestHandler):
    data = None

    def do_GET(self):
        if self.path == "/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.data, indent=2, ensure_ascii=False).encode("utf-8"))
            return
        if self.path == "/" or self.path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = build_preview_html(self.data)
            self.wfile.write(html.encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC
    print(f"Loading dashboard for: {doc_id}")

    try:
        data = get_dashboard_data(doc_id)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {OUTPUT_JSON}")

    # Stats
    print(f"  Highlights: {len(data.get('highlights', []))}")
    print(f"  Core reasoning: {len(data.get('core_reasoning', []))}")
    print(f"  Not found: {len(data.get('not_found_with_reasoning', []))}")
    print(f"  Agreement: {len(data.get('agreement_summary', []))}")

    PreviewHandler.data = data
    server = HTTPServer(("127.0.0.1", PORT), PreviewHandler)
    print(f"\nPreview server: http://127.0.0.1:{PORT}")
    print("Press Ctrl+C to stop\n")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
