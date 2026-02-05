#!/usr/bin/env python3
"""
Visualization script for extracted clinical trial data.
Generates an HTML report showing filled/null columns, grouped by category.
Serves the HTML via Python HTTP server and opens in browser.

Usage:
    python visualize_extraction.py <path_to_extracted_table.csv>
    python visualize_extraction.py test_results/new/NCT01715285_Fizazi_LATITUTE_NEJM'17/extracted_table.csv
"""

import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import webbrowser
import http.server
import socketserver
import threading
import time
import socket
import os

# Ensure visualizer dir is on path when run from project root (e.g. python visualizer/visualize_extraction.py ...)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Import column mapping
from column_mapping import get_category_for_column, get_ordered_categories, COLUMN_CATEGORIES

# categorize_column function removed - now using column_mapping.py

def load_metadata(csv_path):
    """Load metadata JSON if available."""
    metadata_path = Path(csv_path).parent / "extraction_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def load_gold_table(trial_name):
    """Load gold table and find row for this trial."""
    try:
        # Get gold table path from project root (up one level from visualizer/)
        project_root = Path(__file__).parent.parent
        gold_path = project_root / "dataset" / "GoldTable.csv"
        
        if not gold_path.exists():
            print(f"⚠️  Gold table not found: {gold_path}")
            return None
        
        # Load gold table
        gold_df = pd.read_csv(gold_path)
        
        # Match by document name (trial_name + .pdf)
        document_name = f"{trial_name}.pdf"
        matching_rows = gold_df[gold_df['Document Name'] == document_name]
        
        if matching_rows.empty:
            print(f"⚠️  No gold data found for trial: {document_name}")
            return None
        
        print(f"✅ Found gold data for: {document_name}")
        return matching_rows.iloc[0]  # Return first matching row as Series
        
    except Exception as e:
        print(f"⚠️  Error loading gold table: {e}")
        return None


def is_value_null(value):
    """Check if value is null/empty/NA."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in ['', 'NA', 'N/A', 'na', 'null']:
        return True
    return False


def get_comparison_status(extracted_val, gold_val):
    """
    Determine comparison status.
    Returns: 'both_null', 'missing', 'correct', 'incorrect'
    """
    extracted_null = is_value_null(extracted_val)
    gold_null = is_value_null(gold_val)
    
    if extracted_null and gold_null:
        return 'both_null'  # Gray
    elif extracted_null and not gold_null:
        return 'missing'  # Red - extracted is null but gold exists
    elif not extracted_null and gold_null:
        return 'incorrect'  # Yellow - extracted has value but no gold to compare
    else:
        # Both have values - compare them
        # Normalize strings for comparison
        extracted_str = str(extracted_val).strip().lower()
        gold_str = str(gold_val).strip().lower()
        if extracted_str == gold_str:
            return 'correct'  # Green - values match
        else:
            return 'incorrect'  # Yellow - values don't match

def generate_html_report(csv_path, output_path=None, trial_name=None):
    """Generate HTML visualization of extraction results.
    trial_name: optional; if not set, uses csv_path.parent.name (for gold table lookup and title).
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        return None
    
    # Read data
    df = pd.read_csv(csv_path)
    metadata = load_metadata(csv_path)
    
    # Get trial name (e.g. PDF stem when CSV is in .../extraction/extracted_table.csv)
    if trial_name is None:
        trial_name = csv_path.parent.name
    
    # Load gold table
    gold_row = load_gold_table(trial_name)
    
    # Analyze data
    total_cols = len(df.columns)
    
    # Count by comparison status
    both_null_count = 0
    missing_count = 0
    correct_count = 0
    incorrect_count = 0
    
    # Group columns by category
    categories = defaultdict(list)
    for col in df.columns:
        extracted_val = df[col].iloc[0]
        
        # Get gold value
        gold_val = None
        if gold_row is not None and col in gold_row.index:
            gold_val = gold_row[col]
        
        # Get comparison status
        status = get_comparison_status(extracted_val, gold_val)
        
        if status == 'both_null':
            both_null_count += 1
        elif status == 'missing':
            missing_count += 1
        elif status == 'correct':
            correct_count += 1
        elif status == 'incorrect':
            incorrect_count += 1
        
        # Get evidence from metadata if available
        evidence = None
        page = None
        if metadata and col in metadata:
            evidence = metadata[col].get('evidence')
            page = metadata[col].get('page')
        
        category = get_category_for_column(col)
        categories[category].append({
            'name': col,
            'value': extracted_val if not is_value_null(extracted_val) else None,
            'gold_value': gold_val if not is_value_null(gold_val) else None,
            'status': status,
            'evidence': evidence,
            'page': page
        })
    
    # Calculate percentages
    has_value_count = correct_count + incorrect_count
    completion_pct = (has_value_count / total_cols * 100) if total_cols > 0 else 0
    
    # Sort columns within each category according to mapping order
    for category in categories:
        if category in COLUMN_CATEGORIES:
            # Create a mapping of column name to its position in the defined order
            order_map = {col: idx for idx, col in enumerate(COLUMN_CATEGORIES[category])}
            # Sort the columns according to the mapping order
            # Columns not in mapping go to end
            categories[category].sort(key=lambda x: order_map.get(x['name'], 999999))
    
    # Get categories in order from mapping
    category_order = get_ordered_categories()
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Report: {trial_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header .trial-name {{
            font-size: 18px;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .summary {{
            display: flex;
            justify-content: space-around;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .stat-card {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 150px;
        }}
        
        .stat-card .number {{
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-card .label {{
            font-size: 14px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card.correct .number {{ color: #28a745; }}
        .stat-card.incorrect .number {{ color: #ffc107; }}
        .stat-card.missing .number {{ color: #dc3545; }}
        .stat-card.both-null .number {{ color: #6c757d; }}
        .stat-card.total .number {{ color: #667eea; }}
        
        .progress-bar {{
            width: 100%;
            height: 40px;
            background: #e9ecef;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease-out;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .category {{
            margin-bottom: 40px;
        }}
        
        .category-header {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            color: #2d3748;
        }}
        
        .category-stats {{
            font-size: 14px;
            color: #6c757d;
            margin-left: 10px;
        }}
        
        .columns-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 15px;
        }}
        
        .column-card {{
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            transition: all 0.2s;
        }}
        
        .column-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .column-card.correct {{
            background: #d4edda;
            border-color: #28a745;
        }}
        
        .column-card.incorrect {{
            background: #fff3cd;
            border-color: #ffc107;
        }}
        
        .column-card.missing {{
            background: #f8d7da;
            border-color: #dc3545;
        }}
        
        .column-card.both_null {{
            background: #e9ecef;
            border-color: #6c757d;
        }}
        
        .column-name {{
            font-weight: 600;
            margin-bottom: 8px;
            color: #2d3748;
            word-break: break-word;
        }}
        
        .column-value {{
            padding: 8px;
            background: white;
            border-radius: 4px;
            margin-top: 8px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #495057;
            word-break: break-word;
        }}
        
        .gold-value {{
            padding: 8px;
            background: #e3f2fd;
            border-radius: 4px;
            margin-top: 8px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            color: #1565c0;
            word-break: break-word;
            border-left: 3px solid #1976d2;
        }}
        
        .value-label {{
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 4px;
            letter-spacing: 0.5px;
        }}
        
        .null-indicator {{
            color: #dc3545;
            font-style: italic;
            font-size: 14px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 8px;
        }}
        
        .status-badge.correct {{
            background: #28a745;
            color: white;
        }}
        
        .status-badge.incorrect {{
            background: #ffc107;
            color: #000;
        }}
        
        .status-badge.missing {{
            background: #dc3545;
            color: white;
        }}
        
        .status-badge.both_null {{
            background: #6c757d;
            color: white;
        }}
        
        .evidence {{
            margin-top: 8px;
            padding: 8px;
            background: #fff3cd;
            border-left: 3px solid #ffc107;
            border-radius: 4px;
            font-size: 12px;
            color: #856404;
        }}
        
        .page-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-left: 8px;
        }}
        
        .filter-bar {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        
        .filter-button {{
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            color: #667eea;
        }}
        
        .filter-button:hover {{
            background: #667eea;
            color: white;
        }}
        
        .filter-button.active {{
            background: #667eea;
            color: white;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Clinical Trial Extraction Report</h1>
            <div class="trial-name">{trial_name}</div>
        </div>
        
        <div class="summary">
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="number" style="color: white;">{has_value_count}</div>
                <div class="label" style="color: white;">Has Value</div>
            </div>
            <div class="stat-card correct">
                <div class="number">{correct_count}</div>
                <div class="label">├─ Correct</div>
            </div>
            <div class="stat-card incorrect">
                <div class="number">{incorrect_count}</div>
                <div class="label">└─ Incorrect</div>
            </div>
            <div class="stat-card missing">
                <div class="number">{missing_count}</div>
                <div class="label">Missing</div>
            </div>
            <div class="stat-card both-null">
                <div class="number">{both_null_count}</div>
                <div class="label">N/A</div>
            </div>
            <div class="stat-card total">
                <div class="number">{total_cols}</div>
                <div class="label">Total Fields</div>
            </div>
        </div>
        
        <div style="padding: 20px 40px;">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {completion_pct:.1f}%">
                    {completion_pct:.1f}% Complete
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="filter-bar">
                <strong>Quick Filter:</strong>
                <button class="filter-button active" onclick="filterColumns('all')">All</button>
                <button class="filter-button" onclick="filterColumns('has_value')">Has Value</button>
                <button class="filter-button" onclick="filterColumns('correct')">├─ Correct</button>
                <button class="filter-button" onclick="filterColumns('incorrect')">└─ Incorrect</button>
                <button class="filter-button" onclick="filterColumns('missing')">Missing</button>
                <button class="filter-button" onclick="filterColumns('both_null')">N/A</button>
            </div>
"""
    
    # Add categories
    for category in category_order:
        if category not in categories:
            continue
        
        cols = categories[category]
        has_value_in_cat = sum(1 for c in cols if c['status'] in ['correct', 'incorrect'])
        total_in_cat = len(cols)
        
        html += f"""
            <div class="category">
                <div class="category-header">
                    {category}
                    <span class="category-stats">({has_value_in_cat}/{total_in_cat} with values)</span>
                </div>
                <div class="columns-grid">
"""
        
        for col in cols:
            status = col['status']
            value_html = ''
            
            # Show extracted value
            if col['value'] is not None:
                value_html += '<div class="value-label">📤 Extracted:</div>'
                value_html += f'<div class="column-value">{col["value"]}</div>'
                if col['evidence']:
                    evidence_preview = col['evidence'][:150] + ('...' if len(col['evidence']) > 150 else '')
                    value_html += f'<div class="evidence">💬 "{evidence_preview}"</div>'
                if col['page']:
                    value_html += f'<span class="page-badge">Page {col["page"]}</span>'
            else:
                value_html += '<div class="value-label">📤 Extracted:</div>'
                value_html += '<div class="null-indicator">⚠️ No value extracted</div>'
            
            # Show gold value
            if col['gold_value'] is not None:
                value_html += '<div class="value-label" style="margin-top: 12px;">🎯 Gold:</div>'
                value_html += f'<div class="gold-value">{col["gold_value"]}</div>'
            else:
                value_html += '<div class="value-label" style="margin-top: 12px;">🎯 Gold:</div>'
                value_html += '<div class="null-indicator" style="font-size: 12px;">N/A</div>'
            
            # Status badge
            status_labels = {
                'correct': '🟢 Has Value, Correct',
                'incorrect': '🟡 Has Value, Incorrect',
                'missing': '🔴 Missing',
                'both_null': '⚪ N/A'
            }
            value_html += f'<div class="status-badge {status}">{status_labels.get(status, status)}</div>'
            
            html += f"""
                    <div class="column-card {status}" data-status="{status}">
                        <div class="column-name">{col['name']}</div>
                        {value_html}
                    </div>
"""
        
        html += """
                </div>
            </div>
"""
    
    # Close HTML
    html += """
        </div>
        
        <footer>
            Generated by CoRal-Map-Make Visualization Tool
        </footer>
    </div>
    
    <script>
        function filterColumns(filter) {
            // Update button states
            document.querySelectorAll('.filter-button').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Filter columns
            document.querySelectorAll('.column-card').forEach(card => {
                if (filter === 'all') {
                    card.style.display = 'block';
                } else if (filter === 'has_value') {
                    // Show both correct and incorrect
                    card.style.display = (card.dataset.status === 'correct' || card.dataset.status === 'incorrect') ? 'block' : 'none';
                } else {
                    card.style.display = card.dataset.status === filter ? 'block' : 'none';
                }
            });
        }
    </script>
</body>
</html>
"""
    
    # Save HTML
    if output_path is None:
        output_path = csv_path.parent / f"{trial_name}_visualization.html"
    output_path = Path(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Visualization saved to: {output_path}")
    return output_path


def _port_available(port):
    """Check if a port is available to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def serve_html(html_path, port=8000):
    """Serve HTML file via HTTP server and open in browser."""
    html_path = Path(html_path).absolute()
    directory = html_path.parent
    filename = html_path.name

    # Find an available port (robust check without TCPServer)
    original_port = port
    max_attempts = 10
    for attempt in range(max_attempts):
        if _port_available(port):
            break
        port += 1
        if attempt == max_attempts - 1:
            print(f"❌ Could not find available port after {max_attempts} attempts")
            return
    if port != original_port:
        print(f"ℹ️  Port {original_port} busy, using port {port} instead")

    # Allow port reuse so we can bind immediately after checking
    socketserver.TCPServer.allow_reuse_address = True

    original_dir = os.getcwd()
    os.chdir(directory)

    Handler = http.server.SimpleHTTPRequestHandler
    Handler.extensions_map['.html'] = 'text/html; charset=UTF-8'

    url = f"http://127.0.0.1:{port}/{filename}"
    file_url = html_path.as_uri()  # file:///path/to/file.html

    print("\n" + "=" * 60)
    print("🚀 HTTP Server Started!")
    print("=" * 60)
    print(f"📍 Open in browser:  {url}")
    print(f"📂 Or open file:      {file_url}")
    print(f"📁 HTML file:         {html_path}")
    print("=" * 60)
    print("\n🌐 Opening in browser in 2 seconds...")
    print("   (If nothing opens, copy the URL above or open the file path.)\n")

    def open_browser():
        time.sleep(2)  # Give server time to start
        webbrowser.open(url)

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print("✨ Visualization is now live! Press Ctrl+C to stop the server.\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("✅ Goodbye!")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_extraction.py <path_to_extracted_table.csv>")
        print("\nExample:")
        print("  python visualize_extraction.py test_results/new/NCT01715285_Fizazi_LATITUTE_NEJM'17/extracted_table.csv")
        print("\nThis will:")
        print("  1. Generate HTML visualization")
        print("  2. Start a local HTTP server")
        print("  3. Open the visualization in your browser")
        print("  4. Keep serving until you press Ctrl+C")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    html_path = generate_html_report(csv_path)
    
    if html_path:
        print("\n" + "="*60)
        serve_html(html_path)
