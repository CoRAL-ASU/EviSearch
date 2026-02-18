#!/usr/bin/env python3
"""
main_app.py

Modern web interface for Clinical Trial Data Extraction.
Provides endpoints for PDF upload, query submission, and result retrieval.

Run from project root: python web/main_app.py
Then open http://127.0.0.1:5000
"""
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from web.extraction_service import ExtractionService

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = PROJECT_ROOT / 'web' / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Global extraction service instance
extraction_service = None
current_pdf_info = {}


def get_extraction_service() -> ExtractionService:
    """Get or create extraction service instance (lazy initialization)."""
    global extraction_service
    if extraction_service is None:
        try:
            extraction_service = ExtractionService()
        except Exception as e:
            # If initialization fails, return None - will be handled by routes
            print(f"Warning: Could not initialize ExtractionService: {e}")
            return None
    return extraction_service


@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload a PDF file for extraction."""
    global current_pdf_info
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        # Load PDF into extraction service
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.upload_pdf(str(filepath))
        
        if result.get("success"):
            current_pdf_info = {
                "filename": filename,
                "filepath": str(filepath),
                "message": result.get("message")
            }
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500


@app.route('/api/columns', methods=['GET'])
def get_columns():
    """Get list of all available columns."""
    try:
        service = get_extraction_service()
        columns = service.get_available_columns()
        return jsonify({
            "success": True,
            "columns": columns,
            "total": len(columns)
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/extract/single', methods=['POST'])
def extract_single():
    """Extract a single column value."""
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    column_name = data.get('column_name')
    definition = data.get('definition')
    
    if not column_name:
        return jsonify({"success": False, "error": "column_name is required"}), 400
    
    try:
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.extract_single_column(column_name, definition)
        return jsonify(result), 200 if result.get("success") else 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/available', methods=['GET'])
def get_available_documents():
    """Get list of documents with existing extractions."""
    try:
        results_dir = PROJECT_ROOT / 'experiment-scripts' / 'baselines_file_search_results' / 'gemini_native'
        
        documents = []
        
        if results_dir.exists():
            # Look for extraction_metadata.json files
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    for doc_dir in model_dir.iterdir():
                        if doc_dir.is_dir():
                            extraction_file = doc_dir / 'extraction_metadata.json'
                            if extraction_file.exists():
                                documents.append({
                                    'id': f"{model_dir.name}/{doc_dir.name}",
                                    'name': doc_dir.name,
                                    'model': model_dir.name,
                                    'path': str(extraction_file)
                                })
        
        # Sort by document name
        documents.sort(key=lambda x: x['name'])
        
        return jsonify({
            "success": True,
            "documents": documents,
            "count": len(documents)
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/documents/<path:doc_id>/extraction', methods=['GET'])
def get_document_extraction(doc_id):
    """Get extraction data for a specific document."""
    try:
        results_dir = PROJECT_ROOT / 'experiment-scripts' / 'baselines_file_search_results' / 'gemini_native'
        extraction_file = results_dir / doc_id / 'extraction_metadata.json'
        
        if not extraction_file.exists():
            return jsonify({
                "success": False,
                "error": f"Extraction file not found for document: {doc_id}"
            }), 404
        
        # Load extraction metadata
        with open(extraction_file, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        # Transform to web interface format
        results = {}
        for col_name, col_data in extraction_data.items():
            results[col_name] = {
                "value": col_data.get("value", "not found"),
                "page_number": col_data.get("page", "N/A"),
                "modality": col_data.get("plan_source_type", "unknown"),
                "evidence": col_data.get("evidence", ""),
                "definition": ""  # Could load from definitions if needed
            }
        
        # Try to load summary metrics if available
        summary_file = extraction_file.parent / 'evaluation' / 'summary_metrics.json'
        summary_info = None
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_info = json.load(f)
        
        return jsonify({
            "success": True,
            "document_id": doc_id,
            "results": results,
            "total_columns": len(results),
            "summary": summary_info
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/extract/csv', methods=['POST'])
def extract_from_csv():
    """Extract columns from uploaded CSV file."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No CSV file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"success": False, "error": "Only CSV files are allowed"}), 400
    
    try:
        # Read CSV file
        stream = io.StringIO(file.stream.read().decode("UTF-8"), newline=None)
        csv_reader = csv.DictReader(stream)
        
        # Validate CSV headers
        headers = csv_reader.fieldnames
        if not headers:
            return jsonify({"success": False, "error": "CSV file is empty"}), 400
        
        # Check for required columns (case-insensitive)
        headers_lower = [h.lower() for h in headers]
        has_column_name = 'column_name' in headers_lower or 'column name' in headers_lower
        has_definition = 'definition' in headers_lower
        
        if not (has_column_name and has_definition):
            return jsonify({
                "success": False,
                "error": "CSV must contain 'column_name' (or 'Column Name') and 'definition' (or 'Definition') columns"
            }), 400
        
        # Read all rows
        csv_data = list(csv_reader)
        
        if not csv_data:
            return jsonify({"success": False, "error": "CSV file contains no data rows"}), 400
        
        # Extract columns
        service = get_extraction_service()
        if service is None:
            return jsonify({
                "success": False, 
                "error": "Extraction service not available. Please ensure GEMINI_API_KEY is set in your environment."
            }), 500
        
        result = service.extract_from_csv(csv_data)
        
        return jsonify(result), 200 if result.get("success") else 400
        
    except Exception as e:
        return jsonify({"success": False, "error": f"CSV processing failed: {str(e)}"}), 500


@app.route('/api/pdf/info', methods=['GET'])
def get_pdf_info():
    """Get information about the currently loaded PDF."""
    global current_pdf_info
    
    if not current_pdf_info:
        return jsonify({
            "success": False,
            "error": "No PDF loaded"
        }), 404
    
    return jsonify({
        "success": True,
        **current_pdf_info
    }), 200


@app.route('/api/export/<format>', methods=['POST'])
def export_results(format):
    """Export extraction results in various formats."""
    data = request.get_json()
    
    if not data or 'results' not in data:
        return jsonify({"success": False, "error": "No results to export"}), 400
    
    results = data['results']
    
    try:
        if format == 'json':
            return jsonify(results), 200
        
        elif format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Column Name', 'Value', 'Page Number', 'Modality', 'Evidence', 'Definition'])
            
            # Write rows
            for col_name, col_data in results.items():
                writer.writerow([
                    col_name,
                    col_data.get('value', ''),
                    col_data.get('page_number', ''),
                    col_data.get('modality', ''),
                    col_data.get('evidence', ''),
                    col_data.get('definition', '')
                ])
            
            output.seek(0)
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=extraction_results.csv'
            }
        
        else:
            return jsonify({"success": False, "error": f"Unsupported format: {format}"}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Export failed: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({"success": False, "error": "File is too large. Maximum size is 50MB"}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    return jsonify({"success": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Clinical Trial Data Extraction - Web Interface")
    print("=" * 60)
    print("\nServer starting at: http://127.0.0.1:5000")
    print("\nFeatures:")
    print("  • Upload PDF files for extraction")
    print("  • Extract single column or all 133 columns")
    print("  • Upload CSV with custom queries")
    print("  • View results with evidence and location")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
