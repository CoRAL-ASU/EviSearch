# Clinical Trial Data Extraction - Web Interface

A modern, interactive web interface for extracting clinical trial data from PDF documents using AI-powered extraction.

## Features

### 1. PDF Upload
- Drag-and-drop or browse to upload PDF files
- Support for files up to 50MB
- Real-time upload status

### 2. Query Methods

#### Single Column Extraction
- Select from 133 predefined columns
- Or provide custom column name and definition
- Fast extraction for specific data points

#### All 133 Columns Extraction
- Automatically extracts all 133 predefined columns
- Uses definitions from `Definitions_open_ended.csv` or `Definitions_with_eval_category.csv`
- Loads definitions via `definitions.py` approach
- Results displayed in table or card view
- Toggle between view modes for large datasets
- No CSV upload required - fully automated

#### Custom CSV Upload
- Upload a CSV file with custom/non-standard columns
- Required CSV format:
  ```csv
  column_name,definition
  Custom Field,"Custom field definition..."
  ```
- Batch extraction with validation
- Use for columns not in standard 133 set

### 3. Results Display
- **Value**: Extracted data from the PDF
- **Location**: Page number where the value was found
- **Modality**: Type of content (text, table, figure)
- **Evidence**: AI reasoning and evidence from the document
- **Definition**: Column definition used for extraction

### 4. Export Options
- Export results as JSON
- Export results as CSV
- Preserves all metadata including evidence and location

## Setup Instructions

### Prerequisites
- Python 3.8+
- GEMINI_API_KEY environment variable set

### Installation

1. Install dependencies:
```bash
cd web/
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Add to .env file in project root
GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the Application

From the project root directory:

```bash
python web/main_app.py
```

The application will start at: **http://127.0.0.1:5000**

## API Endpoints

### Upload PDF
- **POST** `/api/upload`
- Body: `multipart/form-data` with `file` field
- Returns: Success status and filename

### Get Available Columns
- **GET** `/api/columns`
- Returns: List of all 133 available columns with definitions

### Extract Single Column
- **POST** `/api/extract/single`
- Body: `{"column_name": "...", "definition": "..."}`
- Returns: Extracted value with location and evidence

### Extract from CSV
- **POST** `/api/extract/csv`
- Body: `multipart/form-data` with CSV file
- Returns: Results for all columns in CSV

### Extract All Columns
- **POST** `/api/extract/all`
- Returns: Results for all 133 columns

### Export Results
- **POST** `/api/export/json` or `/api/export/csv`
- Body: `{"results": {...}}`
- Returns: Formatted export file

## Architecture

```
web/
├── main_app.py              # Flask application with API endpoints
├── extraction_service.py    # Service wrapper for Gemini extraction
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Main interface template
├── static/
│   ├── css/
│   │   └── styles.css      # Modern, responsive styles
│   └── js/
│       └── app.js          # Frontend JavaScript logic
└── uploads/                # Temporary PDF storage
```

## Usage Flow

1. **Upload PDF**: User uploads a clinical trial PDF document
2. **Select Method**: Choose extraction method (single, CSV, or all columns)
3. **Configure Query**: 
   - For single: Select or enter column name
   - For CSV: Upload CSV with columns to extract
   - For all: Confirm extraction of all 133 columns
4. **Extract**: Click extract button to start AI-powered extraction
5. **View Results**: Results display with values, locations, and evidence
6. **Export**: Download results as JSON or CSV

## Technical Details

### Backend
- **Framework**: Flask 3.0
- **AI Model**: Google Gemini 2.0 Flash
- **Integration**: Uses baseline_file_search_gemini_native.py methodology
- **JSON Schema**: Structured extraction with location tracking

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Gradient design, smooth animations, intuitive UX
- **Real-time Feedback**: Loading states, progress indicators

### Extraction Format
Each extraction returns:
```json
{
  "column_name": {
    "value": "extracted value",
    "page_number": 5,
    "modality": "table",
    "evidence": "Found in Table 1...",
    "definition": "Column definition..."
  }
}
```

## Troubleshooting

### PDF Upload Fails
- Check file size (max 50MB)
- Ensure file is a valid PDF
- Check disk space in uploads/ folder

### Extraction Takes Too Long
- Large PDFs with complex tables may take longer
- All columns extraction can take 5-10 minutes
- Check API rate limits and credits

### API Key Issues
- Verify GEMINI_API_KEY is set in environment
- Check API key has sufficient credits
- Ensure API key has required permissions

## Future Enhancements

- [ ] Progress bars for long extractions
- [ ] Caching of extraction results
- [ ] Multiple PDF comparison
- [ ] Custom extraction templates
- [ ] User authentication
- [ ] Result history and management
