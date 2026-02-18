# Web Interface - Usage Guide

## Quick Start

### 1. Start the Server

From the project root directory:

```bash
# Option 1: Use the startup script (recommended)
./start_web_interface.sh

# Option 2: Manual start
source .venv/bin/activate
python web/main_app.py
```

The server will start at **http://127.0.0.1:5000**

### 2. Open Your Browser

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Step-by-Step Workflow

### Step 1: Upload a PDF

1. **Drag and Drop**: Drag your PDF file onto the upload area
   - OR -
2. **Browse**: Click "Browse Files" to select a PDF from your computer

**Requirements:**
- File must be in PDF format
- Maximum size: 50MB
- File should be a clinical trial document

**What happens:**
- The PDF is uploaded to the server
- Gemini loads the PDF into memory for extraction
- You'll see a green checkmark with the filename when successful

---

### Step 2: Choose Your Query Method

After uploading, you'll see three options:

#### Option A: Single Column 🎯

**Best for:** Extracting one specific value quickly

**How to use:**
1. Click the "Single Column" card
2. Either:
   - Select a column from the dropdown (133 predefined columns), OR
   - Enter a custom column name
3. The definition will auto-fill for predefined columns
4. For custom columns, optionally provide a definition
5. Click "Extract Value"

**Example:**
- Column: `NCT`
- Definition: "What national Clinical Trial identifier..."
- Result: Returns NCT number with page location and evidence

**Response time:** ~5-10 seconds

---

#### Option B: All 133 Columns 📋

**Best for:** Complete extraction of all available clinical trial data

**How to use:**
1. Click the "All 133 Columns" card
2. Read the information about the extraction
3. Click "Extract All Columns"
4. Confirm in the popup dialog
5. Wait for completion (progress shown)

**What it extracts:**
- All 133 predefined columns from `Definitions_open_ended.csv` or `Definitions_with_eval_category.csv`
- Uses definitions from the pipeline's `definitions.py` loader
- Organized by label groups for efficiency
- Complete clinical trial data table

**Display options:**
- **Table View**: Clean tabular format with sortable columns
- **Card View**: Detailed cards with full evidence and definitions

**Response time:** ~5-10 minutes

**⚠️ Note:** This uses significant API credits. Use for complete extraction needs only.

---

#### Option C: Custom CSV 📊

**Best for:** Extracting custom or non-standard columns

**How to use:**
1. Click the "Custom CSV" card
2. Prepare your CSV file with this format:

```csv
column_name,definition
Custom Field 1,"Description of what to extract..."
Custom Field 2,"Another custom field definition..."
```

**CSV Requirements:**
- Must have headers: `column_name` and `definition`
- Use for columns NOT in the standard 133 set
- Each row = one column to extract

3. Click "Choose CSV File" and select your CSV
4. Click "Extract from CSV"

**Response time:** ~30 seconds to 2 minutes (depends on # of columns)

---

## Understanding Results

### Result Display

Results can be viewed in two formats:

#### Card View (Default for <10 results)

Each extracted column shows:

```
Column Name                    📄 Page 5    📊 table

Value
The extracted value from the document

Evidence:
"Found in Table 1 on page 5. The trial included 1000 participants..."

Definition:
What the total number of participants included in the clinical trial...
```

#### Table View (Default for all 133 columns)

Clean tabular format showing:

| Column Name | Value | Page | Modality | Evidence |
|-------------|-------|------|----------|----------|
| NCT | NCT00309985 | 1 | text | Found in header... |
| Trial Name | CHAARTED | 1 | text | Found in title section... |
| ... | ... | ... | ... | ... |

**Toggle between views** using the "📊 Table View" / "📋 Card View" button.

### Result Components

1. **Column Name**: The field that was extracted (e.g., "Total Participants - N")

2. **Value**: The actual extracted data
   - Exact text/numbers from the PDF
   - "not found" if not present in document
   - "Not reported" if explicitly stated as unreported

3. **Page Number**: Where the value was found
   - Shows specific page number
   - "Page unknown" if location couldn't be determined

4. **Modality**: Type of content source
   - `text`: Found in regular text paragraphs
   - `table`: Found in a table
   - `figure`: Found in a figure or chart
   - `Unknown modality`: If type couldn't be determined

5. **Evidence**: AI's reasoning
   - Explains where it found the value
   - Describes how it interpreted the data
   - Shows relevant context from the document

6. **Definition**: The query definition used
   - Shows what the AI was looking for
   - Useful for understanding extraction logic

### Statistics

At the top of results:

- **Total Columns**: Number of columns extracted
- **Values Found**: Columns with actual values (not "not found")
- **Success Rate**: Percentage of found values

---

## Exporting Results

### Export to JSON

```javascript
{
  "NCT": {
    "value": "NCT00309985",
    "page_number": 1,
    "modality": "text",
    "evidence": "Found in header...",
    "definition": "What national Clinical..."
  },
  // ... more columns
}
```

**Use for:**
- Further programmatic processing
- Integration with other tools
- Preserves all metadata

### Export to CSV

```csv
Column Name,Value,Page Number,Modality,Evidence,Definition
NCT,NCT00309985,1,text,"Found in header...","What national..."
```

**Use for:**
- Excel analysis
- Easy human review
- Spreadsheet import

---

## Common Use Cases

### 1. Quick Single Value Lookup

**Scenario:** "What's the NCT number in this trial?"

**Steps:**
1. Upload PDF
2. Single Column → Select "NCT"
3. Extract → Get result instantly

### 2. Custom Data Extraction

**Scenario:** "Extract a field not in the 133 predefined columns"

**Steps:**
1. Upload PDF
2. Single Column → Enter custom column name
3. Provide custom definition
4. Extract

**Example:**
- Column: `Follow-up Protocol`
- Definition: "What is the follow-up protocol for patients?"

### 3. Batch Processing Multiple Trials

**Scenario:** "Extract same 10 columns from 20 trials"

**Steps:**
1. Create one CSV with your 10 columns
2. For each trial:
   - Upload PDF
   - Upload same CSV
   - Extract
   - Export results
   - Download and save

### 4. Complete Trial Analysis

**Scenario:** "Full extraction for meta-analysis"

**Steps:**
1. Upload PDF
2. All Columns → Extract all 133
3. Export to CSV
4. Use in your analysis pipeline

---

## Tips & Best Practices

### Upload Tips

✅ **Do:**
- Use clear, text-based PDFs (not scanned images)
- Ensure PDF contains the full trial document
- Check file size is under 50MB

❌ **Don't:**
- Upload password-protected PDFs
- Use extremely low-quality scans
- Upload non-trial documents

### Query Tips

✅ **Do:**
- Use specific, clear definitions for custom columns
- Include units in your definition (e.g., "months", "number", "percentage")
- Be explicit about what format you expect

❌ **Don't:**
- Use vague definitions like "get the data"
- Combine multiple questions in one column
- Expect extraction of data not in the PDF

### Performance Tips

- **Single columns**: Fast, great for iterative exploration
- **CSV batch**: Medium speed, good for targeted extraction
- **All columns**: Slow but comprehensive, use once per trial

### Accuracy Tips

1. **Check the evidence**: Review what the AI found
2. **Verify page numbers**: Spot-check a few extractions
3. **Compare definitions**: Ensure they match your intent
4. **Handle "not found"**: Doesn't always mean it's truly absent
   - Sometimes formatting issues prevent extraction
   - Try rephrasing the definition
   - Check if it's in an image/figure

---

## Troubleshooting

### Problem: Upload fails

**Possible causes:**
- File too large (>50MB)
- Not a valid PDF
- Disk space full

**Solutions:**
- Compress PDF if too large
- Verify file integrity
- Check server logs

### Problem: Extraction returns "not found"

**Possible causes:**
- Data truly not in document
- Definition too vague or specific
- Data in unsupported format (images)

**Solutions:**
- Verify data exists in PDF manually
- Rephrase definition to be clearer
- Check evidence field for clues

### Problem: Extraction is slow

**Normal for:**
- All columns extraction (5-10 min)
- Large PDFs (200+ pages)
- Complex tables and figures

**Not normal:**
- Single column taking >30 seconds
- CSV with 5 columns taking >5 minutes

**Solutions:**
- Check internet connection
- Verify API key has credits
- Restart the server

### Problem: Wrong values extracted

**Solutions:**
- Check the evidence to understand why
- Refine the column definition
- Verify page number matches
- Consider if there's ambiguity in the PDF

---

## Advanced Features

### Custom Column Definitions

You can create highly specific queries:

```
Column: "Treatment Duration in Experimental Arm"
Definition: "What is the median duration in months that patients in the 
experimental arm (not control) received treatment? Look for 'median 
treatment duration' or similar phrases in the treatment arm section."
```

### CSV Templates

Create reusable CSV templates for common extraction patterns:

**basic_info.csv:**
```csv
column_name,definition
NCT,"What national Clinical Trial identifier..."
Trial Name,"What the title..."
Year,"Publication year..."
```

**outcomes.csv:**
```csv
column_name,definition
Primary Outcome,"What is the primary outcome..."
OS Median | Treatment,"What is the median overall survival..."
```

### Programmatic Access

All functionality is available via API endpoints. See API documentation in README.md.

---

## Support & Resources

- **Full documentation**: See `web/README.md`
- **API reference**: See `web/README.md` → API Endpoints
- **Issues**: Check server logs in terminal
- **Column definitions**: See `src/table_definitions/Definitions_with_eval_category.csv`

---

## Example Workflow

Complete example from start to finish:

```bash
# 1. Start server
./start_web_interface.sh

# 2. Open browser → http://127.0.0.1:5000

# 3. Upload: dataset/NCT00104715_Gravis_GETUG_EU'15.pdf

# 4. Select: Upload CSV

# 5. Upload CSV with 5 key columns:
#    - NCT, Trial Name, Year, Total Participants - N, Primary Outcome

# 6. Click: Extract from CSV

# 7. Review results:
#    - Check values make sense
#    - Read evidence for confidence
#    - Note any "not found" columns

# 8. Export: Download as CSV

# 9. Use in analysis pipeline

# Done! 🎉
```

---

**Happy extracting! 🏥📊**
