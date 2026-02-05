# Pipeline V2 Web Interface

Simple web UI to run the full pipeline and view extraction results.

**Run from project root:**

```bash
cd /path/to/Mayo
pip install flask   # if not already installed
python web/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- **PDF path**: Server path to the PDF (relative to project root or absolute), e.g. `dataset/NCT00309985_Kriayako_CHAARTED_JCO'18.pdf`
- **Pipeline option**: Choose 1–6 (same as the CLI). After a run that produces extraction (3, 5, or 6), the extraction report opens automatically in the same style as `visualize_extraction.py`.
