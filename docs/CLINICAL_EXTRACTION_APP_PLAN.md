# Clinical Trial Extraction App — Implementation Plan

## Overview

Build a web application for clinicians in two phases:

### Phase 1: Display Only (current focus)
- **View** multiple attribution formats (evidence, sources, confidence)
- **Display** comparison across methods (Gemini, Landing AI, Pipeline with/without keywords)
- **Show** analysis reports (document-level, group-wise, column-wise insights)
- **No user input** — read-only views of existing extraction results

### Phase 2: User Interaction (later)
- Collect and store clinician confirmations
- Edit/override values
- Export confirmed data

---

## Current State

| Component | Location | Purpose |
|-----------|----------|---------|
| **main_app.py** | `web/` | Flask app: PDF upload, single-column extraction (Gemini), CSV extraction |
| **extraction_service.py** | `web/` | Wraps `baseline_file_search_gemini_native.py` — Gemini native extraction |
| **app.py** | `web/` | Simple pipeline runner (chunking → planning → extraction → eval) |
| **extract_with_landing_ai.py** | `experiment-scripts/` | Pipeline: Landing AI chunks + LLM, group-wise multi-candidate |
| **plan_and_extract_columns.py** | `experiment-scripts/` | Plan + extract with optional `--use-keywords` (BM25) |
| **Baselines** | `experiment-scripts/` | `baseline_landing_ai_w_gemini.py`, `baseline_file_search_gemini_native.py` |

**Extraction methods to integrate:**
1. **Gemini native** — PDF + column definition → Gemini (file search)
2. **Landing AI + pipeline** — Landing AI chunks → plan → extract (group-wise)
3. **Pipeline + keywords** — Same as above but with BM25 keyword retrieval
4. **Pipeline (no keywords)** — Same as #2 without keyword supplement

---

## Architecture: API-First + Web UI

**Recommendation:** Build a **FastAPI backend** with REST endpoints, then a **React/Vue or enhanced Flask** frontend. This allows:
- Clinicians to use the web UI
- Scripts/tools to call APIs directly
- Future mobile or other clients

### Option A: Extend Flask (main_app.py)
- Add new API routes for each extraction method
- Add comparison and confirmation endpoints
- Simpler, reuses existing structure

### Option B: New FastAPI Service
- Cleaner async support for long-running extractions
- OpenAPI docs auto-generated
- Better for multiple concurrent extraction jobs

**Suggested:** Start with **Option A** (extend Flask) for speed; refactor to FastAPI later if needed.

---

## 1. Attribution for Clinicians

### Multiple Attribution Formats

| Format | Description | Clinician Use |
|--------|-------------|---------------|
| **Evidence excerpt** | Short text snippet from the document | "Where did this come from?" |
| **Page + modality** | e.g. "Table 1, page 5" | Quick location reference |
| **Chunk ID / anchor** | Stable reference to source chunk | Reproducibility, audit |
| **Confidence** | high / medium / low | Trust signal |
| **Assumptions** | When value is inferred or ambiguous | Transparency |
| **Multi-candidate** | 1–4 plausible values with evidence each | When source is ambiguous |

### UI Presentation Ideas

- **Card per column:** Value + expandable "Evidence" / "Sources" / "Assumptions"
- **Highlight in PDF viewer:** Click evidence → scroll to page/section (future)
- **Badge:** Confidence (color-coded), "Multiple candidates" indicator
- **Side-by-side:** Value vs. evidence for quick scan

---

## 2. API Endpoints

### Phase 1: Display Only (implement first)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List documents with existing extractions (scan result dirs) |
| GET | `/api/documents/{doc_id}/status` | Which methods have run for this doc |
| GET | `/api/documents/{doc_id}/comparison` | Results from all methods, side-by-side |
| GET | `/api/documents/{doc_id}/comparison/group/{group_name}` | Group-wise comparison |
| GET | `/api/documents/{doc_id}/comparison/column/{column_name}` | Single column across methods |
| GET | `/api/documents/{doc_id}/report` | Document analysis report (summary stats) |
| GET | `/api/documents/{doc_id}/report/groups` | Group-wise summary |
| GET | `/api/documents/{doc_id}/report/columns` | Column-wise with attribution |

All read-only. No POST/PUT for extraction or confirmation.

### Phase 2: User Interaction (later)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/extract/*` | Run extraction methods |
| PUT | `/api/documents/{doc_id}/confirm` | Save clinician confirmations |
| POST | `/api/documents/{doc_id}/export` | Export confirmed data |

---

## 3. Data Model for Comparison & Confirmation

### Unified Result Shape (per column, per method)

```json
{
  "column_name": "Median PFS (mo) | High volume | Treatment",
  "group_name": "Median PFS (mo)",
  "method": "landing_ai",
  "value": "14.9",
  "candidates": [
    {"value": "14.9", "evidence": "Table 2...", "assumptions": null, "confidence": "high"}
  ],
  "primary_value": "14.9",
  "found": true,
  "page": 6,
  "source_type": "table",
  "retrieval_source": "planner_sources",
  "chunk_count": 3,
  "attribution": {
    "evidence": "Table 2, Median Time to CRPC...",
    "sources": [[6, "table"]],
    "confidence": "high",
    "assumptions": null
  }
}
```

### Confirmed Result (Phase 2 — clinician override)

```json
{
  "column_name": "Median PFS (mo) | High volume | Treatment",
  "confirmed_value": "14.9",
  "confirmed_by": "landing_ai",
  "edited_by_user": false,
  "notes": ""
}
```

### Storage Layout

```
new_pipeline_outputs/results/{pdf_stem}/
├── planning/
│   ├── plans_all_columns.json
│   ├── extract_landing_ai/extraction_results.json      # pipeline, no keywords
│   └── plan_extract_columns_with_keywords/            # pipeline, with keywords
├── baselines_file_search_results/gemini_native/       # Gemini
├── baselines_file_search_results/landing_ai_w_gemini/ # Landing AI baseline
└── confirmed/                                         # Phase 2: clinician confirmations
    ├── confirmed_values.json
    └── audit_log.json
```

---

## 4. Web Interface Structure

### Phase 1: Display-Only Views

1. **Document List**
   - List documents that have extraction results (from any method)
   - Show status: which methods have run (Gemini ✓, Pipeline ✓, Pipeline+KW ✗)
   - Click document → open analysis/comparison

2. **Document Analysis Dashboard**
   - Summary: which methods have run, column counts per method
   - High-level stats (e.g. columns found vs not found)
   - Links to group-wise and column-wise views

3. **Comparison View**
   - Tabs: By Group | By Column | Full Table
   - Table: Column | Gemini | Landing AI | Pipeline | Pipeline+KW
   - Click cell → expand attribution (evidence, page, confidence)
   - Read-only — no confirm/edit buttons

4. **Group-Wise View**
   - Groups as cards/sections
   - Columns within each group
   - Side-by-side method comparison (display only)

5. **Column-Wise Detail**
   - Single column: all methods + candidates + evidence
   - Attribution display (evidence, assumptions, confidence)
   - Read-only

### Phase 2: User Interaction (later)
- Confirm button, Edit, Export, Run extraction from UI

---

## 5. Implementation Phases

### Phase 1: Display Only (current focus)

**Backend**
- [ ] Comparison service: load and merge results from all method output dirs
- [ ] Normalize result shape across methods (Gemini, pipeline, pipeline+keywords)
- [ ] GET `/api/documents` — list docs with extraction status
- [ ] GET `/api/documents/{doc_id}/status` — which methods have run
- [ ] GET `/api/documents/{doc_id}/comparison` — unified comparison data
- [ ] GET `/api/documents/{doc_id}/comparison/group/{name}` — group-wise
- [ ] GET `/api/documents/{doc_id}/comparison/column/{name}` — column-wise
- [ ] GET `/api/documents/{doc_id}/report` — document analysis report

**Frontend**
- [ ] Document list page (browse existing extractions)
- [ ] Document dashboard (summary stats, method status)
- [ ] Comparison table (columns × methods, expandable attribution)
- [ ] Group-wise view
- [ ] Column-wise detail view
- [ ] Attribution display (evidence, page, confidence, multi-candidate)

All read-only. No run extraction, no confirm, no storage.

### Phase 2: User Interaction (later)
- [ ] Run extraction from UI
- [ ] Confirm / Edit values
- [ ] Store confirmed data
- [ ] Export

---

## 6. File Paths for Each Method

| Method | Results Path |
|--------|--------------|
| Gemini native | `experiment-scripts/baselines_file_search_results/gemini_native/{model}/{pdf_stem}/extraction_metadata.json` |
| Landing AI baseline | `experiment-scripts/baselines_file_search_results/landing_ai_w_gemini/` or similar |
| Pipeline (no keywords) | `new_pipeline_outputs/results/{pdf_stem}/planning/extract_landing_ai/extraction_results.json` |
| Pipeline (with keywords) | `new_pipeline_outputs/results/{pdf_stem}/planning/plan_extract_columns_with_keywords/` |

The comparison service must know these paths and normalize the JSON formats.

---

## 7. Tech Stack Summary

| Layer | Current | Recommendation |
|-------|---------|----------------|
| Backend | Flask | Extend Flask → consider FastAPI later |
| Frontend | Vanilla JS + templates | Keep for Phase 1; consider React/Vue for richer UI |
| Storage | JSON files | JSON files for confirmed; add SQLite if multi-user |
| Extraction | Python scripts | Wrap in service functions, call from API |

---

## 8. Next Steps

**Phase 1 (display only):**
1. **Confirm extraction method paths** — Verify exact output locations for each method.
2. **Build comparison service** — `load_comparison_data(pdf_stem) -> unified_comparison`.
3. **Add read-only API routes** — documents list, status, comparison, report.
4. **Build display UI** — document list, dashboard, comparison table, group/column views, attribution.

**Phase 2 (later):** Confirmation storage, run extraction from UI, export.

---

## Appendix: Extraction Method Invocation

```python
# Gemini native (existing)
from web.extraction_service import ExtractionService
service = ExtractionService()
service.upload_pdf(pdf_path)
result = service.extract_from_csv(columns)  # or extract_single_column

# Pipeline (no keywords)
from experiment_scripts.extract_with_landing_ai import main  # or call functions directly
# Uses: load_landing_ai_chunks, select_chunks_for_group_vote_based, extract_group_multi_candidate

# Pipeline (with keywords)
from experiment_scripts.plan_and_extract_columns import plan_and_extract
plan_and_extract(pdf_path, column_names, results_root=..., use_keywords=True)
```
