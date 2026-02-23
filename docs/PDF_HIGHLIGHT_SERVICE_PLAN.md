# PDF Highlight Service — Plan

## Goal

Show **bounding boxes on the PDF page** when a user selects a column in the comparison dashboard. Full transparency: "here is exactly where this value came from."

---

## Current State

| Component | Has positions? | Location |
|-----------|---------------|----------|
| **Landing AI parse output** | ✅ Yes — `grounding.box` (left, right, top, bottom, 0–1), `grounding.page` (0-based) | `new_pipeline_outputs/results/{pdf_stem}/chunking/landing_ai_parse_output.json` |
| **Extraction results** | ❌ No — only `page`, `source_type`, `evidence` (text) | `.../planning/extract_landing_ai/extraction_results.json` |
| **Pipeline chunks** | ❌ No — `landing_ai_chunks.py` drops grounding when converting | — |
| **Dashboard** | ❌ No PDF viewer — only evidence text in cards | `comparison.html`, `comparison.js` |

---

## Data Flow (What We Need)

```
User selects column "Median PFS | Treatment"
    → column has: page=6, source_type=table
    → Highlight service: find chunks matching page+type in landing_ai_parse_output
    → Return: [{ page: 6, box: { left, top, right, bottom } }, ...]
    → Frontend: PDF viewer overlays semi-transparent rectangles at those positions
```

---

## Architecture

### 1. Highlight Service (Backend)

**New module:** `web/highlight_service.py`

**Responsibilities:**
- Load `landing_ai_parse_output.json` for a document
- Given `(pdf_stem, column_name)` or `(pdf_stem, page, source_type)`:
  - Filter chunks by page (1-based) and type (table/text/figure)
  - Return list of `{ page, box }` where `box` is `{ left, top, right, bottom }` (normalized 0–1)

**Coordinate format:** Landing AI uses normalized (0–1). PDF.js viewport can convert these to screen pixels. We keep normalized for simplicity.

**Data source path:**
```
new_pipeline_outputs/results/{pdf_stem}/chunking/landing_ai_parse_output.json
```

**Chunk matching logic:** Reuse the same rules as `find_chunks_for_column_tiered`:
- Page: `grounding.page` is 0-based → compare with 1-based `page` from extraction
- Type: map Landing AI types (text, table, figure, logo, …) to pipeline types (text, table, figure)

**Optional:** If extraction stored `chunk_ids` at runtime, we could use those. For now, re-derive from page+type.

---

### 2. API Endpoint

**New route:** `GET /api/documents/<doc_id>/highlights`

**Query params:**
- `column_name` (optional) — resolve from comparison data to get page/source_type
- OR `page` + `source_type` (optional) — direct lookup

**Response:**
```json
{
  "doc_id": "NCT00268476_Attard_STAMPEDE_Lancet'23",
  "column_name": "Median PFS (mo) | Overall | Treatment",
  "page": 6,
  "source_type": "table",
  "highlights": [
    {
      "page": 6,
      "box": { "left": 0.05, "top": 0.2, "right": 0.95, "bottom": 0.6 }
    }
  ],
  "available": true
}
```

If `landing_ai_parse_output.json` doesn't exist for this doc, return `available: false`.

---

### 3. PDF Viewer (Frontend)

**Options:**

| Option | Pros | Cons |
|-------|------|------|
| **PDF.js** | Mozilla, widely used, viewport API for coords | Need to handle overlay positioning on zoom/scroll |
| **react-pdf** | React wrapper, easier integration | Adds React if not already used |
| **iframe + PDF URL** | Simplest | No overlay control |
| **Pre-rendered PNGs** | No PDF.js, simple overlays | Large files, no text selection |

**Recommended:** PDF.js (vanilla or via CDN). The comparison page is vanilla JS; PDF.js works without a framework.

**Overlay approach:**
1. Render PDF page to canvas
2. For each highlight: `viewport.convertToViewportRectangle([left*W, top*H, right*W, bottom*H])` → screen rect
3. Create `<div>` with `position: absolute`, `background: rgba(34, 211, 238, 0.25)`, positioned over the canvas
4. Recompute on `pagerendered` and resize/scroll events

---

### 4. UI Integration

**Layout change:** Add a PDF viewer panel to the document detail view.

**Options:**
- **A) Split view:** Left = sidebar + column detail, Right = PDF viewer with highlights
- **B) Tab/modal:** "View in PDF" button opens modal with PDF + highlights
- **C) Inline below column detail:** PDF viewer appears when a column is selected, scrolls to page, shows highlights

**Recommended:** A or B. Split view gives constant context; modal keeps the main view clean.

**Flow:**
1. User selects column in sidebar
2. Column detail loads (value, evidence, methods)
3. `GET /api/documents/{doc_id}/highlights?column_name=...` is called
4. If highlights exist: PDF viewer shows the relevant page(s) with overlay rectangles
5. If no highlights: show "Source positions not available for this document" (e.g. no Landing AI parse)

---

## Implementation Phases

### Phase 1: Highlight Service + API (Backend)

1. Create `web/highlight_service.py`:
   - `load_landing_ai_parse(doc_id: str) -> dict | None`
   - `get_highlights_for_column(doc_id, column_name, comparison_data) -> list[dict]`
   - `get_highlights_by_page_type(doc_id, page, source_type) -> list[dict]`
2. Add route `GET /api/documents/<doc_id>/highlights` in `main_app.py`
3. Wire comparison_service to resolve column → page/source_type when column_name is given

**Files to create/modify:**
- `web/highlight_service.py` (new)
- `web/main_app.py` (add route)

---

### Phase 2: PDF Viewer + Overlays (Frontend)

1. Add PDF.js to comparison page (CDN or static)
2. Add PDF viewer container to `comparison.html`
3. Implement in `comparison.js`:
   - `loadPdfViewer(docId, pdfUrl)` — fetch PDF, render first page
   - `showHighlights(highlights)` — create overlay divs from box coords
   - `goToPage(pageNum)` — render specific page, re-apply highlights
4. On column select: fetch highlights, show PDF viewer, go to first highlight page, draw overlays

**PDF serving:** Need a route to serve the PDF file. Path: `dataset/{pdf_stem}.pdf` or `new_pipeline_outputs/.../` — confirm where PDFs live for comparison docs.

---

### Phase 3: Polish

1. Handle multi-page highlights (e.g. column from pages 5 and 6)
2. Zoom/scroll: recompute overlay positions
3. Fallback: when no grounding, show "Page X" badge only (no boxes)
4. Optional: evidence text highlighting in a text snippet view (separate from PDF)

---

## Coordinate Reference

**Landing AI `grounding.box`:**
- `left`, `right`: 0–1 (fraction of page width)
- `top`, `bottom`: 0–1 (fraction of page height)
- Origin: top-left

**PDF.js viewport:**
- `viewport.convertToViewportRectangle([x0, y0, x1, y1])` — PDF points → screen pixels
- For normalized coords: `x0 = left * pageWidth`, `y0 = top * pageHeight`, etc., then convert

---

## Scope Limits

- **Landing AI only:** Highlights require `landing_ai_parse_output.json`. Pipeline and Gemini native extractions don't have chunk-level grounding. We can still show page number; boxes only when Landing AI parse exists.
- **Chunk-level:** We highlight whole chunks, not sub-spans (evidence substring). Finer granularity would need character→position mapping or LLM span extraction.
- **Documents:** Only docs under `new_pipeline_outputs/results/` with `chunking/landing_ai_parse_output.json` will have highlight support.

---

## File Summary

| File | Action |
|------|--------|
| `web/highlight_service.py` | Create — load parse output, filter chunks, return boxes |
| `web/main_app.py` | Add `/api/documents/<doc_id>/highlights`, `/api/documents/<doc_id>/pdf` (serve PDF) |
| `web/templates/comparison.html` | Add PDF viewer container |
| `web/static/js/comparison.js` | Add PDF.js init, highlight overlay logic, column→highlight fetch |

---

## PDF Path Resolution

Comparison docs use `pdf_stem` (e.g. `NCT00268476_Attard_STAMPEDE_Lancet'23`). PDFs are in:
- `dataset/{pdf_stem}.pdf` (primary)
- Fallback: check `new_pipeline_outputs/results/{pdf_stem}/` for symlinks or copies

Add route `GET /api/documents/<doc_id>/pdf` to serve the PDF file for the viewer.

---

## Open Questions

1. **PDF path:** Confirm `dataset/{pdf_stem}.pdf` exists for all comparison docs.
2. **Multi-method:** When comparing methods, do we show highlights only for pipeline/landing_ai (which use chunks), or also try to infer for Gemini native?
3. **Table cells:** Landing AI parse has per-cell positions in some outputs. Use chunk-level only for v1, or explore cell-level later?
