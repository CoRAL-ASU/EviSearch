# Comparison Dashboard Plan

## Overview

Redesign the comparison view into a dashboard with:
- **Top**: 3 method cards (Landing AI, Gemini Native, Pipeline) with stats
- **Left sidebar**: Column groups as navigation
- **Main content**: Group snapshot when group clicked; column detail with attribution when column clicked
- **Optional**: PDF page snapshot for evidence (Landing AI chunks)

---

## Phase 1: Scope & Decisions

### Methods to Show
- **Landing AI** (`landing_ai_baseline`)
- **Gemini Native** (`gemini_native`)
- **Pipeline** (`pipeline` — extract_landing_ai)

### Top Section Stats (per method)
- Total columns requested
- Found (non-empty)
- Empty
- Empty groups (groups where all columns are empty for that method)

### Left Sidebar
- All column groups (from `by_group`)
- Expandable sections (Flowbite collapse)
- Click group → show group snapshot in main area
- Click column → show column detail with attribution

### Main Content
- **Group view**: Table of all columns in group with values from each method
- **Column view**: Attribution, evidence, reasoning; optionally page snapshot

### PDF Page Snapshot
- **Idea**: When a column has `page` (e.g. page 9), show a thumbnail/snapshot of that PDF page
- **Requires**: API endpoint to render PDF page to image (PyMuPDF); PDF path from `dataset/{doc_id}.pdf`
- **Scope**: Phase 2 or later — nice-to-have

---

## Phase 2: Implementation Order

1. **Backend**
   - [ ] Add `get_dashboard_report()` — returns 3 methods only, with `empty_groups` per method
   - [ ] Add `GET /api/documents/<doc_id>/dashboard` — returns dashboard payload (report + comparison + by_group)
   - [ ] (Later) Add `GET /api/documents/<doc_id>/page/<n>` — returns PNG of PDF page N

2. **Frontend — Dependencies**
   - [ ] Add Flowbite (CDN: CSS + JS) to `comparison.html`

3. **Frontend — Layout**
   - [ ] Restructure document detail view: sidebar + main content
   - [ ] Left: Flowbite sidebar with groups (collapsible)
   - [ ] Right: Main content area (group snapshot or column detail)

4. **Frontend — Top Section**
   - [ ] 3 method cards: Landing AI, Gemini Native, Pipeline
   - [ ] Each card: Total | Found | Empty
   - [ ] Each card: list of empty groups (collapsible or tooltip)

5. **Frontend — Sidebar**
   - [ ] Render groups from `by_group`
   - [ ] Each group expandable; columns listed under it
   - [ ] Click group → load group snapshot into main area
   - [ ] Click column → load column detail into main area

6. **Frontend — Main Content**
   - [ ] Group snapshot: table (columns × methods)
   - [ ] Column detail: value per method, evidence, reasoning, confidence
   - [ ] (Later) Page snapshot placeholder / image when available

---

## Phase 3: Out of Scope (for now)

- PDF page image API and rendering
- User interactions (confirm, edit, save)
- Additional methods (pipeline_plan_extract, pipeline_keywords)

---

## Data Flow

```
User opens document
  → GET /api/documents/<id>/comparison  (existing)
  → GET /api/documents/<id>/report      (existing, or new dashboard endpoint)

Dashboard payload:
  - by_method: { landing_ai: { total, found, empty, empty_groups }, ... }
  - by_group: { "Adverse Events": [...], ... }
  - comparison: [ { column_name, group_name, methods: {...} }, ... ]
```

---

## Files to Touch

| File | Changes |
|------|---------|
| `web/comparison_service.py` | Add `get_dashboard_report()`, `DASHBOARD_METHODS` |
| `web/main_app.py` | Add `/api/documents/<id>/dashboard` (optional) |
| `web/templates/comparison.html` | Flowbite, new layout, sidebar |
| `web/static/js/comparison.js` | New rendering logic for dashboard |

---

## Notes

- Revert or simplify the `get_dashboard_report` / `_is_not_found` changes in `comparison_service.py` if we want to implement incrementally
- Flowbite sidebar: use `data-collapse-toggle` for expandable groups
- Keep existing Report / Comparison / Groups tabs? Or replace with new layout? (TBD)
