# Attribution Module — Analysis & Root Cause

## Executive Summary

**Problem:** Columns with correct extracted values often highlight the wrong part of the PDF. The user sees the right answer but the highlighted region doesn't match where it actually came from.

**Root cause:** The highlight logic prioritizes **value substring search** over **page+source_type** from extraction. Value search often returns the wrong chunk because:
1. Short values (e.g. "NE", "42.4", "2023") appear in many chunks
2. We pick the **smallest bounding box** among matches — captions and footnotes have tiny boxes
3. **Evidence is never used** — we have a narrative of the source but don't use it for highlighting

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EXTRACTION (per method)                                                     │
│  • Pipeline: extract_landing_ai → extraction_results.json                   │
│  • Gemini/Landing AI baseline: extraction_metadata.json                      │
│  Output: value, evidence, page, source_type, candidates                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON SERVICE (web/comparison_service.py)                              │
│  • Merges all methods into comparison rows                                   │
│  • Normalizes: _normalize_pipeline_result, _normalize_gemini_result         │
│  • Output: row.methods[pipeline] = { value, page, source_type, attribution }│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  HIGHLIGHT SERVICE (web/highlight_service.py)                                │
│  • get_highlights_for_column(doc_id, column_name)                            │
│  • Strategy: 1) value substring in chunks → 2) fallback page+type            │
│  • Uses: landing_ai_parse_output.json (chunks with grounding.box)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (comparison.js)                                                    │
│  • Shows evidence, confidence, page in column detail                         │
│  • Draws cyan overlay boxes on PDF from highlights API                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Where Does Attribution Come From?

### Pipeline (extract_landing_ai)

| Field | Source | Notes |
|-------|--------|-------|
| `page` | Extraction plan | Plan says "Table A1 on page 16" → page=16 |
| `source_type` | Extraction plan | table / text / figure |
| `evidence` | LLM output | From candidates[0].evidence |
| `value` | LLM output | Extracted value |

**Important:** `page` is from the **plan** (where we looked), not from the **chunk** that actually contained the answer. The LLM might have seen multiple chunks; we don't store which chunk ID was used.

### Gemini / Landing AI Baseline

| Field | Source | Notes |
|-------|--------|-------|
| `page` | Always "Not applicable" | No chunk retrieval; full-doc context |
| `source_type` | plan_source_type or "text" | Often "Not applicable" |
| `evidence` | LLM output | Narrative, often mentions "page 1" in text |
| `value` | LLM output | Extracted value |

**Important:** We have no page/source_type. Evidence text sometimes says "page 1" but we don't parse it.

---

## Highlight Logic: `get_highlights_for_column`

### Current order
1. **Value-based** (if value exists): search for value in chunk markdown → pick smallest box
2. **Page+type** (fallback): filter chunks by page and source_type → pick smallest box

### Why value-first often fails

| Scenario | What happens |
|----------|--------------|
| Value "NE" | Matches 20+ chunks (captions, tables, figures). Smallest box = caption "NE, not estimable" |
| Value "42.4" | Matches 5+ chunks. Smallest box = narrow caption or footnote |
| Value "2023" | Matches 30+ chunks. Smallest box = year in header/footer |
| Value "standard of care" | Matches 27 chunks. Smallest box = tiny mention in caption |

**The "smallest box" heuristic assumes:** smaller = more specific = correct. **Reality:** captions and footnotes have tiny boxes but are often wrong (they mention the term, not the source).

### When page+type would work

Pipeline has `page=16`, `source_type=table` for Median OS. If we used that first:
- Filter chunks: page 16, type=table
- Pick smallest table chunk on page 16
- **Correct** — we'd highlight the actual Table A1

But we never get there because value search "succeeds" first (finds matches) and returns the wrong one.

---

## Evidence Is Never Used

**Evidence** is stored and displayed but never used for highlighting:
- "Table A1 (Figure A1) on page 16, 'Overall' row, 'Darolutamide + ADT + Docetaxel Median' column"
- "Figure 2, Panel A (page 6), 'Darolutamide + ADT + docetaxel' line description"

We could:
- Parse "page 16" from evidence when page is missing
- Filter value matches by page when evidence mentions a page
- Use evidence to rank chunks (prefer chunks whose content overlaps with evidence)

---

## Files & Responsibilities

| File | Role |
|------|------|
| `web/comparison_service.py` | Loads extraction results, normalizes to unified shape. `_normalize_pipeline_result`, `_normalize_gemini_result`. |
| `web/highlight_service.py` | Resolves value/page for column, runs value or page+type search, returns boxes. |
| `web/explainability_service.py` | Builds reasoning blocks; uses `_best_evidence`, `_best_confidence` from attribution. |
| `web/static/js/comparison.js` | Displays evidence, confidence, page; fetches highlights, draws overlays. |
| `experiment-scripts/extract_with_landing_ai.py` | Produces extraction_results.json with page, source_type from plan. |

---

## Recommended Fixes (Priority Order)

### 1. **Use page+source_type first when available** (high impact)

When pipeline (or any method) has valid `page` and `source_type`, use page+type for highlighting **before** value search. Only fall back to value when page/type are missing.

```python
# In get_highlights_for_column:
# 1. If page and source_type are valid (from pipeline): get_highlights_by_page_type first
# 2. If that returns highlights: use them
# 3. Else: try value-based
```

### 2. **Skip value search for very short values** (medium impact)

Values like "NE", "42.4", "2023" (< 4 chars) match too many chunks. Skip value search for these; use page+type only.

### 3. **Filter value matches by page when we have it** (medium impact)

When we have page from extraction: run value search, then **filter** matches to that page only. Among those, pick smallest box.

### 4. **Parse page from evidence** (low impact, helps baselines)

When page is "Not applicable", try `re.search(r'page\s+(\d+)', evidence)` and use that for page+type fallback.

### 5. **Store chunk_id in extraction** (future improvement)

If extract_with_landing_ai stored the chunk ID(s) used for each column, we could highlight the exact chunk. Requires pipeline changes.

---

## Quick Reference: Method Data Availability

| Method | page | source_type | evidence | Value for highlight |
|--------|------|-------------|----------|---------------------|
| Pipeline | ✅ (from plan) | ✅ | ✅ | ✅ |
| Landing AI baseline | ❌ "Not applicable" | ❌ | ✅ | ✅ |
| Gemini native | ❌ "Not applicable" | ❌ | ✅ | ✅ |
| plan_extract | ✅ | ✅ | ✅ | ✅ |

Pipeline and plan_extract have the best attribution data; baselines have only evidence and value.
