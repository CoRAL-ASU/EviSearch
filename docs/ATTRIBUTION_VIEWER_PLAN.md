# Attribution Viewer Upgrade Plan

## Goals

1. **Show the whole document** (all pages) instead of only the first highlighted page
2. **Show top 2–3 chunks** (reduce from current 4)
3. **Center attribution on exact values/keywords** from Gemini, Landing AI, Pipeline, and the final reconciled output

---

## 1. Whole-Document PDF View

### Current behavior
- `attribution.html` loads PDF.js and renders **only the first page** where a highlight appears
- `loadPdfAndHighlights` uses `const firstPage = res.highlights[0].page` and renders a single canvas

### Target behavior
- Show all pages of the PDF in a scrollable vertical layout
- Render each page as its own canvas (stacked)
- Draw highlights on the correct page for each chunk
- Use lazy loading if document is large (e.g. render visible pages only)

### Implementation options

| Option | Pros | Cons |
|--------|------|------|
| **A. All pages pre-rendered** | Simple, all highlights visible at once | Heavy for long PDFs (many canvases) |
| **B. Virtual scroll / lazy pages** | Scales to long documents | More complex (visibility detection, render on scroll) |
| **C. Single tall canvas** | One DOM element | Memory issues for large PDFs |

**Recommendation:** Start with **Option A** (all pages) with a modest scale (e.g. 1.2) so we don't blow memory. Add lazy loading later if needed for 50+ page docs.

### Files to change
- `web/templates/attribution.html`
  - Replace single `#pdf-canvas` with a container of page canvases
  - Loop over `pdfDoc.numPages`, create canvas per page
  - Position highlights by mapping `highlight.page` → correct page canvas overlay
  - Scroll container with `overflow-y: auto` and fixed max-height

---

## 2. Top 2–3 Chunks

### Current behavior
- `attribution_service.enrich_reconciled_with_attribution` uses `top_k=4`
- `retrieve_chunks_for_evidence` returns up to 4 chunks per column

### Target behavior
- Use `top_k=3` (or 2) so attribution focuses on the strongest evidence
- Configurable via parameter so we can tune 2 vs 3

### Files to change
- `web/attribution_service.py`: change default `top_k=4` → `top_k=3`
- `experiment-scripts/run_attribution.py`: pass `top_k=3` when calling enrich

---

## 3. Attribution Centered on Exact Values/Keywords from All Methods

### Current behavior
- Attribution uses `final_value` only for value-match
- Evidence text is collated from contributing methods but not their **extracted values**
- Method-specific values (Gemini, Landing AI, Pipeline) live in comparison data, not in reconciled_results.json

### Target behavior
- **Extract matchable parts from:**
  - `final_value` (reconciled output)
  - Gemini value (`row.methods.gemini_native.value`)
  - Landing AI value (`row.methods.landing_ai_baseline.value`)
  - Pipeline value (`row.methods.pipeline.value`)
- Prioritize chunks that contain these exact values/keywords (numbers, percentages, short phrases)
- Fall back to semantic retrieval only when no value-match chunks exist

### Data flow
- Reconciled API returns columns with `final_value`, `contributing_methods`, `attributed_chunks`
- **New:** When running attribution, we need access to comparison rows to get per-method values
- `enrich_reconciled_with_attribution` already receives `comparison_rows` — each row has `methods[method_name].value`

### Algorithm for value match

1. **Collect matchable parts** from:
   - `final_value`
   - For each `m` in `contributing_methods`: `row.methods[m].value` (from comparison)
2. **Deduplicate** and filter out non-matchable strings (`"not found"`, `"Not reported"`, empty, very long text)
3. **Value match:** Prefer chunks that contain the most parts (or all parts from final_value + at least one from a method)
4. **Ranking:** Score = number of parts matched + location boost (pipeline page, type)

### Files to change
- `web/attribution_service.py`:
  - Add `method_values: Optional[Dict[str, str]]` to `retrieve_chunks_for_evidence` (or to the evidence collation)
  - In `enrich_reconciled_with_attribution`: build `all_values = [final_value] + [row.methods[m].value for m in contributing_methods if row.methods.get(m)]`
  - Pass `all_values` into retrieval; `_get_matchable_parts` should consider all of them
  - Update `_get_matchable_parts` to accept multiple value strings and merge parts
- `web/main_app.py` (or comparison service): Ensure the attribution viewer can get comparison data — either:
  - Extend `/api/documents/<doc_id>/reconciled` to optionally include `method_values` per column (by loading comparison_data and merging)
  - Or the attribution is run offline (run_attribution.py) and we'd need to store richer data in reconciled_results.json
- **Offline run:** `run_attribution.py` loads comparison_data and passes it to enrich; reconciled_results.json already has attributed_chunks. The *viewer* just displays them. So the change is in **run_attribution** and **attribution_service** — use method values when computing attribution. Re-running attribution will produce better chunks.

---

## 4. Summary of Changes

| Component | Change |
|-----------|--------|
| **attribution.html** | Multi-page PDF view; render all pages; map highlights to correct page overlay |
| **attribution_service.py** | `top_k=3`; use method values (Gemini, Landing AI, Pipeline, final) for value-match |
| **run_attribution.py** | Pass comparison_rows; ensure top_k=3 |
| **API `/reconciled`** | Optional: include method_values in response for UI display (show which method said what) |

---

## 5. UI Enhancements (Optional)

- **Left panel:** Show method values (Gemini: X, Landing AI: Y, Pipeline: Z, Final: W) when a column is selected
- **Chunk list:** Indicate which value/keyword was matched (e.g. "matches: 192, 38%, 502")
- **Highlight colors:** Different color per chunk rank (1st, 2nd, 3rd) for easier scanning

---

## 6. Implementation Order

1. **top_k=3** — trivial change in attribution_service and run_attribution
2. **Method values in value-match** — extend attribution_service to use Gemini/Landing/Pipeline values when available
3. **Whole-document view** — refactor attribution.html to multi-page canvas + highlight overlays
4. **UI polish** — method values in sidebar, match hints
