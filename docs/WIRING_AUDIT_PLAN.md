# Wiring Audit Plan — Pages & APIs

## 1. Attribution Page

### 1.1 Document List Loading

| Step | What | API/Data | Potential Bug |
|------|------|----------|----------------|
| 1 | Page loads, fetches document list | `GET /api/documents/reconciled` | |
| 2 | Response shape | `{documents: string[]}` | OK — returns sorted doc IDs |
| 3 | Doc dropdown | Populates `<select id="doc-select">` | OK |
| 4 | URL params | `?doc=X&column=Y` — pre-select doc and column | **CHECK:** `selectColumnByName(colParam)` runs in `setTimeout(300)` — race if doc change is async; column might not exist yet in filtered list |

### 1.2 Column List & Reconciled Data

| Step | What | API/Data | Potential Bug |
|------|------|----------|----------------|
| 1 | On doc change | `GET /api/documents/<doc_id>/reconciled` | |
| 2 | Response shape | `{success, doc_id?, columns[], verification_stats?}` | **BUG:** `doc_id` — when loading from `recon_attr_path`, the file has `doc_id`; when from `rec_path` we use `data` from reconciled_results which has `doc_id`; when from `agent_attr_path` the file may or may not have `doc_id`. **Verify** agent_attr_path file structure. |
| 3 | Column structure | Each col: `column_name`, `final_value`, `candidate_a`, `candidate_b`, `reconciliation_reasoning`, `attributed_chunks`, `chunk_ids` | |
| 4 | Human-edited merge | API applies human_edited over `final_value` | OK |
| 5 | **Verification badges** | Frontend expects `c.verification.llm.verdict` or `c.verification.applicable` / `c.verification.verified` | **BUG:** Reconciliation outputs `verification` (e.g. `both_correct`) at column level in `reconciled_results.json`. `_reconciliation_agent_to_columns` does NOT pass `verification` through. `enrich_reconciled_with_attribution` does not add it. So `verification` is never present in API response → badges never show. |

### 1.3 Candidate A/B Sources

| Source | Where | Potential Bug |
|--------|-------|---------------|
| candidate_a | `agent_extractor/extraction_results.json` → `columns[].value` | OK |
| candidate_b | `search_agent/extraction_results.json` → `columns[].value` | **CHECK:** If doc has reconciliation but NO search_agent, `search_cols` is empty → candidate_b = "". Display is fine, but user may expect "not run" vs "not found". |
| comparison_rows | `load_comparison_data` — merges agent, search_agent, etc. | OK — used for method_values and attribution fallback |

### 1.4 Attribution (chunk_ids / highlights)

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Columns have `attributed_chunks` or `chunk_ids` | From `enrich_reconciled_with_attribution` or pre-saved attribution_results |
| 2 | `selectColumn` → `loadPdfAndHighlights(docId, chunkIds)` | |
| 3 | Highlights API | `GET /api/documents/<doc_id>/highlights?chunk_ids=...` | Depends on `landing_ai_parse_output.json` existing |
| 4 | When no chunk_ids | Shows "No highlight available" | OK |

### 1.5 Run Reconciliation Button

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Click | `POST /api/documents/<doc_id>/run-reconciliation` | |
| 2 | Requires | agent_extractor + search_agent both exist | **CHECK:** Button enabled whenever doc selected; if only agent exists, run will fail with "Search agent results not found". UX: could disable when no search_agent. |
| 3 | On success | Re-fetches `API.reconciled(docId)` and re-renders | OK |

### 1.6 Save Human Edit

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Click | `POST /api/documents/<doc_id>/human-edited` with `{columns: {col_name: {value: "..."}}}` | OK |
| 2 | On success | Updates `selectedColumn.final_value` locally | OK — but next doc switch/refresh will re-load from API which does merge human_edited, so persisted correctly |

### 1.7 Refresh Attribution

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Click | `POST /api/documents/<doc_id>/attribution/refresh` | |
| 2 | Writes | `reconciliation_agent/attribution_results.json` or `agent_extractor/attribution_results.json` | |
| 3 | Response | `{success, columns}` — `renderColumnList(res)` expects `res.columns` | **CHECK:** `renderColumnList` expects full reconciled response with `columns`; refresh returns `{columns, doc_id, verification_stats}`. It uses `res.columns` and `res.filteredColumns` — but filteredColumns is set from `res.columns`. `reconciledData` is set to `res`. But `reconciledData.doc_id` — refresh response has doc_id. OK. |

### 1.8 Feedback

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Payload | `source: "attribution"`, `table: {column_name, correct_sources, candidate_a, candidate_b, reconciled, reasoning}` | OK |
| 2 | `reconciledData.doc_id` | Must be set | When from refresh, `res` has `doc_id`. When from reconciled API, `data` has `doc_id`. So `reconciledData = res` and `res.doc_id` should exist. **Verify** refresh returns doc_id. It does: `return jsonify({"success": True, "doc_id": doc_id, "columns": enriched, ...})`. Good. |

---

## 2. Extract Page

### 2.1 Document List

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | `GET /api/documents/selectable` | Returns `{documents: [{id, name, source, has_extraction}]}` |
| 2 | View Attribution button | `href=/attribution?doc=${docId}` | OK |

### 2.2 Extraction Flow

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | SSE stream from extract API | |
| 2 | On completion | Sets `view-attribution-btn.href` | OK |
| 3 | Table | Shows Direct PDF / Search-Based; reconciled not shown on Extract | OK |

---

## 3. QA Page

### 3.1 Prepare Document

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | `POST /api/qa/prepare-document` | Creates landing_ai_parse_output.json |
| 2 | Session invalidation | Boot ID check | OK (from prior work) |

### 3.2 Ask (Quick vs Full)

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Quick | Direct PDF chat, no attribution | OK |
| 2 | Full | Agent + Search + Reconcile, returns chunk_ids | Attribution fallback added earlier |

---

## 4. Tables Report Page

### 4.1 Data Loading

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | `GET /api/report/tables` | Only docs with reconciliation |
| 2 | Human-edited merge | API merges human_edited over reconciled per doc | OK |
| 3 | column_groups | From definitions; columns without group → "Other" | OK |

### 4.2 Filters & Export

| Step | What | Potential Bug |
|------|------|---------------|
| 1 | Group filter | Filters columns by group | OK |
| 2 | Column search | Text filter on column names | OK |
| 3 | Export CSV/Excel | Uses filtered rows/columns | OK |

---

## 5. API Summary — Endpoints Used by Attribution

| Endpoint | Purpose | Response used by |
|----------|---------|------------------|
| `GET /api/documents/reconciled` | List doc IDs | Doc dropdown |
| `GET /api/documents/<id>/reconciled` | Full reconciled data per doc | Column list, cards, highlights |
| `POST /api/documents/<id>/attribution/refresh` | Re-run attribution, save | Refresh button |
| `POST /api/documents/<id>/run-reconciliation` | Run reconciliation agent | Run reconciliation button |
| `POST /api/documents/<id>/human-edited` | Save user-edited value | Save my edit |
| `GET /api/documents/<id>/highlights?chunk_ids=` | PDF highlight boxes | PDF viewer |
| `GET /api/column-groups` | Column → group mapping | Grouping in sidebar |
| `POST /api/feedback` | Record feedback | Feedback modal |

---

## 6. Prioritized Bug List (Attribution Page)

1. **Verification badges never show** — `verification` is not passed from reconciled_results through `_reconciliation_agent_to_columns` or `enrich_reconciled_with_attribution`. Fix: Pass `verification` through to columns in API response. **Format mismatch:** reconciled stores string (`both_correct`, `A_correct_B_wrong`); frontend expects object `{llm: {verdict: "correct"}}` or `{applicable, verified}`. API/frontend must map reconciled strings to expected shape.
2. **Run reconciliation enabled when no search_agent** — Button is enabled for any doc; run fails if search_agent missing. Fix: Disable button when doc has no search_agent, or show clear error.
3. **URL param `column` race** — `selectColumnByName(colParam)` in 300ms timeout may run before column list is rendered. Fix: Call after renderColumnList completes, or use a flag / retry.
4. **agent_attr_path structure** — When loading from `agent_extractor/attribution_results.json`, ensure it has `doc_id`. The refresh writes it — confirm existing files have it.

---

## 7. Next Pages to Audit (after Attribution fixes)

- Extract page: streaming, table updates
- QA page: prepare, ask, session
- Tables Report: filters, export, empty states
- Home page: links
