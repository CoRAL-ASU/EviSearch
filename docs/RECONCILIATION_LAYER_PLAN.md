# Reconciliation Layer — LLM-Based Final Answer Synthesis

## Goal

**Generate a final, exportable dataset** — tabular output where rows = trials/documents, columns = schema columns, cells = reconciled extracted values. The reconciliation layer is the last step before export (CSV, Excel, etc.).

---

## Vision

Add an **intelligent reconciliation layer** that takes outputs from Gemini, Landing AI, and Pipeline, and produces a **final synthesized answer** per column with:
- **Verified chunks** — evidence tied to actual document chunks (extremely accurate attribution)
- **More complete answers** — merge facts when one method missed something
- **Dual-trial handling** — e.g. STAMPEDE reports two trials; Pipeline has abiraterone trial, Landing AI has both → combine
- **Number reconciliation** — when Pipeline says "502 (abiraterone trial)" and Landing AI says "454 (abiraterone+enzalutamide trial)", produce "502 (abiraterone); 454 (abiraterone+enzalutamide)"

---

## Problem Statement

| Current state | Desired state |
|---------------|----------------|
| Three methods produce independent answers; we show all three side-by-side | Single best answer with provenance |
| Evidence is noisy; highlights often wrong | Evidence tied to verified chunks (chunk IDs) |
| Pipeline may miss one arm of dual trial; Landing AI may have it | LLM merges complementary info |
| No cross-method reasoning | LLM explains why it chose/combined sources |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  INPUTS                                                                          │
│  • Gemini native: extraction_metadata.json (value, evidence, page: N/A)         │
│  • Landing AI baseline: extraction_metadata.json (value, evidence, page: N/A)    │
│  • Pipeline: extraction_results.json (value, evidence, page, source_type,      │
│              candidates, extraction_plan)                                        │
│  • Document: PDF + landing_ai_parse_output.json (chunks with grounding.box)    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RECONCILIATION LAYER (new) — Chunk-centric, batched                             │
│                                                                                  │
│  1. Group chunks by page (or page-range)                                        │
│  2. For each page: which columns target it? (from pipeline page, evidence parse)  │
│  3. Build batches: batch = { chunks, columns }                                   │
│  4. ONE LLM call per batch: chunks + all columns + method outputs                │
│  5. Parse response → reconciled[col] for each col; verified_chunk_ids per col    │
│                                                                                  │
│  Total calls: ~5–15 per doc (not 130)                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                          │
│  • reconciled_value                                                              │
│  • reconciled_evidence                                                           │
│  • verified_chunk_ids → direct mapping to highlight boxes                        │
│  • contributing_methods: ["pipeline", "landing_ai"]                               │
│  • fusion_notes: "Pipeline had abiraterone trial; Landing AI added enzalutamide"  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Granularity: Chunk-Centric Batching (Efficient)

**Do NOT call per column or per group.** Instead:

| Approach | Calls per doc | Structure |
|----------|---------------|-----------|
| Per-column | 130+ | ❌ Too many |
| Per-group | 20–30 | ❌ Still too many |
| **Chunk-centric batch** | **5–15** | ✅ One call per chunk-batch |

**Chunk-centric flow:**
1. Load all chunks from `landing_ai_parse_output.json`
2. Group chunks by page (or by page-range to control batch size)
3. For each chunk-batch: determine which columns can be answered from these chunks
   - Pipeline: column.page + column.source_type → columns targeting this page
   - Evidence parsing: "page X" in Gemini/Landing AI evidence → columns
4. **One LLM call per batch:** Send chunks + all columns that reference them + method outputs
5. LLM returns reconciled results for all columns in that batch in a single response

**Result:** ~10–20 pages in a typical paper → 5–15 calls if we batch 2–3 pages together, or 10–20 if 1 page per call.

### 2. Building the Batch: Chunk → Column Mapping

```
For each chunk (id, page, type, content):
  → chunk belongs to batch = page_batch(page)   # e.g. pages 1-3 = batch_0

For each column (from comparison):
  → pipeline has page?     → column targets that page
  → evidence has "page X"? → column targets that page
  → no page?               → column goes to "floating" batch (first N pages)

Invert: batch_id → { chunks: [...], columns: [...] }
```

### 3. Chunk Context: Only Send Relevant Chunks

- Each batch sends **only** the chunks in that batch (not the whole doc)
- Columns in the batch are those whose extraction plans/evidence point to these chunks' pages
- Cap batch size by token count: e.g. max 15 chunks or 8K tokens per call

### 3. Output Schema

```json
{
  "column_name": "Control Arm - N",
  "final_value": "502 (abiraterone trial); 454 (abiraterone and enzalutamide trial)",
  "final_evidence": "Findings, page 1: '1003 patients... standard of care (n=502)... abiraterone (n=501)'. Same page: '916 patients... standard of care (n=454)... abiraterone and enzalutamide (n=462)'.",
  "verified_chunk_ids": ["758030bb-634b-4354-a481-ab7527214b88"],
  "contributing_methods": ["pipeline", "landing_ai_baseline"],
  "fusion_notes": "Pipeline reported only abiraterone trial. Landing AI evidence included both trials; merged.",
  "confidence": "high"
}
```

### 4. Verified Chunk IDs

- Landing AI parse chunks have `id` (UUID).
- LLM is given chunk IDs + content in the prompt.
- LLM returns which chunk IDs support the final answer.
- Highlight service: lookup chunk by ID → get `grounding.box` → **exact highlight**, no heuristic.

---

## Algorithm: Chunk-Centric Batching

```
1. Load:
   - chunks = landing_ai_parse_output.chunks
   - comparison = load_comparison_data(doc_id)
   - column_defs = load column definitions (if available)

2. Build page → columns mapping:
   for each row in comparison.comparison:
     for each method in [pipeline, landing_ai, gemini]:
       col = row.methods[method]
       page = col.page or parse_page_from_evidence(col.evidence)
       if page: columns_by_page[page].add(row.column_name)

3. Build batches (by page or page-range):
   PAGES_PER_BATCH = 2   # or 3; tune for token limit
   for page in sorted(columns_by_page.keys()):
     batch_id = page // PAGES_PER_BATCH
     batches[batch_id].pages.add(page)
     batches[batch_id].columns.update(columns_by_page[page])

4. For each batch:
   - chunks_in_batch = [c for c in chunks if c.grounding.page+1 in batch.pages]
   - columns_in_batch = batch.columns
   - method_outputs = {col: {pipeline: ..., landing_ai: ..., gemini: ...} for col in columns_in_batch}
   - ONE_LLM_CALL(chunks_in_batch, columns_in_batch, method_outputs)
   - Parse response → reconciled[col] for each col in batch

5. Handle "floating" columns (no page):
   - floating = columns that never got a page
   - One extra call with first 5 pages of chunks + floating columns
```

---

## Use Cases

### Use case 1: Dual-trial numbers (STAMPEDE)

**Column:** Control Arm - N

| Method | Value | Evidence |
|--------|-------|----------|
| Pipeline | "502 (Abiraterone trial)" | Table 1, page 5 |
| Landing AI | "502 (Abiraterone trial); 454 (Abiraterone and enzalutamide trial)" | Findings, page 1: both trials |
| Gemini | "502 (Abiraterone trial); 454 (Abiraterone and enzalutamide trial)" | Same |

**Reconciliation:** Landing AI and Gemini are more complete. Final value = combined. Verified chunks = chunks containing both numbers.

### Use case 2: One method missed it

**Column:** Median Follow-Up Duration (mo)

| Method | Value | Evidence |
|--------|-------|----------|
| Pipeline | "Not reported" | Plan said not in PDF |
| Landing AI | "96 months (IQR 86–107) in abiraterone trial; 72 months (61–74) in abiraterone+enzalutamide trial" | Findings, page 1 |
| Gemini | "96 months; 72 months" | Findings section |

**Reconciliation:** Pipeline missed it; Landing AI and Gemini have it. Final value from Landing AI (most detailed). Contributing methods = [landing_ai, gemini].

### Use case 3: Conflicting numbers

**Column:** Median OS (mo) | Overall | Control

| Method | Value | Evidence |
|--------|-------|----------|
| Pipeline | "42.4" | Table A1, page 16 |
| Landing AI | "45.7" | Text, page 1 (different trial arm?) |
| Gemini | "42.4" | Table A1 |

**Reconciliation:** LLM must resolve. Likely 42.4 is correct (Table A1); 45.7 may be from a different subgroup. LLM explains in fusion_notes.

### Use case 4: Verified chunks for accurate highlight

**Current:** Value "42.4" → value search → wrong chunk (caption).

**With reconciliation:** LLM returns `verified_chunk_ids: ["c9265d9a-3a09-4348-b409-fa7d7815ed93"]` (Table A1 chunk). Highlight service: direct lookup by ID → correct box.

---

## Implementation Phases

### Phase 1: Chunk-centric batching (single approach)
- Script: `experiment-scripts/reconcile_extractions.py`
- Build batches: chunks by page → columns that target each page
- One LLM call per batch: chunks + columns + method outputs
- Output: `reconciled_results.json` with `verified_chunk_ids` per column
- No separate "no chunks" phase — chunks are always included (that's the efficiency)

### Phase 2: Highlight service integration
- New path: when reconciled results exist, use `verified_chunk_ids` for highlights
- Lookup: `chunk_id → grounding.box` from landing_ai_parse_output
- Fallback to current value/page logic when no reconciled data

### Phase 3: Dashboard integration
- Comparison view: add "Reconciled" as a fourth "method" or replace with reconciled when available
- Show fusion_notes in column detail
- Use reconciled highlights when available

---

## Prompt Sketch: Batched (Chunks + Multiple Columns)

```
You are reconciling clinical trial extractions. You have DOCUMENT CHUNKS and METHOD OUTPUTS for multiple columns. Produce reconciled results for ALL columns in one response.

=== DOCUMENT CHUNKS (page {page_range}) ===
[Chunk ID: {id}]
{content}
---
[Chunk ID: {id}]
{content}
---
... (repeat for all chunks in batch)

=== COLUMNS TO RECONCILE ===

For each column below, produce final_value, final_evidence, verified_chunk_ids, contributing_methods, fusion_notes, confidence.

Column 1: {column_name_1}
  Definition: {definition_1}
  Pipeline: value="{val}", evidence="{ev}", page={p}
  Landing AI: value="{val}", evidence="{ev}"
  Gemini: value="{val}", evidence="{ev}"

Column 2: {column_name_2}
  ...

Column N: {column_name_N}
  ...

=== INSTRUCTIONS ===
1. Use ONLY the chunks above. For each column, which chunk IDs support the answer?
2. Prefer more complete answers. Merge when one method has partial info (e.g. one trial) and another has both.
3. For dual-trial papers, include values for each trial when reported.
4. If methods conflict, prefer stronger evidence (table > text).
5. verified_chunk_ids must be from the chunk IDs listed above.

=== OUTPUT (JSON) ===
{
  "reconciled": [
    {
      "column_name": "...",
      "final_value": "...",
      "final_evidence": "...",
      "verified_chunk_ids": ["uuid-1", "uuid-2"],
      "contributing_methods": ["pipeline", "landing_ai"],
      "fusion_notes": "...",
      "confidence": "high"
    },
    ...
  ]
}
```

**Token budget:** Cap chunks per batch (e.g. 3 pages, ~15 chunks) and columns per batch (e.g. 20) so total prompt stays under model limit.

---

## Export: Final Dataset Format

**Target output:** One table per run (or per corpus).

| doc_id | column_name | value | evidence | page | source_type | chunk_ids | contradiction |
|--------|--------------|-------|----------|------|-------------|-----------|----------------|
| NCT00268476_Attard_... | Control Arm - N | 502; 454 | ... | 1 | text | uuid-1,uuid-2 | false |
| NCT00268476_Attard_... | Median OS \| Overall \| Treatment | NE | ... | 16 | table | uuid-3 | false |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Export formats:**
- **CSV** — flat table, one row per (doc, column)
- **Wide CSV** — rows = docs, columns = schema columns (pivot)
- **Excel** — same, with optional evidence/citation in separate sheets

**Pipeline:** `extraction (3 methods) → reconciliation → export`

**Simplified reconciliation (no document to LLM):** LLM sees only method outputs (value, evidence, page). Outputs final_value + contradiction flag. Citation (chunk_ids) derived locally from page+source_type. Much cheaper (no chunk content in prompt).

---

## File Structure (Proposed)

```
experiment-scripts/
  reconcile_extractions.py      # Main script
  reconciliation_prompts.py     # Prompt templates
  export_reconciled_dataset.py  # CSV/Excel export

new_pipeline_outputs/results/{doc_id}/
  reconciliation/
    reconciled_results.json     # Per-doc reconciled output

outputs/
  reconciled_dataset.csv        # Final export (all docs)
  reconciled_dataset.xlsx       # Optional Excel

web/
  reconciliation_service.py     # Load reconciled results, merge with comparison
  highlight_service.py         # Add path: highlight by chunk_id when available
```

---

## Open Questions

1. **Batch sizing:** Pages per batch (2–3?) and max columns per batch (15–20?) to stay under token limit.
2. **Model:** Same Gemini 2.5 Flash as extraction? Or stronger model for reconciliation?
3. **When to run:** On-demand (when user opens doc) vs. precomputed (batch job)?
4. **Fallback:** If reconciliation fails or times out, keep current 3-method display?
5. **Floating columns:** Columns with no page (baselines) — one extra call with first N pages?
