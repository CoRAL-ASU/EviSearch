# Value Verifier Layer — Plan

## Current State

### 1. Deterministic Verification (web/verification_service.py)
- **Input**: `final_value`, `attributed_chunks` (full text from Landing AI parse)
- **Logic**: Extract numeric parts from value → check if ALL appear in chunk text
- **Output**: `verified` / `failed` / `na` (no numbers)
- **Limitation**: Exact numeric match only. Fails when:
  - Value is synthesized (e.g. "79.6%" from table row "398/500")
  - Value is paraphrased ("darolutamide + ADT" vs "darolutamide and androgen deprivation therapy")
  - Numbers in different format ("271 [54%]" vs "271 (54%)")

### 2. Plan Verifier (experiment-scripts/verify_plans_with_pdf.py)
- **Purpose**: Verifies **extraction plans** (where to look, extraction_plan), NOT extracted values
- **Input**: Group plan (columns with extraction_plan, page, source_type), chunks
- **Flow**:
  1. `find_relevant_chunks_for_group` — select chunks by page + source_type from plan
  2. `format_chunks` — format as TEXT/TABLE/FIGURE blocks
  3. `build_group_prompt` — LLM evaluates each column's plan against chunks
  4. Output: VERIFIED | WRONG per column, with reasoning, issues, corrected_plan
- **Key prompt rules**:
  - "Your reasoning MUST cite Chunk IDs and include verbatim quotes"
  - "If you cannot quote support from chunks, verdict must be WRONG"
  - Uses OutputStructurer (Qwen) to parse free-text → structured JSON

### 3. Extraction (experiment-scripts/extract_with_landing_ai.py)
- **Purpose**: Extract values from chunks using LLM
- **Output**: `candidates` (value, evidence, assumptions, confidence), `primary_value`, `found`
- **Evidence**: LLM provides short excerpt or location (e.g. "Table 3, Any AE worst grade Grade 3-5")
- **Synthesized values**: Handles "458 (70.2%)" from table — LLM can compute 458/652≈70.2%

---

## Proposed: LLM Value Verifier

### Purpose
Verify that an **extracted value** is supported by the attributed chunks. Handles:
- Exact numbers in text
- Synthesized values (e.g. percentages from table rows, ratios)
- Paraphrased text values
- "Not reported" when chunks confirm absence

### Input
- `column_name`, `definition` (from table_definitions)
- `final_value` (extracted value to verify)
- `attributed_chunks` (full text, not truncated snippets)
- Optional: `method_values` (Gemini, Landing AI, Pipeline, Agent) for context

### Output Schema (Pydantic) — Minimal
```python
class ValueVerificationResult(BaseModel):
    verdict: Literal["correct", "wrong"]  # Value/plan correct or not
    alternative_value: Optional[str] = None  # Another valid answer if one exists
```

### Chunk Formatting
Reuse pattern from verify_plans_with_pdf and extract_with_landing_ai:
- `format_chunks` — group by TEXT / TABLE / FIGURE, include chunk_id, page
- For Landing AI chunks: use `markdown` (full text) from parse, not truncated snippet
- Char limit per chunk: ~8000–16000 (tables can be large)

---

## Integration with Web Verification

### Data Flow (End-to-End)

```
Agent Extractor (extraction_results.json)
        │
        ▼
Attribution (enrich_reconciled_with_attribution)
        │  → attributed_chunks per column (chunk_id, snippet, page, source_type)
        ▼
Deterministic Verification (verify_columns)
        │  → verification.deterministic: { verified, applicable, numeric_parts, failed_parts }
        ▼
LLM Verification (group-wise, ~15 columns at a time)
        │  → verification.llm: { verdict, alternative_value }
        ▼
UI: "verified" | "verified with LLM" | "failed"
```

### LLM Verification: Group-Wise Batching (Column-Centric)

**Flow**: Iterate over **columns** (grouped by definition group). For each batch of ≤15 columns:
- Union all `attributed_chunks` from those columns
- Load full chunk text via `get_full_chunk_texts(doc_id, chunk_ids)`
- One LLM call: chunks + columns → verify each column's value against the chunks

**Grouping logic** (same as agent extractor, `MAX_COLUMNS_BATCH = 15`):
- Columns from `load_definitions()` → groups (e.g. "Add-on Treatment", "Control Arm")
- Batch columns: sum ≤ 15 per LLM call. If a group has > 15 columns, that group gets its own call.

**When to run**: After agent extraction + attribution. Triggered by:
- User clicks "Run LLM verification" on attribution page (runs in background, updates when done)
- Optional: Auto-run after attribution refresh (configurable)

**Two options for which columns to verify**:

| Option | Columns to verify | Use case |
|--------|-------------------|----------|
| **A: Failed only** | Only columns where deterministic `verified = false` (and `applicable = true`) | Lower cost, focus on uncertain cases |
| **B: All extracted** | All columns with attributed chunks | Higher confidence, full audit |

Default: **Option A** (failed only). Configurable via API param `verify_all=false` (default) or `verify_all=true`.

---

### LLM Verification Prompt (Group)

```
You are verifying EXTRACTED VALUES for clinical trial columns using ONLY the provided chunks.
For each column: (1) Is the value/plan correct? (2) Is there an alternative answer?

COLUMNS TO VERIFY:
---
Column 1: {column_name}
Definition: {definition}
Extracted value: {final_value}
---
Column 2: ...
(repeat for each column in batch, up to 15)

RETRIEVED CHUNKS (evidence from attribution step):
{formatted_chunks}

Task for EACH column:
1. VERDICT: Is the extracted value correct? (supported by chunks — exact, synthesized, or paraphrased)
2. ALTERNATIVE: If there is another valid answer (different table, interpretation, or correction), provide it.

Output (one block per column, no code fences):
COLUMN: <exact column_name>
VERDICT: CORRECT|WRONG
ALTERNATIVE_VALUE: (string or null — only if another valid answer exists)
```

### Output Schema (Per Column) — Minimal

```python
class LLMVerificationResult(BaseModel):
    verdict: Literal["correct", "wrong"]  # Value/plan correct or not
    alternative_value: Optional[str] = None  # Another valid answer; show in UI for each column
```

### API Design

```
POST /api/documents/<doc_id>/verify-llm
Body: { "verify_all": false }  # false = only failed deterministic; true = all columns
Response: { "success", "columns": [...], "verification_stats": { "llm_correct", "llm_wrong", "skipped" } }
```

- Runs deterministically first (if not already done)
- Groups columns by definition groups, batches ≤ 15 per call
- Streams or returns when complete
- Persists to `attribution_results.json` or `reconciled_results.json` (add `verification.llm`)

### Data Model Extension

Add to each column:
```json
{
  "column_name": "...",
  "final_value": "...",
  "attributed_chunks": [...],
  "verification": {
    "deterministic": { "verified", "applicable", "numeric_parts", "failed_parts" },
    "llm": {
      "verdict": "correct" | "wrong",
      "alternative_value": "..." | null
    }
  }
}
```

### UI Display

| Deterministic | LLM verdict | Badge / Label |
|---------------|-------------|---------------|
| verified | — | ✓ verified |
| verified | correct | ✓ verified with LLM |
| failed | correct | ✓ verified with LLM |
| failed | wrong | ✗ |
| na | correct | ✓ verified with LLM |
| na | wrong | ✗ |

- **"verified with LLM"**: Shown when `verification.llm.verdict === "correct"` (regardless of deterministic).
- **Alternative value**: For each column, when `verification.llm.alternative_value` exists, show it (e.g. expandable "Alternative: …" or inline).

---

### Chunk Formatting for LLM Verifier

**Input**: Union of attributed chunks from all columns in the batch.
- Each attributed chunk: `{chunk_id, page, source_type, snippet}` (snippet truncated to 200 chars in storage)
- We load **full text** via `get_full_chunk_texts(doc_id, chunk_ids)` from `landing_ai_parse_output.json`

**Format** (similar to extract_with_landing_ai):
```
--- Chunk <chunk_id> (TEXT on page N) ---
<full markdown text>

--- Chunk <chunk_id> (TABLE on page N) ---
<full table markdown>

--- Chunk <chunk_id> (FIGURE on page N) ---
<caption/summary>
```

- Group by type (TEXT, TABLE, FIGURE), sort by page
- Char limit per chunk: ~8000 (TEXT), ~16000 (TABLE) to avoid token overflow
- Use `_chunk_text(chunk)` from highlight_service for markdown → plain text

---

## Chunk Source Compatibility

| Source | Format | Used by |
|--------|--------|---------|
| **verify_plans_with_pdf** | `pdf_chunked.json` (pipeline) — content, table_content, page, type | Plan verifier |
| **extract_with_landing_ai** | Landing AI chunks — content, table_content, page, type | Extraction |
| **Web attribution** | Landing AI `landing_ai_parse_output.json` — markdown, grounding | Attribution, deterministic verification |

For the LLM value verifier, we use **attributed chunks** (from attribution). These come from Landing AI parse. We need to:
- Load full chunk text via `get_full_chunk_texts(doc_id, chunk_ids)`
- Format similarly to extract_with_landing_ai (TEXT/TABLE/FIGURE blocks)
- Landing AI chunks have `markdown`; pipeline chunks have `content` + `table_content`. We'll need a unified formatter or adapter.

---

### Group Batching Logic (Detail)

Same as agent extractor (`MAX_COLUMNS_BATCH = 15`):

1. **Group columns** by definition group (from `load_definitions()`).
2. **Filter** which columns to verify:
   - Option A: only `verification.deterministic.applicable && !verification.deterministic.verified`
   - Option B: all columns with `attributed_chunks`
3. **Build batches**: Sum columns per batch ≤ 15. If a group has > 15 columns, that group gets its own batch. Otherwise combine small groups until 15.
4. **Chunks per batch**: Union of `attributed_chunks` from all columns in the batch → load full text → format.

---

## Implementation Order

| Step | Task | Effort |
|------|------|--------|
| 1 | Create `web/llm_verifier.py` — `verify_group_llm(batch_columns, full_chunk_texts, definitions_map)`, prompt, schema | Medium |
| 2 | Add `format_chunks_for_verification(chunks_by_id, full_texts)` — Landing AI chunks → TEXT/TABLE/FIGURE blocks | Low |
| 3 | Add `build_verification_batches(columns, groups, verify_all)` — group-wise batching | Low |
| 4 | Add API `POST /api/documents/<doc_id>/verify-llm` with `verify_all` param | Medium |
| 5 | Wire into attribution flow: "Run LLM verification" button triggers API | Low |
| 6 | Update attribution UI: show "verified with LLM" badge when `verification.llm.verdict === "correct"`; show `alternative_value` for each column when present | Low |
| 7 | Persist `verification.llm` to attribution/reconciled results | Low |

---

## Files to Reference

- `experiment-scripts/verify_plans_with_pdf.py` — prompt structure, VERIFIED/WRONG pattern, structurer usage
- `experiment-scripts/extract_with_landing_ai.py` — format_chunks, build_extraction_prompt_multi_candidate
- `web/verification_service.py` — deterministic layer, get_full_chunk_texts
- `web/highlight_service.py` — load_landing_ai_parse, get_full_chunk_texts, _chunk_text
- `src/table_definitions/definitions.py` — load_definitions for column definitions
