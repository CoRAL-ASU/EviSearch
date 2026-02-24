# Attribution Algorithm — Data Model & Exact Specification

## 1. Purpose

For each reconciled column (e.g. "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Control"), choose 2–3 chunks from the PDF that best support the extracted `final_value`. These chunks are highlighted in the attribution viewer.

---

## 2. Data Model

### 2.1 Inputs (per column)

| Field | Source | Type | Example |
|-------|--------|------|---------|
| `column_name` | reconciled / comparison | str | `"Adverse Events - N (%) \| All-Cause Grade 3 or Higher \| Control"` |
| `final_value` | reconciled (LLM output) | str | `"Abiraterone trial: 192 (38%) of 502; Abiraterone and enzalutamide trial: 204 (45%) of 454"` |
| `contributing_methods` | reconciled | list[str] | `["pipeline", "landing_ai_baseline", "gemini_native"]` |
| `page` | pipeline row (from planner) | int or None | `1` |
| `source_type` | pipeline row (from planner) | str | `"text"`, `"table"`, or `"figure"` |
| `extraction_plan` | pipeline extraction_results | str | `"Extract the value '192 (38%) of 502' from the 'Findings' section on page 1"` |
| `evidence` | per-method (comparison row) | str | `"In the first 5 years... (192 [38%] of 502 vs 271 [54%] of 498...)"` |
| `method_values` | comparison row (per method) | dict | `{ "pipeline": "192 (38%) of 502", "gemini_native": "192 (38%) of 502", ... }` |

### 2.2 Chunks (from Landing AI parse)

Path: `new_pipeline_outputs/results/{doc_id}/chunking/landing_ai_parse_output.json`

```json
{
  "chunks": [
    {
      "id": "758030bb-634b-4354-a481-ab7527214b88",
      "type": "text",
      "grounding": {
        "page": 0,
        "box": { "left": 0.056, "top": 0.68, "right": 0.81, "bottom": 0.89 }
      },
      "markdown": "Findings Between Nov 15, 2011... 192 [38%] of 502..."
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `id` | UUID, used for highlight lookup |
| `type` | `"text"`, `"table"`, `"figure"`, `"logo"` |
| `grounding.page` | 0-based page index |
| `grounding.box` | Normalized coords (0–1) for PDF overlay |
| `markdown` | Chunk text (with optional `<::...::>` placeholders for images) |

### 2.3 Output (attributed_chunks)

```json
{
  "attributed_chunks": [
    {
      "chunk_id": "758030bb-634b-4354-a481-ab7527214b88",
      "page": 1,
      "source_type": "text",
      "snippet": "Findings Between Nov 15... 192 [38%] of 502...",
      "score": 1.0
    }
  ]
}
```

---

## 3. Matchable Tokens — Design Constraint

**Problem:** Words from values can mislead. E.g. value "Abiraterone acetate plus prednisolone" → token "Abiraterone" may match many unrelated chunks (trial name, author mentions). Free-form value words add noise.

**Rule:** Use only:
1. **Numeric parts** — numbers, decimals, n(%) — from the value
2. **Column-name tokens** — stable, semantic anchors (e.g. "Adverse", "Events", "Grade", "Control")

Do **not** use arbitrary words extracted from the value text.

---

## 4. Token Extraction Spec

### 4.1 Numeric parts (from `final_value` and `method_values`)

| Pattern | Regex | Keep? | Example |
|---------|-------|------|---------|
| Integers ≥3 digits | `[\d]{3,}` | Yes | `192`, `502`, `454` |
| 2-digit numbers | `[\d]{2}` (excl. 00) | Yes | `38`, `54` |
| Decimals | `[\d]+\.[\d]+` | Yes | `45.7`, `76.6` |
| Percentages | `\d+\s*%` | Yes, normalized | `38%`, `(45%)` → `38%`, `45%` |
| Single digits | — | **No** | `4` matches inside `454`, `502` |

Deduplicate and merge from all method values. Primary = from `final_value`; method values add optional tokens for ranking.

### 4.2 Column-name tokens (from `column_name`)

- Split on: `|`, `-`, ` `, `(`, `)`
- Keep tokens: length ≥ 3, not pure numbers, not stopwords
- Stopwords: `"the", "of", "and", "or", "to", "in", "with"`, etc.
- Example: `"Adverse Events - N (%) | All-Cause Grade 3 or Higher | Control"`  
  → `["Adverse", "Events", "All-Cause", "Grade", "Higher", "Control"]`

Use column-name tokens for **ranking/filtering**, not as hard requirements (column names can be abstract; chunk may not contain "Adverse Events" verbatim).

---

## 5. Algorithm — Exact Flow

```
INPUT: doc_id, column { column_name, final_value, page, source_type, extraction_plan, method_values, evidence }

LOAD: landing_ai_parse_output.chunks → valid chunks (text length ≥ 10)
```

### Phase 1: Numeric match (primary)

1. **Extract numeric parts** from `final_value` + `method_values`:
   - required_parts = from final_value only
   - all_numeric_parts = merge from all

2. **Filter:** Chunk is a match iff it contains **all** required numeric parts.
   - Numbers: exact substring or word-boundary match (e.g. 38 not inside 384)
   - Percentages: `38%` or `(38%)` or `38 %` normalized

3. **If any chunk matches:**
   - Rank by: (a) count of all_numeric_parts present, (b) column-name token overlap, (c) location boost (page, source_type)
   - Return top 3.

4. **If no chunk matches** → Phase 2.

### Phase 2: Planner location (page + source_type)

1. **If** `page` ∈ [1, N] and `source_type` ∈ {text, table, figure}:
   - Get chunks on that page with matching type (`get_chunk_ids_by_page_type`).

2. **Rank** those chunks by:
   - Column-name token overlap with chunk text
   - (Optional) extraction_plan quoted strings if present

3. **If any chunk on that page/type:**
   - Return top 3.

4. **If none** → Phase 3.

### Phase 3: Semantic retrieval (fallback)

1. **Embed** all chunks (cached per doc).
2. **Query:** `column_name + " " + final_value + " " + evidence_snippet` (truncated).
3. **Retrieve** top K by cosine similarity.
4. **Re-rank** by planner page/type when available.
5. **Return** top 3.

---

## 6. Current vs Proposed

| Aspect | Current | Proposed |
|--------|---------|----------|
| Value tokens | Numbers + **words from value** (≥4 chars) | **Numbers only** from value |
| Column name | Not used | **Column-name tokens** for ranking |
| Fallback order | Value match → Semantic | Value match → **Planner (page+type)** → Semantic |
| Planner extraction_plan | Not used for attribution | Optional: quoted strings for ranking |

---

## 7. Pseudo-Code (Proposed)

```python
def retrieve_chunks_for_column(doc_id, column, top_k=3):
    chunks = load_valid_chunks(doc_id)

    # Phase 1: Numeric match
    required, all_parts = extract_numeric_parts(column.final_value, column.method_values)
    if required:
        matching = [c for c in chunks if chunk_contains_all(required, c.text)]
        if matching:
            col_tokens = extract_column_tokens(column.column_name)
            ranked = rank_by(matching, all_parts, col_tokens, column.page, column.source_type)
            return ranked[:top_k]

    # Phase 2: Planner location
    if column.page and column.source_type:
        page_chunks = [c for c in chunks if on_page_and_type(c, column.page, column.source_type)]
        if page_chunks:
            col_tokens = extract_column_tokens(column.column_name)
            ranked = rank_by_column_tokens(page_chunks, col_tokens)
            return ranked[:top_k]

    # Phase 3: Semantic
    return semantic_retrieve(doc_id, column, top_k)
```

---

## 8. File References

| Logic | File |
|-------|------|
| Token extraction | `web/attribution_service.py` — `_get_matchable_parts`, `_get_matchable_parts_from_values` |
| Chunk filtering | `web/attribution_service.py` — `_chunk_contains_all_parts` |
| Planner page/type | `web/highlight_service.py` — `get_chunk_ids_by_page_type` |
| Chunk loading | `web/highlight_service.py` — `load_landing_ai_parse` |
| Enrichment loop | `web/attribution_service.py` — `enrich_reconciled_with_attribution` |

---

## 9. Extraction Plan Usage (Future)

`extraction_plan` often contains quoted values, e.g.:
> "Extract the value **'302 [68%] of 445'** from the 'Findings' section on page 1"

We can:
- Parse quoted strings with regex `'([^']+)'` or `"([^"]+)"`
- Use them as additional matchable tokens (same rules: prefer numbers, avoid free-form words)

This would be an optional enhancement; numeric + column-name tokens are the core.
