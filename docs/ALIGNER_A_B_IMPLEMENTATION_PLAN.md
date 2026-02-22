# Aligner A & B — Implementation Plan

**Context:** Extended planner/verifier approach. Two structural aligners with strict attribution. No extraction yet—only mapping chunk ↔ column family.

---

## Relationship to Existing Pipeline

| Component | Current | Aligner A | Aligner B |
|-----------|---------|-----------|------------|
| **Direction** | Schema → Plan (per column) | Chunk → Families | Families → Chunks |
| **Input** | PDF + chunks + group | One chunk + family list | Document + family list |
| **Output** | `page`, `source_type`, `extraction_plan` | `families_detected` + evidence | `source` per family |
| **Attribution** | Loose (page, type) | Strict (evidence_anchor, orientation) | Strict (chunk_id, page, modality) |
| **Purpose** | Plan extraction | Chunk-first mapping | Schema-first mapping |

**Planner** = schema-first (like Aligner B). **Verifier** = checks plan against chunks. Aligners are a **structural-only** layer before planning: validate that chunk↔family mapping is consistent and reproducible.

---

## Phase 0 — Canonical Chunk Format (Prerequisite)

**Current chunk shape** (from `pdf_chunked.json`):

```json
{
  "type": "table",
  "content": "**Table 1 – Characteristics...**",
  "page": 3,
  "table_content": "##Markdown Table##\n| ... |",
  "length": 2047905,
  "source": "image"
}
```

**Gaps:**
- No explicit `chunk_id` (e.g. `Table_1`). Verifier uses index `i`.
- Table/Figure captions are in `content` but not parsed as `chunk_id`.
- Text chunks have `page: "1-3"` (range); tables/figures have `page: 3` (int).

**Canonical unit** (Phase 0 output):

```json
{
  "chunk_id": "Table_1",
  "modality": "table",
  "page": 5,
  "content": "...",
  "table_content": "..."  // if table
}
```

**Phase 0 task:** Add `chunk_id` and `modality` to chunks. Options:
1. **Post-process:** Script that reads `pdf_chunked.json`, infers `chunk_id` from captions (e.g. "Table 1" → `Table_1`), writes enriched chunks.
2. **Chunking change:** Modify `PDFChunker` / `save_chunks_to_json` to emit `chunk_id` at creation time.

For Aligner scripts: **assume Phase 0 is done.** Scripts accept chunks with `chunk_id`, `modality`, `page`. If missing, derive: `chunk_id = f"{modality}_{page}_{idx}"` as fallback.

---

## Script 1: Aligner A (Chunk → Columns)

### Role
Given one chunk, decide which column families have **explicit extractable values** in that chunk. Chunk-first, no schema bias.

### Inputs
| Input | Source | Notes |
|-------|--------|-------|
| `chunk` | Single item from `pdf_chunked.json` | Must have `content`; tables need `table_content` |
| `column_families` | From `Definitions_with_eval_category_sectioned.csv` | Group by `Label`; use unique labels as family names |
| `family_definitions` | Optional short descriptions | One line per family for token efficiency |

### Output Schema (Strict JSON)

```json
{
  "chunk_id": "Table_1",
  "page": 5,
  "modality": "table",
  "families_detected": [
    {
      "column_family": "Race - N (%)",
      "evidence_anchor": "Race, n (%)",
      "orientation": "arms_as_columns"
    },
    {
      "column_family": "PS - N (%)",
      "evidence_anchor": "ECOG performance status",
      "orientation": "arms_as_columns"
    }
  ]
}
```

**Fields:**
- `evidence_anchor`: Exact row label or phrase from chunk (verbatim). Required.
- `orientation`: `arms_as_columns` | `nested_subgroup_rows` | `survival_curve` | `single_value` | `other`

### Prompt Design (Conceptual)
- System: You are mapping chunk content to a fixed schema. Only report families with **explicit** values (numbers, structured text). No inference.
- User: Chunk content + list of column family names (and optional 1-line definitions).
- Instruction: For each family with extractable values, return `column_family`, `evidence_anchor` (exact phrase from chunk), `orientation`.
- Output: JSON only, no free text.

### Invocation
- **Per-chunk:** One LLM call per chunk. Small context.
- **Batch:** Run over all chunks of one document; aggregate into `{doc_id: [aligner_a_results]}`.

### Dependencies
- `load_definitions()` or equivalent to get `Label` → column families.
- Chunk loader (from `pdf_chunked.json`).
- LLM provider (e.g. Gemini) + optional structurer for JSON.

### Output Location
- `aligner_a/{doc_id}/chunk_{chunk_id}.json` or
- `aligner_a/{doc_id}/aligner_a_results.json` (all chunks in one file).

### Evaluation (Manual, 5–10 docs)
- Precision: % of detected families that are actually present.
- Recall: % of present families that were detected.
- No value extraction—only structural mapping.

---

## Script 2: Aligner B (Columns → Chunk)

### Role
Given full document (or structural metadata), for each column family list **candidate sources**. Schema-first.

### Inputs
| Input | Source | Notes |
|-------|--------|-------|
| `document` | PDF or structural metadata | See token-efficiency note below |
| `column_families` | Same as Aligner A | Full list |
| `chunks_summary` | Optional | Captions + page + modality only (no full content) |

### Output Schema (Strict JSON)

```json
{
  "Volume of disease - N (%)": [
    {
      "modality": "table",
      "chunk_id": "Table_1",
      "page": 5,
      "explicit": true,
      "orientation": "nested_subgroup_rows"
    }
  ],
  "Median OS (mo)": [
    {
      "modality": "figure",
      "chunk_id": "Figure_2",
      "page": 7,
      "explicit": true,
      "orientation": "survival_curve"
    }
  ]
}
```

**Fields:**
- `explicit`: `true` if values are directly visible; `false` if implied/inferred.
- `orientation`: Same enum as Aligner A.

### Prompt Design (Conceptual)
- System: For each column family, list where in this document the values appear. Require modality, page, chunk_id.
- User: Document structure (abstract, table captions, figure captions, section headings) OR full PDF.
- Instruction: For each family, return list of sources with `modality`, `chunk_id`, `page`, `explicit`, `orientation`.
- Output: JSON only.

### Token Efficiency
- **Option A (full PDF):** Single call with full PDF. Highest cost.
- **Option B (structural metadata):** Abstract + table captions + figure captions + section headings. Often sufficient for source attribution.
- **Recommendation:** Implement both; compare performance. Start with Option B.

### Invocation
- **Per-document:** One (or few) LLM calls per document.
- Output: One JSON per document.

### Dependencies
- Same definitions as Aligner A.
- PDF loader or metadata extractor (captions, headings).
- Chunk list (for `chunk_id` inventory).

### Output Location
- `aligner_b/{doc_id}/aligner_b_results.json`

### Evaluation (Manual)
- Source attribution accuracy: Does it point to the right chunk?
- Page match: Is page correct?
- Modality match: table/figure/text correct?

---

## Shared Infrastructure

### Column Families Source
- Use `Definitions_with_eval_category_sectioned.csv`.
- `Label` column = column family (e.g. "Race - N (%)", "PS - N (%)").
- Build list of unique `Label` values. Optionally include 1-line definition from first row per label.

### Chunk ID Resolution
- If chunk has `chunk_id` → use it.
- Else: Parse caption from `content` (e.g. "Table 1" → `Table_1`, "Figure 2" → `Figure_2`).
- Else: `{type}_{page}_{index}`.

### Orientation Enum
```
arms_as_columns       # Treatment vs Control as columns
nested_subgroup_rows  # Volume, Region, etc. as nested rows
survival_curve        # Kaplan-Meier in figure
single_value          # One number (e.g. NCT, Phase)
other
```

---

## Execution Order

1. **Phase 0:** Add `chunk_id` / `modality` to chunks (or derive in scripts).
2. **Script 1 (Aligner A):** Implement. Run on 3 documents. Manually check mapping quality.
3. **Script 2 (Aligner B):** Implement. Run on same 3 documents.
4. **Consistency test (Phase 3):** For each (family, chunk_id, page) from B → run A on that chunk → check if family in A's output.

---

## File Layout (Proposed)

```
experiment-scripts/
  aligner_a/
    aligner_a.py          # Main script
    prompts.py            # Prompt templates
    schema.py             # Pydantic models for output
  aligner_b/
    aligner_b.py
    prompts.py
    schema.py
  aligner_shared/
    chunk_utils.py         # Load chunks, resolve chunk_id
    definitions_utils.py   # Load column families
    orientation.py        # Enum, validation
```

Or single module:

```
src/alignment/
  aligner_a.py
  aligner_b.py
  schemas.py
  utils.py
```

---

## Open Decisions

1. **Phase 0:** Post-process vs chunking change?
2. **Aligner B input:** Full PDF vs structural metadata first?
3. **Structurer:** Use local Qwen for JSON, or rely on LLM native JSON?
4. **3 test documents:** Which ones? (e.g. GETUG, CHAARTED, STAMPEDE—subgroup-heavy, follow-up, pooled)

---

## Success Criteria (From Plan)

- > 90% precision in source attribution
- High B→A confirmation rate (consistency)
- Clear identification of problematic families (e.g. vague "Baseline characteristics", multi-table families, appendix-only)
