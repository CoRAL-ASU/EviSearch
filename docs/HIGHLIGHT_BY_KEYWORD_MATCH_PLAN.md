# Highlight by Keyword/Phrase Match — Plan

## Problem with Current (Naive) Approach

The current implementation highlights **all chunks** on a page that match `page + source_type`. That is inaccurate because:

- A page may have many text chunks; we highlight all of them
- The actual evidence may be in only one or two chunks
- For tables, we highlight the whole table even when the value is in a specific cell

## Better Approach: Match Keywords/Phrases → Highlight Only Matching Chunks

**Idea:** Use explicit phrases from the planner and extractor to find which chunks actually contain the evidence, then highlight only those chunks.

---

## Data Sources for Matchable Phrases

| Source | What we have | Example |
|--------|--------------|---------|
| **extraction_plan** | Free text, often with quoted values | "Extract the value **'302 [68%] of 445'** from the 'Findings' section on page 1" |
| **evidence** | Excerpt from the document | "…highest toxic effects were seen…**(302 [68%] of 445** vs 204 [45%] of 454…)" |
| **value** | Extracted value (may be normalized) | "302 (68%) of 445" |

---

## Proposed Changes

### 1. Planner: Add `search_phrases` (Optional Structured Output)

**Schema addition:**
```json
{
  "column_name": "...",
  "page": 1,
  "source_type": "text",
  "extraction_plan": "...",
  "search_phrases": ["302 [68%] of 445", "grade 3-5 toxic effects", "abiraterone and enzalutamide"]
}
```

**Prompt change:** Ask the planner to output 2–5 explicit phrases that appear in the document and indicate where the value is. These can be:
- Quoted values from the plan
- Key terms (e.g. "Median OS", "ADT Plus Docetaxel")
- Row/column identifiers for tables

**Fallback:** If `search_phrases` is missing, derive from `extraction_plan` (regex for quoted strings like `'...'` or `"..."`).

---

### 2. Extractor: Add `evidence_quote` (Optional)

**Schema addition for each candidate:**
```json
{
  "value": "302 (68%) of 445",
  "evidence": "In the first 5 years... (302 [68%] of 445 vs 204 [45%] of 454...)",
  "evidence_quote": "302 [68%] of 445",
  "assumptions": null,
  "confidence": "high"
}
```

**Prompt change:** Ask the extractor to provide the **exact substring** from the source that supports the value, when possible. This is the most reliable match.

**Fallback:** If `evidence_quote` is missing, extract from `evidence` (e.g. the value or a short span around it).

---

### 3. Highlight Service: Phrase-Based Chunk Matching

**New logic in `get_highlights_for_column`:**

1. **Collect search phrases** (priority order):
   - `evidence_quote` from primary candidate (if present)
   - `search_phrases` from planner (if present)
   - Quoted strings from `extraction_plan` (regex: `'([^']+)'` or `"([^"]+)"`)
   - The extracted `value` (with normalization: `[`↔`(`, `%` variations)
   - Short substrings from `evidence` (e.g. first 50 chars containing the value)

2. **Load Landing AI chunks with grounding** (already done)

3. **For each chunk:** Check if `chunk.content` or `chunk.table_content` (or `markdown`) contains any search phrase
   - Use substring match first
   - Optionally: fuzzy match (e.g. `difflib.SequenceMatcher`) for minor variations

4. **Return grounding boxes** only for chunks that match

---

## Implementation Phases

### Phase 1: Use Existing Data (No Pipeline Changes)

- Derive phrases from `extraction_plan` (quoted strings)
- Derive phrases from `evidence` (value, or value with context)
- Derive from `value` (normalize brackets/format)
- Implement substring search over chunks
- **Files:** `web/highlight_service.py` only

### Phase 2: Planner Outputs `search_phrases`

- Update plan generator schema and prompt
- Store `search_phrases` in plans
- Highlight service prefers `search_phrases` when available
- **Files:** `src/planning/plan_generator.py`, plan schema, `highlight_service.py`

### Phase 3: Extractor Outputs `evidence_quote`

- Update extraction schema and prompt
- Store `evidence_quote` per candidate
- Highlight service uses `evidence_quote` as primary match
- **Files:** `experiment-scripts/extract_with_landing_ai.py`, extraction schema, `highlight_service.py`

---

## Phrase Extraction Heuristics (Phase 1)

```python
def extract_search_phrases(column_data: dict) -> list[str]:
    phrases = []
    value = str(column_data.get("value") or column_data.get("primary_value") or "")
    evidence = str(column_data.get("evidence") or "")
    plan = str(column_data.get("extraction_plan") or "")

    # 1. Quoted strings from plan: '302 [68%] of 445', "Table 1"
    for m in re.finditer(r"['\"]([^'\"]{4,80})['\"]", plan):
        phrases.append(m.group(1).strip())

    # 2. Value (and normalized variants: [ ] vs ( ))
    if value and len(value) >= 2:
        phrases.append(value)
        normalized = value.replace("[", "(").replace("]", ")")
        if normalized != value:
            phrases.append(normalized)

    # 3. Evidence: try to find value or a 20–60 char span containing it
    if value and value in evidence:
        phrases.append(value)
    # Or extract a substring around the value
    if value and evidence:
        idx = evidence.find(value)
        if idx >= 0:
            start = max(0, idx - 15)
            end = min(len(evidence), idx + len(value) + 15)
            phrases.append(evidence[start:end].strip())

    return list(dict.fromkeys(p))  # dedupe, preserve order
```

---

## Chunk Content to Search

Landing AI chunks have `markdown` (and sometimes structured content). We need to search the **text** that corresponds to each chunk. Options:

1. **Use `markdown`** from `landing_ai_parse_output.json` — strip HTML/anchor tags, search plain text
2. **Use pipeline chunks** — but they come from the same source; pipeline format has `content` and `table_content`

For the highlight service, we load `landing_ai_parse_output.json` which has `chunks[].markdown`. We search `markdown` for the phrases. If a chunk's markdown contains a phrase, we return its `grounding.box`.

---

## Summary

| Approach | Accuracy | Effort |
|----------|----------|--------|
| **Current (page+type)** | Low — highlights too much | Done |
| **Phase 1 (derive phrases from plan/evidence/value)** | Medium — better, no pipeline changes | Low |
| **Phase 2 (+ planner search_phrases)** | Higher | Medium |
| **Phase 3 (+ extractor evidence_quote)** | Highest | Medium |

**Recommendation:** Implement Phase 1 first. It improves accuracy without touching the planner or extractor. Add Phase 2 and 3 when you want maximum precision.
