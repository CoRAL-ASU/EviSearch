# Attribution Logic — Walkthrough

## What is "Attribution"?

Attribution = **where did this value come from?** It includes:
- **evidence** — excerpt or description of the source
- **confidence** — high/medium/low
- **assumptions** — if the extractor made any
- **page** — page number (when available)
- **source_type** — table/text/figure

---

## Data Flow: Raw → Normalized → Display

```
Raw extraction output (varies by method)
    ↓
comparison_service._normalize_*_result()
    ↓
Unified shape: { value, primary_value, attribution: { evidence, confidence, assumptions }, candidates, page, source_type }
    ↓
Frontend (comparison.js): attr.evidence, attr.confidence, candidates[0].evidence
```

---

## Method 1: Gemini Native

**Source:** `baselines_file_search_results/gemini_native/{model}/{doc}/extraction_metadata.json`

**Raw shape:**
```json
{
  "Control Arm": {
    "value": "standard of care",
    "evidence": "On page 1, in the 'Methods' section, it states: 'Patients were randomly assigned...'",
    "page": "Not applicable",
    "group_name": "Control Arm"
  }
}
```

**Normalization** (`_normalize_gemini_result`):
- `attribution.evidence` ← `data.evidence`
- `attribution.confidence` ← **hardcoded "medium"** (raw has no confidence)
- `attribution.assumptions` ← `None`
- `attribution.sources` ← `[]`
- `page` ← `data.page` (often "Not applicable")
- `source_type` ← `data.plan_source_type` or "text"

**Issue:** Gemini native has no `plan_*` fields and no confidence. We always show "medium" confidence.

---

## Method 2: Landing AI Baseline

**Source:** `baseline_landing_ai_w_gemini/results/{model}/{doc}/extraction_metadata.json`

**Raw shape:** Same as Gemini native — flat `value`, `evidence`, `page: "Not applicable"`.

**Normalization:** Uses **same** `_normalize_gemini_result` as Gemini native.

**Issue:** Same as above. `page` is always "Not applicable" because the baseline uses full-document context, not chunk-level retrieval.

---

## Method 3: Pipeline (extract_landing_ai)

**Source:** `new_pipeline_outputs/results/{doc}/planning/extract_landing_ai/extraction_results.json`

**Raw shape:**
```json
{
  "column_name": "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Treatment",
  "value": "302 (68%) of 445",
  "primary_value": "302 (68%) of 445",
  "page": 1,
  "source_type": "text",
  "candidates": [
    {
      "value": "302 (68%) of 445",
      "evidence": "In the first 5 years of treatment, grade 3–5 toxic effects were higher...",
      "assumptions": null,
      "confidence": "high"
    }
  ]
}
```

**Normalization** (`_normalize_pipeline_result`):
- `attribution.evidence` ← `candidates[0].evidence` (first candidate only)
- `attribution.confidence` ← `candidates[0].confidence`
- `attribution.assumptions` ← `candidates[0].assumptions`
- `attribution.sources` ← `row.sources` (from plan, e.g. `[[1, "text"]]`)
- `page` ← `row.page` (actual page from chunk retrieval)
- `source_type` ← `row.source_type`

**Correct:** Pipeline has real page, source_type, and per-candidate evidence/confidence.

---

## Frontend: How Attribution is Used

**Column detail view** (`showColumnDetail`):
```javascript
const attr = col.attribution || {};
const candidates = col.candidates || [];
const evidence = attr.evidence || (candidates[0] && candidates[0].evidence);
const confidence = attr.confidence || (candidates[0] && candidates[0].confidence);
const assumptions = attr.assumptions || (candidates[0] && candidates[0].assumptions);
```

**Fallback chain:** `attribution` first, then `candidates[0]`. This is correct — attribution is the "primary" summary; candidates hold the raw per-candidate data.

---

## Example Walkthroughs

### Example 1: Pipeline — "Adverse Events | All-Cause Grade 3 | Treatment"

| Step | Value |
|------|-------|
| Raw `candidates[0].evidence` | "In the first 5 years of treatment, grade 3–5 toxic effects were higher when abiraterone was added... (302 [68%] of 445 vs 204 [45%] of 454...)" |
| Raw `candidates[0].confidence` | "high" |
| Raw `page` | 1 |
| Raw `source_type` | "text" |
| `attribution.evidence` | Same as candidates[0].evidence |
| `attribution.confidence` | "high" |
| Display | Evidence + "Page 1 • text" |

**Correct:** Evidence and page match the actual source.

---

### Example 2: Landing AI Baseline — "Control Arm"

| Step | Value |
|------|-------|
| Raw `evidence` | "In the 'Methods' section, under 'Randomisation and masking', it states: 'Patients were randomly assigned...'" |
| Raw `page` | "Not applicable" |
| `attribution.evidence` | Same (good) |
| `attribution.confidence` | **"medium"** (hardcoded — raw has none) |
| Display | Evidence + "Page Not applicable • text" (or nothing for page) |

**Issue:** We show "medium" even when the evidence is strong. We have no real page.

---

### Example 3: Pipeline — Multi-candidate "Add-on Treatment"

| Step | Value |
|------|-------|
| `candidates` | [ { value: "abiraterone acetate plus prednisolone", evidence: "...", confidence: "high" }, { value: "abiraterone acetate and prednisolone plus enzalutamide", ... } ] |
| `attribution.evidence` | From **candidates[0]** only |
| `attribution.confidence` | From candidates[0] |
| Display | `primary_value` = "abiraterone acetate plus prednisolone; abiraterone acetate and prednisolone plus enzalutamide" |
| Evidence shown | Only first candidate's evidence |

**Issue:** When there are multiple candidates, we only show the first candidate's evidence. The second candidate might have different evidence (e.g. different table/figure). We could show "Evidence for primary: ..." or list evidence per candidate.

---

### Example 4: Gemini Native — "Control Arm"

| Step | Value |
|------|-------|
| Raw `evidence` | "On page 1, in the 'Methods' section, it states: 'Patients were randomly assigned...'" |
| Raw `page` | "Not applicable" |
| `attribution.confidence` | **"medium"** (hardcoded) |
| Display | Evidence is good; page is "Not applicable" |

**Issue:** The evidence often mentions "page 1" in the text, but we don't parse that into `page`. We could try to extract page from evidence.

---

## Summary of Issues

| Issue | Location | Fix |
|-------|----------|-----|
| **Gemini/Landing AI: confidence always "medium"** | `_normalize_gemini_result` | Parse from evidence or add confidence to baseline output |
| **Gemini/Landing AI: page always "Not applicable"** | Raw data | Baselines don't do chunk retrieval; could parse "page X" from evidence |
| **Pipeline: multi-candidate shows only first evidence** | `_normalize_pipeline_result` | Attribution could aggregate or we show "Evidence (primary): ..." |
| **sources vs evidence** | `attribution.sources` | `sources` is `[[page, type]]` from plan; rarely used in UI. Could show "Source: Page 5, Table" |

---

## Recommended Attribution Display Logic

1. **Evidence:** Always show. Prefer `attribution.evidence`, fallback to `candidates[0].evidence`.
2. **Confidence:** Show when present. For Gemini/Landing AI, consider "—" instead of "medium" when not available.
3. **Page + source_type:** Show when `page` is a valid number. Hide when "Not applicable" or N/A.
4. **Assumptions:** Show when present (important for multi-candidate).
5. **Multi-candidate:** Consider showing "Evidence for selected value: ..." or list evidence per candidate when expanding.
