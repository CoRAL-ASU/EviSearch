# Explainability Service & Document Dashboard — Plan

## Goal

Replace raw-number displays with a **document mini-dashboard** where:
- **Center**: Core reasoning (evidence, how we got the value, where we looked)
- **Explainability service**: Chooses what to show and how to present it
- **No pointless stats** — focus on interpretable, clinician-relevant content

---

## 1. Explainability Service (Backend)

### Role

Takes comparison data and returns a **curated dashboard payload** that answers:
- What are the most important findings?
- Where did each value come from? (evidence, page, source)
- Where do methods disagree? (needs review)
- What was not found and why? (reasoning for absence)

### Input

- `load_comparison_data(pdf_stem)` output (comparison rows, method results)

### Output Structure

```json
{
  "doc_id": "NCT02799602_Hussain_ARASENS_JCO'23",
  "methods_available": ["gemini_native", "pipeline", ...],

  "highlights": [
    {
      "type": "disagreement",
      "column_name": "Median PFS (mo) | High volume | Treatment",
      "group_name": "Median PFS (mo)",
      "summary": "Gemini and Pipeline report different values",
      "values_by_method": {"gemini_native": "14.9", "pipeline": "15.2"},
      "suggest_review": true
    },
    {
      "type": "multi_candidate",
      "column_name": "Adverse Events - N (%) | All-Cause Grade 3 or Higher",
      "summary": "Multiple plausible interpretations (all-cause vs treatment-related)",
      "candidates": [...]
    },
    {
      "type": "key_finding",
      "column_name": "Treatment Arm 1 Regimen",
      "value": "Darolutamide + docetaxel + ADT",
      "evidence_snippet": "Figure 1 on page 3 explicitly labels...",
      "confidence": "high"
    }
  ],

  "core_reasoning": [
    {
      "column_name": "Treatment Arm 1 Regimen",
      "group_name": "Treatment Arm",
      "primary_value": "Darolutamide + docetaxel + ADT",
      "reasoning": {
        "evidence": "On page 1, in the 'METHODS' section, it states: 'Patients... were randomly assigned to darolutamide or placebo plus androgen-deprivation therapy and docetaxel.' Figure 1 on page 3 explicitly labels the experimental arm as 'Darolutamide + docetaxel + ADT'.",
        "source": "Table/Figure on page 3",
        "how_we_got_it": "Identified from METHODS text and CONSORT diagram labels",
        "confidence": "high",
        "assumptions": null
      },
      "by_method": {
        "gemini_native": {"value": "...", "evidence": "..."},
        "pipeline": {"value": "...", "evidence": "..."}
      }
    },
    {
      "column_name": "Adverse Events - N (%) | Treatment-related Grade 3 or Higher | Treatment",
      "group_name": "Adverse Events - N (%)",
      "primary_value": "Not reported",
      "reasoning": {
        "evidence": "Table 3 lists 'Worst grade' for 'Any AE' (all-cause), but does not specify 'treatment-related' for these grades.",
        "where_we_looked": "Table 3 (page 9), Abstract",
        "why_not_found": "Table reports treatment-emergent (all-cause) AEs only; no treatment-related breakdown provided",
        "confidence": "high",
        "assumptions": null
      },
      "by_method": {...}
    }
  ],

  "not_found_with_reasoning": [
    {
      "column_name": "Median Follow-Up Duration (mo)",
      "where_we_looked": "Document reports median survival and treatment duration, but not follow-up",
      "why": "No explicit 'median follow-up' stated for the study population"
    }
  ],

  "agreement_summary": [
    {
      "column_name": "Total Participants - N",
      "agreed_value": "1305",
      "methods_agreeing": ["gemini_native", "pipeline", "landing_ai_baseline"],
      "evidence": "CONSORT flow diagram, page 3"
    }
  ]
}
```

### Selection Logic (What Goes Into core_reasoning)

1. **Priority columns** (always include if present):
   - Treatment Arm 1 Regimen, Control Arm
   - Total Participants - N, Treatment Arm - N, Control Arm - N
   - Primary Endpoint(s)
   - Median OS, Median PFS (or equivalent)
   - Key AE columns

2. **Disagreements** (methods give different values):
   - Always include in highlights + core_reasoning

3. **Multi-candidate** (ambiguity):
   - Include in highlights + core_reasoning

4. **Not found with good reasoning**:
   - Include in `not_found_with_reasoning` (evidence explains why)

5. **Agreement** (all methods same):
   - Include a few in `agreement_summary` for trust signal

6. **Limit**:
   - `core_reasoning`: ~15–25 items (most important + disagreements + multi-candidate)
   - `highlights`: ~5–10
   - `not_found_with_reasoning`: ~5–10 (best reasoning)
   - `agreement_summary`: ~3–5

### Implementation Notes

- Add `get_document_dashboard(pdf_stem)` in `explainability_service.py` (or extend `comparison_service.py`)
- Reuse `load_comparison_data()` — no new data loading
- Pure Python logic: filter, rank, format

---

## 2. Document Mini-Dashboard (Frontend)

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  [← Back]  NCT02799602_Hussain_ARASENS_JCO'23                   │
├─────────────────────────────────────────────────────────────────┤
│  Highlights (sidebar or top)                                     │
│  • 2 disagreements → review                                     │
│  • 1 multi-candidate column                                     │
│  • 3 key findings                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│              CORE REASONING (center, main content)               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Treatment Arm 1 Regimen                                  │   │
│  │ Darolutamide + docetaxel + ADT                            │   │
│  │                                                           │   │
│  │ Evidence: On page 1, METHODS states... Figure 1 on       │   │
│  │ page 3 labels the arm as 'Darolutamide + docetaxel + ADT' │   │
│  │                                                           │   │
│  │ Source: Page 3, Figure 1  •  Confidence: high             │   │
│  │ [Gemini ✓] [Pipeline ✓] [Landing AI ✓]                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Adverse Events | Treatment-related Grade 3+ | Treatment   │   │
│  │ Not reported                                               │   │
│  │                                                           │   │
│  │ Where we looked: Table 3 (page 9), Abstract               │   │
│  │ Why not found: Table lists all-cause AEs only; no          │   │
│  │ treatment-related breakdown provided                       │   │
│  │                                                           │   │
│  │ [Gemini] [Pipeline] [Landing AI] — all agree: Not reported │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ... more reasoning blocks ...                                   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Not found (collapsible)                                         │
│  Agreement summary (collapsible)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Highlights bar** (top or sidebar)
   - Disagreements (with link/scroll to that block)
   - Multi-candidate
   - Key findings
   - Compact, scannable

2. **Core reasoning blocks** (center)
   - One card per `core_reasoning` item
   - Value prominently shown
   - Evidence / reasoning in readable text (not truncated)
   - Source (page, table/figure)
   - Confidence badge
   - Per-method values when relevant (e.g. disagreement)

3. **Not found with reasoning** (collapsible section)
   - Column name
   - Where we looked
   - Why not found

4. **Agreement summary** (collapsible)
   - Columns where all methods agree
   - Builds trust

### UX Principles

- **Reasoning first** — evidence and “how we got it” are primary
- **Value second** — the extracted value is shown but not the only focus
- **No raw counts** — no “78 found, 55 not found” unless it adds context
- **Expandable** — long evidence can be truncated with “Show more”

---

## 3. API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents/<doc_id>/dashboard` | Full dashboard payload (highlights, core_reasoning, not_found, agreement) |

Returns the explainability service output. Frontend consumes this for the mini-dashboard.

---

## 4. Implementation Order

1. **Explainability service** (`get_document_dashboard`)
   - Selection logic (priority columns, disagreements, multi-candidate)
   - Build `core_reasoning`, `highlights`, `not_found_with_reasoning`, `agreement_summary`

2. **API endpoint** `GET /api/documents/<doc_id>/dashboard`

3. **Frontend**
   - Replace report/comparison tabs with dashboard view
   - Highlights bar
   - Core reasoning cards (center)
   - Collapsible not-found and agreement sections

4. **Optional**: Keep comparison table as a secondary tab (“Full table”) for power users

---

## 5. Priority Column List (for selection)

From schema and clinical relevance:

- ID, Trial
- Treatment Arm(s), Treatment Arm 1 Regimen, Control Arm
- Total Participants - N, Treatment Arm - N, Control Arm - N
- Primary Endpoint(s), Secondary Endpoint(s)
- Median OS (mo), Median PFS (mo) (and variants)
- Adverse Events (key rows)
- Add-on Treatment, Class of Agent in Treatment Arm 1

Use prefix/contains matching for column names (e.g. `"Treatment Arm" in column_name`).

---

## 6. Open Questions

- How many `core_reasoning` items? (suggest 15–25)
- Should we add “confidence” as a filter (e.g. show only high-confidence)?
- Collapsible groups (e.g. by group_name) or flat list?
