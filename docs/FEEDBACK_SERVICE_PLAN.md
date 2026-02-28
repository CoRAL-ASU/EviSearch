# Feedback Service Plan

**Unified feedback for Chat (QA) and Tables flows**

**Page names:** "Ask a question" (QA), "Extract full table" (Extract), "Verify" (Verify). All use same source labels: Direct PDF, Search-Based, Reconciled (where applicable).

---

## 1. Goals

- **Chat (Ask a question)**: Let users rate per Q&A turn — which answers were correct (multi-select)
- **Tables (Attribution page only)**: Let users rate **per column** — which values were correct (multi-select). No feedback on Extract or Verify page.
- **Output**: Store feedback for analysis, model tuning, and quality monitoring

---

## 2. Feedback Types & Granularity

### 2.1 Chat (QA flow)

| Mode   | Feedback target           | Granularity                    |
|--------|---------------------------|--------------------------------|
| Quick  | Single answer bubble      | Per Q&A turn (1 rating/turn)   |
| Full   | 3-box card (all three)    | Per Q&A turn — multi-select which were correct |

**Context captured:**
- `doc_id`, `question`, `answer` (or `candidate_a`, `candidate_b`, `reconciled`)
- `mode` (quick | full)
- `correct_sources`: `["direct_pdf"]`, `["search_based"]`, `["reconciled"]`, or any combo (Attributed only)
- Timestamp

### 2.2 Tables (Attribution page only)

| Page        | Content shown                                                    | Feedback granularity        |
|-------------|------------------------------------------------------------------|-----------------------------|
| Attribution | Direct PDF \| Search-Based \| Reconciled cards + reasoning + PDF  | **Per column** (when selected) |

**No feedback on Extract or Verify page.** Feedback only on Attribution page when user selects a column.

**Context captured per column:**
- `doc_id`, `column_name`
- `candidate_a` (Direct PDF), `candidate_b` (Search-Based), `reconciled`
- `reasoning`
- `correct_sources`: which of Direct PDF, Search-Based, Reconciled were correct (multi-select)
- Timestamp

---

## 3. Feedback Schema

### 3.1 Core payload (all feedback)

```json
{
  "source": "chat" | "attribution",
  "doc_id": "string",
  "timestamp": "ISO8601",
  "comment": "string (max 500 chars, optional)"
}
```

- **Chat:** Uses `correct_sources` (see 3.2)
- **Attribution:** Uses `correct_sources` (see 3.3) — same multi-select pattern

### 3.2 Chat-specific fields

```json
{
  "source": "chat",
  "question": "string",
  "answer": "string",
  "mode": "quick" | "full",
  "correct_sources": ["direct_pdf", "search_based", "reconciled"],
  "candidate_a": "string | null",
  "candidate_b": "string | null",
  "reconciled": "string | null"
}
```

- **Quick mode:** `correct_sources` is `["answer"]` if user marks "Answer was correct", else `[]`
- **Attributed mode:** `correct_sources` is multi-select: any combination of `direct_pdf`, `search_based`, `reconciled`

### 3.3 Attribution-specific fields (column-level only)

```json
{
  "source": "attribution",
  "column_name": "string",
  "correct_sources": ["direct_pdf", "search_based", "reconciled"],
  "candidate_a": "string | null",
  "candidate_b": "string | null",
  "reconciled": "string | null",
  "reasoning": "string | null"
}
```

---

## 4. Storage

### Option A: JSON files per doc (simple, no DB)

- Path: `new_pipeline_outputs/results/<doc_id>/feedback/feedback.jsonl`
- Append-only, one JSON object per line (JSONL)
- Easy to grep, export, and version-control friendly

### Option B: Single aggregate file

- Path: `new_pipeline_outputs/feedback/feedback.jsonl`
- All feedback in one file, filter by `doc_id` when needed
- Simpler for cross-doc analysis

**Recommendation:** Option B — single `feedback/feedback.jsonl` to avoid proliferation of small files and simplify aggregation.

---

## 5. API Design

### 5.1 Endpoint

```
POST /api/feedback
```

**Request body (JSON) — Chat (Attributed mode):**
```json
{
  "source": "chat",
  "doc_id": "NCT00268476_Attard_STAMPEDE_Lancet'23",
  "comment": "Optional user comment",
  "chat": {
    "question": "How many deaths?",
    "mode": "full",
    "correct_sources": ["direct_pdf", "reconciled"],
    "candidate_a": "...",
    "candidate_b": "...",
    "reconciled": "..."
  }
}
```

**Request body (JSON) — Chat (Quick mode):**
```json
{
  "source": "chat",
  "doc_id": "...",
  "comment": "",
  "chat": {
    "question": "...",
    "mode": "quick",
    "correct_sources": ["answer"],
    "answer": "..."
  }
}
```

Or for table (column-level):
```json
{
  "source": "attribution",
  "doc_id": "...",
  "comment": "",
  "table": {
    "column_name": "Total Participants - N",
    "correct_sources": ["direct_pdf", "reconciled"],
    "candidate_a": "...",
    "candidate_b": "...",
    "reconciled": "...",
    "reasoning": "..."
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Feedback recorded"
}
```

### 5.2 Optional: Read feedback (for analytics)

```
GET /api/feedback?doc_id=...&source=chat&limit=100
```

Returns list of feedback entries (paginated). Not required for MVP.

---

## 6. UI Design

### 6.1 Chat (QA) — Feedback button + modal

- **Placement:** Below each assistant answer, next to "View attribution" (when present)
- **Trigger:** "Feedback" button
- **On click:** Show modal/drawer with:
  - **Attributed mode:** "Which answers were correct?" — multi-select checkboxes:
    - [ ] Direct PDF
    - [ ] Search-Based
    - [ ] Reconciled
  - **Quick mode:** "Was the answer correct?" — [ ] Yes
  - **Optional:** Free-text comment at bottom (textarea, max 500 chars)
  - [Submit] button
- **After submit:** Dismiss modal, show "Thanks!" inline, disable button

```
[Answer bubble or 3-box card]
[View attribution]  [Feedback]
                    ↓ (click)
┌─────────────────────────────────────┐
│ Which answers were correct?         │
│ ☐ Direct PDF  ☐ Search-Based       │
│ ☐ Reconciled                        │
│                                     │
│ Optional feedback:                  │
│ ┌─────────────────────────────────┐ │
│ │                                 │ │
│ └─────────────────────────────────┘ │
│              [Submit]               │
└─────────────────────────────────────┘
```

### 6.2 Attribution page — column-level feedback (only)

- **Placement:** Feedback button below reconciliation reasoning card (when column selected)
- **On click:** Modal — "Which values were correct for this column?"
  - ☐ Direct PDF
  - ☐ Search-Based
  - ☐ Reconciled
  - Optional comment
  - [Submit]
- **After submit:** Button shows "Thanks!", disabled (resets when user selects another column)

---

## 7. Implementation Phases

### Phase 1 — Backend + Chat UI
1. Add `web/feedback_service.py`: `record_feedback(payload) -> bool` (append to JSONL)
2. Add `POST /api/feedback` in `main_app.py`
3. Add "Feedback" button below each chat answer in `qa.html` (Quick + Full)
4. Modal on click: multi-select checkboxes (Direct PDF, Search-Based, Reconciled for Attributed; "Answer correct" for Quick) + optional comment
5. Submit → API call → "Thanks!", disable button

### Phase 2 — Attribution UI (column-level)
1. Add "Feedback" button to `attribution.html` in report header (when column selected)
2. Modal: "Which values were correct?" — multi-select (Direct PDF, Search-Based, Reconciled) + optional comment
3. Reuse same API, pass `source: "attribution"` with `table: { column_name, correct_sources, ... }`

### Phase 3 — Optional enhancements
1. GET `/api/feedback` for analytics dashboard
2. Export CSV/JSON for feedback analysis

---

## 8. File Structure (post-implementation)

```
web/
  feedback_service.py    # record_feedback(), load_feedback()
  main_app.py            # POST /api/feedback

new_pipeline_outputs/
  feedback/
    feedback.jsonl       # All feedback entries
```

---

## 9. Open Questions

1. **Anonymous vs identified:** No user/session ID for now. Add later if auth exists.
2. **Edit/retract:** Allow user to change rating? (e.g. within 30s) — defer
3. **Export:** CSV/JSON export for analysis — Phase 3
