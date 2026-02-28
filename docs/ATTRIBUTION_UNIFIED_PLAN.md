# Attribution-Unified Plan: Remove Verify, Merge Flows

**Goal:** Remove the Verify page. Route Extract users directly to Attribution. On Attribution, give users two choices per column: pick their own answer (Direct PDF or Search-Based) OR run reconciliation agent to get the final reconciled answer.

---

## 1. Current Flow (to be changed)

- **Extract page:** Run Agent + Search → Table with Column | Direct PDF | Search-Based → [Go to Verify →]
- **Verify page:** Load/reconcile → Table with Column | Direct PDF | Search-Based | Reconciled | Reasoning → [View Attribution]
- **Attribution page:** Select doc + column → See Direct PDF, Search-Based, Reconciled cards + PDF highlights

**Problem:** Three separate pages; user hops Verify → Attribution. Reconciliation is separate from attribution.

---

## 2. Proposed Flow

- **Extract page:** Run Agent + Search → Table with Column | Direct PDF | Search-Based → [View Attribution →] (no Verify)
- **Attribution page (unified):**
  - Select doc
  - Column list shows columns that have Agent + Search results (and optionally reconciled, if run before)
  - When user selects a column:
    - Show **Direct PDF** | **Search-Based** cards
    - Show **"Your choice"** section with two options:
      - **Option A:** "I'll pick" — radio/buttons: [Direct PDF] [Search-Based] (user selects one)
      - **Option B:** "Run reconciliation" — [Run reconciliation] button → calls reconciliation agent for this column (or batch) → shows Reconciled card + reasoning
    - PDF highlights: from attributed chunks (when reconciled) or from selected method’s attribution
  - Once reconciled, show the three cards as today

**UX:** One place for everything: compare Direct PDF vs Search-Based, pick one manually, or run reconciliation.

---

## 3. How Does the Attribution Page Get "Multiple" Attribution Sources?

Today, Attribution gets data from `GET /api/documents/<doc_id>/reconciled`:

1. **Data sources (in priority order):**
   - `reconciliation_agent/attribution_results.json` (reconciled + enriched)
   - `reconciliation_agent/reconciled_results.json` + `enrich_reconciled_with_attribution()`
   - `agent_extractor/attribution_results.json`
   - Agent extraction only → `_build_agent_attribution()`

2. **attributed_chunks (multiple chunk IDs for PDF highlights):**

   - **When reconciliation exists:** Uses reconciled source (single page, modality, verbatim) → `resolve_chunks_from_reconciled_source()`. For table/figure: returns *all* chunks of that type on the page. For text: returns *all* chunks matching verbatim. So one logical source can yield multiple chunks.
   - **When agent-only (no reconciliation):** Uses `retrieve_chunks_for_evidence()` with `attribution` from the agent. The agent can provide `attribution: [{page, source_type}, {page, source_type}, ...]` (multiple entries). `phase0_attribution_match()` iterates each entry, finds matching Landing AI chunks, merges and dedupes. So multiple attribution entries → multiple chunks.
   - **Result:** `attributed_chunks` = list of `{chunk_id, page, source_type, snippet, score}`. The PDF viewer draws a highlight box per chunk and chunk nav (First/Prev/Next) cycles through them.

3. **Summary:** "Multiple" attribution sources =
   - Multiple **chunks** from one source (e.g. all tables on a page)
   - Or multiple **attribution entries** from the agent (e.g. text on p1 + table on p5) → each resolved to chunks
   - Reconciliation agent currently returns one source per column; agent/Search can have several

---

## 4. Implementation Tasks

### 4.1 Remove Verify page
- Delete or redirect `/verify` → `/attribution`
- Remove Verify from navbar on all pages
- Remove `/verify` route and related APIs if only used by Verify (e.g. `api_run_reconciliation`, `api_verification_data` may still be needed for Attribution’s “Run reconciliation” action)

### 4.2 Extract page
- Change "Go to Verify →" to "View Attribution →"
- Link to `/attribution?doc=<doc_id>` (same as current verify btn target, but attribution)
- No other Extract changes

### 4.3 Attribution page – pre-reconciliation state
- **When no reconciliation exists for doc:** 
  - Load Agent + Search results only (candidate_a, candidate_b)
  - Show column list
  - On column select: show Direct PDF | Search-Based cards
  - No Reconciled card yet
  - Add "Your choice" section:
    - Option A: Radio/buttons to pick Direct PDF or Search-Based
    - Option B: [Run reconciliation] button

### 4.4 Attribution page – run reconciliation
- **Option B (Run reconciliation):** 
  - Either: Run reconciliation for *all* columns (reuse existing `run-reconciliation` API)
  - Or: Run for *current column* only (new endpoint or param?)
  - After run: reload data, show Reconciled card + reasoning
  - PDF highlights from attributed_chunks (as today)

### 4.5 Attribution page – post-reconciliation state
- When reconciled data exists: show all three cards (Direct PDF, Search-Based, Reconciled)
- User can still pick one for feedback / export
- Optional: allow "Override" – user-selected value overrides reconciled

### 4.6 API decisions
- `GET /api/documents/<doc_id>/reconciled` – should return columns even when only Agent + Search (no reconciliation yet). Today it may 404 or require reconciliation.
- `POST /api/documents/<doc_id>/run-reconciliation` – keep for "Run reconciliation" on Attribution. May need to support doc_id from URL when opened from Extract.

---

## 5. Open Questions

1. **Reconciliation scope:** Run for all columns at once, or per-column on demand?
2. **Attribution without reconciliation:** Can we show PDF highlights from Agent or Search attribution when reconciled isn’t run? (Would need attributed_chunks from agent/Search methods.)
3. **Navbar:** Remove "Verify" link everywhere; keep Extract and Attribution.
