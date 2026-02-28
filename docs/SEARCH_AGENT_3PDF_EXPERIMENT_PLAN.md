# Search Agent 3-PDF Experiment Plan

## Goal
Run the search agent for all column groups on 3 PDFs (Hussain ARASENS, Fizazi PEACE1, Attard STAMPEDE), then add Search Agent to the comparison report alongside the existing Agent.

---

## 1. Target PDFs and Prerequisites

| PDF | doc_id | parsed_markdown | Status |
|-----|--------|----------------|--------|
| Hussain ARASENS | `NCT02799602_Hussain_ARASENS_JCO'23` | ✓ baselines_landing_ai_new_results | Ready |
| Fizazi PEACE1 | `NCT01957436_Fizazi_PEACE1_Lancet'22` | ✓ baselines_landing_ai_new_results | Ready |
| Attard STAMPEDE | `NCT00268476_Attard_STAMPEDE_Lancet'23` | ✓ baselines_landing_ai_new_results | Ready |

**Prerequisites:**
- `GEMINI_API_KEY` set (for search agent LLM)
- `OPENAI_API_KEY` set (for embedding retriever)
- Embeddings: run `embed_chunks` for each doc_id (or let search agent trigger on first use)

---

## 2. Run Search Agent for All Groups (3 PDFs)

### 2.1 Per-PDF Commands
```bash
# Hussain ARASENS
python experiment-scripts/run_search_agent.py "NCT02799602_Hussain_ARASENS_JCO'23"

# Fizazi PEACE1
python experiment-scripts/run_search_agent.py "NCT01957436_Fizazi_PEACE1_Lancet'22"

# Attard STAMPEDE
python experiment-scripts/run_search_agent.py "NCT00268476_Attard_STAMPEDE_Lancet'23"
```

Without `--groups`, all 39 groups are run. Batches are built by group (≤15 columns per batch). Resume is enabled by default.

### 2.2 Batch Orchestration Script (Optional)
Create `experiment-scripts/run_search_agent_3pdfs.sh` to run all 3 sequentially:
```bash
#!/bin/bash
DOCS=(
  "NCT02799602_Hussain_ARASENS_JCO'23"
  "NCT01957436_Fizazi_PEACE1_Lancet'22"
  "NCT00268476_Attard_STAMPEDE_Lancet'23"
)
for doc in "${DOCS[@]}"; do
  echo "=== $doc ==="
  python experiment-scripts/run_search_agent.py "$doc"
done
```

### 2.3 Outputs
- `new_pipeline_outputs/results/<doc_id>/search_agent/extraction_results.json`
- `new_pipeline_outputs/results/<doc_id>/search_agent/verification_logs/batch_N_conversation.json`

---

## 3. Add Search Agent to build_comparison_report

### 3.1 Changes to `experiment-analysis/build_comparison_report.py`

1. **Add model ID and loader:**
   - `SEARCH_AGENT_MODEL_ID = "search_agent"`
   - `load_search_agent_columns(pdf_id)` → load from `search_agent/extraction_results.json` (same format as `load_agent_columns`)

2. **Register in models list:**
   - Append `{"id": SEARCH_AGENT_MODEL_ID, "name": "Search Agent"}` to `models_out`

3. **Load data per PDF:**
   - In the loop over `PDF_LIST`, call `load_search_agent_columns(pdf_id)` and populate `data[pdf_id]["models"][SEARCH_AGENT_MODEL_ID]`

4. **Add stats to header band:**
   - `search_agent_stats`: `{"columns_filled": N}` (like `agent_stats`)
   - In `renderHeaderBand`, add a block for Search Agent stats (columns filled)

5. **Summary handling:**
   - Search Agent (like Agent) has no correctness/completeness from eval; set `summary[SEARCH_AGENT_MODEL_ID] = None`

### 3.2 PDF List Scope
- Option A: Keep full `PDF_LIST` (10 PDFs). Search Agent will show N/A for the 7 PDFs not run.
- Option B: Add a focused list `PDF_LIST_3` for the report, or a filter in the UI.

Recommendation: Keep full list; the 3 PDFs will have Search Agent data; others show N/A.

---

## 4. Execution Order

| Step | Action | Est. time |
|------|--------|-----------|
| 1 | Run search agent for Hussain ARASENS (all groups) | ~15–30 min |
| 2 | Run search agent for Fizazi PEACE1 (all groups) | ~15–30 min |
| 3 | Run search agent for Attard STAMPEDE (all groups) | ~15–30 min |
| 4 | Modify build_comparison_report.py (add Search Agent) | — |
| 5 | Run `python experiment-analysis/build_comparison_report.py` | <1 min |
| 6 | Open comparison_report.html (via web app or file) | — |

---

## 5. Verification

- [ ] `extraction_results.json` exists for all 3 doc_ids under `search_agent/`
- [ ] Comparison report shows "Search Agent" column
- [ ] Search Agent values appear for Hussain, Fizazi, Attard
- [ ] Search Agent shows N/A or empty for other PDFs

---

## 6. Files to Create/Modify

| File | Action |
|------|--------|
| `experiment-scripts/run_search_agent_3pdfs.sh` | Create (optional batch runner) |
| `experiment-analysis/build_comparison_report.py` | Modify (add Search Agent loader + model) |
