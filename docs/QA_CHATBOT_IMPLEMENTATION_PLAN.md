# QA Chatbot Implementation Plan

**Evidence-based PDF Q&A — Ask a question**

This document details the implementation for the single-query QA chatbot flow.

---

## 1. Architecture Overview

### Central source of truth
All doc-level artifacts live under `new_pipeline_outputs/results/<doc_id>/`:

| Artifact | Path | Purpose |
|----------|------|---------|
| PDF | `results/<doc_id>/<doc_id>.pdf` | Source document |
| Parsed markdown | `results/<doc_id>/chunking/parsed_markdown.md` | Page-split text for retrieval |
| Parse output (chunks + grounding) | `results/<doc_id>/chunking/landing_ai_parse_output.json` | Highlights, chunk IDs |
| Embeddings cache | `new_pipeline_outputs/chunk_embeddings/<safe_doc_id>_*.npz` | Retriever cache |

### Retriever lookup order
1. `new_pipeline_outputs/results/<doc_id>/chunking/parsed_markdown.md`
2. (fallback) `experiment-scripts/baselines_landing_ai_new_results/<doc_id>/parsed_markdown.md`

### Flow
```
User uploads/selects doc → Parse (if needed) → Embed (if needed) → "Ready"
User asks question → Agent + Search (parallel) → Reconcile → Display (3 boxes + highlights)
```

---

## 2. Phase 1: PDF Processing Pipeline (Parse + Embed)

### 2.1 Landing AI Parse service

**New module:** `web/landing_ai_parse_service.py`

```python
def parse_pdf_for_qa(doc_id: str, pdf_path: Path, on_event: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run Landing AI Parse. Store parsed_markdown.md and landing_ai_parse_output.json
    in results/<doc_id>/chunking/.
    Stream progress via on_event: {"stage": "parsing"|"saving", "message": "..."}
    Returns {"success": bool, "error": str?, "parsed_markdown_path": Path?, "parse_output_path": Path?}
    """
```

- Use `landingai_ade.LandingAIADE().parse(document=pdf_path)` (same as `baseline_landing_ai_new`)
- Save `response.markdown` → `results/<doc_id>/chunking/parsed_markdown.md`
- Save full response (model_dump/dict with chunks, grounding) → `results/<doc_id>/chunking/landing_ai_parse_output.json`

**When to run parse:**
- If `results/<doc_id>/chunking/parsed_markdown.md` and `landing_ai_parse_output.json` both exist and PDF not newer → skip parse
- Else: run Landing AI Parse once, save both files (parse returns markdown + chunks with grounding in one call)
- Dataset docs without prior parse: same flow as uploads (run parse, store in results/)

### 2.2 Embedding service

**Modify:** `src/retrieval/openai_embedding_retriever.py`

- Add `PARSED_MARKDOWN_PATHS: List[Tuple[str, Path]]` — ordered list of (name, path) to check
- Default order: `results/<doc_id>/chunking/parsed_markdown.md`, then `baselines_landing_ai_new_results/<doc_id>/parsed_markdown.md`
- `_get_parsed_markdown_path(doc_id)` → first existing path
- `_parse_mtime(doc_id)` → mtime of that path (for cache invalidation)
- Embeddings cache key already uses doc_id; no change needed

### 2.3 Stream processing API

**New endpoint:** `POST /api/qa/prepare-document`

Request: `{"doc_id": string}`

Response: **SSE stream**

Events:
- `{"type": "stage", "stage": "parsing", "message": "Parsing PDF with Landing AI…"}`
- `{"type": "stage", "stage": "parsing_done", "message": "Parse complete"}`
- `{"type": "stage", "stage": "embedding", "message": "Building embeddings…"}`
- `{"type": "stage", "stage": "embedding_done", "message": "Embeddings ready"}`
- `{"type": "ready", "doc_id": "…"}` — document ready for questions
- `{"type": "error", "error": "…"}` — on failure

Logic:
1. Resolve PDF path (uploads → copy to results if needed, same as extract)
2. If `chunking/parsed_markdown.md` exists and PDF not newer → skip parse, go to embed
3. Run parse, save both files, emit events
4. Call `embed_chunks(doc_id, force=False)` — will use cache if mtime OK
5. Emit `ready`

---

## 3. Phase 2: Ask API (Agent + Search + Reconcile)

### 3.1 Synthetic column adapter

**New module:** `web/qa_adapter.py`

```python
QA_CONTEXT_TURNS = 5  # configurable, default 5

def build_definition_with_context(current_question: str, history: List[Dict]) -> str:
    """
    history: [{"question": str, "answer": str}, ...] (most recent last)
    Include up to last QA_CONTEXT_TURNS. No token cap.
    """
    if not history:
        return current_question
    block = "\n".join(
        f"- Q: {h['question']}\n  A: {h['answer']}"
        for h in history[-QA_CONTEXT_TURNS:]
    )
    return f"Previous Q&A:\n{block}\n\nCurrent question: {current_question}"
```

### 3.2 Single-query extraction

**Agent (Direct PDF):**
- Use `agent_extractor.extract_batch(doc_id, batch_columns, pdf_handle, provider)`
- `batch_columns = [{"column_name": "qa_1", "definition": build_definition_with_context(query, history)}]`
- Synthetic column name: `qa_<turn_id>` or `qa_<timestamp>` — internal only

**Search (Search-Based):**
- Use `run_search_agent(doc_id, batch_columns, definitions_map, ...)` with same single column
- **Retrieval query = current_question only** (no context in search query)
- **Definition passed to extraction =** `build_definition_with_context(current_question, history)` (context in extraction prompt)

**Reconciliation:**
- Use `run_reconciliation_agent` for single batch: one column with agent value + search value
- Or call `web/reconciliation_agent.run_reconciliation_agent` with single-column batch

### 3.3 Run reconciliation for one column

**No modification needed.** `web/reconciliation_agent.run_reconciliation_agent` already accepts:
- `batch_columns`: `[{column_name, definition}]` (single item for QA)
- `source_a_data`: `{col_name: {value, reasoning, attribution}}` — pass agent result in-memory
- `source_b_data`: `{col_name: {value, reasoning, attribution}}` — pass search result in-memory

For QA we do **not** write to `agent_extractor/extraction_results.json` or `search_agent/extraction_results.json`. Agent and Search run in-memory; results passed directly to reconciliation.

### 3.4 Ask endpoint

**New endpoint:** `POST /api/qa/ask`

Request:
```json
{
  "doc_id": "upload_abc123",
  "question": "What was the treatment arm?",
  "history": [{"question": "...", "answer": "..."}]
}
```

Response: **SSE stream**

Events:
- `{"type": "stage", "stage": "direct_pdf", "message": "Extracting with Direct PDF…"}`
- `{"type": "stage", "stage": "direct_pdf_done", "value": "…", "reasoning": "…"}`
- `{"type": "stage", "stage": "search", "message": "Extracting with Search-Based…"}`
- `{"type": "stage", "stage": "search_done", "value": "…", "reasoning": "…"}`
- `{"type": "stage", "stage": "reconciling", "message": "Reconciling…"}`
- `{"type": "stage", "stage": "reconciled", "value": "…", "reasoning": "…"}`
- `{"type": "done", "candidate_a": "…", "candidate_b": "…", "reconciled": "…", "reconciliation_reasoning": "…", "attribution": [...], "verbatim_quote": "…"}`
- `{"type": "error", "error": "…"}`

Logic:
1. Run Agent and Search **in parallel** (threads)
2. Stream `direct_pdf` / `search` start and done
3. Run reconciliation on single column
4. Run attribution for highlights (if `landing_ai_parse_output.json` exists)
5. Emit `done` with full payload

---

## 4. Phase 3: Frontend — QA Page UI

### 4.1 Layout

- **Top:** Doc selector (same as extract: upload or select from list) + "Process" or auto-process on select
- **Processing:** Streaming status area — "Parsing PDF…" → "Embedding…" → "Ready"
- **Chat area:** Scrollable message list
- **Input:** Text box + "Ask" button (disabled until Ready)

### 4.2 Doc selection

- Reuse `/api/documents/selectable` for dropdown
- On doc select (or upload):
  - Call `POST /api/qa/prepare-document` with `doc_id`
  - Consume SSE, update status
  - When `ready` → enable input

### 4.3 Message display

Per user message:
1. User bubble: question
2. Pending: spinner + "Direct PDF…" / "Search-Based…" / "Reconciling…" (streaming updates)
3. Answer card:
   - Three boxes: Direct PDF | Search-Based | Reconciled
   - Reconciliation reasoning
   - PDF viewer with highlights (or "See page N" if no highlights)

### 4.4 Conversation state

- **Session only:** `flask.session["qa_conversations"] = {doc_id: [{"question", "answer", "candidate_a", "candidate_b", "reconciled", ...}]}`
- On doc switch: clear `qa_conversations[previous_doc]`? No — keep per-doc. When user returns to doc, show previous convo.
- Session is per-browser; lost on close. No persistence to disk.

### 4.5 History for context

- When sending `POST /api/qa/ask`, include `history` from session for current doc_id
- `history = last 5 items from session["qa_conversations"][doc_id]` (just question + reconciled answer for context)

---

## 5. Phase 4: PDF Highlights in QA

### 5.1 Attribution resolution

- Reconciliation output includes `verbatim_quote`, `page`, `modality`
- Use `highlight_service.resolve_chunks_from_reconciled_source` (or equivalent) to get chunk IDs
- `highlight_service.get_highlights_by_chunk_ids(doc_id, chunk_ids)` → boxes
- `highlight_service` already looks at `results/<doc_id>/chunking/landing_ai_parse_output.json`

### 5.2 Fallback

- If no `landing_ai_parse_output.json`: show page numbers only ("See page 5") — no overlay
- Mostly won't happen once we run parse for all docs

---

## 6. Implementation Order

| # | Task | Files | Dependencies |
|---|------|-------|--------------|
| 1 | Retriever: multi-path parsed_markdown lookup | `openai_embedding_retriever.py` | — |
| 2 | Landing AI parse service (parse + save to results) | `web/landing_ai_parse_service.py` | — |
| 3 | Copy/symlink baseline parsed_markdown for dataset docs (optional: run parse if missing) | parse service | — |
| 4 | `POST /api/qa/prepare-document` SSE | `main_app.py` | 1, 2 |
| 5 | QA adapter (build_definition_with_context) | `web/qa_adapter.py` | — |
| 6 | Single-query Agent + Search + Reconcile (ensure reconciliation supports 1 col) | `main_app.py`, `run_reconciliation_agent` | 5 |
| 7 | `POST /api/qa/ask` SSE | `main_app.py` | 6 |
| 8 | QA page: doc select, upload, prepare-document stream | `qa.html`, JS | 4 |
| 9 | QA page: chat UI, ask stream, 3-box display | `qa.html`, JS | 7 |
| 10 | PDF viewer + highlights in QA | `qa.html`, reuse attribution PDF component | 5.1 |
| 11 | Session conversation storage | `main_app.py`, `qa.html` | 9 |

---

## 7. Configuration

```python
# web/qa_config.py or in main_app
QA_CONTEXT_TURNS = 5  # number of previous Q&A to inject
```

---

## 8. Error Handling

- Parse fails (Landing AI API error): emit `error`, show message, allow retry
- Embed fails (OpenAI key): emit `error`
- Agent/Search timeout: retry once, else error
- Doc has no parsed_markdown after prepare: do not emit ready; show "Parse failed"

---

## 9. Session Storage Schema

```python
session["qa_conversations"] = {
    "doc_id_1": [
        {
            "question": "What was the treatment arm?",
            "candidate_a": "abiraterone acetate",
            "candidate_b": "abiraterone acetate plus prednisolone",
            "reconciled": "abiraterone acetate plus prednisolone",
            "reconciliation_reasoning": "...",
            "attribution": [...],
            "verbatim_quote": "...",
        }
    ]
}
```

For `history` param: `[{"question": q, "answer": item["reconciled"]} for item in turns[-5:]]`
