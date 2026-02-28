# Verification Agent — Tool-Calling Design

## Goal

Replace the current single-shot LLM verifier with a **tool-calling agent** that:

1. **Loads attributed chunks first** (no extra call) — fast path for columns that verify
2. For columns that don't verify: agent calls `search_chunks(query, column_names)`
3. We send the query to **Gemini** (agent_extractor-style) to get (a) a proper question/answer, (b) **places to look** (page + modality)
4. We fetch chunks from Landing AI using **page + modality** (modality-aware system)
5. Return chunks to agent; agent verifies, revises, or asks another query
6. **Limit tool calls** to prevent runaway behavior

---

## Modality-Aware Evidence (Confirmed)

We use **page + source_type (modality)** when pulling evidence:
- `attribution_matcher._chunk_on_page_and_type(c, page, source_type)` — text/table/figure
- `attribution_service.retrieve_chunks_for_evidence` with `pipeline_page`, `pipeline_source_type`
- Highlights use `get_highlights_by_chunk_ids(doc_id, chunk_ids)` (chunk IDs from attribution)

---

## Flow (Per Group)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load attributed chunks (no extra call)                       │
│    → Columns + full chunk text from attributed sources            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Agent verifies: columns with qualitative evidence → ✅ done   │
│    Take them out. Remaining: need more context / wrong            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Agent calls search_chunks(query, column_names)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. We send query to GEMINI: "Where should agent look?"           │
│    → Returns [{page, source_type}, ...] + reasoning              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Fetch chunks from Landing AI by page + modality               │
│    (modality-aware: _chunk_on_page_and_type, etc.)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Return chunks to agent. Agent verifies or asks again.         │
│    If agent calls search_chunks again: pass excluded_sources =   │
│    all (page, source_type) from chunks already seen.             │
│    When done: submit_verification (revised values for late ones)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tools (Initial Set)

### Tool 1: `get_attributed_chunks`

**Purpose**: Load the full text of chunks attributed to given columns.

**Input**:

```json
{"column_names": ["Col A", "Col B"], "doc_id": "..."}
```

**Output**: Chunks formatted as TEXT/TABLE/FIGURE blocks with page, full text.

**Implementation**: Use `get_full_chunk_texts(doc_id, chunk_ids)` from attributed_chunks.

---

### Tool 2: `search_chunks`

**Purpose**: Agent needs more context. We use Gemini to decide where to look, then fetch by page+modality.

**Input**:

```json
{
  "query": "adverse events grade 3-5 treatment arm",
  "column_names": ["Col A", "Col B"],
  "excluded_sources": [{"page": 3, "source_type": "text"}, {"page": 5, "source_type": "table"}]
}
```

- `excluded_sources`: (page, source_type) pairs the agent has already seen. Tool will NOT return chunks from these. Agent must pass this so we avoid duplicate retrieval.

**Agent instruction** (in system prompt):

> When calling search_chunks, your query should describe **what evidence you're missing** that isn't in the chunks you've seen. Be specific (e.g. "Methods section dosing schedule", "Table 2 control arm demographics") so the tool returns different pages/sources. You MUST pass `excluded_sources` with all (page, source_type) you already have — this prevents the tool from returning the same chunks.

**Implementation** (Option A + Gemini):

1. Send to **Gemini**: "Agent is verifying columns X. Query: '...'. Agent already has chunks from: [excluded_sources]. Suggest OTHER places (page, source_type) to look. Do NOT suggest excluded sources."
2. Gemini returns: `[{page: 7, source_type: "table"}, ...]` — different from excluded
3. Fetch chunks from Landing AI by page+modality (skip any that overlap excluded)
4. Return to agent: `{reasoning, attributions, matched_chunks}`

---

### Tool 3: `query_pdf` (optional / phase 2)

**Purpose**: Search PDF for specific text or pattern. Use when chunks don't contain the answer.

**Input**:

```json
{"query": "458 70.2%", "doc_id": "..."}
```

**Output**: Matching excerpts with page numbers.

**Implementation**: Text search over Landing AI chunk text (grep-like), or PDF text extraction.

---

### Tool 4: `submit_verification`

**Purpose**: Output verified/corrected results. Agent calls this when done (or at limit).

**Input**:

```json
{
  "results": {
    "Col A": {"verdict": "correct", "alternative_value": null, "evidence_quote": "..."},
    "Col B": {"verdict": "wrong", "alternative_value": "X", "evidence_quote": "..."},
    "Col C": {"verdict": "unverifiable", "alternative_value": null}
  }
}
```

**Output**: Persisted to attribution_results / reconciled_results.

---

## Limits (Prevent Runaway)


| Limit                        | Value | Purpose          |
| ---------------------------- | ----- | ---------------- |
| Max tool calls per group     | 10    | Cap iterations   |
| Max search_chunks per column | 2     | Don't over-fetch |
| Max total chunks in context  | 20    | Token budget     |
| Max turns                    | 5     | Agent loops      |


---

## Implementation Order

1. **Scaffold** `web/verification_agent.py` with tool definitions and stub implementations
2. **Tool 1** Load attributed chunks at start (no tool call; pass in initial prompt)
3. **Tool 2** `search_chunks` — Gemini call for "where to look" → fetch by page+modality from Landing AI
4. **Agent loop** — model with tool-calling (Gemini function calling)
5. **Tool 4** `submit_verification` — persist results (revised values for late-verified columns)
6. **Tool 3** `query_pdf` — phase 2 if needed

---

## Model / Provider

- Use Gemini with function calling (tools)
- System prompt: define tools, instruct agent to:
  - Verify only with qualitative evidence; call submit_verification when done
  - When calling search_chunks: write a query for **missing** evidence; pass `excluded_sources` (all page+source_type already seen) so the tool returns different chunks

---

## Chunk Embeddings

- Path: `new_pipeline_outputs/chunk_embeddings/<doc_id>_<model>_meta.json` has `chunk_ids`
- Need to locate/store actual embedding vectors for semantic search
- Fallback: BM25 over Landing AI chunk text (no embeddings required)

