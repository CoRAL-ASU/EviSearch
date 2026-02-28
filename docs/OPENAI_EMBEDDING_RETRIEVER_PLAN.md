# OpenAI text-embedding-3-large as Retriever — Plan

## 1. Getting the Model via API

### Prerequisites
- **OpenAI API key**: Set `OPENAI_API_KEY` in `.env` (already used for chunking)
- **openai package**: Already in `requirements.txt`

### API Usage
```python
from openai import OpenAI
client = OpenAI()
embedding = client.embeddings.create(
    input="Your text",
    model="text-embedding-3-large"
).data[0].embedding
```

### Model Specs
- **Dimensions**: 3072 (default)
- **Max input**: 8192 tokens per input, 300k tokens per request
- **Pricing**: ~$0.13 per 1M input tokens
- **Batch**: Pass list of strings for efficiency

### Optional Parameters
- `dimensions`: Can reduce to 256, 1024, etc. for smaller storage
- `encoding_format`: `"float"` (default) or `"base64"`

---

## 2. Current Retrieval in Codebase

| Location | Method | Uses Embeddings? |
|----------|--------|------------------|
| `web/verification_agent.py` `_search_chunks_enhanced` | Token + fuzzy match (SequenceMatcher) | No |
| `web/attribution_service.py` `retrieve_chunks_for_evidence` | Attribution → numeric match → planner | No |
| `src/chunking/utils_chunking.py` | `embedding_model` loaded (all-MiniLM-L6-v2) | **Never used** |
| `src/config/config.py` | `RETRIEVAL_STRATEGY = "hybrid"` | **Config only, not wired** |

---

## 3. Integration Plan

### Option A: New Semantic Retriever Module
Create `src/retrieval/openai_embedding_retriever.py`:
- `embed_chunks(doc_id)` — embed Landing AI chunks via API, cache to disk
- `search_chunks(doc_id, query, top_k)` — embed query, cosine similarity, return top chunks

### Option B: Extend Verification Agent Search
Add semantic path to `_search_chunks_enhanced`:
- If `EMBEDDING_PROVIDER == "openai"` and embeddings cached → use cosine similarity
- Fallback to current token-based search

### Option C: Standalone Script for Testing
Create `experiment-scripts/test_openai_embedding_retrieval.py`:
- Load doc chunks, embed via OpenAI, run a few queries, print top matches
- Validates API + retrieval quality before full integration

---

## 4. Recommended Order

1. **Get API access** — Ensure `OPENAI_API_KEY` works
2. **Standalone script** — Test embedding + retrieval on one doc
3. **Retriever module** — Reusable `openai_embedding_retriever.py`
4. **Wire into verification agent** — Add `--use-semantic` or config flag

---

## 5. Caching Strategy

- **Path**: `new_pipeline_outputs/chunk_embeddings/<doc_id>_text-embedding-3-large.npz` or `.json`
- **Format**: `{chunk_id: [float, ...]}` or numpy array
- **Invalidation**: Recompute if `landing_ai_parse_output.json` changes (mtime)
