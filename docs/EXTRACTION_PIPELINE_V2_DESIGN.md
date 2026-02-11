# Clinical Trial Data Extraction Pipeline V2

## Design Document

**Version:** 2.0  
**Date:** January 2026  
**Status:** Draft  

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Approach & Limitations](#2-current-approach--limitations)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Stage 1: Extraction Map Generation](#4-stage-1-extraction-map-generation)
5. [Stage 2: Smart Chunk Selection](#5-stage-2-smart-chunk-selection)
6. [Stage 3: Targeted Extraction](#6-stage-3-targeted-extraction)
7. [Data Structures & Schemas](#7-data-structures--schemas)
8. [Implementation Plan](#8-implementation-plan)
9. [File Organization](#9-file-organization)
10. [Configuration](#10-configuration)
11. [Testing Strategy](#11-testing-strategy)
12. [Future Improvements](#12-future-improvements)

---

## 1. Problem Statement

### Goal
Extract 134 structured data fields from clinical trial publications (PDFs) with high accuracy. Fields include demographics, disease characteristics, outcomes, and safety data.

### Challenges
1. **Dense medical documents**: Clinical papers pack extensive data into tables, figures, and text
2. **Variable terminology**: Same concept expressed differently across papers
3. **Complex tables**: Multi-level headers, merged cells, stratified data requiring aggregation
4. **Context dependency**: Understanding paper structure crucial for correct extraction

---

## 2. Current Approach & Limitations

### Current Pipeline (V1)
```
PDF → Chunking → Generic Guide Generation → RAG Retrieval → LLM Extraction
```

### Limitations

| Issue | Description | Impact |
|-------|-------------|--------|
| **RAG Misses** | Retriever often selects wrong chunks | Values from wrong tables/sections |
| **Generic Guide** | One guide for all columns, not column-specific | LLM guesses where to look |
| **Context Loss** | Chunking breaks paper understanding | Can't reason across sections |
| **Table Extraction** | No special handling for structured tables | Wrong columns, missed aggregations |
| **No Source Awareness** | Treats tables, text, figures the same | Suboptimal extraction strategies |

### Performance (V1)
- Accuracy: ~60-70% on well-structured papers
- Table extraction: ~40-50% (major pain point)
- Text extraction: ~70-80%

---

## 3. Proposed Architecture

### Core Philosophy: **Simulate Human Extraction**

**Human approach:**
1. **Shallow Pass**: Skim paper, build mental map of where data lives
2. **Deep Pass**: Go to specific location, extract precise values

**Pipeline V2:**
1. **Stage 1 (Shallow)**: Generate extraction map - WHERE is each data point?
2. **Stage 2 (Select)**: Use map to select correct chunks (no LLM needed)
3. **Stage 3 (Deep)**: Extract with full context + precise chunk + specific instructions

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTRACTION PIPELINE V2                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │   PDF       │                                                        │
│  │  (bytes)    │                                                        │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              STAGE 1: EXTRACTION MAP GENERATION                  │   │
│  │                                                                  │   │
│  │  Input: Full PDF + Column Groups (2-3 per call)                 │   │
│  │  Output: Per-column extraction map with locations               │   │
│  │  LLM Calls: ~15-20 (batched)                                    │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│                   ┌────────────────────────┐                           │
│                   │   extraction_map.json   │                           │
│                   │   (cached for reuse)    │                           │
│                   └────────────┬───────────┘                           │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               STAGE 2: SMART CHUNK SELECTION                     │   │
│  │                                                                  │   │
│  │  Input: Extraction map + Chunked PDF                            │   │
│  │  Logic: Heuristic matching (table name, page, keywords)         │   │
│  │  Output: Selected chunks per group                              │   │
│  │  LLM Calls: 0 (pure heuristics)                                 │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 STAGE 3: TARGETED EXTRACTION                     │   │
│  │                                                                  │   │
│  │  Input: PDF + Selected chunks + Map instructions                │   │
│  │  Process: 8 parallel workers                                    │   │
│  │  Output: Extracted values                                       │   │
│  │  LLM Calls: ~30-40 (only for present groups)                    │   │
│  └─────────────────────────────┬───────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│                   ┌────────────────────────┐                           │
│                   │  extracted_table.csv    │                           │
│                   └────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Stage 1: Extraction Map Generation

### Purpose
Analyze the full PDF to create a detailed map of WHERE each data point lives and HOW to extract it.

### Input
- Full PDF (uploaded as bytes)
- Column group definitions (2-3 groups per LLM call)

### Process
```python
def generate_extraction_map(pdf_bytes, groups, batch_size=3):
    """
    Generate extraction map for all column groups.
    
    Args:
        pdf_bytes: PDF file as bytes
        groups: Dict of {group_name: [column_definitions]}
        batch_size: Groups per LLM call (default: 3)
    
    Returns:
        Dict: Complete extraction map
    """
    extraction_map = {}
    group_batches = batch_groups(groups, batch_size)
    
    for batch in group_batches:
        # Single LLM call with PDF + batch of groups
        batch_map = call_llm_for_map(pdf_bytes, batch)
        extraction_map.update(batch_map)
    
    return extraction_map
```

### LLM Prompt Template
```
You are analyzing a clinical trial publication to create an extraction map.

For each column group provided, identify:
1. Whether the data is PRESENT in this paper
2. The SOURCE TYPE: "table", "text", "abstract", or "figure"
3. The exact LOCATION: table name, page number, section
4. The TERMINOLOGY used in this paper for each concept
5. Any EXTRACTION NOTES (aggregation needed, format parsing, etc.)

=== COLUMN GROUPS TO MAP ===
{group_definitions}

=== OUTPUT FORMAT (JSON) ===
{
  "group_name": {
    "present_in_paper": true/false,
    "columns": {
      "Column Name": {
        "present": true/false,
        "source_type": "table" | "text" | "abstract" | "figure",
        "location": {
          "page": <number>,
          "table_name": "Table X" (if applicable),
          "figure_name": "Figure X" (if applicable),
          "section": "Results" / "Methods" / etc.,
          "row_header": "exact row text" (for tables),
          "col_header": "exact column text" (for tables)
        },
        "search_hints": ["keyword1", "keyword2"],
        "paper_terminology": "how this paper refers to this concept",
        "extraction_note": "specific instructions for this value"
      }
    }
  }
}

Be EXHAUSTIVE - check tables, text, figures, and abstract.
If data is NOT present, set present: false and explain why.
```

### Output Schema
```json
{
  "Demographics - Treatment": {
    "present_in_paper": true,
    "columns": {
      "Median Age - Treatment": {
        "present": true,
        "source_type": "table",
        "location": {
          "page": 4,
          "table_name": "Table 1",
          "section": "Baseline Characteristics",
          "row_header": "Age, years - Median (IQR)",
          "col_header": "Darolutamide + ADT + docetaxel (n=651)"
        },
        "search_hints": ["Table 1", "Age", "Baseline"],
        "paper_terminology": "Age, years - Median (IQR)",
        "extraction_note": "Extract number before parenthesis. Format: integer"
      },
      "Region - North America - Treatment": {
        "present": false,
        "source_type": null,
        "location": null,
        "search_hints": [],
        "paper_terminology": null,
        "extraction_note": "Paper only reports 'Europe' and 'Rest of World'. No North America breakdown."
      }
    }
  },
  "Primary Outcomes": {
    "present_in_paper": true,
    "columns": {
      "OS HR - Overall": {
        "present": true,
        "source_type": "text",
        "location": {
          "page": 6,
          "section": "Results",
          "table_name": null
        },
        "search_hints": ["hazard ratio", "overall survival", "OS", "HR"],
        "paper_terminology": "hazard ratio for death",
        "extraction_note": "Look for 'HR' followed by value and 95% CI in parentheses"
      }
    }
  }
}
```

### Caching
- Cache as `extraction_map.json` in output directory
- Regenerate only if PDF changes (check file hash)

---

## 5. Stage 2: Smart Chunk Selection

### Purpose
Use the extraction map to select the correct chunk(s) for each group. **No LLM needed** - pure heuristics.

### Input
- Extraction map from Stage 1
- Chunked PDF (list of chunks with content and page metadata)

### Selection Strategies by Source Type

#### 5.1 Table Extraction
```python
def select_chunk_for_table(chunks, location):
    """
    Find chunk containing the specified table.
    
    Priority:
    1. Exact table name match in content ("Table 1")
    2. Page number match
    3. Keyword fallback
    """
    table_name = location.get("table_name")  # e.g., "Table 1"
    page = location.get("page")
    
    # Strategy 1: Table name in content
    if table_name:
        for chunk in chunks:
            if table_name in chunk["content"]:
                return chunk
    
    # Strategy 2: Page match
    if page:
        for chunk in chunks:
            if page in chunk.get("pages", []):
                return chunk
    
    return None
```

#### 5.2 Text Extraction
```python
def select_chunk_for_text(chunks, location, search_hints):
    """
    Find chunk containing text-based data.
    
    Priority:
    1. Section header match
    2. Page number match
    3. Keyword search (BM25)
    """
    section = location.get("section")  # e.g., "Results"
    page = location.get("page")
    
    # Strategy 1: Section in content
    if section:
        for chunk in chunks:
            if section.lower() in chunk["content"].lower():
                return chunk
    
    # Strategy 2: Page match
    if page:
        for chunk in chunks:
            if page in chunk.get("pages", []):
                return chunk
    
    # Strategy 3: BM25 keyword search
    if search_hints:
        return bm25_search(chunks, search_hints, top_n=1)[0]
    
    return None
```

#### 5.3 Abstract Extraction
```python
def select_chunk_for_abstract(chunks):
    """Abstract is typically in first chunk or page 1."""
    # Strategy 1: Page 1
    for chunk in chunks:
        if 1 in chunk.get("pages", []):
            return chunk
    
    # Fallback: First chunk
    return chunks[0] if chunks else None
```

#### 5.4 Figure Extraction
```python
def select_chunk_for_figure(chunks, location, search_hints):
    """
    Find chunk containing figure reference.
    Note: We extract from text near figures, not the figure itself.
    """
    figure_name = location.get("figure_name")  # e.g., "Figure 2"
    page = location.get("page")
    
    # Strategy 1: Figure name in content
    if figure_name:
        for chunk in chunks:
            if figure_name in chunk["content"]:
                return chunk
    
    # Strategy 2: Page match
    if page:
        for chunk in chunks:
            if page in chunk.get("pages", []):
                return chunk
    
    # Fallback: Keyword search
    if search_hints:
        return bm25_search(chunks, search_hints, top_n=1)[0]
    
    return None
```

### Master Selection Function
```python
def select_chunks_for_group(extraction_map_entry, chunks):
    """
    Select relevant chunks for a column group.
    
    Returns:
        List of unique chunks needed for this group
    """
    if not extraction_map_entry.get("present_in_paper"):
        return []  # Skip groups not in paper
    
    selected_chunks = []
    
    for col_name, col_info in extraction_map_entry["columns"].items():
        if not col_info.get("present"):
            continue
        
        source_type = col_info.get("source_type")
        location = col_info.get("location", {})
        hints = col_info.get("search_hints", [])
        
        chunk = None
        
        if source_type == "table":
            chunk = select_chunk_for_table(chunks, location)
        elif source_type == "text":
            chunk = select_chunk_for_text(chunks, location, hints)
        elif source_type == "abstract":
            chunk = select_chunk_for_abstract(chunks)
        elif source_type == "figure":
            chunk = select_chunk_for_figure(chunks, location, hints)
        
        if chunk and chunk not in selected_chunks:
            selected_chunks.append(chunk)
    
    return selected_chunks
```

---

## 6. Stage 3: Targeted Extraction

### Purpose
Extract values with full context (PDF) + precise text (chunks) + specific instructions (map).

### Input per Group
1. **PDF bytes**: Full paper for global context
2. **Selected chunks**: Markdown text containing target data
3. **Map instructions**: Column-specific extraction details

### Inference Structure
- **Fresh instance per group** (not multi-turn chat)
- **8 parallel workers** for throughput
- **Only process groups with `present_in_paper: true`**

### LLM Call Structure
```python
def extract_group(pdf_bytes, chunks, map_entry, group_columns):
    """
    Extract all columns for a single group.
    
    Args:
        pdf_bytes: Full PDF for context
        chunks: Selected markdown chunks
        map_entry: Extraction map for this group
        group_columns: Column definitions
    
    Returns:
        Dict of extracted values
    """
    # Build extraction prompt
    prompt = build_extraction_prompt(map_entry, group_columns, chunks)
    
    # Create PDF part
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
    
    # Call LLM with PDF + prompt
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[pdf_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4000,
            system_instruction=EXTRACTION_SYSTEM_PROMPT
        )
    )
    
    return parse_extraction_response(response.text)
```

### Extraction Prompt Template
```
You are extracting clinical trial data. You have:
1. The FULL PDF for understanding context and terminology
2. PRECISE TEXT CHUNKS containing the target data
3. SPECIFIC INSTRUCTIONS for each column

IMPORTANT: Extract values from the TEXT CHUNKS (more accurate than PDF parsing).
Use the PDF only for understanding context.

=== GROUP: {group_name} ===

=== TERMINOLOGY MAPPING ===
- Treatment arm in this paper: "{treatment_term}"
- Control arm in this paper: "{control_term}"

=== TEXT CHUNKS (extract from here) ===
--- Chunk from {chunk_source} ---
{chunk_content}
---

=== COLUMNS TO EXTRACT ===
{for each column in group}
{idx}. {column_name}
   - Source: {source_type} ({location_detail})
   - Look for: {paper_terminology}
   - Format: {extraction_note}
   - If not found: return null
{/for}

=== OUTPUT FORMAT ===
Return valid JSON:
{
  "Column Name": {
    "value": <extracted value or null>,
    "evidence": "<exact text snippet used>",
    "reasoning": "<brief explanation>"
  }
}
```

### System Prompt
```
You are an expert clinical data extractor. 

Rules:
1. Extract values ONLY from the provided text chunks
2. Use the PDF for context and terminology understanding
3. For tables: Match exact row and column headers
4. For text: Find the specific sentence containing the value
5. Return null if value is not explicitly present
6. Never infer or calculate unless explicitly instructed
7. Include evidence (exact quote) for every extracted value
```

### Parallel Execution
```python
def run_extraction_stage(pdf_bytes, chunks, extraction_map, groups):
    """
    Run extraction for all groups in parallel.
    """
    results = {}
    
    # Filter to only present groups
    present_groups = {
        name: entry for name, entry in extraction_map.items()
        if entry.get("present_in_paper")
    }
    
    def process_group(group_name):
        map_entry = extraction_map[group_name]
        group_columns = groups[group_name]
        selected_chunks = select_chunks_for_group(map_entry, chunks)
        
        if not selected_chunks:
            return group_name, {col["Column Name"]: null_result() for col in group_columns}
        
        extracted = extract_group(pdf_bytes, selected_chunks, map_entry, group_columns)
        return group_name, extracted
    
    # Parallel execution with 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_group, name): name for name in present_groups}
        
        for future in as_completed(futures):
            group_name, group_results = future.result()
            results.update(group_results)
    
    return results
```

---

## 7. Data Structures & Schemas

### 7.1 Chunk Schema (from chunking stage)
```json
{
  "chunk_id": 3,
  "content": "| Column1 | Column2 |\n|---------|---------|...",
  "pages": [4, 5],
  "type": "table" | "text" | "mixed",
  "metadata": {
    "source_pdf": "paper.pdf",
    "chunk_index": 3
  }
}
```

### 7.2 Extraction Map Schema
```json
{
  "<group_name>": {
    "present_in_paper": boolean,
    "columns": {
      "<column_name>": {
        "present": boolean,
        "source_type": "table" | "text" | "abstract" | "figure" | null,
        "location": {
          "page": number | null,
          "table_name": string | null,
          "figure_name": string | null,
          "section": string | null,
          "row_header": string | null,
          "col_header": string | null
        } | null,
        "search_hints": [string],
        "paper_terminology": string | null,
        "extraction_note": string | null
      }
    }
  }
}
```

### 7.3 Extraction Result Schema
```json
{
  "<column_name>": {
    "value": any | null,
    "evidence": string | null,
    "reasoning": string,
    "source_chunk": number,
    "source_type": "table" | "text" | "abstract" | "figure"
  }
}
```

### 7.4 Final Output Schema
```csv
Column1,Column2,Column3,...
value1,value2,value3,...
```

Plus metadata JSON:
```json
{
  "<column_name>": {
    "value": any,
    "evidence": string,
    "reasoning": string,
    "source_chunk": number,
    "source_type": string,
    "extraction_map_entry": {...}
  }
}
```

---

## 8. Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `src/extraction_v2/` directory
- [ ] Implement `map_generator.py` - Stage 1
- [ ] Implement `chunk_selector.py` - Stage 2
- [ ] Implement `extractor.py` - Stage 3
- [ ] Implement `pipeline.py` - Orchestration

### Phase 2: Integration
- [ ] Add new pipeline option to `main.py`
- [ ] Update `config.py` with V2 settings
- [ ] Implement caching for extraction map

### Phase 3: Testing
- [ ] Test with 1 paper, 5 groups (prototype)
- [ ] Compare accuracy with V1 pipeline
- [ ] Full test with all 46 groups

### Phase 4: Optimization
- [ ] Tune batch sizes for map generation
- [ ] Optimize chunk selection heuristics
- [ ] Add error handling and retries

---

## 9. File Organization

```
src/
├── extraction_v2/           # NEW - V2 Pipeline
│   ├── __init__.py
│   ├── map_generator.py     # Stage 1: Generate extraction map
│   ├── chunk_selector.py    # Stage 2: Heuristic chunk selection
│   ├── extractor.py         # Stage 3: Targeted extraction
│   ├── pipeline.py          # Orchestrate all stages
│   ├── prompts.py           # Prompt templates
│   └── schemas.py           # Data structure definitions
│
├── fill_table/              # OLD - V1 Pipeline (keep for comparison)
│   └── fill_table.py
│
├── chunking/                # Unchanged
│   └── chunking.py
│
├── LLMProvider/             # Unchanged
│   └── provider.py
│
├── config/
│   └── config.py            # Add V2 configs
│
└── main/
    └── main.py              # Add V2 pipeline option
```

---

## 10. Configuration

### New Config Options (`config.py`)
```python
# ============== EXTRACTION V2 CONFIGS ==============

# Pipeline version: "v1" (RAG-based) or "v2" (map-based)
EXTRACTION_PIPELINE_VERSION = "v2"

# Stage 1: Map Generation
MAP_GENERATION_BATCH_SIZE = 3  # Groups per LLM call
MAP_GENERATION_MODEL = "gemini-2.5-flash"
MAP_CACHE_ENABLED = True

# Stage 2: Chunk Selection
CHUNK_SELECTION_FALLBACK_TOP_N = 2  # BM25 fallback results

# Stage 3: Extraction
EXTRACTION_WORKERS = 8  # Parallel workers
EXTRACTION_MODEL = "gemini-2.5-flash"
EXTRACTION_MAX_RETRIES = 2
```

---

## 11. Testing Strategy

### Unit Tests
- `test_map_generator.py`: Test map generation with mock PDF
- `test_chunk_selector.py`: Test each selection strategy
- `test_extractor.py`: Test extraction with mock data

### Integration Tests
- Single paper, single group
- Single paper, all groups
- Compare V1 vs V2 accuracy

### Evaluation Metrics
- **Accuracy**: % of correctly extracted values
- **Coverage**: % of columns with non-null values
- **Precision by source type**: Table vs text vs figure accuracy
- **Cost**: Total LLM tokens used
- **Time**: End-to-end pipeline duration

### Test Papers
1. `NCT02799602_Hussain_ARASENS_JCO'23.pdf` - Primary test
2. Additional papers from dataset for generalization

---

## 12. Future Improvements

### Short-term
- [ ] Vision-based table extraction for complex tables
- [ ] Confidence scores for extracted values
- [ ] Self-correction loop for low-confidence extractions

### Medium-term
- [ ] Learning from corrections (fine-tuning)
- [ ] Cross-paper consistency checks
- [ ] Automated schema validation

### Long-term
- [ ] Multi-paper extraction (batch processing)
- [ ] Real-time extraction feedback
- [ ] Integration with downstream analysis tools

---

## Appendix A: Example Walkthrough

### Paper: ARASENS Trial (NCT02799602)

**Stage 1 Output (partial):**
```json
{
  "Demographics - Treatment": {
    "present_in_paper": true,
    "columns": {
      "Median Age - Treatment": {
        "present": true,
        "source_type": "table",
        "location": {
          "page": 4,
          "table_name": "Table 1",
          "row_header": "Age, years - Median (IQR)",
          "col_header": "Darolutamide (N=651)"
        },
        "extraction_note": "Extract number before parenthesis"
      }
    }
  }
}
```

**Stage 2 Output:**
```
Group "Demographics - Treatment" → Chunk 3 (contains Table 1, pages 4-5)
```

**Stage 3 Output:**
```json
{
  "Median Age - Treatment": {
    "value": "67",
    "evidence": "Age, years - Median (IQR) | 67 (60-72)",
    "reasoning": "Extracted median value from Table 1, treatment column"
  }
}
```

---

## Appendix B: Error Handling

### Map Generation Errors
- **Empty response**: Retry with smaller batch
- **Malformed JSON**: Parse what's possible, log errors
- **Timeout**: Increase timeout, retry

### Chunk Selection Errors
- **No match found**: Fall back to BM25, log warning
- **Multiple matches**: Select first, log for review

### Extraction Errors
- **LLM error**: Retry up to MAX_RETRIES
- **Parse error**: Return null with error note
- **Rate limit**: Exponential backoff

---

*End of Design Document*



