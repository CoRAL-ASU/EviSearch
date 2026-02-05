# Pipeline V2 Integration Design
**Plan-Based Extraction with Category-Aware Evaluation**

---

## 📋 Executive Summary

This document outlines the integration of the new plan-based extraction pipeline into the main workflow, replacing the RAG-based approach with a more robust, interpretable, and human-reviewable system.

**Key Changes:**
- Replace RAG retrieval with explicit extraction planning
- Category-aware evaluation (exact_match, numeric_tolerance, structured_text)
- Versioned outputs for reproducibility
- Standardized LLMProvider with PDF upload support
- Menu-based CLI maintaining familiar UX

---

## 🎯 Design Goals

1. **Interpretability**: Every extraction decision is traceable
2. **Reliability**: Category-aware evaluation reduces false negatives
3. **Efficiency**: Parallel processing with proper batching
4. **Maintainability**: Single unified pipeline, not scattered scripts
5. **Extensibility**: Easy to add human-in-the-loop later
6. **Reproducibility**: Versioned outputs, logged LLM calls

---

## 📊 Current State Analysis

### Current Pipeline (main.py)
```
┌─────────────┐
│  Chunking   │ process_pdf() → pdf_chunked.json
└──────┬──────┘
       │
┌──────▼──────┐
│ Extraction  │ fill_table_with_retrieval() → extracted_table.csv
│   (RAG)     │                              → extraction_metadata.json
└──────┬──────┘
       │
┌──────▼──────┐
│ Evaluation  │ Evaluator (old) → evaluation_results.txt
│   (LLM)     │                  → evaluation_summary.json
└─────────────┘
```

**Issues:**
- ❌ RAG retrieval is opaque (no plan visibility)
- ❌ Binary evaluation (Equivalent/Not Equivalent)
- ❌ No tolerance for numeric rounding
- ❌ Scattered experiment scripts not integrated
- ❌ No versioning (overwrites)

### New Components (Experiment Scripts)

```
experiment-scripts/
├── generate_extraction_plan_v2.py     # Planning stage
├── run_extraction_with_plans_v2.py   # Execution stage
└── extraction_providers.py            # OpenAI/Gemini wrappers
```

```
src/evaluation/
└── evaluator_v2.py                    # New category-aware evaluator
```

**Strengths:**
- ✅ Explicit extraction plans (reviewable)
- ✅ Correctness/completeness scoring
- ✅ Numeric tolerance (±0.1, ±2%)
- ✅ Label-based batching
- ✅ Parallel processing

---

## 🏗️ New Architecture

### High-Level Pipeline
```
┌────────────────────────────────────────────────────────────────┐
│                      MAIN PIPELINE V2                          │
└────────────────────────────────────────────────────────────────┘
       │
       ├─► Stage 1: CHUNKING
       │   Input:  PDF
       │   Output: pdf_chunked.json
       │   Status: ✅ Already good (keep as-is)
       │
       ├─► Stage 2: PLANNING
       │   Input:  PDF + pdf_chunked.json
       │   Output: extraction_plans/{group}_plan.json (one per column group)
       │   LLM:    Gemini 2.0 Flash (multimodal)
       │   Logic:  Scan PDF, find where each column value is located
       │
       ├─► Stage 3: EXTRACTION
       │   Input:  PDF + pdf_chunked.json + extraction_plans/
       │   Output: extraction_metadata.json
       │   LLM:    GPT-4o or Gemini (configurable)
       │   Logic:  Execute plans, extract values from identified locations
       │
       └─► Stage 4: EVALUATION
           Input:  extraction_metadata.json + ground_truth.json
           Output: evaluation_results.json + summary_metrics.json
           LLM:    Gemini 2.0 Flash (eval) + Qwen (structurer)
           Logic:  Category-aware scoring (correctness, completeness)
```

### Directory Structure (Versioned)
```
results/{pdf_name}/
├── run_2026-02-04_16-30/
│   ├── chunking/
│   │   └── pdf_chunked.json
│   ├── planning/
│   │   ├── Add-on_Treatment_plan.json
│   │   ├── Treatment_Arm_N_plan.json
│   │   └── ... (133 plan files)
│   ├── extraction/
│   │   └── extraction_metadata.json
│   └── evaluation/
│       ├── evaluation_results.json
│       ├── summary_metrics.json
│       └── llm_logs/
│           ├── gemini_calls.jsonl
│           └── structurer_calls.jsonl
└── latest/ → symlink to most recent run
```

---

## 🔧 Required Changes

### 1. LLMProvider Enhancements

**Current State:**
- ✅ Supports text generation: `provider.generate(prompt)`
- ✅ Supports images: `provider.generate_with_image(prompt, image)`
- ❌ No unified PDF upload

**Required Methods:**

```python
class LLMProvider:
    def upload_pdf(self, pdf_path: str) -> PDFHandle:
        """
        Upload PDF to provider's file API.
        Returns provider-specific handle (file_id for OpenAI, Part for Gemini).
        
        Implementation:
        - OpenAI: Use Files API → return file_id
        - Gemini: Upload via File API → return file Part
        - Local/Groq: Convert PDF to images → return list of image Parts
        """
        
    def generate_with_pdf(
        self,
        prompt: str,
        pdf_handle: PDFHandle,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 8000
    ) -> LLMResponse:
        """
        Generate text response with PDF context.
        
        Implementation:
        - OpenAI: Use Responses API with file_id
        - Gemini: Pass PDF Part in contents
        - Local: Pass images as context
        """
        
    def cleanup_pdf(self, pdf_handle: PDFHandle):
        """Delete uploaded PDF file (if applicable)."""
```

**PDFHandle Structure:**
```python
@dataclass
class PDFHandle:
    provider: str  # "openai", "gemini", etc.
    file_id: Optional[str] = None  # OpenAI file ID
    file_part: Optional[Any] = None  # Gemini Part
    image_parts: Optional[List[Any]] = None  # Fallback images
```

### 2. Config Updates

**Add to `src/config/config.py`:**
```python
# ============== PIPELINE MODES ==============
EXTRACTION_MODE = "plan"  # "plan" (new) or "rag" (legacy)

# ============== PLANNING CONFIGS ==============
PLANNING_PROVIDER = "gemini"
PLANNING_MODEL = "gemini-2.0-flash-001"
PLANNING_WORKERS = 10  # Parallel groups

# ============== EXTRACTION CONFIGS ==============
EXTRACTION_PROVIDER = "openai"  # Already exists
EXTRACTION_MODEL = "gpt-4o"     # Already exists
EXTRACTION_WORKERS = 10

# ============== EVALUATION CONFIGS ==============
EVALUATION_PROVIDER = "gemini"  # Already exists
EVALUATION_MODEL = "gemini-2.0-flash-001"
EVALUATION_WORKERS = 5  # Parallel batches

# ============== OUTPUT VERSIONING ==============
VERSION_OUTPUTS = True  # Create timestamped run directories
```

### 3. Modularized Pipeline Stages

**Convert experiment scripts to importable modules:**

```
src/
├── chunking/
│   └── chunking.py                    # ✅ Already modular
├── planning/
│   └── plan_generator.py              # NEW: Move from experiment-scripts/generate_extraction_plan_v2.py
├── extraction/
│   └── plan_executor.py               # NEW: Move from experiment-scripts/run_extraction_with_plans_v2.py
└── evaluation/
    ├── evaluator.py                   # OLD: Legacy RAG evaluation
    └── evaluator_v2.py                # ✅ NEW: Category-aware evaluation
```

**Each module should expose a clean API:**

```python
# src/planning/plan_generator.py
class PlanGenerator:
    def __init__(self, provider: LLMProvider, definitions: dict):
        ...
    
    def generate_plans(
        self, 
        pdf_path: Path, 
        chunks: dict, 
        output_dir: Path,
        workers: int = 10
    ) -> dict:
        """
        Generate extraction plans for all column groups.
        Returns: {group_name: plan_data}
        """

# src/extraction/plan_executor.py
class PlanExecutor:
    def __init__(self, provider: LLMProvider, structurer: OutputStructurer):
        ...
    
    def execute_plans(
        self,
        pdf_path: Path,
        chunks: dict,
        plans: dict,
        output_path: Path,
        workers: int = 10
    ) -> dict:
        """
        Execute extraction plans and save results.
        Returns: extraction_metadata
        """

# src/evaluation/evaluator_v2.py (already done)
class EvaluatorV2:
    def run(self) -> dict:
        """Run evaluation and return results."""
```

---

## 🔄 New Main Pipeline (main_v2.py)

### Menu System
```
╔════════════════════════════════════════════════════════════╗
║          CLINICAL TRIAL EXTRACTION PIPELINE V2             ║
╚════════════════════════════════════════════════════════════╝

Select pipeline stage to run:
  1. Chunking only
  2. Planning only (requires chunks)
  3. Extraction only (requires chunks + plans)
  4. Evaluation only (requires extraction)
  5. Complete pipeline (all stages)
  6. Planning → Extraction → Evaluation (resume from chunks)
  
Enter choice (1-6):
```

### Implementation Structure

```python
# src/main/main_v2.py

from src.chunking.chunking import process_pdf
from src.planning.plan_generator import PlanGenerator
from src.extraction.plan_executor import PlanExecutor
from src.evaluation.evaluator_v2 import EvaluatorV2
from src.LLMProvider.provider import LLMProvider
from src.LLMProvider.structurer import OutputStructurer
from src.config.config import *

def create_versioned_output_dir(base_dir: Path) -> Tuple[Path, Path]:
    """Create timestamped run directory and update 'latest' symlink."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    latest_link = base_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    
    return run_dir, latest_link

def run_chunking(pdf_path: Path, output_dir: Path):
    """Stage 1: Chunk PDF into text/image/table chunks."""
    logger.info("="*60)
    logger.info("STAGE 1: CHUNKING")
    logger.info("="*60)
    
    chunk_dir = output_dir / "chunking"
    chunk_dir.mkdir(exist_ok=True)
    chunk_file = chunk_dir / "pdf_chunked.json"
    
    if chunk_file.exists():
        logger.info(f"✓ Chunks already exist: {chunk_file}")
        choice = input("Rerun chunking? (y/n): ").strip().lower()
        if choice != 'y':
            return chunk_file
    
    process_pdf(str(pdf_path), output_path=str(chunk_file))
    logger.info(f"✅ Chunks saved to {chunk_file}")
    
    return chunk_file

def run_planning(pdf_path: Path, chunk_file: Path, output_dir: Path):
    """Stage 2: Generate extraction plans for all columns."""
    logger.info("="*60)
    logger.info("STAGE 2: PLANNING")
    logger.info("="*60)
    
    plan_dir = output_dir / "planning"
    plan_dir.mkdir(exist_ok=True)
    
    # Initialize provider
    provider = LLMProvider(
        provider=PLANNING_PROVIDER,
        model=PLANNING_MODEL
    )
    
    # Load definitions
    definitions = load_definitions()
    
    # Generate plans
    planner = PlanGenerator(provider, definitions)
    plans = planner.generate_plans(
        pdf_path=pdf_path,
        chunks=json.load(open(chunk_file)),
        output_dir=plan_dir,
        workers=PLANNING_WORKERS
    )
    
    logger.info(f"✅ Generated {len(plans)} extraction plans in {plan_dir}")
    
    # FUTURE: Human review checkpoint
    # if ENABLE_HUMAN_REVIEW:
    #     print(f"\n📝 Plans saved to: {plan_dir}")
    #     print("Review and edit plans if needed, then press Enter to continue...")
    #     input()
    
    return plan_dir

def run_extraction(pdf_path: Path, chunk_file: Path, plan_dir: Path, output_dir: Path):
    """Stage 3: Execute extraction plans."""
    logger.info("="*60)
    logger.info("STAGE 3: EXTRACTION")
    logger.info("="*60)
    
    extraction_dir = output_dir / "extraction"
    extraction_dir.mkdir(exist_ok=True)
    extraction_file = extraction_dir / "extraction_metadata.json"
    
    # Initialize providers
    extraction_provider = LLMProvider(
        provider=EXTRACTION_PROVIDER,
        model=EXTRACTION_MODEL
    )
    
    structurer = OutputStructurer(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Load plans
    plans = {}
    for plan_file in plan_dir.glob("*_plan.json"):
        with open(plan_file) as f:
            plan_data = json.load(f)
            plans[plan_data['group_name']] = plan_data
    
    # Execute plans
    executor = PlanExecutor(extraction_provider, structurer)
    extraction_data = executor.execute_plans(
        pdf_path=pdf_path,
        chunks=json.load(open(chunk_file)),
        plans=plans,
        output_path=extraction_file,
        workers=EXTRACTION_WORKERS
    )
    
    logger.info(f"✅ Extraction complete: {extraction_file}")
    
    return extraction_file

def run_evaluation(extraction_file: Path, pdf_name: str, output_dir: Path):
    """Stage 4: Evaluate extraction against ground truth."""
    logger.info("="*60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("="*60)
    
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Check if ground truth exists for this document
    gt_path = PROJECT_ROOT / "dataset" / "Manual_Benchmark_GoldTable_cleaned.json"
    
    if not gt_path.exists():
        logger.warning("⚠️  Ground truth file not found, skipping evaluation")
        return None
    
    # Check if this document has ground truth
    with open(gt_path) as f:
        gt_data = json.load(f)
        doc_names = [row['Document Name']['value'] for row in gt_data['data']]
        if f"{pdf_name}.pdf" not in doc_names:
            logger.warning(f"⚠️  No ground truth for {pdf_name}, skipping evaluation")
            return None
    
    # Run evaluation
    evaluator = EvaluatorV2(
        extraction_file=str(extraction_file),
        ground_truth_file=str(gt_path),
        definitions_file=str(PROJECT_ROOT / "src/table_definitions/Definitions_with_eval_category.csv"),
        document_name=pdf_name,
        output_dir=str(eval_dir)
    )
    
    results = evaluator.run()
    logger.info(f"✅ Evaluation complete: {eval_dir}")
    
    return results

def main():
    print("\n" + "="*60)
    print("CLINICAL TRIAL EXTRACTION PIPELINE V2")
    print("="*60)
    
    # Get PDF path
    pdf_path = input("\nEnter PDF path: ").strip()
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)
    
    pdf_name = pdf_path.stem
    
    # Create versioned output directory
    base_output = PROJECT_ROOT / "results" / pdf_name
    run_dir, latest_link = create_versioned_output_dir(base_output)
    
    print(f"\n📁 Output directory: {run_dir}")
    print(f"📁 Latest link: {latest_link}")
    
    # Show menu
    print("\nSelect pipeline stage to run:")
    print("  1. Chunking only")
    print("  2. Planning only (requires chunks)")
    print("  3. Extraction only (requires chunks + plans)")
    print("  4. Evaluation only (requires extraction)")
    print("  5. Complete pipeline (all stages)")
    print("  6. Planning → Extraction → Evaluation (resume from chunks)")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    # Execute based on choice
    chunk_file = None
    plan_dir = None
    extraction_file = None
    
    if choice in ['1', '5']:
        chunk_file = run_chunking(pdf_path, run_dir)
    
    if choice in ['2', '5', '6']:
        if not chunk_file:
            # Try to find in latest/
            chunk_file = base_output / "latest" / "chunking" / "pdf_chunked.json"
            if not chunk_file.exists():
                print("❌ Chunks not found. Run chunking first.")
                sys.exit(1)
        plan_dir = run_planning(pdf_path, chunk_file, run_dir)
    
    if choice in ['3', '5', '6']:
        if not chunk_file:
            chunk_file = base_output / "latest" / "chunking" / "pdf_chunked.json"
        if not plan_dir:
            plan_dir = base_output / "latest" / "planning"
        
        if not chunk_file.exists() or not plan_dir.exists():
            print("❌ Chunks or plans not found. Run earlier stages first.")
            sys.exit(1)
        
        extraction_file = run_extraction(pdf_path, chunk_file, plan_dir, run_dir)
    
    if choice in ['4', '5', '6']:
        if not extraction_file:
            extraction_file = base_output / "latest" / "extraction" / "extraction_metadata.json"
        
        if not extraction_file.exists():
            print("❌ Extraction not found. Run extraction first.")
            sys.exit(1)
        
        run_evaluation(extraction_file, pdf_name, run_dir)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Results: {run_dir}")
    print(f"Latest:  {latest_link} → {run_dir.name}")
    print("="*60)

if __name__ == "__main__":
    main()
```

---

## 📦 Data Formats

### 1. pdf_chunked.json (unchanged)
```json
{
  "chunks": [
    {"type": "text", "content": "...", "page": "1-3"},
    {"type": "table", "content": "...", "page": "5"}
  ]
}
```

### 2. extraction_plans/{group}_plan.json
```json
{
  "group_name": "Treatment Arm - N",
  "columns": [
    {
      "column_index": 16,
      "column_name": "Treatment Arm - N",
      "found_in_pdf": true,
      "page": 5,
      "source_type": "table",
      "confidence": "high",
      "extraction_plan": "Extract from Table 1, Darolutamide column..."
    }
  ]
}
```

### 3. extraction_metadata.json
```json
{
  "Treatment Arm - N": {
    "value": 651,
    "evidence": "From Table 1: 'Darolutamide (n = 651)'",
    "chunk_id": "Treatment Arm - N::1",
    "page": "5",
    "column_index": 16,
    "group_name": "Treatment Arm - N",
    "plan_found_in_pdf": true,
    "plan_confidence": "high"
  },
  ...
}
```

### 4. evaluation_results.json
```json
{
  "document_name": "NCT00104715_Gravis_GETUG_EU'15.pdf",
  "evaluation_timestamp": "2026-02-04T16:30:00",
  "columns": {
    "Treatment Arm - N": {
      "correctness": 1.0,
      "completeness": 1.0,
      "overall": 1.0,
      "reason": "Both values are 192, exact match",
      "category": "numeric_tolerance",
      "definition": "Total number of participants in treatment arm",
      "ground_truth": "192 (ADT + Docetaxel)",
      "predicted": "192"
    },
    ...
  }
}
```

### 5. summary_metrics.json
```json
{
  "overall": {
    "avg_correctness": 0.85,
    "avg_completeness": 0.82,
    "avg_overall": 0.835,
    "total_columns": 133
  },
  "by_category": {
    "exact_match": {
      "avg_correctness": 0.93,
      "avg_completeness": 0.93,
      "avg_overall": 0.93,
      "column_count": 14
    },
    "numeric_tolerance": {
      "avg_correctness": 0.84,
      "avg_completeness": 0.81,
      "avg_overall": 0.825,
      "column_count": 107
    },
    "structured_text": {
      "avg_correctness": 0.79,
      "avg_completeness": 0.75,
      "avg_overall": 0.77,
      "column_count": 12
    }
  }
}
```

---

## 🚀 Migration Plan

### Phase 1: Module Refactoring (Week 1)
- [ ] Create `src/planning/plan_generator.py` from `generate_extraction_plan_v2.py`
- [ ] Create `src/extraction/plan_executor.py` from `run_extraction_with_plans_v2.py`
- [ ] Move provider wrappers to `src/LLMProvider/pdf_utils.py`
- [ ] Test each module independently

### Phase 2: LLMProvider Enhancement (Week 1)
- [ ] Add `upload_pdf()` method
- [ ] Add `generate_with_pdf()` method
- [ ] Add `cleanup_pdf()` method
- [ ] Update `PDFHandle` dataclass
- [ ] Test with OpenAI + Gemini

### Phase 3: Main Integration (Week 2)
- [ ] Create `src/main/main_v2.py`
- [ ] Implement menu system
- [ ] Add versioned output logic
- [ ] Add stage resumption logic
- [ ] Test end-to-end pipeline

### Phase 4: Testing & Validation (Week 2)
- [ ] Run on all 11 ground truth studies
- [ ] Compare results with experiment-scripts outputs
- [ ] Validate evaluation metrics
- [ ] Performance benchmarking

### Phase 5: Documentation & Cleanup (Week 3)
- [ ] Update README.md with V2 usage
- [ ] Archive old RAG code (don't delete, keep for reference)
- [ ] Create migration guide for existing users
- [ ] Add troubleshooting guide

---

## 🎨 Human-in-the-Loop Design (Future)

### Design Principles
- **Non-blocking**: Human review is optional, not required
- **File-based**: Edit JSON files directly (familiar format)
- **Checkpoints**: Natural pause points between stages

### Implementation Strategy

**Option 1: Interactive Mode**
```python
if ENABLE_HUMAN_REVIEW:
    print(f"\n📝 Plans saved to: {plan_dir}")
    print("Options:")
    print("  1. Continue to extraction")
    print("  2. Review and edit plans (opens editor)")
    print("  3. Abort")
    choice = input("Enter choice: ")
```

**Option 2: Manual Resume (Simpler)**
```bash
# Step 1: Run planning
python main_v2.py --pdf NCT001... --stages planning

# Step 2: User manually reviews/edits plans in results/NCT001.../latest/planning/

# Step 3: Resume extraction
python main_v2.py --pdf NCT001... --stages extraction,evaluation
```

**Recommended: Option 2** (simpler, more flexible)

### Review UI (Future Enhancement)
- Web-based plan viewer/editor
- Syntax highlighting for JSON
- Validation on save
- Diff view (before/after edits)

---

## 🔍 Error Handling & Robustness

### Failure Recovery
```python
# Each stage saves status.json
{
  "stage": "planning",
  "status": "success" | "failed" | "partial",
  "timestamp": "2026-02-04T16:30:00",
  "error": null | "error message",
  "completed_items": 85,
  "total_items": 133
}
```

**Resume Logic:**
- If status = "success": Skip or prompt to rerun
- If status = "failed": Auto-rerun from scratch
- If status = "partial": Resume from completed_items

### LLM Call Resilience
```python
# All LLM calls should:
1. Log request + response to jsonl
2. Check response.success before using
3. Retry on transient errors (3x with backoff)
4. Gracefully degrade on persistent failures
```

---

## 📈 Performance Optimization

### Current Performance (Sequential)
- Chunking: ~30s
- Planning: ~120s (133 groups × ~1s)
- Extraction: ~180s (133 groups × ~1.5s)
- Evaluation: ~60s (21 batches × ~3s)
- **Total: ~390s (~6.5 minutes)**

### Optimized Performance (Parallel)
- Chunking: ~30s (can't parallelize)
- Planning: ~12s (10 workers, 133/10 = 14 batches)
- Extraction: ~18s (10 workers, 133/10 = 14 batches)
- Evaluation: ~12s (5 workers, 21/5 = 5 batches)
- **Total: ~72s (~1.2 minutes)**

**5x speedup** 🚀

---

## 🧪 Testing Strategy

### Unit Tests
```python
# test/test_plan_generator.py
def test_plan_generator_single_group()
def test_plan_validation()
def test_plan_resumption()

# test/test_plan_executor.py
def test_execute_single_plan()
def test_missing_plan_handling()

# test/test_evaluator_v2.py (already exists)
def test_exact_match_category()
def test_numeric_tolerance()
def test_structured_text()
```

### Integration Tests
```python
# test/test_pipeline_e2e.py
def test_complete_pipeline_on_sample_pdf()
def test_stage_resumption()
def test_versioned_outputs()
```

### Regression Tests
- Run V2 on all 11 ground truth studies
- Compare metrics with experiment-scripts baseline
- Ensure evaluation scores improve (better tolerance)

---

## 📚 API Reference

### PlanGenerator
```python
class PlanGenerator:
    def __init__(self, provider: LLMProvider, definitions: dict):
        """Initialize with LLM provider and column definitions."""
    
    def generate_plan_for_group(
        self,
        group: ColumnGroup,
        pdf_path: Path,
        chunks: dict
    ) -> dict:
        """Generate extraction plan for one column group."""
    
    def generate_plans(
        self,
        pdf_path: Path,
        chunks: dict,
        output_dir: Path,
        workers: int = 10
    ) -> dict:
        """Generate plans for all groups in parallel."""
```

### PlanExecutor
```python
class PlanExecutor:
    def __init__(self, provider: LLMProvider, structurer: OutputStructurer):
        """Initialize with LLM provider and structurer."""
    
    def execute_plan_for_group(
        self,
        group_plan: dict,
        pdf_path: Path,
        chunks: dict
    ) -> dict:
        """Execute extraction for one group based on plan."""
    
    def execute_plans(
        self,
        pdf_path: Path,
        chunks: dict,
        plans: dict,
        output_path: Path,
        workers: int = 10
    ) -> dict:
        """Execute all plans in parallel and save results."""
```

### EvaluatorV2
```python
class EvaluatorV2:
    def __init__(
        self,
        extraction_file: str,
        ground_truth_file: str,
        definitions_file: str,
        document_name: str,
        output_dir: str
    ):
        """Initialize evaluator with all required files."""
    
    def evaluate_all(self, max_workers: int = 5):
        """Run evaluation on all columns with parallel processing."""
    
    def run(self) -> dict:
        """Execute full evaluation pipeline."""
```

---

## 🛠️ Implementation Checklist

### Core Implementation
- [x] Create evaluator_v2.py with category-aware scoring
- [x] Tag all columns with eval_category
- [x] Create ground truth JSON (Manual_Benchmark_GoldTable_cleaned.json)
- [ ] Refactor generate_extraction_plan_v2.py → src/planning/plan_generator.py
- [ ] Refactor run_extraction_with_plans_v2.py → src/extraction/plan_executor.py
- [ ] Add PDF upload methods to LLMProvider
- [ ] Create main_v2.py with menu system
- [ ] Add versioned output logic

### Configuration
- [ ] Update config.py with all new settings
- [ ] Validate all provider/model combinations
- [ ] Test local Qwen structurer connectivity

### Testing
- [ ] Unit tests for each module
- [ ] Integration test for full pipeline
- [ ] Regression test on 11 ground truth studies

### Documentation
- [ ] Update README.md with V2 usage
- [ ] Add troubleshooting guide
- [ ] Document evaluation categories
- [ ] Migration guide for existing users

---

## 🔮 Future Enhancements

### Immediate Next Steps
1. Implement label-based batching fix in evaluator_v2.py
2. Test evaluation on all 11 ground truth studies
3. Analyze category-specific performance

### Medium-Term
1. Web UI for plan review/editing
2. Automated error analysis (what columns fail most?)
3. Adaptive tolerance tuning based on column type
4. Confidence-weighted evaluation

### Long-Term
1. Active learning: Use evaluation feedback to improve prompts
2. Multi-annotator ground truth support
3. Real-time extraction monitoring dashboard
4. Automatic prompt optimization based on eval results

---

## 📞 Contact & Support

**Created:** 2026-02-04  
**Author:** Pipeline V2 Design Team  
**Status:** 🚧 In Development  

**Current Progress:**
- ✅ Evaluation framework complete
- ✅ Ground truth data cleaned
- ✅ Column categorization complete
- 🚧 Module refactoring in progress
- ⏳ LLMProvider enhancements pending
- ⏳ Main integration pending

---

## 🎯 Success Criteria

**The integration is successful when:**
1. ✅ All 11 ground truth studies score >80% overall accuracy
2. ✅ Pipeline completes in <2 minutes per PDF
3. ✅ All stages are resumable without re-running
4. ✅ Outputs are versioned and reproducible
5. ✅ Code is modular and testable
6. ✅ Documentation is complete

---

*End of Design Document*
