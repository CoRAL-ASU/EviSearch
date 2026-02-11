# src/config/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============== API KEYS ==============
# These are loaded from .env file
# For Gemini, we use Vertex AI with service account (config.json in LLMProvider/)
GROQ_API_KEY = os.getenv("LLAMA_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # <-- Add this if not present
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")


# ============== PER-TASK LLM CONFIG ==============
# Each task can use a different provider and model
# Supported providers: "gemini", "openai", "novita", "groq", "deepinfra"

# Chunking (image/table analysis - requires multimodal)
CHUNKING_PROVIDER = "openai"
CHUNKING_MODEL = "gpt-4.1"

# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS_CSV_PATH = PROJECT_ROOT / "src" / "table_definitions" / "Definitions_open_ended.csv"
GOLD_TABLE_PATH = PROJECT_ROOT / "dataset" / "GoldTable.csv"
EVALUATION_PROMPT_PATH = PROJECT_ROOT / "src" / "evaluation" / "llm_judge.txt"

# ============== CHUNKING CONFIGS ==============
TEXT_CHUNK_MIN_SIZE = 5000  # Larger chunks for 4-5 chunks per PDF
TEXT_CHUNK_OVERLAP = 0      # No overlap (deprecated, but kept for compatibility)
CHUNKING_MODE = "paragraph" # 'paragraph' (default), 'sentence' (legacy)
PATTERN_SAMPLE_PAGES = 5
TOP_MARGIN = 60
BOTTOM_MARGIN = 60
TOP_THRESHOLD_RATIO = 0.1
BOTTOM_THRESHOLD_RATIO = 0.9
HEURISTIC_MAX_LENGTH = 150

# ============== IMAGE PROCESSING ==============
PIXMAP_RESOLUTION = 6

# ============== EMBEDDINGS ==============
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============== RETRIEVAL CONFIGS ==============
# Enable/disable retrieval-based extraction (vs brute-force all chunks)
USE_RETRIEVAL = True

# Retrieval strategy: "bm25" (keyword), "semantic" (embedding), "hybrid" (both)
RETRIEVAL_STRATEGY = "hybrid"  # Options: "bm25", "semantic", "hybrid"

# Number of chunks to retrieve per group
RETRIEVAL_TOP_N = 3

# For hybrid strategy: weights for BM25 and semantic scores (should sum to ~1.0)
RETRIEVAL_BM25_WEIGHT = 0.7
RETRIEVAL_SEMANTIC_WEIGHT = 0.3

# Maximum number of chunks to combine in a single LLM call
# Set to None to combine all retrieved chunks
RETRIEVAL_MAX_COMBINED_CHUNKS = None

# ============== CONTEXT GENERATION CONFIGS (deprecated)==============
# Context generation settings for extraction guide
USE_FILE_API_CONTEXT = True  # Use Gemini File API for context (vs old 2-page text)
CONTEXT_GENERATION_PROVIDER = "gemini"  # Provider for context generation
CONTEXT_GENERATION_MODEL = "gemini-2.5-flash"  # Model for context generation
CONTEXT_MAX_RETRIES = 3  # Retry attempts for context generation


## Table Filling Configs
MAX_WORKERS = 8

# ============== PAGE CLASSIFICATION CONFIGS ==============
# LLM-based page classification for targeted chunking (requires structurer at STRUCTURER_BASE_URL)
# Set to False if vLLM is not running (chunking will proceed without table/figure page hints)
USE_LLM_PAGE_CLASSIFICATION = True  # Use Gemini to identify table/figure pages
PAGE_CLASSIFICATION_MODEL = "gemini-2.5-flash"  # Gemini model for classification
STRUCTURER_MODEL = "Qwen/Qwen3-8B"  # Local model for structuring responses
STRUCTURER_BASE_URL = "http://localhost:8001/v1"  # Local LLM endpoint for structuring

# ============== PIPELINE V2 (Plan-Based Extraction) ==============
EXTRACTION_MODE = "plan"  # "plan" (V2) or "rag" (legacy)

# Planning stage (multimodal: PDF + chunks)
PLANNING_PROVIDER = "gemini"
PLANNING_MODEL = "gemini-2.5-flash"
PLANNING_WORKERS = 10

# Extraction stage (execute plans)
EXTRACTION_PROVIDER_V2 = "gemini"
EXTRACTION_MODEL_V2 = "gemini-2.0-flash-001"
EXTRACTION_WORKERS = 10

# Evaluation stage (category-aware)
EVALUATION_PROVIDER_V2 = "gemini"
EVALUATION_MODEL_V2 = "gemini-2.0-flash-001"
EVALUATION_WORKERS = 5

# Output versioning (timestamped run directories)
VERSION_OUTPUTS = False

# Results base directory (pipeline writes to RESULTS_BASE_DIR / {pdf_name} / ...)
RESULTS_BASE_DIR = PROJECT_ROOT / "new_pipeline_outputs" / "results"

# Skip pipeline stages when output already exists (no re-run unless forced)
SKIP_STAGE_IF_EXISTS = False

# Ground truth for V2 evaluation
GOLD_TABLE_JSON_PATH = PROJECT_ROOT / "dataset" / "Manual_Benchmark_GoldTable_cleaned.json"
DEFINITIONS_EVAL_CATEGORY_PATH = PROJECT_ROOT / "src" / "table_definitions" / "Definitions_with_eval_category.csv"


