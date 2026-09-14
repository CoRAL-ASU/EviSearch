# src/config/config.py
"""
Selected options for this run.

Every model, option and server named here must exist in src/config/catalog.yaml; the selection is
validated when this module is imported. Each value can be overridden with the environment variable
shown next to it. List everything that is available with:  python -m src.config
"""
from pathlib import Path

from dotenv import load_dotenv

from src.config.catalog import env, load_catalog, role_overrides_from_env

# Load environment variables from .env file
load_dotenv()

CATALOG = load_catalog()

# ============== INFERENCE ==============
PRESET = env("EVISEARCH_PRESET", "local")  # local | offline | cloud

# Per-role model overrides on top of the preset (role -> model key from the catalog).
# Environment: EVISEARCH_ROLE_<ROLE>=<model key>, e.g. EVISEARCH_ROLE_JUDGE=gemini-2.5-pro
ROLE_OVERRIDES = {
    # "judge": "gemini-2.5-pro",
    **role_overrides_from_env(),
}

OPTIONS = {
    "pdf_query_input": env("EVISEARCH_PDF_QUERY_INPUT", "markdown"),  # markdown | pdf
    "reconciliation_page_images": env("EVISEARCH_RECONCILIATION_PAGE_IMAGES", "auto"),  # auto | never
}

# Output token budget per role
MAX_TOKENS = {
    "pdf_query": 16000,
    "search_agent": 8192,
    "reconciliation": 8192,
    "qa": 4096,
    "judge": 32000,
    "baseline": 16000,
    "structurer": 4096,
}

# ============== GPUS (local vLLM servers) ==============
# GPUs this project may use, and where each server runs: a list of GPU indices, or "auto" to pick the
# least-used GPUs from the pool when the server starts. Environment: EVISEARCH_GPU_POOL="0,1,2,3",
# EVISEARCH_GPUS="qwen36_27b=0;qwen3_embed_8b=1;qwen3_rerank_8b=1"
GPU_POOL = env("EVISEARCH_GPU_POOL", [0, 1, 2, 3, 4, 5, 6, 7])
GPUS = env("EVISEARCH_GPUS", {
    "qwen36_27b": "auto",
    "qwen3_embed_8b": "auto",
    "qwen3_rerank_8b": "auto",
})
GPU_MAX_MEMORY_FRACTION = 0.95  # launcher refuses a GPU if used memory + requested fraction exceeds this
VLLM_BIN = env("EVISEARCH_VLLM_BIN", "vllm")

SELECTION = CATALOG.resolve(PRESET, ROLE_OVERRIDES, OPTIONS, GPUS, GPU_POOL)

# ============== AGENTS ==============
BATCH_MAX_COLUMNS = 15  # columns per LLM call / agent run
AGENT_MAX_TURNS = 25
AGENT_MAX_TOOL_CALLS = 15
PDF_QUERY_MAX_MARKDOWN_CHARS = 0  # 0 = send the whole parsed markdown
RECONCILIATION_MAX_PAGE_IMAGES = 6  # page images attached per reconciliation batch

# ============== RETRIEVAL ==============
RETRIEVAL_TOP_K = 5  # pages returned by search_chunks
RERANK_CANDIDATES = 12  # embedding hits passed to the reranker when one is selected
SEARCH_PAGE_MAX_CHARS = 15000  # page text returned per hit

# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS_CSV_PATH = PROJECT_ROOT / "src" / "table_definitions" / "Definitions_with_eval_category.csv"
DEFINITIONS_EVAL_CATEGORY_PATH = DEFINITIONS_CSV_PATH
GOLD_TABLE_JSON_PATH = PROJECT_ROOT / "dataset" / "Manual_Benchmark_GoldTable_cleaned.json"
