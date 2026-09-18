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
PRESET = env("EVISEARCH_PRESET", "local")  # local | local_mistral | offline | cloud

# Per-role model overrides on top of the preset (role -> model key from the catalog).
# Environment: EVISEARCH_ROLE_<ROLE>=<model key>, e.g. EVISEARCH_ROLE_JUDGE=gemini-2.5-pro
ROLE_OVERRIDES = {
    # "judge": "gemini-2.5-pro",
    **role_overrides_from_env(),
}

OPTIONS = {
    # Arm A input, identical for every provider (the PDF file itself is never sent to a model).
    "pdf_query_input": env("EVISEARCH_PDF_QUERY_INPUT", "markdown_images"),  # markdown_images | markdown
    "reconciliation_page_images": env("EVISEARCH_RECONCILIATION_PAGE_IMAGES", "auto"),  # auto | never
    "extraction_rules": env("EVISEARCH_EXTRACTION_RULES", "v2"),  # v2 | v1 | none (the E0 prompts)
}

# Output token budget per role
MAX_TOKENS = {
    "pdf_query": 8000,  # Arm A answers use ~3k tokens per batch
    "search_agent": 8192,
    "reconciliation": 8192,
    "verifier": 4096,  # reconciler's verify_attribution tool (reconciliation role's model), <= 12 claims per call
    "reader": 4096,  # reconciler's ask_document tool: whole paper in, <= 8 answers out
    "qa": 4096,
    "judge": 32000,
    "baseline": 16000,
    "structurer": 4096,
}

# ============== GPUS (local vLLM servers) ==============
# GPUs this project may use, and where each server runs: a list of GPU indices, or "auto" to place it when it
# starts. "auto" keeps our servers together (the local preset's three servers fit on one H200) and otherwise picks
# the least-used GPU in the pool. Environment: EVISEARCH_GPU_POOL="4,5,6,7",
# EVISEARCH_GPUS="qwen36_27b=4;qwen3_embed_8b=4;qwen3_rerank_8b=4"
GPU_POOL = env("EVISEARCH_GPU_POOL", [4, 5, 6, 7])  # GPUs 0-3 are reserved for other groups
GPUS = env("EVISEARCH_GPUS", {
    "qwen36_27b": "auto",
    "mistral_small_24b": "auto",
    "qwen3_embed_8b": "auto",
    "qwen3_rerank_8b": "auto",
})
GPU_MAX_MEMORY_FRACTION = 0.95  # launcher refuses a GPU if used memory + requested fraction exceeds this
VLLM_BIN = env("EVISEARCH_VLLM_BIN", "vllm")  # found on PATH or next to the running Python (project venv)

SELECTION = CATALOG.resolve(PRESET, ROLE_OVERRIDES, OPTIONS, GPUS, GPU_POOL)

# ============== AGENTS ==============
BATCH_MAX_COLUMNS = 15  # columns per LLM call / agent run
AGENT_MAX_TURNS = 25
AGENT_MAX_TOOL_CALLS = 15
PAGE_IMAGE_SCALE = 2.0  # render scale for every page image sent to a model (Arm A, reconciliation); 2 = 144 dpi
PDF_QUERY_MAX_PAGE_IMAGES = 32  # page images per Arm A call; keep in step with --limit-mm-per-prompt in catalog.yaml
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
