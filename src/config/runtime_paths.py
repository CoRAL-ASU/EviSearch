from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_runtime_root = os.getenv("EVISEARCH_RUNTIME_ROOT", "").strip()
RUNTIME_ROOT = Path(_runtime_root) if _runtime_root else None


def _resolve_runtime_path(env_name: str, default_relative: str, default_repo_path: Path) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        return Path(configured)
    if RUNTIME_ROOT is not None:
        return RUNTIME_ROOT / default_relative
    return default_repo_path


UPLOADS_DIR = _resolve_runtime_path(
    "EVISEARCH_UPLOADS_DIR",
    "uploads",
    PROJECT_ROOT / "web" / "uploads",
)
RESULTS_ROOT = _resolve_runtime_path(
    "EVISEARCH_RESULTS_ROOT",
    "results",
    PROJECT_ROOT / "new_pipeline_outputs" / "results",
)
CHUNK_EMBEDDINGS_DIR = _resolve_runtime_path(
    "EVISEARCH_CHUNK_EMBEDDINGS_DIR",
    "chunk_embeddings",
    PROJECT_ROOT / "new_pipeline_outputs" / "chunk_embeddings",
)
FEEDBACK_DIR = _resolve_runtime_path(
    "EVISEARCH_FEEDBACK_DIR",
    "feedback",
    PROJECT_ROOT / "new_pipeline_outputs" / "feedback",
)
DATASET_DIR = Path(os.getenv("EVISEARCH_DATASET_DIR", str(PROJECT_ROOT / "dataset")))


def ensure_runtime_dirs() -> None:
    for path in (UPLOADS_DIR, RESULTS_ROOT, CHUNK_EMBEDDINGS_DIR, FEEDBACK_DIR):
        path.mkdir(parents=True, exist_ok=True)
