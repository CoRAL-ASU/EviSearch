from __future__ import annotations

import os
import shutil
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


# Outputs shipped with the repo, and the runtime directory each one is served from.
SEEDED_DIRS = (
    (PROJECT_ROOT / "new_pipeline_outputs" / "results", RESULTS_ROOT),
    (PROJECT_ROOT / "new_pipeline_outputs" / "chunk_embeddings", CHUNK_EMBEDDINGS_DIR),
    (PROJECT_ROOT / "new_pipeline_outputs" / "feedback", FEEDBACK_DIR),
)


def seed_runtime_dirs(pairs=SEEDED_DIRS) -> int:
    """Copy repo outputs into relocated runtime dirs (e.g. a Fly volume), never overwriting.

    A no-op when the runtime dirs are the repo dirs. Files already on the volume win, so
    uploads and human edits survive redeploys while new benchmark outputs still appear.
    Returns the number of files copied.
    """
    copied = 0
    for source, target in pairs:
        if not source.is_dir() or source.resolve() == target.resolve():
            continue
        for path in source.rglob("*"):
            dest = target / path.relative_to(source)
            if not path.is_file() or dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied += 1
    return copied
