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
SCHEMAS_DIR = _resolve_runtime_path(
    "EVISEARCH_SCHEMAS_DIR",
    "schemas",
    PROJECT_ROOT / "new_pipeline_outputs" / "schemas",
)
KNOWLEDGE_DIR = _resolve_runtime_path(
    "EVISEARCH_KNOWLEDGE_DIR",
    "knowledge",
    PROJECT_ROOT / "new_pipeline_outputs" / "knowledge",
)
JOBS_DIR = _resolve_runtime_path(
    "EVISEARCH_JOBS_DIR",
    "jobs",
    PROJECT_ROOT / "new_pipeline_outputs" / "jobs",
)
DATASET_DIR = Path(os.getenv("EVISEARCH_DATASET_DIR", str(PROJECT_ROOT / "dataset")))


def ensure_runtime_dirs() -> None:
    for path in (UPLOADS_DIR, RESULTS_ROOT, CHUNK_EMBEDDINGS_DIR, FEEDBACK_DIR, SCHEMAS_DIR, KNOWLEDGE_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---- a deployment's runtime data (e.g. a Fly volume at EVISEARCH_RUNTIME_ROOT) -----------------------------------------
# The image ships the repository's outputs; the app serves them from the runtime directories. SEED_VERSION, written by
# shell-scripts/deploy_fly.sh, names the data an image carries.
SEED_SOURCE = PROJECT_ROOT / "new_pipeline_outputs"
SEED_VERSION_FILE = SEED_SOURCE / "SEED_VERSION"
SEEDED_DIRS = (
    (SEED_SOURCE / "results", RESULTS_ROOT),
    (SEED_SOURCE / "chunk_embeddings", CHUNK_EMBEDDINGS_DIR),
    (SEED_SOURCE / "feedback", FEEDBACK_DIR),
    (SEED_SOURCE / "schemas", SCHEMAS_DIR),
    (SEED_SOURCE / "knowledge", KNOWLEDGE_DIR),
    (SEED_SOURCE / "benchmark_runs", RESULTS_ROOT.parent / "benchmark_runs"),
)


def seed_runtime_dirs(pairs=SEEDED_DIRS, version_file: Path = SEED_VERSION_FILE, root: Path | None = None) -> int:
    """Serve the image's outputs from relocated runtime directories. Returns the number of files copied.

    A no-op unless EVISEARCH_RUNTIME_ROOT is set. When the image carries data the volume has not seen
    (its SEED_VERSION differs from the volume's .seed_version), the volume's copies are moved to
    _previous/<old version>/ - kept, never deleted - and the image's are copied in whole, so the deployed app shows what
    the local app shows. Otherwise only missing files are copied, so what visitors added since the deploy (uploads,
    runs, reviews, note edits) stays.
    """
    import shutil
    import time

    root = root or RUNTIME_ROOT
    if root is None:  # only a deployment (EVISEARCH_RUNTIME_ROOT) serves from a separate place
        return 0
    pairs = [(source, target) for source, target in pairs if source.is_dir() and source.resolve() != target.resolve()]
    if not pairs:
        return 0
    marker = root / ".seed_version"
    image_version = version_file.read_text().strip() if version_file.exists() else ""
    volume_version = marker.read_text().strip() if marker.exists() else ""
    fresh = bool(image_version) and image_version != volume_version
    if fresh:
        keep = marker.parent / "_previous" / f"{volume_version or 'unversioned'}-{int(time.time())}"
        for _, target in pairs:
            if target.exists() and any(target.iterdir()):
                keep.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(keep / target.name))
    copied = 0
    for source, target in pairs:
        for path in source.rglob("*"):
            dest = target / path.relative_to(source)
            if not path.is_file() or dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied += 1
    if fresh:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(image_version + "\n")
    return copied
