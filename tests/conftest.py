from __future__ import annotations

import os
from pathlib import Path

# The offline tests script their models; they are written against the local preset's model names, whatever preset
# this machine defaults to (src/config/config.py reads EVISEARCH_PRESET when it is first imported).
os.environ.setdefault("EVISEARCH_PRESET", "local")
# Scripted models hand out their replies in order, so the offline tests run one batch and one stage at a time unless a
# test sets the schedule itself.
os.environ.setdefault("EVISEARCH_STAGE_CONCURRENCY", "1")
os.environ.setdefault("EVISEARCH_STAGE_PARALLEL", "0")

import pytest


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    import web.main_app as main_app

    results_root = tmp_path / "results"
    dataset_dir = tmp_path / "dataset"
    upload_dir = tmp_path / "uploads"

    results_root.mkdir()
    dataset_dir.mkdir()
    upload_dir.mkdir()

    monkeypatch.setattr(main_app, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(main_app, "DATASET_DIR", dataset_dir)
    from src.config import runtime_paths  # services read paths from here (runs, jobs, reviews)

    monkeypatch.setattr(runtime_paths, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(runtime_paths, "JOBS_DIR", tmp_path / "jobs")
    main_app.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=upload_dir,
    )

    return main_app


@pytest.fixture
def client(isolated_app):
    return isolated_app.app.test_client()


@pytest.fixture
def definitions_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "Definitions_with_eval_category.csv"
    csv_path.write_text(
        "Column Name,Definition,Label\n"
        "Overall Survival,Median OS,Outcomes\n"
        "Treatment Arm,Intervention arm,Design\n",
        encoding="utf-8",
    )
    return csv_path
