from __future__ import annotations

from pathlib import Path

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
    main_app.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=upload_dir,
    )
    main_app.current_pdf_info = {}
    main_app.extraction_service = None

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
