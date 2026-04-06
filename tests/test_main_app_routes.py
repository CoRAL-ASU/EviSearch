from __future__ import annotations

import io
import json

import src.table_definitions.definitions as definitions_module


def test_api_documents_selectable_lists_dataset_results_and_uploads(isolated_app, client):
    dataset_pdf = isolated_app.DATASET_DIR / "dataset_doc.pdf"
    dataset_pdf.write_bytes(b"%PDF-1.4 dataset")

    extracted_dir = isolated_app.RESULTS_ROOT / "extracted_doc" / "agent_extractor"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "extraction_results.json").write_text(
        json.dumps({"columns": {"Overall Survival": {"value": "12.1"}}}),
        encoding="utf-8",
    )

    upload_pdf = isolated_app.app.config["UPLOAD_FOLDER"] / "upload_abc123.pdf"
    upload_pdf.write_bytes(b"%PDF-1.4 uploaded")

    response = client.get("/api/documents/selectable")
    payload = response.get_json()

    assert response.status_code == 200
    docs = {doc["id"]: doc for doc in payload["documents"]}
    assert docs["dataset_doc"]["source"] == "dataset"
    assert docs["extracted_doc"]["source"] == "extracted"
    assert docs["extracted_doc"]["has_extraction"] is True
    assert docs["upload_abc123"]["source"] == "upload"


def test_api_report_tables_applies_human_edits_and_column_groups(
    isolated_app,
    client,
    definitions_csv,
    monkeypatch,
):
    monkeypatch.setattr(definitions_module, "DEFINITIONS_CSV_PATH", definitions_csv)

    doc_dir = isolated_app.RESULTS_ROOT / "doc-1"
    recon_dir = doc_dir / "reconciliation_agent"
    human_dir = doc_dir / "human-edited"
    recon_dir.mkdir(parents=True)
    human_dir.mkdir(parents=True)

    (recon_dir / "reconciled_results.json").write_text(
        json.dumps(
            {
                "columns": {
                    "Overall Survival": {"value": "12.1"},
                    "Treatment Arm": {"value": "ADT"},
                }
            }
        ),
        encoding="utf-8",
    )
    (human_dir / "human_edited_results.json").write_text(
        json.dumps({"columns": {"Overall Survival": {"value": "13.0"}}}),
        encoding="utf-8",
    )

    response = client.get("/api/report/tables")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["document_count"] == 1
    assert payload["total_filled_values"] == 2
    assert payload["column_groups"]["Overall Survival"] == "Outcomes"
    assert payload["rows"] == [{"doc_id": "doc-1", "Overall Survival": "13.0", "Treatment Arm": "ADT"}]


def test_upload_extract_saves_uploaded_pdf(client, isolated_app):
    response = client.post(
        "/api/upload/extract",
        data={"file": (io.BytesIO(b"%PDF-1.4 fake pdf"), "trial.pdf")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["doc_id"].startswith("upload_")
    saved_path = isolated_app.app.config["UPLOAD_FOLDER"] / f"{payload['doc_id']}.pdf"
    assert saved_path.exists()


def test_api_document_pdf_serves_resolved_pdf(client, isolated_app, monkeypatch, tmp_path):
    pdf_path = tmp_path / "served.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 served")
    monkeypatch.setattr(isolated_app, "resolve_pdf_path", lambda doc_id: pdf_path)

    response = client.get("/api/documents/doc-1/pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-1.4")
