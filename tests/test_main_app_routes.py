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
    assert docs["dataset_doc"]["has_cached_embeddings"] is False
    assert docs["extracted_doc"]["has_cached_parse"] is False


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
    assert payload["rows"] == [{"doc_id": "doc-1", "pdf_exists": False, "Overall Survival": "13.0", "Treatment Arm": "ADT"}]


def test_upload_extract_saves_uploaded_pdf(client, isolated_app):
    response = client.post(
        "/api/upload/extract",
        data={"file": (io.BytesIO(b"%PDF-1.4 fake pdf"), "trial.pdf")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["doc_id"].startswith("pdf_")
    assert payload["upload_doc_id"].startswith("upload_")
    raw_upload = isolated_app.app.config["UPLOAD_FOLDER"] / f"{payload['upload_doc_id']}.pdf"
    canonical_pdf = isolated_app.RESULTS_ROOT / payload["doc_id"] / f"{payload['doc_id']}.pdf"
    assert raw_upload.exists()
    assert canonical_pdf.exists()
    assert payload["reused_existing_doc"] is False


def test_upload_extract_reuses_canonical_doc_for_duplicate_pdf(client, isolated_app):
    payloads = []
    for filename in ("trial-a.pdf", "trial-b.pdf"):
        response = client.post(
            "/api/upload/extract",
            data={"file": (io.BytesIO(b"%PDF-1.4 fake pdf"), filename)},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        payloads.append(response.get_json())

    first, second = payloads
    assert first["doc_id"] == second["doc_id"]
    assert first["upload_doc_id"] != second["upload_doc_id"]
    assert second["reused_existing_doc"] is True


def test_api_document_pdf_serves_resolved_pdf(client, isolated_app, monkeypatch, tmp_path):
    pdf_path = tmp_path / "served.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 served")
    monkeypatch.setattr(isolated_app, "resolve_pdf_path", lambda doc_id: pdf_path)

    response = client.get("/api/documents/doc-1/pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-1.4")


def test_api_refresh_attribution_returns_reconciled_page_shape(client, isolated_app):
    doc_dir = isolated_app.RESULTS_ROOT / "doc-1"
    recon_dir = doc_dir / "reconciliation_agent"
    agent_dir = doc_dir / "agent_extractor"
    search_dir = doc_dir / "search_agent"
    recon_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)

    (recon_dir / "reconciled_results.json").write_text(
        json.dumps(
            {
                "columns": {
                    "Overall Survival": {
                        "value": "12.1",
                        "reasoning": "Reconciled answer",
                        "source": {"page": 5, "modality": "text"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "extraction_results.json").write_text(
        json.dumps(
            {
                "columns": {
                    "Overall Survival": {
                        "value": "12.1",
                        "reasoning": "Agent answer",
                        "attribution": [{"page": 5, "modality": "text"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (search_dir / "extraction_results.json").write_text(
        json.dumps(
            {
                "columns": {
                    "Overall Survival": {
                        "value": "12.1",
                        "reasoning": "Search answer",
                        "attribution": [{"page": 5, "modality": "text"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    response = client.post("/api/documents/doc-1/attribution/refresh")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["doc_id"] == "doc-1"
    assert len(payload["columns"]) == 1
    assert payload["columns"][0]["column_name"] == "Overall Survival"
    assert payload["columns"][0]["candidate_a"] == "12.1"
    assert payload["columns"][0]["candidate_b"] == "12.1"


def test_api_reconciled_resolves_agent_and_search_page_attribution_to_chunks(
    client,
    isolated_app,
    monkeypatch,
):
    import src.evisearch.services.highlight as highlight_service

    monkeypatch.setattr(highlight_service, "PIPELINE_RESULTS", isolated_app.RESULTS_ROOT)

    doc_dir = isolated_app.RESULTS_ROOT / "doc-1"
    recon_dir = doc_dir / "reconciliation_agent"
    agent_dir = doc_dir / "agent_extractor"
    search_dir = doc_dir / "search_agent"
    chunk_dir = doc_dir / "chunking"
    recon_dir.mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    search_dir.mkdir(parents=True)
    chunk_dir.mkdir(parents=True)

    (chunk_dir / "landing_ai_parse_output.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "id": "table-chunk-1",
                        "type": "table",
                        "markdown": "Table 1. Baseline characteristics ADT Alone (N=393)",
                        "grounding": {
                            "page": 4,
                            "box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (recon_dir / "reconciled_results.json").write_text(
        json.dumps(
            {
                "columns": {
                    "Control Arm - N": {
                        "value": "393",
                        "reasoning": "Reconciled answer",
                        "source": {"page": 5, "modality": "table"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    for out_dir, reasoning in ((agent_dir, "Agent answer"), (search_dir, "Search answer")):
        (out_dir / "extraction_results.json").write_text(
            json.dumps(
                {
                    "columns": {
                        "Control Arm - N": {
                            "value": "393",
                            "reasoning": reasoning,
                            "attribution": [{"page": 5, "modality": "table"}],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    response = client.get("/api/documents/doc-1/reconciled")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    column = payload["columns"][0]
    assert column["chunk_ids_a"] == ["table-chunk-1"]
    assert column["chunk_ids_b"] == ["table-chunk-1"]
    assert column["chunk_ids"] == ["table-chunk-1"]
    assert "page_5" not in column["chunk_ids_a"]
    assert "page_5" not in column["chunk_ids_b"]
