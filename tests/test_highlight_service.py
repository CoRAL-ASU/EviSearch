from __future__ import annotations

import json

import web.highlight_service as highlight_service


def test_get_highlights_by_chunk_ids_returns_boxes(tmp_path, monkeypatch):
    pipeline_results = tmp_path / "results"
    chunk_dir = pipeline_results / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "landing_ai_parse_output.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "id": "chunk-1",
                        "markdown": "Alpha",
                        "grounding": {
                            "page": 1,
                            "box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(highlight_service, "PIPELINE_RESULTS", pipeline_results)

    result = highlight_service.get_highlights_by_chunk_ids("doc-1", ["chunk-1"])

    assert result["available"] is True
    assert result["highlights"] == [
        {
            "page": 2,
            "box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8},
            "chunk_id": "chunk-1",
        }
    ]


def test_get_chunks_by_page_type_filters_page_and_modality(tmp_path, monkeypatch):
    pipeline_results = tmp_path / "results"
    chunk_dir = pipeline_results / "doc-1" / "chunking"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "landing_ai_parse_output.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "id": "text-1",
                        "type": "text",
                        "markdown": "Page 1 text",
                        "grounding": {"page": 0},
                    },
                    {
                        "id": "table-1",
                        "type": "table",
                        "markdown": "Page 1 table",
                        "grounding": {"page": 0},
                    },
                    {
                        "id": "text-2",
                        "type": "text",
                        "markdown": "Page 2 text",
                        "grounding": {"page": 1},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(highlight_service, "PIPELINE_RESULTS", pipeline_results)

    result = highlight_service.get_chunks_by_page_type("doc-1", page=1, source_type="text")

    assert result == [
        {
            "chunk_id": "text-1",
            "page": 1,
            "source_type": "text",
            "text": "Page 1 text",
        }
    ]
