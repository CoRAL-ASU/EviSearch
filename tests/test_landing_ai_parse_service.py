from __future__ import annotations

import json

import web.landing_ai_parse_service as parse_service


class _FakeParseResponse:
    markdown = "# Parsed markdown"

    def model_dump(self):
        return {
            "markdown": self.markdown,
            "chunks": [{"id": "chunk-1", "markdown": "Some text"}],
        }


class _FakeClient:
    def parse(self, document, model):
        assert document.exists()
        assert model == "dpt-2-latest"
        return _FakeParseResponse()


def test_parse_pdf_for_qa_writes_markdown_and_json(tmp_path, monkeypatch):
    pdf_path = tmp_path / "trial.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test pdf\n")

    results_root = tmp_path / "results"
    monkeypatch.setattr(parse_service, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(parse_service, "_ensure_api_key", lambda: None)
    monkeypatch.setattr(parse_service, "_init_client", lambda: _FakeClient())

    result = parse_service.parse_pdf_for_qa("doc-1", pdf_path)

    assert result["success"] is True

    markdown_path = results_root / "doc-1" / "chunking" / "parsed_markdown.md"
    parse_output_path = results_root / "doc-1" / "chunking" / "landing_ai_parse_output.json"

    assert markdown_path.read_text(encoding="utf-8") == "# Parsed markdown"
    assert json.loads(parse_output_path.read_text(encoding="utf-8"))["chunks"][0]["id"] == "chunk-1"


def test_parse_pdf_for_qa_returns_error_when_pdf_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(parse_service, "RESULTS_ROOT", tmp_path / "results")

    result = parse_service.parse_pdf_for_qa("doc-1", tmp_path / "missing.pdf")

    assert result["success"] is False
    assert "PDF not found" in result["error"]
