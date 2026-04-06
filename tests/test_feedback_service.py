from __future__ import annotations

import web.feedback_service as feedback_service


def test_record_and_load_feedback_round_trip(tmp_path, monkeypatch):
    feedback_dir = tmp_path / "feedback"
    feedback_file = feedback_dir / "feedback.jsonl"

    monkeypatch.setattr(feedback_service, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(feedback_service, "FEEDBACK_FILE", feedback_file)

    assert feedback_service.record_feedback(
        {
            "source": "chat",
            "doc_id": "doc-1",
            "comment": "x" * 700,
            "chat": {"question": "Q", "answer": "A"},
        }
    )
    assert feedback_service.record_feedback(
        {
            "source": "attribution",
            "doc_id": "doc-2",
            "comment": "second",
        }
    )

    loaded = feedback_service.load_feedback(doc_id="doc-1", source="chat")

    assert len(loaded) == 1
    assert loaded[0]["doc_id"] == "doc-1"
    assert loaded[0]["source"] == "chat"
    assert len(loaded[0]["comment"]) == 500
    assert "timestamp" in loaded[0]


def test_load_feedback_returns_most_recent_first(tmp_path, monkeypatch):
    feedback_dir = tmp_path / "feedback"
    feedback_file = feedback_dir / "feedback.jsonl"

    monkeypatch.setattr(feedback_service, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(feedback_service, "FEEDBACK_FILE", feedback_file)

    feedback_service.record_feedback({"source": "chat", "doc_id": "old"})
    feedback_service.record_feedback({"source": "chat", "doc_id": "new"})

    loaded = feedback_service.load_feedback(limit=2)

    assert [entry["doc_id"] for entry in loaded] == ["new", "old"]
