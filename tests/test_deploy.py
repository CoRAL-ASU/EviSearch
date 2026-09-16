from __future__ import annotations

import base64

from src.config.runtime_paths import seed_runtime_dirs


def _basic(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_demo_password_unset_leaves_app_open(client, monkeypatch):
    monkeypatch.delenv("EVISEARCH_DEMO_PASSWORD", raising=False)

    assert client.get("/api/qa/session-info").status_code == 200


def test_demo_password_requires_basic_auth_except_healthz(client, monkeypatch):
    monkeypatch.setenv("EVISEARCH_DEMO_PASSWORD", "s3cret")
    monkeypatch.delenv("EVISEARCH_DEMO_USER", raising=False)

    denied = client.get("/api/qa/session-info")
    assert denied.status_code == 401
    assert denied.headers["WWW-Authenticate"].startswith("Basic")
    assert client.get("/api/qa/session-info", headers=_basic("evisearch", "wrong")).status_code == 401
    assert client.get("/api/qa/session-info", headers=_basic("evisearch", "s3cret")).status_code == 200
    assert client.get("/healthz").status_code == 200


def test_seed_runtime_dirs_copies_missing_files_without_overwriting(tmp_path):
    source = tmp_path / "repo" / "results"
    (source / "doc" / "chunking").mkdir(parents=True)
    (source / "doc" / "chunking" / "parsed_markdown.md").write_text("shipped", encoding="utf-8")
    (source / "doc" / "human-edited.json").write_text("shipped", encoding="utf-8")
    target = tmp_path / "volume" / "results"
    (target / "doc").mkdir(parents=True)
    (target / "doc" / "human-edited.json").write_text("edited on volume", encoding="utf-8")

    assert seed_runtime_dirs([(source, target)]) == 1
    assert (target / "doc" / "chunking" / "parsed_markdown.md").read_text(encoding="utf-8") == "shipped"
    assert (target / "doc" / "human-edited.json").read_text(encoding="utf-8") == "edited on volume"
    assert seed_runtime_dirs([(source, target)]) == 0
    assert seed_runtime_dirs([(source, source)]) == 0
