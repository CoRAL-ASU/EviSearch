"""A deployment serves the image's outputs from its volume: a new image's data replaces the volume's (which is kept),
and between deploys what visitors add is never overwritten."""
from __future__ import annotations

from src.config.runtime_paths import seed_runtime_dirs


def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_seeding_is_a_no_op_without_a_runtime_root(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    _tree(source, {"a.json": "1"})
    assert seed_runtime_dirs([(source, target)], tmp_path / "SEED_VERSION", root=None) == 0
    assert not target.exists()


def test_a_new_image_replaces_the_volume_data_and_keeps_the_old_copy(tmp_path):
    image, volume = tmp_path / "image", tmp_path / "data"
    source, target = image / "results", volume / "results"
    version = image / "SEED_VERSION"
    _tree(source, {"doc/runs/r1/x.json": "new"})
    _tree(target, {"doc/runs/old/x.json": "old", "doc/agent_extractor/y.json": "legacy"})
    version.write_text("v2\n")

    assert seed_runtime_dirs([(source, target)], version, root=volume) == 1
    assert (target / "doc/runs/r1/x.json").read_text() == "new"
    assert not (target / "doc/runs/old").exists()                       # the old deploy's outputs are gone from view
    kept = list((volume / "_previous").glob("unversioned-*/results/doc/runs/old/x.json"))
    assert kept and kept[0].read_text() == "old"                          # ... but kept
    assert (volume / ".seed_version").read_text().strip() == "v2"


def test_between_deploys_visitors_data_survives_a_restart(tmp_path):
    image, volume = tmp_path / "image", tmp_path / "data"
    source, target = image / "results", volume / "results"
    version = image / "SEED_VERSION"
    _tree(source, {"doc/runs/r1/x.json": "shipped"})
    version.write_text("v2\n")
    seed_runtime_dirs([(source, target)], version, root=volume)
    _tree(target, {"doc/runs/visitor/x.json": "made on the web"})
    (target / "doc/runs/r1/x.json").write_text("reviewed on the web")

    assert seed_runtime_dirs([(source, target)], version, root=volume) == 0  # same image: nothing replaced
    assert (target / "doc/runs/visitor/x.json").read_text() == "made on the web"
    assert (target / "doc/runs/r1/x.json").read_text() == "reviewed on the web"


def test_the_health_check_answers_without_the_demo_password(monkeypatch):
    import web.main_app as main_app

    monkeypatch.setenv("EVISEARCH_DEMO_PASSWORD", "secret")
    client = main_app.app.test_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/stats").status_code == 401
    ok = client.get("/api/stats", headers={"Authorization": "Basic ZXZpc2VhcmNoOnNlY3JldA=="})  # evisearch:secret
    assert ok.status_code != 401
