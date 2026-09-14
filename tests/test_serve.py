from __future__ import annotations

import subprocess

import pytest

from src.config.catalog import ConfigError, load_catalog
import src.inference.serve as serve
from src.inference.serve import PROJECT_ROOT, GpuInfo, build_command, plan_placement, query_gpus, resolve_vllm_bin, server_env

CATALOG = load_catalog()
GIB = 1024
SERVERS = ["qwen36_27b", "qwen3_embed_8b", "qwen3_rerank_8b"]


def _gpus(*used_gib):
    return [GpuInfo(index, int(used * GIB), 140 * GIB) for index, used in enumerate(used_gib)]


def test_query_gpus_parses_nvidia_smi():
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="0, 0, 143771\n4, 22307, 143771\n")

    assert query_gpus(fake_run) == [GpuInfo(0, 0, 143771), GpuInfo(4, 22307, 143771)]


def test_auto_placement_prefers_least_used_gpus_in_pool():
    selection = CATALOG.resolve("local", gpu_pool=[0, 1, 2])
    placement = plan_placement(selection, SERVERS, _gpus(30, 0, 0), max_fraction=0.95)
    assert placement == {"qwen36_27b": [1], "qwen3_embed_8b": [2], "qwen3_rerank_8b": [0]}


def test_auto_placement_packs_small_servers_onto_shared_gpu_when_pool_is_small():
    selection = CATALOG.resolve("local", gpu_pool=[0, 1])
    placement = plan_placement(selection, SERVERS, _gpus(0, 0), max_fraction=0.95)
    assert placement["qwen36_27b"] == [0]
    assert placement["qwen3_embed_8b"] == [1]
    assert placement["qwen3_rerank_8b"] == [1]


def test_explicit_busy_gpu_is_rejected_with_memory_details():
    selection = CATALOG.resolve("local", gpus={"qwen36_27b": [1]}, gpu_pool=[0, 1])
    with pytest.raises(ConfigError) as exc:
        plan_placement(selection, ["qwen36_27b"], _gpus(0, 22), max_fraction=0.95)
    assert "GPU 1: 22.0/140.0 GiB in use" in str(exc.value)


def test_auto_placement_fails_when_pool_is_full():
    selection = CATALOG.resolve("local", gpu_pool=[0])
    with pytest.raises(ConfigError, match="qwen3_rerank_8b needs 1 GPU"):
        plan_placement(selection, SERVERS, _gpus(0), max_fraction=0.95)


def test_vllm_bin_falls_back_to_the_active_python_environment(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "vllm").write_text("#!/bin/sh\n")
    monkeypatch.setattr(serve.sys, "executable", str(bin_dir / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert resolve_vllm_bin("vllm") == str(bin_dir / "vllm")
    assert resolve_vllm_bin("/opt/custom/vllm") == "/opt/custom/vllm"
    (bin_dir / "vllm").unlink()
    assert resolve_vllm_bin("vllm") == "vllm"


def test_server_env_sets_gpus_catalog_env_and_vllm_tools_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")

    catalog_env = CATALOG.servers["qwen36_27b"].env
    env = server_env(str(bin_dir / "vllm"), [0, 6], catalog_env)

    assert catalog_env == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PATH"].split(":")[:2] == [str(bin_dir), "/usr/bin"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,6"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert server_env("vllm", [1])["PATH"] == "/usr/bin"
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in server_env("vllm", [1], CATALOG.servers["qwen3_embed_8b"].env)


def test_build_command_uses_served_name_parsers_and_absolute_template_path():
    selection = CATALOG.resolve("local", gpu_pool=[0])
    chat = build_command(selection, "qwen36_27b", "/opt/vllm/bin/vllm")
    assert chat[:3] == ["/opt/vllm/bin/vllm", "serve", CATALOG.servers["qwen36_27b"].model_path]
    assert chat[chat.index("--served-model-name") + 1] == "Qwen/Qwen3.6-27B"
    assert chat[chat.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert chat[chat.index("--port") + 1] == "8002"

    rerank = build_command(selection, "qwen3_rerank_8b", "vllm")
    template = rerank[rerank.index("--chat-template") + 1]
    assert template == str(PROJECT_ROOT / "src/config/templates/qwen3_reranker.jinja")
    assert '"is_original_qwen3_reranker": true' in rerank[rerank.index("--hf-overrides") + 1]
