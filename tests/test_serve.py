from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

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


def test_local_servers_fit_on_one_h200():
    total = sum(CATALOG.servers[server].gpu_memory_utilization for server in SERVERS)
    idle = 1 / 140  # an idle H200 reports ~0.5-1 GiB in use
    assert total + idle <= 0.95


def test_auto_placement_puts_all_local_servers_on_the_least_used_free_gpu():
    selection = CATALOG.resolve("local", gpu_pool=[0, 1, 2])
    placement = plan_placement(selection, SERVERS, _gpus(30, 0, 0), max_fraction=0.95)
    assert placement == {"qwen36_27b": [1], "qwen3_embed_8b": [1], "qwen3_rerank_8b": [1]}


def test_mistral_preset_places_mistral_with_embedding_and_reranker_on_one_gpu():
    selection = CATALOG.resolve("local_mistral", gpu_pool=[4, 5, 6, 7])
    servers = selection.servers_needed()
    assert servers == ["mistral_small_24b", "qwen3_embed_8b", "qwen3_rerank_8b"]
    idle = 1 / 140
    assert sum(CATALOG.servers[server].gpu_memory_utilization for server in servers) + idle <= 0.95

    placement = plan_placement(selection, servers, _gpus(0, 0, 0, 0, 30, 1, 1, 1), max_fraction=0.95)
    assert placement == {"mistral_small_24b": [5], "qwen3_embed_8b": [5], "qwen3_rerank_8b": [5]}


def test_mistral_server_command_loads_mistral_format_weights():
    selection = CATALOG.resolve("local_mistral", gpu_pool=[0])
    command = build_command(selection, "mistral_small_24b", "vllm")
    spec = CATALOG.servers["mistral_small_24b"]
    assert command[2] == spec.model_path and spec.port not in (8002, 8003, 8004, 8005, 8007)
    assert command[command.index("--served-model-name") + 1] == "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    for flag in ("--tokenizer-mode", "--config-format", "--load-format", "--tool-call-parser"):
        assert command[command.index(flag) + 1] == "mistral", flag
    assert command[command.index("--max-model-len") + 1] == str(CATALOG.models["mistral-small-3.2-24b"].context_tokens)
    assert "--enable-auto-tool-choice" in command and "--enable-prompt-tokens-details" in command
    shim = PROJECT_ROOT / "src" / "inference" / "vllm_compat"
    assert serve.catalog_env(selection, "mistral_small_24b") == {"VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTHONPATH": str(shim)}
    assert server_env("vllm", [4], serve.catalog_env(selection, "mistral_small_24b"))["PYTHONPATH"] == str(shim)


def test_vllm_compat_gives_a_module_raising_stand_ins_for_names_it_lacks(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("vllm_compat_shim", PROJECT_ROOT / "src/inference/vllm_compat/sitecustomize.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.PIXTRAL == "transformers.models.pixtral.modeling_pixtral"
    (tmp_path / "fakepixtral").mkdir()
    (tmp_path / "fakepixtral" / "__init__.py").write_text("")
    (tmp_path / "fakepixtral" / "modeling.py").write_text("kept = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", [shim.AliasFinder("fakepixtral.modeling", ["kept", "removed"]), *sys.meta_path])

    module = importlib.import_module("fakepixtral.modeling")

    assert module.kept == 1
    with pytest.raises(NotImplementedError, match="fakepixtral.modeling.removed"):
        module.removed()
    monkeypatch.delitem(sys.modules, "fakepixtral.modeling")
    monkeypatch.delitem(sys.modules, "fakepixtral")


def test_auto_placement_spills_to_the_next_gpu_when_the_shared_one_is_full():
    selection = CATALOG.resolve("local", gpu_pool=[0, 1])
    placement = plan_placement(selection, SERVERS, _gpus(10, 30), max_fraction=0.95)
    assert placement == {"qwen36_27b": [0], "qwen3_embed_8b": [0], "qwen3_rerank_8b": [1]}


def test_auto_placement_joins_the_gpu_where_our_servers_already_run():
    selection = CATALOG.resolve("local", gpu_pool=[0, 1])
    placement = plan_placement(selection, ["qwen3_embed_8b"], _gpus(0, 85), max_fraction=0.95, ours=[1])
    assert placement == {"qwen3_embed_8b": [1]}


def test_explicit_busy_gpu_is_rejected_with_memory_details():
    selection = CATALOG.resolve("local", gpus={"qwen36_27b": [1]}, gpu_pool=[0, 1])
    with pytest.raises(ConfigError) as exc:
        plan_placement(selection, ["qwen36_27b"], _gpus(0, 60), max_fraction=0.95)
    assert "GPU 1: 60.0/140.0 GiB in use" in str(exc.value)


def test_auto_placement_fails_when_pool_is_full():
    selection = CATALOG.resolve("local", gpu_pool=[0])
    with pytest.raises(ConfigError, match="qwen3_rerank_8b needs 1 GPU"):
        plan_placement(selection, SERVERS, _gpus(20), max_fraction=0.95)


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


def test_port_taken_by_another_model_is_an_error_not_a_running_server(monkeypatch, capsys):
    monkeypatch.setattr(serve, "_port_in_use", lambda host, port: True)
    monkeypatch.setattr(serve, "served_models", lambda host, port: ["qwen3-embedding-8b"])
    assert serve.main(["--only", "qwen3_embed_8b"]) == 2
    assert "serves ['qwen3-embedding-8b'], not 'Qwen/Qwen3-Embedding-8B'" in capsys.readouterr().err

    monkeypatch.setattr(serve, "served_models", lambda host, port: ["Qwen/Qwen3-Embedding-8B"])
    assert serve.main(["--only", "qwen3_embed_8b"]) == 0
    assert "already running" in capsys.readouterr().out


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
