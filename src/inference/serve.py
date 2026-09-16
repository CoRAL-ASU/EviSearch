"""Start the local vLLM servers the current selection needs, on the GPUs chosen in src/config/config.py.

  python -m src.inference.serve                 # start, wait until healthy, stay in the foreground (Ctrl+C stops)
  python -m src.inference.serve --dry-run       # show GPU placement and commands only
  python -m src.inference.serve --detach        # start in the background (logs in .cache/serve/)
  python -m src.inference.serve --status        # show servers started with --detach
  python -m src.inference.serve --stop          # stop servers started with --detach
  python -m src.inference.serve --only qwen36_27b
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import httpx

from src.config.catalog import ConfigError, Selection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".cache" / "serve"
STATE_FILE = STATE_DIR / "servers.json"
HEALTH_TIMEOUT_S = 1800
MIB_PER_GIB = 1024


@dataclass(frozen=True)
class GpuInfo:
    index: int
    used_mib: int
    total_mib: int

    @property
    def used_fraction(self) -> float:
        return self.used_mib / self.total_mib if self.total_mib else 1.0


def query_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> List[GpuInfo]:
    try:
        completed = run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"Could not query GPUs with nvidia-smi: {exc}") from exc
    gpus = []
    for line in completed.stdout.strip().splitlines():
        index, used, total = (int(float(value)) for value in line.split(","))
        gpus.append(GpuInfo(index, used, total))
    return gpus


def plan_placement(
    selection: Selection,
    servers: Sequence[str],
    gpus: Sequence[GpuInfo],
    max_fraction: float,
) -> Dict[str, List[int]]:
    """Choose GPUs for each server. Explicit assignments are checked first, then "auto" servers are packed
    onto the least-used GPUs in the pool. Raises ConfigError explaining why a server does not fit."""
    catalog = selection.catalog
    info = {gpu.index: gpu for gpu in gpus}
    committed = {gpu.index: gpu.used_fraction for gpu in gpus}
    placement: Dict[str, List[int]] = {}
    errors: List[str] = []

    def describe(index: int) -> str:
        gpu = info[index]
        return f"GPU {index}: {gpu.used_mib / MIB_PER_GIB:.1f}/{gpu.total_mib / MIB_PER_GIB:.1f} GiB in use"

    explicit = [s for s in servers if selection.gpus.get(s, "auto") != "auto"]
    automatic = [s for s in servers if selection.gpus.get(s, "auto") == "auto"]

    for server in explicit:
        spec = catalog.servers[server]
        indices = list(selection.gpus[server])
        problems = []
        for index in indices:
            if index not in info:
                problems.append(f"GPU {index} does not exist (found {sorted(info)})")
            elif committed[index] + spec.gpu_memory_utilization > max_fraction:
                problems.append(
                    f"{describe(index)}, plus {spec.gpu_memory_utilization:.0%} for {server} exceeds the {max_fraction:.0%} limit"
                )
        if problems:
            errors.append(f"{server} (GPUS={indices}): " + "; ".join(problems))
            continue
        for index in indices:
            committed[index] += spec.gpu_memory_utilization
        placement[server] = indices

    for server in automatic:
        spec = catalog.servers[server]
        candidates = sorted(
            (index for index in selection.gpu_pool if index in info),
            key=lambda index: (committed[index], index),
        )
        fitting = [index for index in candidates if committed[index] + spec.gpu_memory_utilization <= max_fraction]
        if len(fitting) < spec.tensor_parallel:
            states = "; ".join(describe(index) for index in candidates) or "no pool GPUs found"
            errors.append(
                f"{server} needs {spec.tensor_parallel} GPU(s) with {spec.gpu_memory_utilization:.0%} free memory in "
                f"GPU_POOL {selection.gpu_pool} ({states})"
            )
            continue
        chosen = sorted(fitting[: spec.tensor_parallel])
        for index in chosen:
            committed[index] += spec.gpu_memory_utilization
        placement[server] = chosen

    if errors:
        raise ConfigError("Cannot place local servers:\n  - " + "\n  - ".join(errors))
    return placement


def served_model_name(selection: Selection, server: str) -> str:
    catalog = selection.catalog
    return catalog.models[catalog.models_on_server(server)[0]].name


def served_models(host: str, port: int) -> Optional[List[str]]:
    """Model names an OpenAI-compatible server on this port reports, or None when it does not answer /v1/models."""
    host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        response = httpx.get(f"http://{host}:{port}/v1/models", timeout=3.0)
        response.raise_for_status()
        return [item.get("id") for item in response.json().get("data", [])]
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


def build_command(selection: Selection, server: str, vllm_bin: str) -> List[str]:
    catalog = selection.catalog
    spec = catalog.servers[server]
    served = served_model_name(selection, server)
    command = [
        vllm_bin, "serve", spec.model_path,
        "--served-model-name", served,
        "--host", spec.host,
        "--port", str(spec.port),
        "--tensor-parallel-size", str(spec.tensor_parallel),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
    ]
    if spec.max_model_len:
        command += ["--max-model-len", str(spec.max_model_len)]
    if spec.max_num_seqs:
        command += ["--max-num-seqs", str(spec.max_num_seqs)]
    for arg in spec.args:
        candidate = PROJECT_ROOT / arg
        if not arg.startswith(("-", "{", "/")) and candidate.exists():
            arg = str(candidate)
        command.append(arg)
    return command


def resolve_vllm_bin(configured: str) -> str:
    """A configured path as-is; otherwise `vllm` from PATH, or the one installed next to this Python (the venv)."""
    if os.sep in configured:
        return configured
    found = shutil.which(configured)
    if found:
        return found
    sibling = Path(sys.executable).with_name(configured)
    return str(sibling) if sibling.is_file() else configured


def server_env(vllm_bin: str, gpus: Sequence[int], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a server process: its GPUs, offline model loading, the server's catalog `env`, and the vllm
    environment's bin/ first on PATH so tools vLLM runs itself (e.g. ninja) are found."""
    env = dict(os.environ)
    if os.sep in vllm_bin:
        env["PATH"] = os.pathsep.join([str(Path(vllm_bin).resolve().parent), env.get("PATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["HF_HUB_OFFLINE"] = "1"
    env.update(extra or {})
    return env


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) == 0


def _health_url(selection: Selection, server: str) -> str:
    spec = selection.catalog.servers[server]
    host = "127.0.0.1" if spec.host in ("0.0.0.0", "") else spec.host
    return f"http://{host}:{spec.port}/health"


def _read_state() -> Dict[str, dict]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state(state: Dict[str, dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _wait_healthy(url: str, process: subprocess.Popen, log_path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}. Last log lines:\n{_tail(log_path)}")
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError(f"server not healthy after {timeout_s:.0f}s ({url}). Last log lines:\n{_tail(log_path)}")


def _stop(pids: Dict[str, int]) -> None:
    for server, pid in pids.items():
        try:
            os.killpg(pid, signal.SIGTERM)
            print(f"Stopping {server} (pid {pid})")
        except OSError:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start local vLLM servers for the current selection")
    parser.add_argument("--only", nargs="+", help="Start only these servers")
    parser.add_argument("--dry-run", action="store_true", help="Show placement and commands without starting anything")
    parser.add_argument("--detach", action="store_true", help="Start in the background and return")
    parser.add_argument("--status", action="store_true", help="Show servers started with --detach")
    parser.add_argument("--stop", action="store_true", help="Stop servers started with --detach")
    args = parser.parse_args(argv)

    try:
        from src.config import config
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    selection = config.SELECTION

    if args.status or args.stop:
        state = _read_state()
        if not state:
            print("No servers were started with --detach.")
            return 0
        if args.stop:
            _stop({name: entry["pid"] for name, entry in state.items() if _alive(entry["pid"])})
            STATE_FILE.unlink(missing_ok=True)
            return 0
        for name, entry in state.items():
            try:
                healthy = httpx.get(entry["health"], timeout=2.0).status_code == 200
            except httpx.HTTPError:
                healthy = False
            status = "healthy" if healthy else ("starting" if _alive(entry["pid"]) else "stopped")
            print(f"{name:<16} {status:<9} pid={entry['pid']} GPUs={entry['gpus']} log={entry['log']}")
        return 0

    needed = selection.servers_needed()
    servers = [s for s in needed if not args.only or s in args.only]
    unknown = [s for s in (args.only or []) if s not in selection.catalog.servers]
    if unknown:
        print(f"Unknown server(s): {', '.join(unknown)}. Servers: {', '.join(selection.catalog.servers)}", file=sys.stderr)
        return 2
    if args.only:
        servers = list(dict.fromkeys(args.only))
    if not servers:
        print(f"Preset '{selection.preset}' uses no local servers; nothing to start.")
        return 0

    pending = []
    for server in servers:
        spec = selection.catalog.servers[server]
        if not _port_in_use(spec.host, spec.port):
            pending.append(server)
            continue
        expected, found = served_model_name(selection, server), served_models(spec.host, spec.port)
        if found is not None and expected not in found:
            print(
                f"{server}: port {spec.port} is in use by a server that serves {found}, not '{expected}' "
                f"(someone else's server?). Give {server} a free port in src/config/catalog.yaml.",
                file=sys.stderr,
            )
            return 2
        print(f"{server}: '{expected}' is already running on port {spec.port}; skipping it.")
    if not pending:
        return 0

    try:
        placement = plan_placement(selection, pending, query_gpus(), config.GPU_MAX_MEMORY_FRACTION)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    vllm_bin = resolve_vllm_bin(config.VLLM_BIN)
    have_vllm = Path(vllm_bin).is_file()
    for server in pending:
        command = build_command(selection, server, vllm_bin)
        overrides = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, placement[server])), **selection.catalog.servers[server].env}
        print(f"\n{server}: " + " ".join(f"{k}={v}" for k, v in overrides.items()))
        print("  " + shlex.join(command))
    if args.dry_run:
        if not have_vllm:
            print(f"\nNote: vLLM executable '{vllm_bin}' was not found; set EVISEARCH_VLLM_BIN before starting.")
        return 0
    if not have_vllm:
        print(
            f"vLLM executable '{vllm_bin}' was not found. Install it into this environment with "
            "pip install -r requirements-local.txt, or set EVISEARCH_VLLM_BIN to an existing vllm executable.",
            file=sys.stderr,
        )
        return 2

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state()
    started: Dict[str, subprocess.Popen] = {}
    try:
        for server in pending:
            log_path = STATE_DIR / f"{server}.log"
            with open(log_path, "ab") as log:
                process = subprocess.Popen(
                    build_command(selection, server, vllm_bin),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=server_env(vllm_bin, placement[server], selection.catalog.servers[server].env),
                    cwd=PROJECT_ROOT,
                    start_new_session=True,
                )
            started[server] = process
            state[server] = {"pid": process.pid, "gpus": placement[server], "log": str(log_path), "health": _health_url(selection, server)}
            _write_state(state)
            print(f"Started {server} (pid {process.pid}); log: {log_path}")
        for server, process in started.items():
            print(f"Waiting for {server} to become healthy...")
            _wait_healthy(_health_url(selection, server), process, STATE_DIR / f"{server}.log", HEALTH_TIMEOUT_S)
            print(f"{server} is ready at {_health_url(selection, server).removesuffix('/health')}")
    except (RuntimeError, KeyboardInterrupt) as exc:
        if isinstance(exc, RuntimeError):
            print(f"Startup failed: {exc}", file=sys.stderr)
        _stop({name: process.pid for name, process in started.items()})
        for name in started:
            state.pop(name, None)
        _write_state(state)
        return 1

    if args.detach:
        print("\nServers are running in the background. Stop them with: python -m src.inference.serve --stop")
        return 0
    print("\nServers are running. Press Ctrl+C to stop them.")
    try:
        while all(process.poll() is None for process in started.values()):
            time.sleep(5)
        exited = [name for name, process in started.items() if process.poll() is not None]
        print(f"Server(s) exited: {', '.join(exited)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        _stop({name: process.pid for name, process in started.items() if process.poll() is None})
        for name in started:
            state.pop(name, None)
        _write_state(state)


if __name__ == "__main__":
    raise SystemExit(main())
