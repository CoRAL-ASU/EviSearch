"""Long-running work the web app starts, kept on disk so a reload or a server restart never loses it.

JOBS_DIR/<job_id>.json = {id, kind, status, created, started, ended, by, error, result, ...meta}
  kind      draft_schema | revise_schema | prepare | extract | propose ...
  status    queued -> running -> done | error | cancelled; an in-process job whose server stopped reads "interrupted"

Two ways to run:
  run_thread(kind, fn, **meta)       inside the web server (model calls that take a minute: drafting, revising)
  run_process(kind, cmd, log, **meta) a detached child (`python -m src.evisearch.services.job_runner <id>`) that runs
                                      `cmd`, writes the exit status itself and survives server restarts; cancel() stops
                                      its whole process group.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import runtime_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOOT_ID = uuid.uuid4().hex[:12]  # this server process; thread jobs of an earlier process can't still be running
FINAL = ("done", "error", "cancelled", "interrupted")
_LOCK = threading.Lock()


def jobs_dir() -> Path:
    return runtime_paths.JOBS_DIR


def _path(job_id: str) -> Path:
    if not job_id or not all(c.isalnum() for c in job_id):
        raise ValueError(f"bad job id {job_id!r}")
    return jobs_dir() / f"{job_id}.json"


def _write(job: Dict[str, Any]) -> None:
    path = _path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _read(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_path(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def create(kind: str, by: str = "", **meta: Any) -> Dict[str, Any]:
    job = {"id": uuid.uuid4().hex[:12], "kind": kind, "status": "queued", "created": time.time(), "started": None,
           "ended": None, "by": by, "error": None, "result": None, "boot": BOOT_ID, **meta}
    _write(job)
    return job


def update(job_id: str, **fields: Any) -> Dict[str, Any]:
    with _LOCK:
        job = _read(job_id) or {"id": job_id}
        job.update(fields)
        _write(job)
    return job


def _alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False
    try:  # a zombie child of this server still answers kill(0); it has finished
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as handle:
            return handle.read().split(") ", 1)[1][:1] != "Z"
    except OSError:
        return True


def get(job_id: str) -> Optional[Dict[str, Any]]:
    """The job, with its status corrected for work that died without saying so."""
    job = _read(job_id)
    if not job or job.get("status") in FINAL:
        return job
    if job.get("pid"):  # a runner process: running only while it lives
        if not _alive(job["pid"]):
            job = _read(job_id) or job  # the runner may have written its end status just now
            if job.get("status") not in FINAL:
                job = update(job_id, status="error", error=job.get("error") or "the job process ended without reporting",
                             ended=job.get("ended") or time.time())
    elif job.get("boot") != BOOT_ID and job.get("status") in ("queued", "running"):
        job = update(job_id, status="interrupted", error="the web server restarted while this job ran", ended=time.time())
    return job


def list_jobs(**match: Any) -> List[Dict[str, Any]]:
    """Newest first; `match` keeps jobs whose fields equal the given values (e.g. table=..., kind=...)."""
    folder = jobs_dir()
    out = []
    for path in folder.glob("*.json") if folder.exists() else []:
        job = get(path.stem)
        if job and all(job.get(k) == v for k, v in match.items() if v is not None):
            out.append(job)
    return sorted(out, key=lambda j: j.get("created") or 0, reverse=True)


def run_thread(kind: str, fn: Callable[[], Any], by: str = "", **meta: Any) -> Dict[str, Any]:
    job = create(kind, by=by, **meta)

    def work():
        update(job["id"], status="running", started=time.time())
        try:
            result = fn()
            update(job["id"], status="done", result=result, ended=time.time())
        except Exception as exc:  # reported to the page; a job never kills the server
            update(job["id"], status="error", error=f"{type(exc).__name__}: {exc}", ended=time.time())

    threading.Thread(target=work, daemon=True, name=f"job-{job['id']}").start()
    return job


def run_process(kind: str, cmd: List[str], log: Path, by: str = "", env: Optional[Dict[str, str]] = None, **meta: Any) -> Dict[str, Any]:
    """Start `cmd` under the job runner, detached from the server (its own session / process group)."""
    # "running" from the start: only the runner writes the end status, so this process never overwrites it
    job = create(kind, by=by, cmd=cmd, log=str(log), status="running", started=time.time(), **meta)
    log.parent.mkdir(parents=True, exist_ok=True)
    runner = [sys.executable, "-m", "src.evisearch.services.job_runner", job["id"]]
    child_env = dict(os.environ, **(env or {}), EVISEARCH_JOBS_DIR=str(jobs_dir()))
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(runner, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=child_env,
                                start_new_session=True)
    return update(job["id"], pid=proc.pid, pgid=proc.pid)


def cancel(job_id: str, by: str = "") -> Dict[str, Any]:
    job = get(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.get("status") in FINAL:
        return job
    if not job.get("pgid"):
        raise ValueError("only extraction jobs can be cancelled")
    try:
        os.killpg(int(job["pgid"]), signal.SIGTERM)
    except ProcessLookupError:
        pass
    return update(job_id, status="cancelled", ended=time.time(), cancelled_by=by)


def log_tail(job: Dict[str, Any], lines: int = 60) -> str:
    path = Path(job.get("log") or "")
    if not job.get("log") or not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 64_000))
        text = handle.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])
