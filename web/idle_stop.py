"""On-demand hosting: the server stops itself once nobody is using it and no job is running.

Fly starts a stopped machine on the next request (auto_start_machines), so an idle demo costs only its storage. Fly's
own auto-stop is not used: its proxy judges a machine by the HTTP traffic passing through it, and an extraction runs in
a background process long after the request that started it has returned, so the proxy would stop a machine that is
busy extracting and kill the job. This module decides with what the app knows instead: requests in progress, the time
since the last one, and whether any job is queued or running. The platform's health checks (/healthz) do not count.

Enabled by EVISEARCH_IDLE_STOP_MINUTES (set in fly.toml); off everywhere else, so a local server never stops itself.
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)
IGNORED_PATHS = {"/healthz"}  # the platform's health checks are not somebody using the demo

_lock = threading.Lock()
_state = {"last": time.monotonic(), "in_progress": 0}


def should_stop(now: float, last_request: float, in_progress: int, jobs_busy: bool, idle_seconds: float) -> bool:
    """Stop only with no request in progress, no job queued or running, and no request for `idle_seconds`."""
    return in_progress == 0 and not jobs_busy and now - last_request >= idle_seconds


def _stop_server() -> None:
    """End the process the platform watches. Under gunicorn this code runs in a worker, whose parent is the master;
    stopping the master ends the machine's main process with exit code 0, and the machine (restart policy
    on-failure) stays stopped until the next request starts it."""
    parent = os.getppid()
    try:
        is_gunicorn = b"gunicorn" in open(f"/proc/{parent}/cmdline", "rb").read()
    except OSError:
        is_gunicorn = False
    os.kill(parent if is_gunicorn else os.getpid(), signal.SIGTERM)


def install(app, jobs_busy: Callable[[], bool], stop: Optional[Callable[[], None]] = None) -> bool:
    """Watch `app` and stop the server after EVISEARCH_IDLE_STOP_MINUTES without use. Returns whether it is enabled."""
    try:
        minutes = float(os.getenv("EVISEARCH_IDLE_STOP_MINUTES") or 0)
    except ValueError:
        minutes = 0
    if minutes <= 0:
        return False
    idle_seconds = minutes * 60
    from flask import g, request

    @app.before_request
    def _request_started():
        if request.path in IGNORED_PATHS:
            return
        g.idle_stop_counted = True
        with _lock:
            _state["in_progress"] += 1
            _state["last"] = time.monotonic()

    @app.teardown_request
    def _request_ended(_exc=None):  # runs after a streamed response has finished, too
        if not g.get("idle_stop_counted"):
            return
        with _lock:
            _state["in_progress"] = max(0, _state["in_progress"] - 1)
            _state["last"] = time.monotonic()

    def watch():
        interval = min(30.0, idle_seconds / 4)
        while True:
            time.sleep(interval)
            with _lock:
                last, in_progress = _state["last"], _state["in_progress"]
            try:
                busy = jobs_busy()
            except Exception:  # noqa: BLE001 - when in doubt, keep running
                busy = True
            if should_stop(time.monotonic(), last, in_progress, busy, idle_seconds):
                log.warning("unused for %g minutes with no job running: stopping the server", minutes)
                (stop or _stop_server)()
                return

    threading.Thread(target=watch, daemon=True, name="idle-stop").start()
    log.info("the server stops itself after %g minutes without use", minutes)
    return True
