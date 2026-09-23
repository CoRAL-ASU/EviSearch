"""On-demand hosting: the server stops itself only when unused, with no request in progress and no job running."""
import time

from flask import Flask

from src.inference import limits
from web import idle_stop


def test_the_decision_needs_quiet_no_request_and_no_job():
    assert idle_stop.should_stop(now=1000, last_request=0, in_progress=0, jobs_busy=False, idle_seconds=900)
    assert not idle_stop.should_stop(now=1000, last_request=200, in_progress=0, jobs_busy=False, idle_seconds=900)  # used recently
    assert not idle_stop.should_stop(now=1000, last_request=0, in_progress=1, jobs_busy=False, idle_seconds=900)  # a request runs
    assert not idle_stop.should_stop(now=1000, last_request=0, in_progress=0, jobs_busy=True, idle_seconds=900)  # a job runs


def test_it_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("EVISEARCH_IDLE_STOP_MINUTES", raising=False)
    assert idle_stop.install(Flask("off"), lambda: False, stop=lambda: None) is False


def _app(monkeypatch, busy, stopped, minutes="0.01"):
    monkeypatch.setenv("EVISEARCH_IDLE_STOP_MINUTES", minutes)
    idle_stop._state.update(last=time.monotonic(), in_progress=0)
    app = Flask("on")
    app.add_url_rule("/healthz", "healthz", lambda: "ok")
    app.add_url_rule("/page", "page", lambda: "page")
    assert idle_stop.install(app, lambda: busy["value"], stop=lambda: stopped.append(time.monotonic()))
    return app


def test_health_checks_are_not_use_and_pages_are(monkeypatch):
    app = _app(monkeypatch, {"value": True}, [], minutes="60")
    client = app.test_client()
    before = idle_stop._state["last"]
    time.sleep(0.01)
    client.get("/healthz")
    assert idle_stop._state["last"] == before
    client.get("/page")
    assert idle_stop._state["last"] > before and idle_stop._state["in_progress"] == 0


def test_a_running_job_keeps_it_up_and_it_stops_once_the_job_is_done(monkeypatch):
    busy, stopped = {"value": True}, []
    _app(monkeypatch, busy, stopped)  # 0.6 s idle limit, checked every 0.15 s
    time.sleep(1.2)
    assert not stopped  # idle for twice the limit, but a job is running
    busy["value"] = False
    time.sleep(0.6)
    assert len(stopped) == 1


def test_papers_at_once_can_be_capped(monkeypatch):
    monkeypatch.setenv("EVISEARCH_PAPERS_AT_ONCE", "1")
    assert limits.papers_at_once(10) == 1
    monkeypatch.delenv("EVISEARCH_PAPERS_AT_ONCE")
    assert limits.papers_at_once(1) == 1
