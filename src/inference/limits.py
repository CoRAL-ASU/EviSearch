"""How much this process sends at once: one cap on model calls in flight, shared by every paper, stage and batch.

A run fans out as far as its dependencies allow. The PDF Query Agent and the Search Agent read the paper, never each
other, so they run together; each of their column batches is an independent session over its own columns, so a
stage's batches run together; papers are independent, so a job's papers run together. The Reconciliation Agent reads
both agents' saved answers and waits for them. Nothing here changes what is sent: every session gets the prompt it
would get alone, at temperature 0, so running sessions at the same time changes the schedule, not the results.

The cap is what keeps that fan-out from becoming hundreds of simultaneous requests. Hosted models (a selection that
needs no local vLLM server) take many requests at once, so the defaults fan out fully; a local vLLM server has a
fixed number of sequence slots shared with other tenants, so there the defaults stay at what the published runs used.

  EVISEARCH_MAX_INFLIGHT        model calls in flight in this process      hosted 64, local 8
  EVISEARCH_STAGE_CONCURRENCY   a stage's column batches at once           hosted: all of them, local 3
  EVISEARCH_STAGE_PARALLEL      the two extraction agents at the same time on (0 turns it off)
  EVISEARCH_VERIFIER_WORKERS    attribution checks of one call at once     hosted 16, local 4
  EVISEARCH_PAPERS_AT_ONCE      a job's papers at once (run_benchmark)     hosted: all of the job's, local 2

Queueing for the cap is not model time: ChatModel.chat takes its timings inside the cap, so a stage's model_seconds
still measure the model.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

_lock = threading.Lock()
_semaphore: Optional[threading.BoundedSemaphore] = None
_hosted: Optional[bool] = None

HOSTED = {"max_inflight": 64, "batches": 32, "verifier_workers": 16, "papers": 16}
LOCAL = {"max_inflight": 8, "batches": 3, "verifier_workers": 4, "papers": 2}


def hosted() -> bool:
    """True when the selected models are all served by an API (no local vLLM server is needed)."""
    global _hosted
    if _hosted is None:
        from src.config.config import SELECTION

        _hosted = not SELECTION.servers_needed()
    return _hosted


def _defaults() -> dict:
    return HOSTED if hosted() else LOCAL


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    try:
        return max(int(raw), 1) if raw else None
    except ValueError:
        return None


def max_inflight() -> int:
    return _env_int("EVISEARCH_MAX_INFLIGHT") or _defaults()["max_inflight"]


def batch_concurrency() -> int:
    return _env_int("EVISEARCH_STAGE_CONCURRENCY") or _defaults()["batches"]


def verifier_workers() -> int:
    return _env_int("EVISEARCH_VERIFIER_WORKERS") or _defaults()["verifier_workers"]


def papers_at_once(n_papers: int) -> int:
    return max(1, min(n_papers, _env_int("EVISEARCH_PAPERS_AT_ONCE") or _defaults()["papers"]))


def stage_parallel() -> bool:
    raw = os.getenv("EVISEARCH_STAGE_PARALLEL", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


@contextmanager
def inflight() -> Iterator[None]:
    """Hold one of the process's model-call slots for the duration of one request."""
    global _semaphore
    if _semaphore is None:
        with _lock:
            if _semaphore is None:
                _semaphore = threading.BoundedSemaphore(max_inflight())
    with _semaphore:
        yield


def reset() -> None:
    """Forget the cached selection and cap (tests switch presets and limits)."""
    global _semaphore, _hosted
    with _lock:
        _semaphore, _hosted = None, None
