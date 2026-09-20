"""Run a stage's column batches at the same time instead of one after another (and, for a runner that asks, say whether
a document's independent stages may overlap too: `stage_parallel`).

Why this is free. Measured on R3, the pipeline is decode-bound and nothing else: a stage's wall clock is its output
tokens divided by the model's generation speed (Agent A, 1,940 output tokens per batch at 52 tok/s = 37 s per batch,
and `model_seconds` 409.7 against `duration_s` 412.4 - 99.4% of the stage is inside model calls). The prompts are not
the problem: 85% of Agent A's 49k-token input is cached.

Each batch is an independent session over a disjoint set of columns, at temperature 0. So running batches at the same
time changes the schedule and nothing else - same prompts, same replies, same results. That is what separates this from
shortening the generated text, which would change what the model writes and therefore what the table says.

Off by default (`EVISEARCH_STAGE_CONCURRENCY=1`), so every run up to R4 reproduces exactly. The accumulator callback is
called under a lock, so incremental saving keeps working and a crash still leaves the finished batches on disk.

The server is the ceiling, not this: one vLLM instance with `--max-num-seqs 8`, shared with other tenants. Total
in-flight requests are (documents in parallel) x (batches in parallel), so raising both at once oversubscribes the
queue. vLLM queues gracefully rather than failing, but past the slot count the gain flattens.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CONCURRENCY = 1


def stage_concurrency(default: int = DEFAULT_CONCURRENCY) -> int:
    """EVISEARCH_STAGE_CONCURRENCY: how many of a stage's batches run at the same time. 1 reproduces the serial runs."""
    raw = os.getenv("EVISEARCH_STAGE_CONCURRENCY", "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 1)
    except ValueError:
        return default


def stage_parallel(default: bool = False) -> bool:
    """EVISEARCH_STAGE_PARALLEL: whether a document's independent stages run at the same time instead of one after
    another (Arm A reads the paper, Arm B queries the embeddings; neither reads the other's results). Off reproduces the
    serial runs, and an unrecognised value stays off rather than crashing a run.

    This multiplies with everything else in flight: requests on the server are (documents in parallel) x (stages in
    parallel) x (batches in parallel), and one vLLM instance with --max-num-seqs 8 is the ceiling.
    """
    raw = os.getenv("EVISEARCH_STAGE_PARALLEL", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"", "0", "false", "no", "off"}:
        return False if raw else default
    return default


def run_batches(
    items: Sequence[Tuple[int, Any]],
    work: Callable[[int, Any], Any],
    accumulate: Callable[[int, Any, Any], None],
    concurrency: Optional[int] = None,
) -> None:
    """Run `work(index, batch)` for every item, then `accumulate(index, batch, payload)` under a lock.

    `work` runs in a worker thread and must not touch shared state; `accumulate` is serialized, in completion order,
    and is where results are merged and written. An exception in `work` propagates once every other batch has been
    given the chance to finish, so one bad batch does not discard the others' results.
    """
    workers = min(concurrency if concurrency is not None else stage_concurrency(), len(items)) or 1
    if workers == 1:
        for index, batch in items:
            accumulate(index, batch, work(index, batch))
        return

    lock = threading.Lock()
    failures: List[BaseException] = []

    def one(item: Tuple[int, Any]) -> None:
        index, batch = item
        try:
            payload = work(index, batch)
        except BaseException as exc:  # noqa: BLE001 - re-raised after the others finish
            with lock:
                failures.append(exc)
            return
        with lock:
            accumulate(index, batch, payload)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch") as pool:
        list(pool.map(one, items))
    if failures:
        raise failures[0]


def numbered(batches: Iterable[Any], start: int = 0) -> List[Tuple[int, Any]]:
    return list(enumerate(batches, start))
