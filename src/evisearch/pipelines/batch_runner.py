"""Run a stage's column batches at the same time instead of one after another (and, for a runner that asks, say whether
a document's independent stages may overlap too: `stage_parallel`).

Why this changes nothing but the schedule. Each batch is an independent session over a disjoint set of columns, at
temperature 0: the Reconciliation Agent's batch k reads the two agents' answers for its own columns only. Running
batches at the same time therefore sends the same prompts and gets the same replies - which is what separates this
from shortening the generated text, which would change what the model writes and therefore what the table says.

How far it fans out is set in src/inference/limits.py: every batch at once on hosted models, three at a time on a
local vLLM server, with one cap on model calls in flight across the whole process. The accumulator callback is called
under a lock, so incremental saving keeps working and a crash still leaves the finished batches on disk.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


def stage_concurrency(default: Optional[int] = None) -> int:
    """How many of a stage's batches run at the same time (EVISEARCH_STAGE_CONCURRENCY, else limits' default)."""
    from src.inference import limits

    if default is not None and not os.getenv("EVISEARCH_STAGE_CONCURRENCY", "").strip():
        return max(int(default), 1)
    return limits.batch_concurrency()


def stage_parallel(default: Optional[bool] = None) -> bool:
    """Whether a document's independent stages run at the same time (the two extraction agents read the paper, never
    each other). On unless EVISEARCH_STAGE_PARALLEL turns it off."""
    from src.inference import limits

    if default is not None and not os.getenv("EVISEARCH_STAGE_PARALLEL", "").strip():
        return bool(default)
    return limits.stage_parallel()


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
