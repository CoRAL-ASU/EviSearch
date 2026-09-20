#!/usr/bin/env python3
"""Run only the reconciliation stage, for every gold document of the current run (EVISEARCH_RUN).

Agent A and Agent B must already be in the run - copy them from a finished run with copy_agent_stages.py. This is how
two arbiters are compared on identical agent outputs.

  EVISEARCH_RUN=<run> EVISEARCH_ARBITER=v5 python experiment-scripts/reconcile_run.py --parallel 2
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import claude_scoring as cs  # noqa: E402
from src.evisearch.pipelines import results_store  # noqa: E402
from src.evisearch.pipelines.reconciliation_pipeline import arbiter_module, run_reconciliation_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--docs", default="all")
    args = parser.parse_args()

    run = results_store.current_run()
    if not run:
        parser.error("set EVISEARCH_RUN to the run to reconcile")
    docs = cs.select_docs(args.docs, cs.load_gold())
    print(f"[reconcile] run {run}, arbiter {arbiter_module().RECONCILER_VERSION}, {len(docs)} documents", flush=True)

    started = time.time()
    failures = []

    def one(doc_id: str):
        began = time.time()
        result = run_reconciliation_pipeline(doc_id)
        note = result.get("error") or f"{len(result.get('columns') or {})} columns"
        print(f"[reconcile] {doc_id}: {note} ({time.time() - began:.0f}s)", flush=True)
        if result.get("error"):
            failures.append((doc_id, result["error"]))

    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as pool:
        list(pool.map(one, docs))

    print(f"[reconcile] {len(docs) - len(failures)}/{len(docs)} documents ran in {(time.time() - started) / 60:.1f} min", flush=True)
    for doc_id, error in failures:
        print(f"[reconcile] FAILED {doc_id}: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
