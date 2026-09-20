#!/usr/bin/env python3
"""Copy Agent A's and Agent B's finished stages from one run into another.

Used to compare two arbiters on identical inputs: the agents are re-run once, and each arbiter reconciles the same
outputs. Without this the two arbiters would also be reading two different samples of the agents, and the difference
between them would not be an arbiter difference.

  python experiment-scripts/copy_agent_stages.py <source-run> <target-run>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evisearch.pipelines import results_store  # noqa: E402

STAGES = ("agent_extractor", "search_agent")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, target = sys.argv[1], sys.argv[2]
    if source == target:
        print("source and target must differ")
        return 2
    root = results_store.RESULTS_ROOT
    copied, skipped = 0, []
    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        src_run = doc_dir / "runs" / source
        if not src_run.is_dir():
            continue
        for stage in STAGES:
            src = src_run / stage
            if not src.is_dir():
                skipped.append(f"{doc_dir.name}/{stage}")
                continue
            dst = doc_dir / "runs" / target / stage
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
            copied += 1
    print(f"copied {copied} stage directories from {source} to {target}")
    if skipped:
        print(f"missing in the source run ({len(skipped)}): {', '.join(skipped[:8])}")
    return 0 if copied and not skipped else (1 if skipped else 0)


if __name__ == "__main__":
    raise SystemExit(main())
