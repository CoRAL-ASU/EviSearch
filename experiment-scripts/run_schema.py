#!/usr/bin/env python3
"""
Extract papers under a locked schema version with the EviSearch pipeline: the three agents read that version's
definitions and the knowledge notes, frozen for the run, and results go to the run schema-<id>-v<N>. This is what the
web app runs.

  python experiment-scripts/run_schema.py --schema <id> --docs "<doc>,<doc>"

Two reproductions of comparisons reported in the paper, not offered in the web app:
  --system B1   the single-pass baseline (one call per definition group over the parsed text)
  --kb off      the knowledge notes replaced by the fixed extraction guidelines (Table 2's schema ladder)

Runs experiment-scripts/run_benchmark.py in a child process with EVISEARCH_DEFINITIONS_CSV pointing at the version's CSV,
so nothing process-global leaks into a web server. Scoring still uses the hand-written definitions (the gold table's),
which the definitions override never changes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evisearch.schema import store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", required=True, help="schema id (see /schema or new_pipeline_outputs/schemas)")
    parser.add_argument("--version", type=int, help="locked version (default: the latest locked one)")
    parser.add_argument("--system", default="E", choices=["E", "B1"], help="E = EviSearch (default); B1 = the baseline")
    parser.add_argument("--docs", required=True)
    parser.add_argument("--kb", default="on", choices=["on", "off"],
                        help="on (default): the knowledge notes, frozen for the run; off: the fixed extraction guidelines")
    parser.add_argument("--run", help="run name (default: schema-<id>-v<N>, with -kboff / -b1 suffixes when they apply)")
    parser.add_argument("--parallel", type=int, default=0, help="papers at once (default: run_benchmark's)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-warnings", action="store_true", help="passed to run_benchmark (see there)")
    args = parser.parse_args()

    schema = store.load(args.schema)
    versions = schema.get("locked_versions", [])
    version = args.version or (max(versions) if versions else None)
    if version is None or version not in versions:
        parser.error(f"schema {args.schema} has no locked version {args.version or ''} (locked: {versions})")
    csv_path = store.schema_dir(args.schema) / "versions" / f"v{version}.csv"
    run = args.run or store.run_name(args.schema, version) + ("" if args.kb == "on" else "-kboff") + \
        ("" if args.system == "E" else f"-{args.system.lower()}")

    env = dict(os.environ, EVISEARCH_DEFINITIONS_CSV=str(csv_path), EVISEARCH_KB=args.kb)
    env.setdefault("EVISEARCH_EXTRACTION_RULES", "v5")  # the fixed guidelines, read only when the knowledge base is off
    snapshot = None
    if args.kb == "on":  # the run reads the notes as they are now, whatever anyone edits while it runs
        from src.evisearch.knowledge import notes

        snapshot = notes.snapshot()
        env["EVISEARCH_KB_NOTES_SNAPSHOT"] = str(snapshot)
        print(f"[run_schema] knowledge notes frozen for this run: {snapshot}", flush=True)
    cmd = [sys.executable, str(ROOT / "experiment-scripts" / "run_benchmark.py"), "--system", args.system, "--docs", args.docs,
           "--run", run] + (["--parallel", str(args.parallel)] if args.parallel else [])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.check_warnings:
        cmd.append("--check-warnings")
    print(f"[run_schema] schema {args.schema} v{version} ({csv_path}) kb={args.kb} -> run {run}", flush=True)
    if args.dry_run:  # nothing runs, so nothing goes into the feedback log
        return subprocess.call(cmd, cwd=ROOT, env=env)
    store.record_event("extraction_start", args.schema, version=version, run=run, system=args.system, docs=args.docs, kb=args.kb,
                       kb_snapshot=snapshot.name if snapshot else None)
    code = subprocess.call(cmd, cwd=ROOT, env=env)
    store.record_event("extraction_end", args.schema, version=version, run=run, exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
