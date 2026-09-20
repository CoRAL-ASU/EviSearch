#!/usr/bin/env python3
"""Append double-blind judgements to the label store.

The store is append-only and the last record for a cell id wins, so a re-judgement supersedes an earlier one without
losing it. Every record carries the scorer and batch so a number can be traced back to who judged it.

  python experiment-analysis/merge_judgements.py judgements.json <batch-tag>

judgements.json: {"judgements": [{"id", "correctness", "completeness", "reason"}, ...]}
`id` must be a cell id from collect_unjudged.py; anything else is refused rather than guessed at.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/mnt/data1/nahuja11_home/EviSearch")
from src.evaluation import claude_scoring as cs  # noqa: E402

ALLOWED = {0.0, 0.25, 0.5, 0.75, 1.0}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tag = sys.argv[2]
    rows = payload.get("judgements") or payload.get("items") or []
    known = {it["id"]: it for it in (payload.get("source") or [])}

    labels = cs.load_labels(cs.labels_path())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    good, bad = [], []
    for r in rows:
        cid = str(r.get("id") or "")
        try:
            corr, comp = float(r["correctness"]), float(r["completeness"])
        except (KeyError, TypeError, ValueError):
            bad.append((cid, "correctness and completeness must be numbers"))
            continue
        if corr not in ALLOWED or comp not in ALLOWED:
            bad.append((cid, f"scores must be one of {sorted(ALLOWED)}, got {corr}/{comp}"))
            continue
        if len(cid) != 16:
            bad.append((cid, "not a cell id"))
            continue
        rec = {
            "id": cid,
            "doc": r.get("doc") or known.get(cid, {}).get("doc", ""),
            "column": r.get("column") or known.get(cid, {}).get("column", ""),
            "category": r.get("category") or known.get(cid, {}).get("category", ""),
            "gold": r.get("gold", known.get(cid, {}).get("gold", "")),
            "pred": r.get("pred", known.get(cid, {}).get("pred", "")),
            "correctness": corr,
            "completeness": comp,
            "reason": str(r.get("reason") or "")[:600],
            "scorer": r.get("scorer") or "claude-opus-5",
            "batch": tag,
            "at": now,
            "supersedes": bool(cid in labels),
        }
        good.append(rec)

    if bad:
        print(f"REFUSED {len(bad)} records:")
        for cid, why in bad[:10]:
            print(f"  {cid or '<no id>'}: {why}")
        if not good:
            return 1
    with open(cs.labels_path(), "a", encoding="utf-8") as fh:
        for rec in good:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    resupers = sum(1 for r in good if r["supersedes"])
    print(f"appended {len(good)} judgements as batch {tag} ({resupers} supersede an earlier judgement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
