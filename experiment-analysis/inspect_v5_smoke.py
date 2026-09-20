"""Inspect one v5 run's outputs: did phase 1 really answer, are the verdicts filled, can an absence win?"""
import json
import sys
from collections import Counter
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "r4-smoke-v5"
ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")

docs = [d for d in sorted(ROOT.iterdir()) if (d / "runs" / RUN / "reconciliation_agent" / "reconciled_results.json").exists()]
print(f"run {RUN}: {len(docs)} documents\n")
for doc in docs:
    cols = json.loads((doc / "runs" / RUN / "reconciliation_agent" / "reconciled_results.json").read_text())["columns"]
    n = len(cols)
    own = [c for c in cols.values() if isinstance(c.get("own_finding"), dict)]
    own_val = [c for c in own if (c["own_finding"].get("value") or "").strip()]
    own_none = [c for c in own if not (c["own_finding"].get("value") or "").strip() and not c["own_finding"].get("unread")]
    unread = [c for c in own if c["own_finding"].get("unread")]
    print(f"{doc.name[:34]:34} {n} columns")
    print(f"  own_finding present on {len(own)}/{n};  own answered {len(own_val)}, own found nothing {len(own_none)}, unread {len(unread)}")
    print(f"  final_source: {dict(Counter(c.get('final_source') for c in cols.values()))}")
    print(f"  own_verdict : {dict(Counter(c.get('own_verdict') for c in cols.values()))}")
    print(f"  a_verdict   : {dict(Counter(c.get('a_verdict') for c in cols.values()))}")
    print(f"  b_verdict   : {dict(Counter(c.get('b_verdict') for c in cols.values()))}")
    print(f"  decided_by  : {dict(Counter(c.get('decided_by') for c in cols.values()))}")
    empties = [k for k, c in cols.items() if str(c.get("value", "")).strip().lower().startswith("not reported")]
    own_backed = [k for k in empties if not (cols[k].get("own_finding") or {}).get("value")]
    print(f"  shipped 'Not reported' on {len(empties)}; of those, own reading also found nothing: {len(own_backed)}")
    flagged = [k for k in empties if cols[k].get("needs_review")]
    print(f"  of those absences, flagged because an agent had a value: {len(flagged)}")
    # the behaviour v4 could not do at all
    won = [k for k in own_backed if cols[k].get("decided_by") == "own_reading"]
    print(f"  absences accepted on the strength of its own reading (impossible in v4): {len(won)}")
    if won[:3]:
        for k in won[:3]:
            c = cols[k]
            print(f"     {k[:52]:52} looked at {c['own_finding'].get('looked_at')}  review={bool(c.get('needs_review'))}")

    log = doc / "runs" / RUN / "reconciliation_agent" / "verification_logs" / "batch_0_conversation.json"
    if log.exists():
        d = json.loads(log.read_text())
        p1 = d.get("phase1") or {}
        seq1 = [t["name"] for t in (p1.get("tool_calls_sequence") or [])]
        seq2 = [t["name"] for t in (d.get("tool_calls_sequence") or [])]
        print(f"  batch 0 phase 1 tools: {seq1}")
        print(f"  batch 0 phase 2 tools: {seq2}")

