"""Show exactly what the arbiter receives for one batch, and how big each part is."""
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results")
RUN = sys.argv[1] if len(sys.argv) > 1 else "schema-mhspc-trials-20260919020503-v4"
BATCH = sys.argv[2] if len(sys.argv) > 2 else "batch_0"

log = next((p for p in sorted(ROOT.glob(f"*/runs/{RUN}/reconciliation_agent/verification_logs/{BATCH}_conversation.json"))), None)
if log is None:
    print("no conversation log found")
    raise SystemExit(1)
d = json.loads(log.read_text())
doc = log.parts[log.parts.index("results") + 1]
conv = d.get("conversation") or []
user = next((m for m in conv if m.get("role") == "user"), None)
text = user.get("content") or ""

print(f"document: {doc}")
print(f"log: {log.name}\n")

# the per-column blocks
blocks = text.split("\n---\n")
head, cols = blocks[0], blocks[1:]
print(f"=== SIZES (characters; ~4 chars per token) ===")
print(f"  preamble                    {len(head):7,}")
print(f"  {len(cols)} column blocks{'':14} {sum(len(b) for b in cols):7,}")
print(f"  mean per column block       {sum(len(b) for b in cols)//max(len(cols),1):7,}")
print(f"  largest column block        {max((len(b) for b in cols), default=0):7,}")
print(f"  WHOLE user message          {len(text):7,}  (~{len(text)//4:,} tokens)")

# reported usage for the first call of this batch
calls = d.get("calls") or []
if calls:
    c0 = calls[0]
    u = c0.get("usage") or c0
    print(f"\n=== what the server saw on the first call of this batch ===")
    for k in ("input_tokens", "cached_input_tokens", "output_tokens", "input_images"):
        if k in u:
            print(f"  {k:22} {u[k]:>9,}")

print(f"\n=== VERBATIM: the smallest and the largest column block ===")
ordered = sorted(cols, key=len)
for label, block in (("SMALLEST", ordered[0]), ("LARGEST", ordered[-1])):
    print(f"\n----- {label} ({len(block):,} chars) -----")
    print(block[:2600].rstrip())
    if len(block) > 2600:
        print(f"... [{len(block)-2600:,} more chars]")
