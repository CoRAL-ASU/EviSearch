"""Did any tool call in a run come back as a swallowed error? tool_loop turns an exception into {"error": ...}, so a
broken reader looks like a quiet loss of answers rather than a failure.

  python experiment-analysis/check_tool_errors.py <run> [doc-prefix]
"""
import glob
import json
import sys
from collections import Counter

RUN = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
ROOT = "/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs/results"

pattern = f"{ROOT}/{PREFIX}*/runs/{RUN}/reconciliation_agent/verification_logs/*_conversation.json"
files = sorted(glob.glob(pattern))
print(f"{len(files)} batch logs under run {RUN}" + (f" for {PREFIX}*" if PREFIX else ""))

total = errors = 0
kinds: Counter = Counter()
examples = []
for path in files:
    data = json.loads(open(path).read())
    for block in (data.get("phase1") or {}, data):
        for message in block.get("conversation") or []:
            if message.get("role") != "tool":
                continue
            total += 1
            response = message.get("response")
            text = ""
            if isinstance(response, dict) and response.get("error"):
                text = str(response["error"])
            elif isinstance(response, str) and "error" in response.lower()[:200]:
                text = response[:200]
            if text:
                errors += 1
                kinds[text.split(":")[0][:40]] += 1
                if len(examples) < 6:
                    examples.append((path.split("/")[-1], message.get("name", "?"), text[:110]))

print(f"tool results inspected: {total}")
print(f"with an error field:    {errors}")
if kinds:
    print("kinds:", dict(kinds))
    for name, tool, text in examples:
        print(f"  {name} [{tool}] {text}")
else:
    print("no swallowed tool errors found")
