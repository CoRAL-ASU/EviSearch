"""Smoke-check arbiter v5: imports, module selection, role-gated knowledge, prompt assembly."""
import os
import sys

sys.path.insert(0, ".")

os.environ["EVISEARCH_KB"] = "notes"
os.environ.setdefault("EVISEARCH_PRESET", "local")

from src.evisearch.services import reconciliation, reconciliation_v5  # noqa: E402
from src.evisearch.services.extraction_rules import kb_format, rules_setting, shared_rules  # noqa: E402
from src.evisearch.pipelines import reconciliation_pipeline as pipe  # noqa: E402

print("v4 version:", reconciliation.RECONCILER_VERSION)
print("v5 version:", reconciliation_v5.RECONCILER_VERSION)
print("kb_format:", kb_format())
print("rules_setting:", rules_setting())

os.environ.pop("EVISEARCH_ARBITER", None)
print("\ndefault arbiter module ->", pipe.arbiter_module().RECONCILER_VERSION)
os.environ["EVISEARCH_ARBITER"] = "v5"
print("EVISEARCH_ARBITER=v5  ->", pipe.arbiter_module().RECONCILER_VERSION)

cols = ["Median OS (mo) | Overall | Treatment", "Region - N (%) | Europe | Control"]
agent_text = shared_rules(columns=cols, role="agent")
auditor_text = shared_rules(columns=cols, role="auditor")
print(f"\nknowledge delivered to agent : {len(agent_text):6} chars")
print(f"knowledge delivered to auditor: {len(auditor_text):6} chars")
only_agent = [n for n in ("figures-and-panels", "tables-and-subgroups", "thresholds-and-arithmetic", "arm-level-values")
              if f"### {n}" in agent_text and f"### {n}" not in auditor_text]
print("extraction notes withheld from the auditor:", only_agent)
shared = [n for n in ("statistics-and-units", "endpoints", "regions") if f"### {n}" in auditor_text]
print("definition notes the auditor does get :", shared)

print("\nv5 prompt sizes:")
print("  FINDINGS_PROMPT ", len(reconciliation_v5.FINDINGS_PROMPT), "chars")
print("  RECONCILE_PROMPT", len(reconciliation_v5.RECONCILE_PROMPT), "chars")
spec = reconciliation_v5.findings_spec(cols)
print("  submit_findings tool:", spec.name, sorted(spec.parameters["properties"]["findings"]["items"]["properties"]))

print("\nv4 prompt still mentions 'few calls':", "few calls" in reconciliation.SYSTEM_PROMPT)
print("v5 findings prompt mentions 'few calls':", "few calls" in reconciliation_v5.FINDINGS_PROMPT)
print("v5 reconcile prompt licenses an empty cell:", "Not reported" in reconciliation_v5.RECONCILE_PROMPT)
