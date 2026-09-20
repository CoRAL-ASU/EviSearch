"""Check the notes tree parses, is role-gated, and accounts for every active convention."""
import sys

sys.path.insert(0, ".")
from src.evisearch.knowledge import conventions, notes  # noqa: E402

ns = notes.load_notes("all")
print("notes dir:", notes.notes_dir())
print(f"{len(ns)} notes loaded, fingerprint {notes.fingerprint(ns)}\n")
for n in ns:
    print(f"  {n.role:11} {n.scope:6} {n.id:32} supersedes={list(n.supersedes)}")

auditor, agent = notes.load_notes("auditor"), notes.load_notes("agent")
print(f"\nauditor sees {len(auditor)} notes ({', '.join(n.id for n in auditor)})")
print(f"agent sees   {len(agent)} notes")

cov = notes.coverage()
active = {c["id"] for c in conventions.active()}
missing = sorted(active - set(cov))
extra = sorted(set(cov) - active)
print(f"\nactive conventions {len(active)}, claimed by notes {len(cov)}")
print(f"NOT covered by any note: {missing}")
print(f"claimed but not active : {extra}")

print("\n--- scoped delivery check ---")
for cols in (["Region - N (%) | Europe | Treatment"], ["Median OS (mo) | Overall | Treatment"], ["Author"]):
    got = notes.select_for(agent, cols)
    print(f"  batch {cols[0][:44]:44} -> {len(got)} notes: {', '.join(n.id for n in got)}")

print("\n--- rendered length by role ---")
for role in ("agent", "auditor"):
    text = notes.render(notes.load_notes(role))
    print(f"  {role:8} {len(text):6} chars")
