"""Where every run the paper reports lives.

  new_pipeline_outputs/results/     the system's runs: the configuration the web app runs, and the runs it shows
  new_pipeline_outputs/paper_runs/  runs the paper reports that are not the system: the single-pass baseline
                                    (Table 1), the agreement-gated admission variant (Appendix B), and the
                                    schema-alignment ladder run without the knowledge base (Table 2)

The ladder runs (ladder-*) hold the extraction agents' outputs of the schema-alignment runs, copied unchanged, and the
current Reconciliation Agent's reconciliation of them, so every row of Table 2 reports the system as it is. Their
check_run.log fails on the copied stages' metadata, which names the run that produced them; ladder-v4-r1 and
ladder-v4-r2 also record three verifier calls cut off at max_tokens, whose claims count as not verified.

Both use the same layout, <paper>/runs/<run>/<stage>/, so any script can score a run from either with `root_of(run)`.
The system runs' agent stages were produced once and reused by their reconciliation stage (see each run header's
`stages_from`); the agreement-gated variant reconciles the same agent outputs, byte for byte.
"""
from pathlib import Path

OUT = Path("/mnt/data1/nahuja11_home/EviSearch/new_pipeline_outputs")
RESULTS, ARCHIVE = OUT / "results", OUT / "paper_runs"

SCHEMA = "schema-mhspc-trials-20260919020503"
SYSTEM_RUNS = (f"{SCHEMA}-v4", f"{SCHEMA}-v4-r2")
BASELINE_RUNS = ("r4-notes-b1", "r4-notes-b2")
AGREEMENT_GATE_RUNS = ("r4-guard-r1", "r4-guard-r2")
# Table 2: auto-drafted schema, schema after two and after three review rounds (knowledge base off), final + knowledge
LADDER = [("draft", ("ladder-v0draft-r1", "ladder-v0draft-r2")),
          ("rev", ("ladder-v3-r1", "ladder-v3-r2")),
          ("revfour", ("ladder-v4-r1", "ladder-v4-r2")),
          ("kb", SYSTEM_RUNS)]


def root_of(run: str) -> Path:
    return ARCHIVE if next(ARCHIVE.glob(f"*/runs/{run}"), None) else RESULTS
