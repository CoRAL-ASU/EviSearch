"""Column and trial conventions added to the extraction prompts (option `extraction_rules`).

The same text goes to every system that extracts columns — Agent A, Agent B and the parsed-markdown baseline — so a
comparison between systems measures the architecture, not the instructions. v1 comes from the error analysis of the
first full run (E0) on the development papers, revised after a three-lens review: schema conventions only, with no
paper, trial, drug or value names. `none` reproduces the E0 prompts.
"""
from __future__ import annotations

from typing import Dict, Optional

RULES: Dict[str, str] = {
    "none": "",
    "v1": """

COLUMN AND TRIAL CONVENTIONS (apply to every column):
- The column name says which statistic goes in the cell: "Rate (%)" is a percentage of patients that the paper
  states (give its timepoint; do not compute a rate from counts or medians), "N (%)" is a count and/or its percentage
  (give whichever the paper reports if it gives only one), "(mo)" is a duration in months, "N" is a count. When the
  definition text asks for a different statistic than the column name, follow the column name. If the paper reports
  only a different kind of statistic for the column (for example only a hazard ratio or a median where a Rate (%) is
  asked), answer "Not reported".
- A median that was not reached or is not estimable is a value: write "Not reached" or "Not estimable", never
  "Not reported".
- A per-arm column (its name contains "| Treatment" or "| Control") holds each arm's own value; when several arms fit,
  list each one. Never put a between-arm statistic (hazard ratio, odds ratio, difference, p-value) in a per-arm
  column.
- Total-participant and arm-size counts are the numbers randomised into the population this paper reports on (for
  example only the metastatic patients when the paper analyses that cohort of a larger trial), not a safety,
  per-protocol or evaluable subset of it, unless the column asks for analysed patients.
- Thresholds ("grade 3 or higher", ">=X"): use a total the paper prints for that threshold (a "grade >=3", "grade 3
  or worse" or "grade 3-5" row or sentence) as it is. When one table gives mutually exclusive worst-grade rows, add the
  rows at or above the threshold, grade 5 included. Never add a separately reported fatal or grade 5 count to a
  printed total.
- Add counts only. Never add rows for different event types, and never add or average rates, medians or durations.
- A paper is a follow-up when it reports further, updated, long-term, post hoc or secondary analyses of a trial whose
  primary results were published earlier (for example "as previously reported"), whatever its article type. A post
  hoc table inside the trial's primary report does not make that report a follow-up.
- The add-on treatment is the agent or agents added to the shared backbone in the experimental arm(s), not the
  backbone itself.""",
}


def rules_version() -> str:
    from src.config.config import SELECTION

    return SELECTION.option("extraction_rules")


def shared_rules(version: Optional[str] = None) -> str:
    return RULES[version or rules_version()]


def rules_setting(version: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Run-settings entry for resume and reuse checks. `none` is recorded as None, which also matches results saved
    before the option existed (no key), while v1 and none results can never be mixed in either direction."""
    version = version or rules_version()
    return {"extraction_rules": None if version == "none" else version}
