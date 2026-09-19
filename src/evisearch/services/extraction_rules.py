"""Column and trial conventions added to the extraction prompts (option `extraction_rules`).

The same text goes to every system that extracts columns — Agent A, Agent B and the parsed-markdown baseline — so a
comparison between systems measures the architecture, not the instructions. v1 comes from the error analysis of the
first full run (E0) on the development papers, revised after a three-lens review; v2 adds named endpoint subtypes after
v1 was measured on Agent A; v3 adds values stated for every patient and subgroup sizes, from the cells where the
markdown baseline beat EviSearch; v4 drops v3's every-patient bullet after it was measured on Agent A and the baseline.
Schema conventions only, with no paper, trial, drug or value names. `none` reproduces the E0 prompts.
"""
from __future__ import annotations

import os
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


# v2 = v1 + named endpoint subtypes. Measured on Agent A (qwen_b2_v1): under v1 the "only a different kind of
# statistic" clause made the model answer "Not reported" for PFS columns when the paper reports PFS only as labelled
# subtypes (biochemical / radiographic PFS), which the E0 prompts had right (6 dev cells on one paper).
RULES["v2"] = RULES["v1"].replace(
    """  asked), answer "Not reported".""",
    """  asked), answer "Not reported". A named subtype of the endpoint the column asks for (for example biochemical,
  radiographic or clinical progression-free survival for a PFS column) is that endpoint, not a different statistic:
  report each labelled subtype (for example "bPFS X months; rPFS Y months").""",
)
assert RULES["v2"] != RULES["v1"]

# v3 = v1 + three bullets.
# (1) Named endpoint variants as a bullet of their own. The v2 sentence did not work: on Agent A (qwen_b2_v2) the model
#     still answered "Not reported" for all 6 PFS cells of the paper that reports only bPFS/rPFS ("PFS ... is distinct
#     from bPFS or rPFS"), because the sentence sat inside the clause that ends in "answer Not reported".
# (2), (3) From the development cells where the markdown baseline beat EviSearch in both E0 runs, Agent A answered
#     "Not reported" when the paper says every patient in an arm received a treatment but prints no count, and when a
#     characteristic is missing from the baseline table but a subgroup analysis gives each subgroup's patients per arm
#     ("events/N"), which Agent A read as event counts. The zero clause covers treatments only: the benchmark leaves
#     other categories of a characteristic empty rather than 0. The one-country example was added after a single-paper
#     check (run archived as qwen_b2_v3probe), where the model called a one-country trial's region "not explicitly
#     reported".
RULES["v3"] = RULES["v1"].replace(
    """  asked), answer "Not reported".
""",
    """  asked), answer "Not reported".
- An endpoint keeps its identity when the paper names a variant of it: biochemical, radiographic, clinical or PSA
  progression-free survival is progression-free survival, and a paper that reports only such variants reports that
  endpoint. Give each variant with its label (for example "bPFS X months; rPFS Y months"). Never answer "Not
  reported" because the paper's name for the endpoint adds a qualifier.
""",
    1,
).replace(
    """  column.
""",
    """  column.
- A value the paper states for every patient is reported even without a printed count: when the paper says that all
  patients in an arm received a treatment or share a characteristic (by design, eligibility or allocation, for example
  a trial that enrolled patients in one country only), give the arm size with 100%. When an arm by design received no
  such treatment, give 0 (0%).
- Counts of patients with a characteristic can come from a subgroup analysis: when the baseline table does not list
  the characteristic but a subgroup forest plot or table gives each subgroup's patients per arm (for example the N in
  "events/N"), that N is the number of patients with that characteristic in that arm; give it as the count.
""",
    1,
)
assert RULES["v3"].count("\n- ") == RULES["v1"].count("\n- ") + 3

# v4 = v3 without the every-patient bullet. Measured on the development papers (Claude-scored against the gold): on
# Agent A (6 papers) the bullet gained 3.5 cells and lost 14, on the markdown baseline (7 papers) it gained 4.5 and lost
# 8. The losses are design-implied values the benchmark leaves empty (other regions "0 (0%)" in a one-country trial,
# mode of metastases 100% / 0% in an all-synchronous trial, prior local therapy 0 (0%)); the model applied the zero
# clause to characteristics although the bullet limits it to treatments. The gold is not consistent about such values
# (it fills region and docetaxel ones), so the bullet is dropped rather than tuned to it.
# Also unquoted: the endpoint-variant example. Agent A copied the quoted example into the reasoning string of its JSON
# reply, where the raw quote ended the string and the constrained reply looped until the token limit (the same batch,
# twice, at temperature 0).
_EVERY_PATIENT = RULES["v3"][RULES["v3"].index("- A value the paper states for every patient"):RULES["v3"].index("- Counts of patients")]
_QUOTED_EXAMPLE = '(for example "bPFS X months; rPFS Y months")'
RULES["v4"] = RULES["v3"].replace(_EVERY_PATIENT, "", 1).replace(_QUOTED_EXAMPLE, "(for example: bPFS X months; rPFS Y months)", 1)
assert RULES["v4"].count("\n- ") == RULES["v1"].count("\n- ") + 2 and "every patient" not in RULES["v4"]
assert _QUOTED_EXAMPLE in RULES["v3"] and _QUOTED_EXAMPLE not in RULES["v4"]

# v5 = v4 + three bullets against over-strictness. From the development cells where the markdown baseline beat E1
# (reconciler v3 on the v4 agents): both agents answered "Not reported" because the paper gave a median in years where the
# column asks for months, because the paper's subgroups carry other labels and criteria than the column's (10 cells on
# one paper), and because the paper states that an arm's regimen includes a treatment but prints no count (4.5 cells on
# four papers). The baseline reports all of these. The treatment bullet is limited to what the protocol gives an arm:
# v3's broader every-patient bullet was applied to characteristics and eligibility, where the benchmark leaves cells empty.
RULES["v5"] = RULES["v4"].replace(
    """- Total-participant""",
    """- Report a value in the column's unit, converting when the paper uses another unit (for example a median in years
  for a column in months: give both, X years, Y months). A different unit is never a reason for "Not reported".
- A subgroup column is answered from the paper's subgroup that corresponds to it, also when the paper names or defines
  it differently (for example a split by extent of disease under other labels or criteria): give the value with the
  paper's own label.
- When the trial design gives a treatment to every patient in an arm (it defines the arm or is part of the arm's
  protocol regimen), that arm's count for the treatment is the arm size with 100%; when the arm's regimen excludes it,
  0 (0%). This covers treatments the protocol assigns, not patient characteristics or eligibility criteria.
- Total-participant""",
    1,
)
assert RULES["v5"].count("\n- ") == RULES["v4"].count("\n- ") + 3


def rules_version() -> str:
    from src.config.config import SELECTION

    return SELECTION.option("extraction_rules")


def knowledge_base_on() -> bool:
    """EVISEARCH_KB=on: the prompts get the conventions knowledge base (KNOWLEDGE_DIR) instead of a fixed rules text."""
    return os.getenv("EVISEARCH_KB", "").strip().lower() in {"1", "on", "true", "yes"}


def shared_rules(version: Optional[str] = None) -> str:
    if version is None and knowledge_base_on():
        from src.evisearch.knowledge import conventions

        return conventions.render()
    return RULES[version or rules_version()]


def rules_setting(version: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Run-settings entry for resume and reuse checks. `none` is recorded as None, which also matches results saved
    before the option existed (no key), while v1 and none results can never be mixed in either direction. With the
    knowledge base on, its fingerprint is recorded instead, so results made with different conventions never mix."""
    if version is None and knowledge_base_on():
        from src.evisearch.knowledge import conventions

        return {"extraction_rules": f"kb:{conventions.fingerprint()}"}
    version = version or rules_version()
    return {"extraction_rules": None if version == "none" else version}
