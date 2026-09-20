---
id: arm-level-values
role: extraction
scope: column
columns:
  - Median On-Treatment Duration (mo) | Treatment
  - Median On-Treatment Duration (mo) | Control
  - Median Age (years) | Treatment
  - Median Age (years) | Control
  - Median OS (mo) | Overall | Treatment
  - Median OS (mo) | Overall | Control
  - Median PFS (mo) | Overall | Treatment
  - Median PFS (mo) | Overall | Control
supersedes: [cv-0022]
changed: >
  narrowed. The prohibition on combining subgroup medians is kept, but cv-0022 was also refusing medians the paper
  does print for the arm somewhere other than the main results table - 4 cells per run, and it collided with the rules
  that license assembling subgroup counts. It no longer overrides a printed arm-level value, and the search now
  explicitly includes the discussion.
---

# Arm-level medians

- Do not combine subgroup medians, and do not copy a subgroup median or a pooled across-arms median into an arm's
  cell. Medians cannot be added or averaged.
- But a median the paper states for this arm **anywhere** is a printed arm-level value and belongs in the cell,
  including when it appears in the text, the abstract, a supplementary table, a figure caption, or a discussion
  sentence comparing this trial with another. Look there before answering "Not reported".
- Only when the paper gives this arm's median nowhere - so that the sole route to it would be combining or borrowing
  from subgroups or from the other arm - is the cell "Not reported".
- This note governs medians only. It does not restrict counts and percentages, which may be assembled from printed
  mutually exclusive rows under the threshold and subgroup rules.
