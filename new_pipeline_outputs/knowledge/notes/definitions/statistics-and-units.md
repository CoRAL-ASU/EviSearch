---
id: statistics-and-units
role: definitions
scope: global
supersedes: [cv-0001, cv-0004, cv-0006, cv-0009]
changed: none - wording carried over verbatim from the seed conventions
---

# Which statistic a cell holds, and in which unit

- The column name says which statistic goes in the cell: "Rate (%)" is a percentage of patients that the paper
  states (give its timepoint; do not compute a rate from counts or medians), "N (%)" is a count and/or its percentage
  (give whichever the paper reports if it gives only one), "(mo)" is a duration in months, "N" is a count. When the
  definition text asks for a different statistic than the column name, follow the column name. If the paper reports
  only a different kind of statistic for the column (for example only a hazard ratio or a median where a Rate (%) is
  asked), answer "Not reported".
- A per-arm column (its name contains "| Treatment" or "| Control") holds each arm's own value; when several arms fit,
  list each one. Never put a between-arm statistic (hazard ratio, odds ratio, difference, p-value) in a per-arm
  column.
- Report a value in the column's unit, converting when the paper uses another unit (for example a median in years
  for a column in months: give both, X years, Y months). A different unit is never a reason for "Not reported".
- Total-participant and arm-size counts are the numbers randomised into the population this paper reports on (for
  example only the metastatic patients when the paper analyses that cohort of a larger trial), not a safety,
  per-protocol or evaluable subset of it, unless the column asks for analysed patients.
