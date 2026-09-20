---
id: endpoints
role: definitions
scope: global
supersedes: [cv-0002, cv-0003]
changed: >
  cv-0003 narrowed to the column's own population and arm (7 cells per run wrote "Not reached" from a
  secondary-population subgroup table where the owner's table leaves the cell empty); an explicit
  time-to-PSA-progression alias added (4 cells per run answered "Not reported" because the paper names the
  endpoint "time to CRPC").
---

# Endpoint identity

- An endpoint keeps its identity when the paper names a variant of it: biochemical, radiographic, clinical or PSA
  progression-free survival is progression-free survival, and a paper that reports only such variants reports that
  endpoint. Give each variant with its label (for example: bPFS X months; rPFS Y months). Never answer "Not
  reported" because the paper's name for the endpoint adds a qualifier.
- Time to PSA progression is the endpoint a TTPSA column asks for. When the paper reports time to castration-resistant
  prostate cancer and defines that event by a PSA rise (alone or together with clinical or radiographic progression),
  that is this endpoint: give the value with the paper's own label (for example: 22.0 months (time to CRPC, PSA rise
  or clinical progression)). Only a time to an event the paper defines without any PSA criterion is a different
  endpoint.

# A median that was not reached

- A median the paper reports as not reached or not estimable **for the column's own population and arm** is a value:
  write "Not reached" or "Not estimable", never "Not reported".
- A not-reached median printed for a different population, analysis set, subgroup or timepoint than the column asks
  for is not this cell's value. When that is the only place the paper shows it, the cell is "Not reported". In
  particular, do not carry a not-reached median out of a subgroup or secondary-population table into a column that
  names no subgroup.
