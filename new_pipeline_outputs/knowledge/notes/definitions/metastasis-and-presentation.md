---
id: metastasis-and-presentation
role: definitions
scope: family
family: Mode of metastases
also_families: [Metastases - N (%)]
supersedes: [cv-0015, cv-0019, cv-0021]
changed: >
  cv-0019 rewritten. Its prior-local-therapy equivalence was the single worst-scoring rule in the knowledge base -
  4 cells per run, identically in both runs: the arbiter summed "primary radiation" and "prostatectomy" rows into a
  metachronous count, showed correct arithmetic, cited the rule, and scored zero every time. Prior local therapy is
  not a statement about how the disease presented. Its "sum the counts from those subgroups" clause is also removed:
  it manufactured arm totals the papers never print. cv-0021's M1/M0 mapping is kept but limited to the count and
  rate columns, so it no longer collides with the yes/no reporting columns.
---

# Synchronous and metachronous presentation

- Synchronous means the patient had metastatic disease at initial diagnosis (de novo). Metachronous means the
  metastases appeared later, after non-metastatic disease.
- Answer these columns **only** from the paper's own reporting of how the disease presented: the words de novo,
  synchronous or metachronous, or the stage at initial diagnosis (see the M1/M0 mapping below).
- Prior local therapy is **not** evidence of presentation, and the absence of prior local therapy is not evidence of
  de novo disease. A patient may receive prostatectomy or radiotherapy and still have presented with metastases, and
  many patients who presented de novo receive local therapy afterwards. Never derive a synchronous or metachronous
  count from prostatectomy, primary radiation or "no local therapy" rows.
- When the paper reports only prior local therapy, or reports presentation for no population the column covers, the
  cell is "Not reported". Do not add up subgroup rows to manufacture an arm total the paper does not print.

# Stage at initial diagnosis (count and rate columns only)

- In the count and rate columns for presentation, map "M1" or "distant metastasis at initial diagnosis" to the
  Synchronous columns and "M0" (with later metastasis) to the Metachronous columns; exclude "MX" or "not assessed"
  rows from the count and percentage, and keep the paper's original label in parentheses.
- This mapping does not apply to the yes/no reporting columns, which ask whether the paper reports outcomes for the
  subgroup - see the prognostic-group-reporting note.

# Metastatic sites

- Map M1 substages to specific site columns: assign M1a (non-regional lymph nodes only) to Nodal, M1b (bone, with or
  without nodes) to Bone, and M1c (visceral) to Liver. Set Lungs to "Not reported" if only M1 substages are provided.
