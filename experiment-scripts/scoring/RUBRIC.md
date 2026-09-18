# Scoring rubric

The evaluator_v2 prompts, verbatim. Use the section matching the batch category.

## Category: exact_match

```text
You are evaluating clinical trial data extraction for exact match columns (identifiers, categorical values, binary fields).

Compare Ground Truth (GT) vs Predicted (Pred) values for exact/semantic equivalence.

CRITICAL - Empty / missing (use for BOTH GT and Pred):
- Treat as equivalent to empty: "", "Not reported", "not reported", "not found", "Not found", "not applicable", "N/A", "na", "NaN", "not present", "Not reported", "Not applicable"
- When GT is empty and Pred is any of the above → correctness=1.0, completeness=1.0 (no hallucination)
- When GT is empty and Pred has a substantive value → correctness=0.0, completeness=0.0 (hallucination)
- When GT has a value and Pred is empty → correctness=0.0, completeness=0.0 (missing extraction)

Rules:
- Case-insensitive comparison
- Accept synonyms: "Yes"/"Y", "No"/"N", "Full Pub"/"Full Publication", "Phase 3"/"3.0", etc.
- For identifiers (NCT, Trial Name): must match exactly (ignoring case/whitespace)
- Consider the column definition to understand acceptable variations

For each column, evaluate:
- If values are equivalent (by rules above): correctness=1.0, completeness=1.0
- If values differ: correctness=0.0, completeness=0.0

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## Category: numeric_tolerance

```text
You are evaluating clinical trial data extraction for numeric columns.

Compare Ground Truth (GT) vs Predicted (Pred) values using the column Definition to determine what is required.

CRITICAL RULES:

1) **What's required vs optional (check the Definition)**:
   - If definition says "include count AND percentage" or "N (%)": both the count (N) and the percentage numeric value are REQUIRED. The literal "%" symbol is OPTIONAL—e.g. Pred="103 (55)" has both N and percentage value; do NOT dock for missing "%".
   - If definition says "at X years" or "X-year rate": the timepoint is REQUIRED.
   - If definition says "median" without specifying CI/IQR: only the median number is required; CI, IQR, SD, range, p-values are OPTIONAL context.
   - **Multiple values in GT**: If GT contains multiple distinct required values (e.g. "bPFS 12.9 mo; rPFS 15.3 mo" or "X and Y"), pred must include ALL of them for completeness=1.0. If pred reports only one value and GT has two or more, score correctness based on whether the reported value matches one in GT, but set completeness=0.5 (missing other required value(s)).
   - Default: primary numbers are required; statistical context (CI, IQR, p-values) is optional unless definition explicitly asks for it.

2) **Tolerance**: ±0.1 for absolute values, ±2% relative for percentages.

3) **Correctness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all predicted numbers match GT numbers (within tolerance); extra optional stats (IQR, CI) in pred are fine
   - 0.5 = some predicted numbers match, some don't or contradict GT
   - 0.0 = no predicted numbers match GT, or predicted numbers contradict GT

4) **Completeness** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all REQUIRED numbers from GT are present in pred (check definition to determine what's required)
   - 0.5 = some required numbers present, some missing
   - 0.0 = required numbers missing

5) **Empty handling** (treat as empty for BOTH GT and Pred):
   - Empty-equivalent: "", "Not reported", "not found", "Not found", "not applicable", "N/A", "na", "NaN", "not present"
   - When GT is empty and Pred is empty-equivalent → correctness=1.0, completeness=1.0 (no hallucination)
   - When GT is empty and Pred has a number or substantive value → correctness=0.0, completeness=0.0 (hallucination)
   - When GT has a value and Pred is empty-equivalent → correctness=0.0, completeness=0.0 (missing extraction)
   - When both have substantive values: apply tolerance and required-vs-optional rules above

EXAMPLES:
- GT="35.1 months (95% CI 29.9–43.6)", Pred="35.1 months", Def="median OS" → correctness=1.0 (35.1 matches), completeness=1.0 (CI is optional)
- GT="70% at 5 years", Pred="70%", Def="OS rate at 5 years" → correctness=1.0 (70% matches), completeness=0.5 (missing required timepoint)
- GT="250 (63.6%)", Pred="250", Def="include count and percentage" → correctness=1.0 (N matches), completeness=0.5 (missing percentage value)
- GT="103 (55%)", Pred="103 (55)", Def="include count and percentage" → correctness=1.0, completeness=1.0 (both N and percentage value present; "%" symbol optional)
- GT="250 (63.6%)", Pred="250 (64%)", Def="include count and percentage" → correctness=1.0 (both within tolerance), completeness=1.0 (both present)
- GT="High-volume: 92 (48%) Low-volume: 100 (52%)", Pred="92 (48%)", Def="by volume subgroup" → correctness=1.0 (high-volume matches), completeness=0.5 (missing low-volume)
- GT="bPFS 12.9 mo; rPFS 15.3 mo", Pred="12.9 mo", Def="median PFS (months)" → correctness=1.0 (12.9 matches), completeness=0.5 (missing rPFS 15.3)
- GT="bPFS 12.9 mo; rPFS 15.3 mo", Pred="12.9 mo; 15.3 mo" or "bPFS 12.9; rPFS 15.3", Def="median PFS" → correctness=1.0, completeness=1.0 (both values present)

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## Category: structured_text

```text
You are evaluating clinical trial data extraction for structured text columns (treatments, regimens, endpoints, classifications).

Compare Ground Truth (GT) vs Predicted (Pred) for semantic information content using the column Definition to determine what is required.

CRITICAL RULES:

1) **Correctness = no contradiction** (use only 0.0, 0.5, or 1.0):
   - 1.0 = all information in Pred matches or is compatible with GT; extra correct detail (e.g., drug mechanism, trial name, expanded abbreviations) is fine and does NOT lower the score
   - 0.5 = some predicted info correct, some contradicts GT (mixed: core fact right but extra detail wrong)
   - 0.0 = predicted info contradicts GT or core facts are wrong

2) **Completeness = required facts only** (use only 0.0, 0.5, or 1.0):
   - Use the Definition to identify what information is REQUIRED for this column
   - For "treatment regimen": drug name and combination partner (e.g., ADT) are typically required; dose and schedule are required if definition implies detail (e.g., "describe the regimen") but optional if definition just asks "what treatment"
   - For "endpoint": endpoint name is required; timepoints/thresholds are required only if definition specifies
   - Use medical judgment based on the definition to decide what's required vs optional context
   - 1.0 = all required facts from GT are present in pred
   - 0.5 = some required facts present, some missing
   - 0.0 = required facts missing
   - Do NOT penalize for missing optional context (mechanism, rationale, historical notes, expanded forms)

3) **Empty handling** (treat as empty for BOTH GT and Pred):
   - Empty-equivalent: "", "Not reported", "not found", "Not found", "not applicable", "N/A", "na", "not present"
   - When GT is empty and Pred is empty-equivalent → correctness=1.0, completeness=1.0 (no hallucination)
   - When GT is empty and Pred has substantive text → correctness=0.0, completeness=0.0 (hallucination)
   - When GT has a value and Pred is empty-equivalent → correctness=0.0, completeness=0.0 (missing extraction)

4) **General**:
   - Be lenient with abbreviations (e.g., "ADT" = "Androgen Deprivation Therapy")
   - Accept rephrasing if meaning is preserved

EXAMPLES:
- GT="ADT", Pred="ADT (LHRH agonist for testosterone suppression)", Def="control arm regimen" → correctness=1.0 (extra mechanism is fine, no contradiction), completeness=1.0 (core fact "ADT" present)
- GT="Docetaxel 75 mg/m² every 21 days + ADT", Pred="Docetaxel + ADT", Def="treatment regimen (describe)" → correctness=1.0 (no contradiction), completeness=0.5 (missing dose/schedule which are required by "describe")
- GT="ADT", Pred="ADT with docetaxel", Def="control arm" → correctness=0.5 (ADT correct but extra "docetaxel" contradicts), completeness=1.0 (ADT present)
- GT="Overall survival", Pred="Overall survival at 3 years", Def="primary endpoint" → correctness=1.0 (extra timepoint is fine), completeness=1.0 (endpoint name present)

Columns to evaluate:

For each column, provide your evaluation with reasoning, then output the final scores.
Be thorough in your reasoning but concise.

IMPORTANT: When returning results, use the EXACT column name shown above (without the number prefix).
For example, if the column is shown as "1. Control Arm - N:", you must return column name as "Control Arm - N".
```

## How to return scores

You are scoring one queued batch. Every item is one column of one paper: its definition, the gold value (GT) and a
predicted value (Pred). Apply the rules above for the batch's category to each item on its own. You do not know
which system produced a prediction, and you must not look anything up: judge only GT vs Pred under the rules.

## Conventions (fixed after the pilot; they resolve ambiguities in the rules above, for every system alike)

- C1. Tolerance, by kind of number. We are not looking for exact decimals, but counts are different populations when
  they differ:
  counts (N) must match exactly (654 vs 655: no match; 1305 vs 1306: no match);
  percentages match within ±1 percentage point or ±2% relative, whichever is more lenient (4.3 vs 4.2: match;
  23 vs 22.3: match; 19.3 vs 18.7: match; 63.5 vs 67.5: no match);
  other values (medians, durations, rates in the paper's units) match within ±0.1 or at a different rounding
  (35 vs 35.1: match; 41 vs 41.9: no match).
- C2. For completeness, a required GT number counts as present only if Pred has it within tolerance; a wrong number is
  not "present". A count that matches with a percentage outside tolerance is 0.5 / 0.5 when the Definition requires
  both (see C7 for "count and/or percentage"). When GT gives only a pooled total, per-arm numbers whose sum equals it do
  not make it present; Pred must state the total. When GT lists each arm's number and also their total, and the
  Definition asks for arms separately, the per-arm numbers are what is required and the total is optional.
- C3. "Not reached", "NR (not reached)" and "not estimable" are substantive values, not empty-equivalent: GT empty and
  Pred "Not reached" is 0/0; GT "Not reached" and Pred "Not reached" is 1/1. A bare "NR" in a median-survival or
  time-to-event column means "not reached" (so GT "NR" vs Pred "Not reported" is a miss, 0/0).
- C4. Structured text: a category label implied by the components Pred names counts as present (e.g. GT "Triplet:
  ARPI + ADT + docetaxel" vs Pred "darolutamide plus ADT and docetaxel" is complete).
- C5. Exact match: parenthetical context in GT is optional ("2 arms (A vs B)" vs "2" matches); "Surname et al" formats
  match when the definition allows them.
- C6. In the yes/no columns ("Quality of Life reported", "Reporting by prognostic groups - Y/N | ..."), "No"/"N" and an
  empty value mean the same thing (not reported): GT empty vs Pred "No" is 1/1; GT "No" vs Pred "Not reported" is 1/1;
  "Yes" vs "No" or vs empty is 0/0.
- C7. When the Definition says "count and/or percentage", either the count or the percentage alone is complete (GT
  "143 (36.4%)" vs Pred "143" is 1/1). If Pred gives both and only one is within tolerance, correctness is 0.5 and
  completeness 1.0 (GT "397 (100%)" vs Pred "390 (98.2%)" is 0.5 / 1.0). When it says "count and percentage" or
  "N (%)" without "or", both are required.
  A Definition that says "title or identifier" is satisfied by the identifier alone.

Write a JSON file with exactly this shape, one entry per item id in the batch:

    {"batch": "<batch name>",
     "results": [{"id": "<item id>", "correctness": 0.0 | 0.5 | 1.0, "completeness": 0.0 | 0.5 | 1.0,
                  "reason": "<one short sentence>"}]}
