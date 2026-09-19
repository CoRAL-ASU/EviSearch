"""Schema layer: a table schema (column definitions) generated from a spreadsheet of column headers and one example row,
reviewed by a human, locked, and reused for every extraction under it.

- ingest: read the spreadsheet (headers + example rows).
- facets: split each header into what it asks for (statistic, unit, characteristic, arm, subgroup, category).
- grounding: find each example value in the example paper, with the page text around it.
- generator: the schema agent that drafts a definition per column (and questions for the reviewer).
- store: schemas on disk, versions, review edits, lock, and export to the definitions CSV the pipelines read.
"""
