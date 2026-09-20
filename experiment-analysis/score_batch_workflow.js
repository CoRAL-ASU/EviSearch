export const meta = {
  name: 'score-unjudged-cells',
  description: 'Judge unjudged extraction cells double-blind against gold, one agent per document',
  phases: [{ title: 'Judge' }, { title: 'Adjudicate' }],
}

// args: { file: "<path to unjudged.json>", tag: "<batch tag>" }
const FILE = (args && args.file) || '/mnt/data1/nahuja11/dotfiles/.claude/jobs/3397edac/tmp/unjudged.json'
const TAG = (args && args.tag) || 'r4'
const WT = '/mnt/data1/nahuja11_home/EviSearch/.claude/worktrees/kb-notes-arbiter-v2'

const BASE = `
You are scoring clinical-trial table extraction, double blind.

THE RUBRIC IS AUTHORITATIVE AND YOU MUST READ IT FIRST:
  ${WT}/experiment-scripts/scoring/RUBRIC.md
It holds the evaluator_v2 prompts per category (exact_match, numeric_tolerance, structured_text) and conventions
C1-C7, which settle the ambiguous cases. Apply the section matching each item's own "category" field.

The items to judge are in ${FILE}, under "items". Each item has: id, doc, column, category, definition, gold, pred.
Read that file and select only the items for the document named below.

RULES OF THE EXERCISE:
- Judge ONLY gold (GT) against pred, under the rubric. You do not know which system produced a prediction, you must
  not try to infer it, and you must not open the paper or look anything up. The prediction is right or wrong against
  gold, full stop.
- correctness and completeness are each exactly one of 0.0, 0.5, 1.0.
- Apply the conventions exactly. The ones that bite most often: C1 counts must match exactly but percentages have
  tolerance; C3 "Not reached" is a value, not empty; C6 in yes/no columns "No" and empty mean the same thing;
  C7 "count and/or percentage" is satisfied by either alone, "count and percentage" needs both.
- Give one short sentence of reasoning naming the rule you applied.
- Return one entry per item you were given. Do not invent ids and do not skip items.`

const SCHEMA = {
  type: 'object',
  properties: {
    doc: { type: 'string' },
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          column: { type: 'string' },
          correctness: { type: 'number' },
          completeness: { type: 'number' },
          reason: { type: 'string' },
          uncertain: { type: 'boolean', description: 'true when the rubric genuinely does not settle this item' },
        },
        required: ['id', 'correctness', 'completeness', 'reason'],
      },
    },
  },
  required: ['doc', 'results'],
}

phase('Judge')
const DOCS = (args && args.docs) || []
if (!DOCS.length) throw new Error('pass args.docs: the list of document ids to judge')

const judged = await parallel(DOCS.map(doc => () => agent(`${BASE}

YOUR DOCUMENT: ${doc}
Judge every item in ${FILE} whose "doc" equals exactly that string. Report how many you found and judged.`,
  { label: `judge:${String(doc).slice(0, 26)}`, phase: 'Judge', schema: SCHEMA, effort: 'high' })))

phase('Adjudicate')
const all = judged.filter(Boolean).flatMap(r => (r.results || []).map(x => ({ ...x, doc: r.doc })))
const shaky = all.filter(x => x.uncertain)

let settled = []
if (shaky.length) {
  settled = await parallel(shaky.map(x => () => agent(`${BASE}

One item was flagged as not settled by the rubric. Judge it independently, from the rubric alone. Do not defer to the
earlier judgement; reach your own and say which rule decides it.

  id:         ${x.id}
  document:   ${x.doc}
  column:     ${x.column || '(see the file)'}
  The item's definition, category, gold and pred are in ${FILE} - find it by id and read them there.

Earlier judgement was correctness=${x.correctness}, completeness=${x.completeness} ("${x.reason}"). It may be wrong.`,
    { label: `adjudicate:${String(x.id).slice(0, 10)}`, phase: 'Adjudicate', schema: SCHEMA, effort: 'high' })))
}

const revised = new Map()
for (const r of settled.filter(Boolean)) for (const x of (r.results || [])) revised.set(x.id, x)

const final = all.map(x => {
  const r = revised.get(x.id)
  return r ? { ...x, correctness: r.correctness, completeness: r.completeness, reason: `${r.reason} [adjudicated]` } : x
})

log(`${final.length} judgements, ${shaky.length} adjudicated a second time`)
return { batch: TAG, judgements: final, counts: { judged: all.length, adjudicated: shaky.length } }
