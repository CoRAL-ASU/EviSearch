# Speed configuration guide

How to make an EviSearch extraction run faster, what each lever costs, and which levers can change the numbers in the
table. Every figure below is either measured (marked with its source) or arithmetic on measured figures. Estimates are
labelled and shown with their working.

Read this before changing a serving flag or an environment variable on a run that will be compared with another run.

---

## 1. Where the time goes

One document, system E (Arm A -> Arm B -> arbiter), Qwen3.6-27B on the local vLLM instance:

| stage | wall | calls | output tok/call | output tokens |
|---|---|---|---|---|
| Arm A (`agent`, pdf_query) | 6.9 min | 11.2 | 1,940 | 21,728 |
| Arm B (`search`, search_agent) | 10.2 min | 62.8 | 488 | 30,646 |
| arbiter (`reconciliation`) | 10.2 min | 58.1 | 614 | 35,673 |
| **end to end** | **27.3 min** | 132 | | **88,047** |

The three stages run one after another inside `run_doc` (`experiment-scripts/run_benchmark.py`), so the per-document
wall clock is their sum: 6.9 + 10.2 + 10.2 = 27.3 min.

**It is decode-bound.** Per-stream decode is 53.8 output tok/s (measured; aggregate 210 tok/s across 3.9 concurrent
stages, 210 / 3.9 = 53.8). Dividing each stage's output tokens by that rate reproduces its wall clock:

- Arm A: 21,728 / 53.8 = 404 s = 6.7 min, against 6.9 min measured (97.6%).
- Arm B: 30,646 / 53.8 = 570 s = 9.5 min, against 10.2 min measured (93%).
- arbiter: 35,673 / 53.8 = 663 s = 11.1 min, against 10.2 min measured (108% - the arbiter beats one stream because
  it already fans its verifier calls out four ways; see §2.1).

The live run confirms it. Averaged over the documents finished so far in `r4-notes-r1`
(`new_pipeline_outputs/results/*/runs/r4-notes-r1/*/extraction_metadata.json`):

| stage | docs | model_seconds | duration_s | model/wall | calls | output tok | out/call | tok/s |
|---|---|---|---|---|---|---|---|---|
| agent_extractor | 8 | 389 | 391 | 0.996 | 10.4 | 19,988 | 1,927 | 51.3 |
| search_agent | 7 | 598 | 603 | 0.991 | 58.0 | 30,015 | 517 | 50.2 |
| reconciliation_agent | 6 | 772 | 705 | **1.095** | 62.7 | 39,314 | 627 | 50.9 |

Two things to take from that table: the wall clock is the model's time (0.99 of it for A and B), and the arbiter is the
only stage with any internal overlap at all (1.095).

### Why the prompt is not the problem

Arm A sends 49k input tokens per call, 85.5% of it cached, so 49,000 x 0.145 = 7,105 tokens are actually prefilled.
Take the measured Agent A stage with `model_seconds` 409.7 inside `duration_s` 412.4:

```
decode  21,728 / 53.8      = 403.9 s   98.6% of model_seconds
prefill + tool parse       =   5.8 s   409.7 - 403.9, over 11.2 calls = 0.52 s/call
outside the model          =   2.7 s   412.4 - 409.7
------------------------------------------------------------------
everything that is not decode = 8.5 s of 412.4 s = 2.1%
```

So a prompt cut to *zero bytes* would save at most 2.1% of Arm A's stage. Halving it saves under 1%. Prompt
engineering for speed is not worth an hour of anyone's time here; **the only thing that matters is how many tokens the
model writes and how many streams write at once.**

A useful by-product: 7,105 uncached tokens in <= 0.52 s means prefill runs at >= 13.7k tok/s (estimate, derived from
the two measured numbers above). That is ~260x the decode rate, and it is why cache-warming effects are noise.

### The unit of work

`BATCH_MAX_COLUMNS = 15` (`src/config/config.py`) over 133 columns in 39 definition groups gives **11 batches per
stage**, with sizes `[15, 15, 15, 10, 12, 14, 8, 8, 10, 12, 14]` (verified by `build_batches`, and by the 11
`verification_logs/batch_*.json` files each stage writes). Arm A makes one call per batch; Arm B and the arbiter run a
tool loop, ~5 calls per batch.

11 is therefore the hard ceiling on within-stage concurrency, and the largest batch (15 columns) is the floor on the
stage's wall clock no matter how many workers you give it.

---

## 2. The levers

### 2.1 Batch concurrency within a stage - `EVISEARCH_STAGE_CONCURRENCY`

**What it does.** `src/evisearch/pipelines/batch_runner.py` runs a stage's 11 column batches in a thread pool instead
of a loop. Default `1` = serial = every run up to R4 reproduces. Wired into all three pipelines
(`pdf_query_pipeline.py:112`, `search_pipeline.py:97`, `reconciliation_pipeline.py:127`).

**Expected factor.** Bounded by three things: the worker count, the 11 batches, and the makespan of unequal batches.
Using column counts as a proxy for output tokens per batch (estimate - batches are not exactly proportional to their
column count), longest-processing-time packing of `[15,15,15,14,14,12,12,10,10,8,8]` (sum 133) gives:

| workers | ideal 133/W | LPT makespan | factor |
|---|---|---|---|
| 2 | 66.5 | 67 | 2.0x |
| 4 | 33.3 | 35 | 3.8x |
| 8 | 16.6 | 20 | 6.7x |

The server, not this, decides whether you get it (§3).

One-time cost: with W workers the first W batches each prefill the document prefix instead of one filling it for the
rest. At >= 13.7k tok/s that is ~41,895 x (W-1) / 13,700 ~= 3 s per extra worker (estimate), against 37 s of decode per
batch. Negligible.

**Can it change accuracy? No, by construction.** Each batch is a separate session over a disjoint set of columns at
temperature 0. Same prompts, same tools, same order of turns within a batch. Only the order in which batches are
scheduled changes. `accumulate` is called under a lock, so incremental saving and crash-resume still work.

Two residuals that are *not* accuracy but are worth knowing:

1. `results_store.save_columns` writes `json.dumps(..., indent=2)` with no `sort_keys`, and `accumulate` does
   `columns.update(results)` in completion order. So the **key order** in `extraction_results.json` changes. Compare
   parsed dicts, never bytes.
2. Greedy decoding is not bit-identical across batch sizes: vLLM's matmul shapes change with the number of sequences
   in flight, and bf16 reductions are not associative. This can only ever flip a token where two candidates were
   already tied. It is checked by a cheap re-run diff (§5), not by a re-score.

### 2.2 Stage parallelism - Arm A alongside Arm B (`EVISEARCH_STAGE_PARALLEL`)

**Is it legal?** Yes, by data dependency. `search_pipeline.py` reads only `doc_id` and its column batch - it never
touches Arm A's results. Only the arbiter reads both arms. The two stages write to different method directories.
`run_doc` currently runs them in sequence purely because it iterates `SYSTEMS[system]`.

**Expected factor.** Per document, `max(A, B) + arbiter` instead of `A + B + arbiter`:

```
serial:   6.9 + 10.2 + 10.2 = 27.3 min
A || B:   max(6.9, 10.2) + 10.2 = 20.4 min      ->  1.34x, saving 6.9 min (25%)
```

That 1.34x is available **only if the server has a free slot**. If the card is already saturated, overlapping A and B
moves work around without finishing it sooner: the same 88k tokens still have to be decoded.

**Can it change accuracy? No, by construction** - same inputs, no shared mutable state, and the arbiter still starts
only after both arms are complete. The risk is operational, not statistical: it doubles a document's in-flight
requests, and a failure now has two stages in progress to resume.

`batch_runner.stage_parallel()` reads `EVISEARCH_STAGE_PARALLEL` (off by default) and `run_benchmark.py` imports it;
treat the runner side as the thing to verify before relying on it.

### 2.3 Documents in parallel - `--parallel`

**What it does.** `run_benchmark.py` maps documents over a `ThreadPoolExecutor(max_workers=args.parallel)`; default 2.
Documents are wholly independent: separate result directories, separate manifests, a failure isolated to one document.

**Expected factor.** Up to `--parallel`x on aggregate throughput while slots last. For 10 papers at 27.3 min each:

```
--parallel 2:  ceil(10/2) = 5 waves x 27.3 = 137 min
--parallel 4:  ceil(10/4) = 3 waves x 27.3 =  82 min     ->  1.7x
```

Real makespan is worse than waves arithmetic because documents differ: in `r4-notes-r1` per-document wall clock ranged
1,512 s (Gravis GETUG) to 2,498 s (Hussain ARASENS), so the last wave is set by the longest paper left.

**Can it change accuracy? No, by construction** - different documents share nothing but the server.

### 2.4 Raising `--max-num-seqs` (currently 8)

**What it does.** The number of sequences vLLM will schedule at once. Useless unless you are actually sending more
than 8 requests; harmful past what the KV cache holds.

**The cache is the real ceiling, and it is below 8.** From `src/config/catalog.yaml` (`qwen36_27b`): at
`gpu_memory_utilization: 0.60` on the shared 140 GiB H200, weights 51.1 GiB + ~3.2 GiB activations leave ~30 GiB of
cache = ~470k tokens at ~67 KB/token = **~5 concurrent requests of the largest Arm A document** (84k tokens). Raising
`--max-num-seqs` without raising `gpu_memory_utilization` buys preemption and recompute, not throughput. Arm B and
arbiter requests are smaller, so more of those fit; Arm A is the one that does not.

**Can it change accuracy? No** - it is a scheduling parameter, with the same bf16-reduction residual as §2.1.
**But it requires restarting the server**, which is forbidden while an experiment is live, and the GPU is shared with
other tenants whose requests occupy the same slots.

### 2.5 Tensor parallelism across GPUs (`tensor_parallel`, currently 1)

**What it does.** Splits the weights across N GPUs, shrinking per-token latency for a single stream. This is the only
lever that makes *one* request faster; everything in §2.1-2.3 makes *more* requests run at once.

**Expected factor: unmeasured here.** TP=2 halves the per-GPU weight read per token, so the ceiling is 53.8 -> ~107
tok/s, but all-reduce per layer eats into that and real-world TP=2 scaling on a single node is under-linear (estimate,
not measured on this box - do not quote it as a result). Cost: two GPUs at 0.60 utilisation each, a server restart, and
`serve.py` must find two GPUs with that much free in `GPU_POOL` or it raises `ConfigError`.

**Can it change accuracy? In principle no** (same weights, same math in exact arithmetic) - but TP changes reduction
order, which is the same bf16 caveat as above, one notch larger. Treat a TP change on a rung of the ladder as worth a
re-run diff.

### 2.6 FP8 / AWQ quantisation

**Expected factor.** Decode is weight-bandwidth-bound, so halving the bytes per weight is the single largest available
win - roughly 1.5-2x on paper (estimate, vendor-typical, **not measured here**). It also frees ~25 GiB, which turns
straight into KV cache and lifts the §2.4 ceiling.

**Can it change accuracy? YES.** Different weights produce different text: different numbers extracted, different
evidence quoted, different reconciliation verdicts. This is not a scheduling change and cannot be waved through.
It requires a full re-score of both replicates (§5).

### 2.7 Generating fewer output tokens

**Expected factor: exactly proportional**, because wall clock = output tokens / 53.8. The tokens live here:

```
arbiter  35,673  (41%)   614 out/call over 58.1 calls
Arm B    30,646  (35%)   488 out/call over 62.8 calls
Arm A    21,728  (25%) 1,940 out/call over 11.2 calls
```

Arm B and the arbiter together are 75% of the budget, and both are tool loops where much of the output is reasoning
(`--reasoning-parser qwen3`). Cutting the total from 88k to 60k would give 88/60 = 1.47x.

**Can it change accuracy? YES**, and it is the most tempting way to fool yourself. Shorter reasoning, tighter
`max_tokens`, fewer verifier claims, dropping the arbiter's read-first phase - each changes what the model writes.
Any of these is a pipeline change wearing a speed costume, and belongs on the ladder as its own rung with its own
score, not in a config file.

### 2.8 A serverless open-model provider

**Expected factor.** Concurrency stops being scarce: the 10-paper run collapses towards the wall clock of the single
longest document (~27-42 min for the range seen in `r4-notes-r1`) instead of `ceil(10/parallel)` waves. Per-stream
decode is usually higher than 53.8 tok/s too.

**The cost is on the input side.** Local prefix caching is what makes the 49k-token Arm A prompts free; a provider
bills them. Per document from the live run (Gravis / Attard):

```
Gravis  A 344k + B 762k + arbiter 606k = 1.71M input tokens,  80k output
Attard  A 714k + B 2.11M + arbiter 989k = 3.81M input tokens, 100k output
10 papers ~= 17-38M input tokens and ~880k output tokens per run
```

Multiply by the provider's rates - and check whether their cache discount applies to the tool-loop shape we send,
because without it the input side dominates the bill.

**Can it change accuracy? YES, in practice.** Same weights are not the same system: a different serving stack means
different kernels, often a different quantisation, and a different tool-call and reasoning parser. Even at temperature
0 the sampled text can differ, and the tool loop can fail differently. A provider is a new rung, not a setting.

---

## 3. The interaction that bites

```
in-flight requests = (documents in parallel)
                   x (stages overlapping per document)     1, or 2 with EVISEARCH_STAGE_PARALLEL
                   x (batches in parallel per stage)       EVISEARCH_STAGE_CONCURRENCY
                   x (verifier fan-out, arbiter only)      up to 4
```

That last factor is easy to forget: `src/evisearch/services/evidence_check.py:35` sets `MAX_WORKERS = 4` and the
arbiter's claim verification already runs four calls at once inside a single batch. It is where the arbiter's 1.095
model/wall ratio comes from. **So the arbiter stage is already multiplying whatever you set by up to 4.**

Against `--max-num-seqs 8`, and a KV cache that holds only ~5 large Arm A requests, on a card shared with other
tenants:

| configuration | A/B stages in flight | arbiter stage in flight | verdict |
|---|---|---|---|
| `--parallel 2`, conc 1 (today, one run) | 2 | up to 8 | at the slot count |
| two runs of `--parallel 2` (today, r1+r2) | 4 | up to 16 | over; queues |
| `--parallel 4`, conc 1 | 4 | up to 16 | over in the arbiter only |
| `--parallel 2`, conc 4 | 8 | up to 32 | oversubscribed, and over the Arm A cache limit |
| `--parallel 1`, conc 4 | 4 | up to 16 | fine for A/B, arbiter queues |

vLLM queues rather than failing, and the degradation is mild: `r4-notes-r1` and `r4-notes-r2` are running concurrently
right now (4 documents in flight, up to 16 requests during arbiter phases) and average 29.8 and 28.6 min per document
against the 27.3 min measured for a single run - 5-9% slower, not 2x. **Past the slot count you do not lose, you just
stop gaining.**

### Recommended: batch experiment over 10 papers (throughput is what matters)

```
--parallel 4
EVISEARCH_STAGE_CONCURRENCY=1      (unset)
EVISEARCH_STAGE_PARALLEL           off
```

Estimated wall: 3 waves x 27.3 = 82 min against 137 min at `--parallel 2`, so ~1.7x (estimate; the real last wave is
set by the longest remaining paper, 2,498 s in `r4-notes-r1`).

Why documents rather than batches, when both give you 4 streams: documents are independent and resumable, a crash
costs one paper, the arbiter's internal x4 is not multiplied by a second factor, and each document's Arm A batches keep
running against a warm 75k-token prefix (85.5% cache hit rate - the thing `--mamba-cache-mode all` was set for).
Drop to `--parallel 3` if another tenant is busy on the card.

### Recommended: single-paper demo (latency is what the user feels)

```
--parallel 1                       (one document by definition)
EVISEARCH_STAGE_CONCURRENCY=4
EVISEARCH_STAGE_PARALLEL=1         once the runner side is verified
```

Estimated wall (all estimates, from the §2.1 LPT factors and the §2.2 arithmetic):

```
Arm A   6.9 / 3.8 = 1.8 min
Arm B  10.2 / 3.8 = 2.7 min          runs alongside A -> costs max(1.8, 2.7) = 2.7 min
arbiter 10.2 / 2   = 5.1 min          conc 2 only: x4 internal fan-out already, 2 x 4 = 8 = the slot count
------------------------------------------------------------------------------
stages serial:   1.8 + 2.7 + 5.1 =  9.6 min   ->  2.8x
A || B:          2.7 + 5.1       =  7.8 min   ->  3.5x
```

Two caveats. Arm A at 4 concurrent 49k-token requests is close to the ~5-request Arm A cache limit, so 4 is the
ceiling, not a starting point. And a shared card means another tenant's streams come out of the same 8 slots - a demo
is exactly when you cannot control that, so quote the serial number to anyone who needs a guarantee.

---

## 4. Accuracy-neutral by construction, and not

| lever | neutral? | why |
|---|---|---|
| `EVISEARCH_STAGE_CONCURRENCY` | yes, by construction | identical independent requests, disjoint columns, temperature 0; only the schedule changes |
| `EVISEARCH_STAGE_PARALLEL` (A \|\| B) | yes, by construction | B never reads A; arbiter still waits for both |
| `--parallel` documents | yes, by construction | documents share nothing but the server |
| `--max-num-seqs` | yes | scheduler capacity |
| tensor parallelism | yes in exact arithmetic | same weights; reduction order changes (see caveat) |
| FP8 / AWQ quantisation | **no** | different weights -> different text |
| fewer output tokens | **no** | different reasoning, different evidence, different verdicts |
| serverless provider | **no** | different kernels, quantisation and tool parsers |

The caveat that applies to every row in the "yes" half: greedy decoding is not bit-identical across batch sizes,
because the matmul shapes change with the number of sequences in flight and bf16 reductions are not associative. This
can flip a token only where candidates were already tied. It is a reproducibility fact, not an accuracy mechanism -
handle it with a diff (§5), not with a re-score.

---

## 5. What to re-measure

**After an accuracy-neutral change** (anything in the "yes" half above), one cheap check is enough:

1. Pick one document. Run the stage into a scratch run name serially, then again with the new setting.
2. Diff the **parsed** `extraction_results.json` column values, not the bytes (key order follows completion order, §2.1).
3. Expect identical values. A handful of differing cells means you found a tie flip; a systematic difference means the
   change is not what you thought it was - stop and find out why before running the ladder.
4. `tests/test_agents_offline.py` already pins the two properties that matter offline: that concurrency produces the
   same per-batch results as serial, and that one failing batch does not discard the others' results.

**After an accuracy-affecting change** (quantisation, shorter generations, a different provider or model), there is no
cheap check. Re-run **both replicates** of every rung you want to compare, and re-score all of it: per-column accuracy,
the flag rate, and the `both_wrong` counts.

**Why a change mid-ladder invalidates the comparison.** The ladder's whole design is pairwise: rung against rung, two
replicates each, so that run-to-run variance can be separated from a real effect. 10 papers x 133 columns is a small
enough sample that replicate-to-replicate disagreement *is* the noise floor. If R3 ran in bf16 and R4 in FP8, a
difference between them is the schema change, or the quantisation, or noise, and nothing you can compute afterwards
will tell you which - the confound is in the data, not in the analysis. A speed change that touches what the model
writes therefore has to be either applied to every rung from the start, or measured as its own rung with its own two
replicates.

Corollary for the run in flight: `r4-notes-r1` and `r4-notes-r2` are the R4 pair. Do not change a serving flag, a
generation budget or `gpu_memory_utilization` until both have finished and been scored, and do not restart the server.
Concurrency settings are safe to change between rungs, but note them in the run manifest anyway - `--parallel` is
already recorded there, and the two environment variables should be too if you start using them.
