# Fresh-seed re-measurement v9

**Measurement only.** Authorized by the user on 2026-09-28, as the next step
recommended in the v8 report
(`../20260926-anchor-diagnostic-v8/REPORT.md`). There is no training and no
checkpoint choice, and the final set is not touched. **The baseline stays
the deliverable whatever the result.**

## Question

Do an1's H10 gain (v8: 9/16 vs 2/16) and its dev01 H50 loss survive fresh
seeds once the hardware platform is controlled? Does the answer depend on
the platform?

v8 left three reasons to doubt its H10 pass:
- Every campaign had reused seeds 200–207.
- The GPU model (A40 vs Quadro RTX 8000) was assigned by the scheduler, and
  it changes outcomes on some rows. dev06's +2 came from an A40-only
  baseline failure.
- The r1/r2 replicates were coupled by that hardware allocation.

## Design

**Policies.** Both are pinned by SHA-256 in `manifest.json`, all 8 files
each:
- the untouched baseline `bc_smolvla_raster_ft_full`;
- v8's `an1-step000300`, byte for byte.

**Rows.** 8 development poses × H10/H50 × 3 fresh seeds gives 48 rows.
- Seeds are `97000 + 100·k + pose` for k = 0..2.
- A seed is shared by both horizons and both policies.
- They were checked against the manifests, plans, request/status records,
  analysis, audit and ledger JSON, and `--seed` arguments of every earlier
  campaign: 182 prior seeds, max 96207, no collision.

**Blocks.** Each block is one interleaved Slurm array: even index =
baseline, odd = candidate on the same row, so each row's pair starts
adjacently. Each block is pinned to one GPU model with `--constraint`:

| block | role | episodes | expected GPU-h |
|---|---|---|---|
| `a40` | **primary**, the approved design | 96 | ~5.7 |
| `rtx8000` | secondary replication on a second platform | 96 | ~7.8 |

The RTX 8000 block was added because every public A40 node was drained for
maintenance at launch. It can start before the maintenance ends.

The blocks compare hardware **platforms**: the GPU model plus its node pool
(CPU, ISA and driver). The comparison is unpaired but balanced on rows and
seeds, and driver and host are reported per block.

QoS limits concurrency to 5 tasks on `gpu` and 2 GPUs on `ampere`.

**Provenance.** Every task checks, before Isaac starts:
- the manifest digest the driver submitted under;
- every pinned source, including the LeHome checkout;
- its checkpoint's 8 files.

It checks the GPU model before and after the run. A task on the wrong model
is an infrastructure error.

## How it is read (preregistered; `manifest.json` → `protocol`)

**Per block.** Each complete block is a reading in its own right; the
**a40 reading is the headline**. The reading is matched_verdict's clauses
on 24 vs 24 per horizon:
- H10: margin ≥ 4 **and** one-sided Fisher p < 0.05. At n = 24, p < 0.05
  already needs a margin of at least 5.
- H50: candidate ≥ baseline.
- All 96 episodes valid on the pinned model under one manifest digest.

**Cross-platform label for the H10 gain:**

| H10 clause holds on | label |
|---|---|
| both blocks | replicates on both platforms |
| exactly one block | H10 clause holds on X, not on Y |
| neither block | not replicated at this power (…) |
| a block is incomplete | not available |

A result on exactly one block is called hardware-dependent **only** if the
preregistered interaction test gives p < 0.05. That test is an exact
two-sided sign-flip over the 8 rows of the a40 minus the rtx8000
(candidate − baseline) difference. "Not established" is not evidence of no
hardware effect. The false-positive rate of "passes in at least one block"
is about 5%.

**Power of the H10 clause at 24 vs 24** (baseline rate taken from its
v5–v8 H10 history on each model):

| block | baseline history | p1 = 0.35 | p1 = 0.45 | p1 = 0.54 |
|---|---|---|---|---|
| a40 | 12/40 | 6% | 20% | 42% |
| rtx8000 | 3/24 | 47% | 74% | 90% |

**dev01 H50** (descriptive, not gated). The loss "recurs" in a block iff
the baseline settles dev01 H50 more often than the candidate (3 vs 3).
Complete blocks are pooled with a one-sided Fisher p, and 3 vs 3 cannot
reach significance.

**Reported, not gated** (every p-value unadjusted and descriptive, each
with its paired n):
- the row-stratified (8 development slots) and cluster-stratified exact
  tests;
- the row sign-flip test and the paired McNemar;
- the block × row stratified test, which cannot override the per-block
  readings;
- per-cluster settled/reached/n;
- P_B H50 per row;
- v8's endpoints E1/E2;
- the drag and early-bimanual-lift detectors (`scripts/analysis.py`; their
  calibration on the 128 v5–v8 H50 episodes is frozen in the manifest);
- terminal margins;
- settled by platform on the same seeds.

**Retry rule.** A task gets one outcome-blind retry iff its status is not
`completed`: missing, infrastructure error (including the wrong GPU model or
a validation failure), or timeout. The driver re-runs `verify_sources.py`
first and records every retried directory with its state. If the retry
fails too, that episode is invalid and its block incomplete.

**Stop rule.** One array per block plus that single retry. Not allowed
after launch: extra seeds, re-running valid rows, another checkpoint, or
any rule change. No manual cancellation: any cancellation other than at
the deadline is logged as a protocol deviation.

**Deadline.** 2026-10-06 00:00 UTC, enforced on each episode's own
completion time. A block still running at the deadline is cancelled and
reported as incomplete.

## What this cannot show

- Generalization: the development poses are corpus poses, and nine
  checkpoints were screened on them.
- Anchor causation.
- An unbiased effect size.
- Performance as shipped (H50) unless H50 holds.

Changing the deliverable would need a separately preregistered holdout that
includes P_B H50 and a declared shipping horizon.

## Running it

```bash
C=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro/campaigns/20260928-fresh-seed-remeasure-v9
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
$PY $C/scripts/driver.py --campaign $C tick --dry-run   # the plan; writes and moves nothing
$PY $C/scripts/driver.py --campaign $C tick             # preflight, submit, schedule ticks
$PY -m pytest $C/tests/test_v9.py -q                    # CPU tests
```

`STATUS.md` is the live page.
