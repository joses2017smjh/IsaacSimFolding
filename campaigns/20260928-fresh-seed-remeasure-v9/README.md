# Fresh-seed re-measurement v9

**Measurement only.** Authorized by the user on 2026-09-28, as the next step
recommended in the v8 report
(`../20260926-anchor-diagnostic-v8/REPORT.md`). There is no training and no
checkpoint choice, and the final set is not touched. **The baseline stays
the deliverable whatever the result.**

## Question

Do an1's H10 gain (v8: 9/16 vs 2/16) and its dev01 H50 loss survive fresh
seeds once the GPU model is controlled? Does the answer depend on the GPU
model?

v8 left three reasons to doubt its H10 pass:
- Every campaign had reused seeds 200–207.
- The GPU model (A40 vs Quadro RTX 8000) was assigned by the scheduler, and
  it changes outcomes on some rows. dev06's +2 came from an A40-only
  baseline failure.
- The r1/r2 replicates were coupled by that hardware allocation.

## Design

**Policies.** Both are pinned by SHA-256 in `manifest.json`:
- the untouched baseline `bc_smolvla_raster_ft_full`;
- v8's `an1-step000300`, byte for byte.

**Rows.** 8 development poses × H10/H50 × 3 fresh seeds gives 48 rows.
- Seeds are `97000 + 100·k + pose` for k = 0..2.
- A seed is shared by both horizons and both policies.
- `build_manifest.py` proves no earlier campaign used any of them.

**Blocks.** Each block is one interleaved Slurm array: even index =
baseline, odd = candidate on the same row, throttled at 8. Each block is
pinned to one GPU model with `--constraint`:
- `rtx8000`: 96 episodes;
- `a40`: 96 episodes, the approved design.

At launch every public A40 node was drained for maintenance, and both RTX
8000 nodes were full. Each block runs when its hardware frees up. The two
blocks share rows and seeds, so they also measure the GPU effect directly.

**GPU check.** Each task checks its GPU twice: `nvidia-smi` before Isaac
starts, and the device table Isaac prints after. A task on the wrong model
is an infrastructure error and gets one outcome-blind retry.

## How it is read (preregistered; `manifest.json` → `protocol`)

**Per block.** matched_verdict's clauses on 24 vs 24 per horizon:
- H10: margin ≥ 4 **and** one-sided Fisher p < 0.05;
- H50: candidate ≥ baseline;
- all 96 episodes valid on the pinned model.

**Classification of the H10 gain:**

| H10 clause holds on | classification |
|---|---|
| both complete blocks | replicates on both GPU models |
| exactly one block | replicates on that model only (hardware-conditional) |
| neither block | does not replicate |
| a block is incomplete | partially measured |

**Reported, not gated:**
- pose-stratified exact and pose sign-flip tests;
- paired (row, seed) McNemar;
- the block × pose stratified test;
- per-cluster settled/reached/n;
- P_B H50 per row;
- v8's endpoints E1/E2;
- the drag and early-bimanual-lift detectors (`scripts/analysis.py`; their
  calibration on all 256 v5–v8 episodes is frozen in the manifest);
- terminal margins;
- settled by GPU model on the same seeds.

**Stop rule.** One array per block, and one retry only for tasks that left
no completed rollout. Not allowed after launch: extra seeds, re-running
valid rows, another checkpoint, or any rule change. A block not finished by
2026-10-06 00:00 UTC is cancelled and reported as incomplete.

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
$PY $C/scripts/driver.py --campaign $C tick --dry-run   # the plan, writes nothing
$PY $C/scripts/driver.py --campaign $C tick             # preflight, submit, schedule ticks
$PY -m pytest $C/tests/test_v9.py -q                    # CPU tests
```

`STATUS.md` is the live page.
