# Closed-loop training v2 — live status

<!-- The four fields below are the contract. Update them, in place, after
     every submission and every completed job. Everything under "Record" is
     append-only history. -->

| | |
|---|---|
| **Active job** | none submitted yet |
| **Latest result** | none — campaign frozen at `f2422ef`, 64 sources verified, 13 CPU tests pass |
| **Blocker** | none |
| **Next milestone** | end-to-end smoke (`slurm/smoke.sbatch`) |

## What this campaign is

One bounded rollout-driven AWR iteration from the untouched BC baseline,
measured on closed-loop folding rather than offline loss.

The horizon-pilot campaign is closed at `0e83a7b`. It established that fixed
RNG, RTC, retained queues, observation substitution, static H50-suffix
training, mixed replay and a fresh-H50 rescue oracle all fail to repair the
H10 replanning collapse — the oracle recovered **0/6** student-visited roots.
H50 plans are therefore prohibited as training labels here.

`20260922-closed-loop-training-v1` was a draft of this campaign. It was
reviewed, found to have five correctness defects, and never launched. This
campaign is its repair, rebuilt from committed sources.

## Frozen before submission

Everything decidable in advance is in `manifest.json` and was fixed before any
job ran. The point is that no threshold, seed or selection rule can be chosen
after an outcome is visible.

- **Budget** — 44 GPU tasks for iteration 1 (1 smoke, 8 collect, 1 train, 16
  development, 16 frozen test, 1 boundary, plus 1 CPU compile). 8 more if the
  preregistered expansion fires; 30 more only if a second iteration is earned.
- **Collection** — 4 garment classes x 2 real demonstration poses, seeds
  4200-4207. Every trajectory is retained whatever it scores. No seed
  selection, no dropped failures.
- **Expansion** — a second 8-row draw (seeds 4300-4307, poses 2-3) declared
  now, so a degenerate first collection has a fixed response rather than one
  chosen once the outcomes are known.
- **AWR** — beta 1.0, w_max 3.0, w_min 1e-6. Importance *resampling* with an
  unchanged supervised loss, because SmolVLA's flow-matching forward exposes
  one scalar batch loss and no per-example likelihood. RECAP conditioning is
  **not** used: its gradient update is unimplemented in this repository.
- **Gates** — >=3 distinct rewards, advantage std > 0.02, episode ESS <= n-1
  and >= 3, sample ESS fraction <= 0.95. Two-sided on purpose: ESS is
  *maximised* by uniform weights, so "ESS is high" is equally consistent with
  a strong signal and with no signal at all.
- **Training** — 300 steps, batch 4, 50% AWR-resampled rollout + 50% raster BC
  anchor, lr 1e-5, AdamW, boundary unfreeze, seed 4242, checkpoint every 100.
- **Selection** — evaluate `step_000300` on the full development set; fall
  back to `step_000200` only if `step_000300` breaches the raster retention
  guard. One candidate, decided before training.
- **Targets** — development H10 >= 4/8 primary, >= 6/8 stretch with no H50
  regression below the baseline 4/8, then confirmation on the frozen test set.

## Evaluation design

The eight-pose set is **development data, not an untouched test**. Its H10
failure is what motivated this campaign, so a gain on it is not an unbiased
estimate of anything.

| set | rows | role |
|---|---|---|
| development | 8 poses x {H10, H50} | matched comparison; targets are stated against it |
| frozen test | 8 garments, seeds 9000-9007, H10 | excluded from collection, expansion, development and selection |
| raster held-out | 4 files x 8 frames | regression guard only, never an optimisation target |

Baseline development numbers are **reused** from the horizon pilot: H10 0/8,
H50 4/8. Identical garments, poses, seeds 200-207, checkpoint SHA, 600/60/60
protocol and runner. The only runner change since is trajectory capture, which
is inert unless `--trajectory_out` is passed and is not passed for benchmark
rows. Baseline on the **frozen test set is run fresh** under this campaign,
for both checkpoints.

## Reproducibility

`manifest.json` records the git commit and the SHA-256 of all 64 executed
sources taken from the **committed blob**, not the working tree.
`build_manifest.py` refuses to run against a dirty tree;
`verify_sources.py` re-checks every source and the immutable baseline before
each production submission.

This exists because the horizon pilot's frozen manifest drifted: it recorded
hashes for `policy_rollout51.py` matching neither HEAD nor the working tree,
and `run_improvement_task.py` hard-fails on that mismatch today. That drift is
recorded in `CLAUDE.md` §6.1 and has **not** been hidden by regenerating the
historical manifest.

## Record

append-only; newest last
