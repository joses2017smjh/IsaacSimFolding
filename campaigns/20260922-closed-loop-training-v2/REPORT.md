# Closed-loop training v2 — final report

**Outcome: target not met. The untouched baseline checkpoint is retained.**
Three rollout-driven AWR iterations, each changing one factor chosen from the
previous iteration's closed-loop evidence, produced no checkpoint that
improves development H10 over the matched baseline. One regressed H50.

| | |
|---|---|
| Target | development H10 settled >= 6/8 (stretch 8/8), no H50 regression, retention guard |
| Achieved | best eligible candidate H10 **1/8** (iter3) vs matched baseline **2/8**; no candidate qualified |
| Delivered checkpoint | the baseline, `/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-data/outputs/train/bc_smolvla_raster_ft_full` (`model.safetensors` 4a37d4dce3d96e39…) |
| Frozen test (baseline) | H10 settled **0/8** on 8 garments excluded from all training and selection (mean conditions 3.000; one latched fold undone by settling) |
| Budget used | 105 of 170 GPU tasks, 8.1 of 45 GPU-hours, 15G on disk (cap 60 GB); finished 2026-09-23, well before the 2026-09-24T18:00Z iteration cutoff |

## Development benchmark

Eight Pant_Short poses (garments 0/3/7/9, pose keys 0-1, seeds 200-207), each
run at H10 and H50, 600 actions, 60 settling steps. **Settled** = the official
checker passes after settling; the latched "ever" verdict is recorded but is
not the target. Per-row data: `analysis/development-results.csv`.

| checkpoint | H10 settled | H10 mean conditions | H50 settled | H50 mean conditions | eligible |
|---|---|---|---|---|---|
| baseline, horizon-pilot run (same seeds) | 0/8 | 2.375 | 4/8 | 3.000 | — |
| **baseline, this campaign (matched)** | **2/8** | **3.000** | **4/8** | **3.125** | — |
| iter1-step000300 | 0/8 | 2.125 | 4/8 | 3.250 | yes, not better |
| iter2-step000300 | 0/8 | 2.500 | 3/8 | 3.125 | no — H50 regression |
| iter3-step000300 | 1/8 | 2.375 | 4/8 | 3.250 | yes, not better |

**Read these with the noise in mind.** The same baseline, seeds, checkpoint
and controller scored H10 0/8 in the horizon pilot and 2/8 here; its H10 mean
conditions ranged 2.375-3.000. Every candidate falls within or just below that
range. So the supportable statement is that **no candidate improved H10**;
whether any degraded it is not established at n=8 (iteration 1, with 7 of 8
rows at exactly 2/4, is the only one slightly outside the baseline's range).
An earlier statement in STATUS that iteration 1 "systematically degraded" H10
compared it with the favourable run alone and is corrected there.

## What each iteration tested

**Iteration 1 — as preregistered.** 8 autonomous H10 rollouts from the
baseline across four garment classes, two poses each. Gate passed (rewards
`[0.75 x4, 0.6, 0.4 x3]`, ESS 5.36/8). No success anywhere in the data, and
the reward was identical within every garment except one pose pair: **the
advantage measured garment difficulty, not action quality.** The gate passes
such a confound by construction, since it checks spread and concentration.

**Iteration 2 — success density (amendment A2).** 16 H50 rollouts on
Pant_Short garments 4/5/6/8 (no development or test garment), mirroring the
development pose mix. Only **2/16 settled** — the baseline's H50 competence
proved garment-specific (8/10 on the development garments' P_B/P_C poses, at
most 1/10 on the collection garments' same poses). Ineligible on an H50
regression. A2's preregistered test — an H10 gain on P_B/P_C rows — failed.

**Iteration 3 — advantage sharpness (amendment A3).** Iteration 2's data held
fixed; beta 1.0 -> 0.5 with w_max 3 -> 20. Under iteration 2's constants the
cap had held the two successes to 28.6% of the sampling mass while six 3/4
failures carried 54%; the new constants moved success mass to 63% (ESS 4.57,
every gate threshold unchanged). No improvement. A3 records that this change
was post-hoc, made after two candidates' development results.

## Remaining limitation

**The policy does not produce enough successful behaviour on its own for
outcome-weighted self-imitation to learn from.** On non-development garments
it settled about 1 episode in 8 at either horizon. Every dataset the campaign
could generate autonomously was either failure-dominated by sampling mass
(iterations 1-2) or, once sharpened, rested on two successful episodes
(iteration 3). Across horizon, garment class and weighting, AWR importance
resampling from 16 rollouts did not move H10. What would change that is a
source of *successful* actions at the states the H10 policy actually visits —
teleoperated or scripted recovery (DAgger), or a learned within-episode value
signal that credits partial progress — neither of which exists in this
headless setup. The one training-side lever left untested is update size
(fewer steps or a lower learning rate); it can limit drift but cannot supply
successes.

## Limits of the evidence

- n=8 per horizon per checkpoint, one seed per pose; identical seeds do not
  reproduce outcomes (GPU cloth physics is not bitwise deterministic).
- The frozen test was run for the baseline only, because no candidate
  qualified; it measures the delivered checkpoint, not a candidate.
- Development rows dev00/dev01 share garment and pose with iteration 1's
  Pant_Short_Seen_0 collection rows (a v1 design carried into v2), so the
  development set was not fully held out from iteration-1 training.
  Iterations 2-3 trained on no development or test garment.
- **The raster retention guard was vacuous in practice.** Held-out loss fell
  35-43% under every run, because the guard's frames (the first 8 of 4
  Top_Short demonstrations) share a distribution with the anchor (the first 8
  of 16 demonstrations), which is half of every batch. It never came close to
  binding, so step_000300 was selected every time. The matched H50
  closed-loop check did the real regression testing: it caught iteration 2.

## Reproduction

```bash
git clone https://github.com/joses2017smjh/IsaacSimFolding.git && cd IsaacSimFolding
C=$PWD/campaigns/20260922-closed-loop-training-v2
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python   # Python 3.11 with numpy
PYTHONPATH=src $PY tests/test_pure.py                       # 56 root tests
$PY -m pytest $C/tests -q                                   # campaign + driver tests
$PY $C/scripts/verify_sources.py --campaign $C              # 73 sources + immutable baseline
$PY $C/scripts/driver.py --campaign $C status               # full resumable state
```

A fresh run needs a new campaign directory: every stage refuses to overwrite
retained output, and `build_manifest.py` refuses to overwrite a frozen
manifest. The protocol is `manifest.json` plus `amendments/A1-A3`; iteration
plans are `plans/iteration{1,2,3}.resolved.json`. The workspace resources —
container `bhl.sif`, the Isaac Sim 5.1 venv, `lehome51-site`, the LeHome
asset pack and the baseline checkpoint — are hash-pinned but not in Git.

## Provenance

| stage | manifest commit | notes |
|---|---|---|
| smoke `21400590` (failed) | `f2422ef` | in Git history at that commit |
| smoke `21400605` (failed) | `a313e4a` | in Git history at that commit |
| smoke `21400623` (completed), collect:iteration1 `21400641` (completed), compile:iteration1 `21400642` (completed), train:iteration1 `21400643` (completed) | `d3422c9` | archived `manifests/manifest_d3422c9.json` |
| baseline.dev `21400694` (completed), iter1.reload `21400695` (failed) | `b319349` | archived `manifests/manifest_b319349.json` |
| iter1.reload.retry1 `21400710` (completed) | `c799da8` | archived `manifests/manifest_c799da8.json` |
| iter1.evaluate `21400738` (completed) | `87a21e5` | archived `manifests/manifest_87a21e5.json` |
| iter2.collect `21400812` (completed), iter2.compile `21400892` (completed), iter2.train `21400898` (completed) | `fecd4dc` | archived `manifests/manifest_fecd4dc.json` |
| iter2.reload `21400907` (completed), iter2.evaluate `21400910` (completed) | `2ed70ce` | archived `manifests/manifest_2ed70ce.json` |
| iter3.compile `21401005` (completed), iter3.train `21401009` (completed), iter3.reload `21401026` (completed), iter3.evaluate `21401028` (completed), final.test.baseline `21401084` (completed) | `4e7e789` | archived `manifests/manifest_4e7e789.json` |

Datasets (compiled `datasets/*.npz`, not in Git; hashes in `analysis/*-dataset-provenance.json`):
- `iteration1`: 888 samples from 8 episodes, beta 1.0, w_max 3.0, sha256 `314f4762085c2028…`
- `iteration2`: 1776 samples from 16 episodes, beta 1.0, w_max 3.0, sha256 `00b8d756af88b2a8…`
- `iteration3`: 1776 samples from 16 episodes, beta 0.5, w_max 20.0, sha256 `90a6e5c5cf45aa21…`

Selected checkpoints (`model.safetensors`; weights not in Git, full per-file hashes in `training/iteration*/training.json`):
- `iter1-step000300`: `c231b7569be1fdf0…`, raster held-out 0.0823 vs baseline 0.1333
- `iter2-step000300`: `561e20c26449cca9…`, raster held-out 0.0830 vs baseline 0.1333
- `iter3-step000300`: `69fe2be02f2b2596…`, raster held-out 0.0813 vs baseline 0.1333

Amendments, each recorded before the iteration it governs was trained or evaluated. A1 and A2 also preceded the evidence they depend on; A3 was made after two candidates' development results and says so:
- `amendments/A1-autonomous-mandate.json` (`1898c2b664bb…`)
- `amendments/A2-iteration2-success-density.json` (`1c9f422b6e14…`)
- `amendments/A3-iteration3-advantage-sharpness.json` (`f90189b5b35e…`)

Slurm: `ledger/slurm-jobs.json`, reconciled from `sacct` by
`scripts/reconcile_ledger.py`. 20 stage jobs (3 failed, 17 completed) and 17 support jobs (driver ticks, submission probes). The three failures are the two smokes whose causes were fixed (read-only logs, unloadable checkpoint) and the CPU reload (checkpoints need a GPU); each was diagnosed, fixed and rerun, and is kept in the ledger.

## Infrastructure defects found and fixed during the campaign

Each has a regression test in `tests/`.

1. The AWR cap could never bind (max-subtraction before `exp`); now clipped in log space.
2. `save_pretrained` drops the draccus `type` key, so every saved checkpoint was unloadable.
3. LeHome's logger needs a writable checkout; the campaign mounted it read-only.
4. `slurm/_common.sh` (the container invocation) was not covered by the source pin.
5. Checkpoints load only on a GPU (the preprocessor pins `cuda`); the CPU reload gate failed.
6. The site `sbatch` wrapper expands `$@` unquoted, word-splitting every argument.
7. A driver tick `scancel`'d its own job and died before saving state; ledger adoption now closes that window.
8. A killed tick's lock blocked all ticks; holders are now checked for liveness.
9. `--begin` timestamps were read in local time (UTC-7); begin times are now relative.
10. A budget-skipped final test would have been scored 0/0; it is now reported as not run.
11. A dry-run tick wrote a resolved plan to disk.
12. The held-out evaluation reseeded the global RNG mid-training; state is now saved and restored.
