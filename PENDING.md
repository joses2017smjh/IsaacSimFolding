# Pending tasks and stretch goals

_Updated 2026-09-29. The deliverable is still the untouched baseline
`bc_smolvla_raster_ft_full`. Per-campaign detail is in `campaigns/*/REPORT.md`._

## In flight

**v9: fresh-seed, GPU-pinned re-measurement** of the baseline against
v8's `an1-step000300`
([status](campaigns/20260928-fresh-seed-remeasure-v9/STATUS.md) ·
[design](campaigns/20260928-fresh-seed-remeasure-v9/README.md)).
- It is measurement only: 48 fresh-seed dev rows in two pinned blocks. The
  **A40 block** (job 21462341) is primary and the RTX 8000 block (21462342)
  is secondary.
- Both blocks are queued. The A40 nodes are drained for maintenance until
  about Oct 1, and the RTX 8000 array is waiting its turn in priority.
- The driver advances by itself. The deadline is **2026-10-06 00:00 UTC**,
  after which an unfinished block is reported as incomplete.

## Pending

| # | Task | Trigger |
|---|---|---|
| 1 | Check the A40 block starts once maintenance ends. If it cannot finish by the deadline, decide whether to accept "incomplete" or refreeze with a later deadline. | after Oct 1 16:00 (cluster time) |
| 2 | Write the v9 `REPORT.md` from `analysis/remeasure/results.json`, then update `README.md` and `CLAUDE.md`. | both blocks finished, or the deadline passes |
| 3 | Choose the next step from v9's preregistered reading (see "After v9" below). | the v9 report |
| 4 | Stop the tree being permanently dirty. `campaigns/20260921-folding-pilot-v5/audit/smoke_gate.json` and `results/stage4_thompson.json` are regenerated as a side effect; either commit them as results or gitignore them and keep their SHAs. | any time |
| 5 | Clean up the repo root: about 30 untracked `slurm-*.out` files; `SLURM_JOBS.md`, stale since Sep 8; and root `scripts/` copies that differ from the campaign copies actually run. | any time |
| 6 | Promote the five pilot-only modules in `campaigns/20260921-horizon-pilot/src/lehome_fold/` to root `src/`, with tests. | any time |

**After v9** (decided by its preregistered rules, not after seeing the numbers):
- **H10 and H50 both hold on A40.** A deliverable change still needs a new,
  separately preregistered holdout (stretch goal 1).
- **Only H10 holds.** Checkpoints ship at H50, so the policy as shipped does
  not improve. Close the an1 line unless shipping at H10 is preregistered.
- **Not replicated.** Close the an1 line and record the v8 H10 pass as not
  reproduced.

## Stretch goals

1. **A holdout that can confirm a gain.** The current final set covers one
   mesh and no P_A or P_B pose, and its power is 8–30%. A new holdout would
   need:
   - unseen poses in every pose cluster;
   - scoring at the shipping horizon (H50);
   - about **72 episodes per arm** for 80% power at the effect size seen so
     far (baseline 30% → 54%), or about 150 per arm for 30% → 45%.
2. **Fix the dev01 H50 late fold miss.** an1 folds early and leaves
   condition 1 about 3 cm short. The corpus has no H50-format labels, so
   H50 recovery supervision must be collected on meshes outside the holdout.
3. **Keep the early H50 plan without the P_A right-arm drag.** v8 showed
   the two come together: restoring the baseline's early plan brought back
   its drag. Breaking that link is open research.
4. **Separate the anchor's three confounded factors** from v8: composition,
   source capture and selected step.
5. **Evaluate beyond Pant_Short.** Every matched campaign has scored only
   Pant_Short. Tops and long pants have their own checkers.
6. **Make results robust to hardware.** Pin or balance the GPU model in
   every future comparison, and measure how far cloth physics diverges
   across GPU models.
7. **Unbuilt pieces from the original plan:**
   - the RECAP gradient update;
   - a value head used in training;
   - a valid Thompson ranking;
   - π0.5 (blocked by the lerobot 0.4.3 import);
   - the RTX renderer, which segfaults here, so the Storm raster gap remains.
