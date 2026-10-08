# Pending tasks and stretch goals

_Updated 2026-10-06. The deliverable remains the immutable `bc_smolvla_raster_ft_full` baseline._

**Latest measured outcome: v9 closed on 30 September 2026; the v8 candidate did not reproduce its gain.** All 192 fresh-seed episodes were valid and GPU-pinned. On the primary A40 block, candidate versus baseline settled H10 **6/24 vs 7/24** and H50 **4/24 vs 5/24**. On RTX 8000, H10 was **6/24 vs 9/24** and H50 **3/24 vs 4/24**. Neither preregistered improvement rule passed. The baseline remains the deliverable; the final set remains untouched. See [v9 measured results](campaigns/20260928-fresh-seed-remeasure-v9/analysis/remeasure/results.json) and [status](campaigns/20260928-fresh-seed-remeasure-v9/STATUS.md).

## New bounded screen

**6 October 2026: runtime control intervention queued.** Compare the immutable baseline with a 10-action execution queue against its shipped 50-action queue. Both settings still predict 50 actions, use the same three RGB views and joint state, and execute the same 600-action budget plus 60-step terminal hold. No checker output guides actions. The bounded design uses one full-episode smoke pair followed by 48 fresh-seed development pairs on the same allocated A40 per pair, with counterbalanced order. The preregistered screen requires five additional settled folds and one-sided exact paired p <= 0.05 across all 48 valid pairs. This is a development screen; held-out generalization and hardware deployment remain unmeasured. No result improvement is claimed before the queued measurements finish.

The [frozen protocol](campaigns/20261006-runtime-horizon/manifest.json), [paired rollout worker](campaigns/20261006-runtime-horizon/scripts/run_pair.py), and [submission ledger](campaigns/20261006-runtime-horizon/ledger/submissions.json) are reviewable in this repository. The authoritative running campaign remains `/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/folding`; the repository directory is a byte-identical source snapshot with site-specific paths. Smoke job **21601323** gates the 48-pair array **21601324**, and **21601325** writes the JSON summary after the array ends. It uses at most one A40 concurrently and creates no new Markdown or text documents.

## Pending

| # | Task | Trigger |
|---|---|---|
| 1 | Read the runtime-horizon screen JSON after its bounded queue finishes. Infrastructure failures remain missing data. | smoke and screen terminal |
| 2 | Record the v9 closure in the existing status and repository documentation; create no new `.md` or `.txt` files. | completed in this update |
| 3 | Close the an1 checkpoint line: v9 did not replicate the earlier H10 pass. Evaluate the independent runtime intervention without replacing the baseline weights. | v9 closed |
| 4 | Stop the tree being permanently dirty. `campaigns/20260921-folding-pilot-v5/audit/smoke_gate.json` and `results/stage4_thompson.json` are regenerated as a side effect; either commit them as results or gitignore them and keep their SHAs. | any time |
| 5 | Clean up the repo root: about 30 untracked `slurm-*.out` files; `SLURM_JOBS.md`, stale since Sep 8; and root `scripts/` copies that differ from the campaign copies actually run. | any time |
| 6 | Promote the five pilot-only modules in `campaigns/20260921-horizon-pilot/src/lehome_fold/` to root `src/`, with tests. | any time |

**Historical v9 preregistered decision rules.** The observed result was **not replicated**, so the an1 checkpoint line is closed:
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
   - scoring at the explicitly selected deployment horizon (H50 for the unchanged baseline; H10 would require its own passed screen and separately preregistered holdout);
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


<!-- folding-runtime-result:34c246b0e347e99db81af24659cd5fcf17e35491d6419f81a4c1c41f10bca5e8:begin -->

**Runtime execution-horizon result: No improvement under the preregistered development rule.** H10 settled **16/48** versus H50 **8/48** across 48/48 valid matched pairs. Difference: +8 folds; one-sided exact paired p = 0.0576591. The unchanged baseline weights remain the deliverable. A development-screen pass requires separately preregistered held-out confirmation before promotion; this update establishes no hardware-deployment or generalization result. All results retain the 600-action policy budget and 60-step terminal settling protocol. Source JSON SHA-256: `4a4e7c365ea6a44166dfc27bc080b796b82f693461a5a0596c355de59337ffce`. [Publication receipt](campaigns/20261006-runtime-horizon/analysis/publication_receipt.json). [Measured result JSON](campaigns/20261006-runtime-horizon/analysis/results.json).

<!-- folding-runtime-result:34c246b0e347e99db81af24659cd5fcf17e35491d6419f81a4c1c41f10bca5e8:end -->
