# Folding status — 21 September 2026

All v3/v4 Slurm tasks finished, but the required smoke gate did not pass. **Three of five v4 smokes passed infrastructure validation; two failed before policy execution. The 24-rollout horizon pilot has not been submitted. No result supports improved folding.**

This supersedes the pending-job status in the [20 September execution report](FOLDING_EXECUTION_2026-09-20.md), which retains the verified 48-garment inventory, exact historical success episodes, and frozen 8-pose × 3-horizon protocol. [Machine-readable findings](../results/folding_status_2026-09-21.json) separate infrastructure status from policy outcomes.

## Completed v4 results

| Garment | Array task | Infrastructure | Official ever / settled terminal | Terminal conditions |
|---|---|---|---|---|
| Top_Short_Seen_3 | 21370102_0 | Passed | false / false | 2/5 |
| Pant_Short_Seen_0 | 21370102_1 | Passed | false / false | 1/4 |
| Pant_Long_Seen_0 | 21370102_2 | Failed before actions | unknown / unknown | unmeasured |
| Top_Long_Seen_0 | 21370102_3 | Passed | false / false | 2/5 |
| Top_Short_Seen_0 | 21370102_4 | Failed before actions | unknown / unknown | unmeasured |

Each valid smoke executed 120 policy actions, three replans at horizon 50, and 60 terminal settling steps; all had finite cloth/robot states and 121 fresh three-camera acquisitions. The three valid v4 smokes are legitimate policy failures at this short budget, not 600-action performance measurements. Pants have four official conditions and tops have five. Last-link proximity gives a conservative failure proxy; actual contacts, first grasps and failed-grasp counts remain unmeasured.

Policy footage, explicitly labeled as failure:

- [Short top](../campaigns/20260920-folding-pilot-v4/outputs/smoke_Top_Short_Seen_3/rollout_policy_failure.mp4)
- [Short pants](../campaigns/20260920-folding-pilot-v4/outputs/smoke_Pant_Short_Seen_0/rollout_policy_failure.mp4)
- [Long top](../campaigns/20260920-folding-pilot-v4/outputs/smoke_Top_Long_Seen_0/rollout_policy_failure.mp4)
- [Long pants from the valid v2 run](../campaigns/20260920-folding-pilot-v2/outputs/smoke_Pant_Long_Seen_0/rollout_policy_failure.mp4), retained separately from the v4 gate.

Thus actual policy footage now covers all four classes, but eight representative clips and a success/failure pair per class are not yet available.

## Remaining defect and repair

Both failed v4 runs passed the new identity-mapping helper, then `bind_landmarks` evaluated `len(to_orig)` even though PhysX had omitted the unwelded remap table. The resulting `TypeError: len() of unsized object` occurred with zero completed policy actions. These are infrastructure errors, not folding failures. The earlier claim that the v4 unwelded repair was ready was premature: its helper-level tests did not exercise this subsequent guard.

The [v5 repair](../campaigns/20260921-folding-pilot-v5/src/lehome_fold/folding_geometry.py) checks the actual live array shape directly after the verified mapping. It preserves the strict source/rest/count checks and the authoritative forward/reverse maps for welded meshes. A new full-binding regression exercises the actual USD data for both failed unwelded garments and the formerly out-of-range welded short top, using a stand-in for GPU schema access. **All 16 CPU regression tests pass. This is not yet GPU certification.** Prior snapshots and their failure evidence remain unchanged.

## Garment-switch finding

Both diagnostics completed: v3 job **21370082** and v4 job **21370103**. The upstream Top_Long_Seen_0 → Top_Long_Seen_1 switch returned after approximately **1.21 s** and **1.23 s**, respectively. The historical hang was **not reproduced**, so its precise cause remains unresolved.

These traces exercise the native switch but do not validate a second policy episode, post-switch scorer rebinding, or Storm camera retargeting; no observer was passed to the diagnostic. Fresh-process evaluation remains the benchmark path.

## New submissions and next steps

At 2026-09-21 20:46:58 UTC, corrected smoke array **21383691** and dependent switch diagnostic **21383692** were submitted. [Exact receipts and output paths](../campaigns/20260921-folding-pilot-v5/submissions.jsonl). At publication, v5 smoke index 0 had passed; index 1 was running and the other indices were pending. The experiment-level gate remained closed. The snapshot retains the same fixed checkpoint, 8 development poses, seeds and horizons 50/10/5. No retraining was launched and unrelated jobs were preserved.

1. Validate all five v5 smoke outputs, especially long pants and the unwelded short top. A successful Slurm exit alone is insufficient.
2. Once the experiment-level gate passes, submit the already frozen **24-rollout pilot**. Compare native ever-success, settled terminal success, geometric margins, cloth motion, replanning costs and camera validity across matched poses.
3. Claim improvement only after the controlled results support it. If neither shorter horizon helps, inspect the legitimate failure footage and contact/proximity evidence before retraining. Do not compare repaired-scorer rates directly against the historical 8/24.

From this repository directory:

```bash
squeue -u "$USER"
sacct -j 21370102,21370103,21383691,21383692 --format=JobID,State,ExitCode,Elapsed -X
python3 campaigns/20260921-folding-pilot-v5/scripts/check_smoke_gate.py
find campaigns/20260921-folding-pilot-v5/outputs -name status.json -print -exec cat {} \;
# Run only after the smoke checker passes (the submitter enforces the gate):
python3 campaigns/20260921-folding-pilot-v5/scripts/submit_experiments.py pilot
```

The previous partition check found dgx2/dgxh account eligibility, but these failures were code errors. Their GPUs remain unvalidated for this physics-plus-Storm pipeline; switching partitions would not fix the missing-remap guard.
