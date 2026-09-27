# Anchor diagnostic v8 — final report

**Outcome: not improved under the preregistered rule. The untouched baseline
remains the deliverable, and the final set stays unspent.** Measured side by
side in the same Slurm arrays (v5's protocol, two runs each),
`an1-step000300` settled **9/16 vs 2/16 at H10** (margin 7, one-sided Fisher
p = 0.0117). That is the first H10 pass in any matched comparison. It settled
**6/16 vs 8/16 at H50**, failing non-regression. The whole H50 deficit is one
row, dev01 H50: baseline 2/2, an1 0/2. All 64 episodes are valid.

**The anchor hypothesis is supported, on the preregistered mechanistic
endpoints, which were read before the counts.**

## What ran

v7's diagnosis traced every fine-tune's H50 deficit to a shift they all share
early in H50 episodes, and named the byte-identical BC anchor (12 tops +
4 pants) as the main suspect. v8 replaced only that anchor, with 16 pants
demonstrations (8 Pant_Short + 8 Pant_Long, the same size and frame schedule,
each pinned by SHA-256). Everything else was reused unchanged:
- v7's 6,375-label corpus, byte for byte;
- v6's recipe (lr 3.3e-6, fp32 master weights, batch 4 × grad-accum 8, 300 steps);
- the latest-guard-passing selection rule;
- v5's matched evaluation.

The pre-launch review confirmed 10 findings and fixed all of them before any GPU task
(`analysis/review/prelaunch-review.json`).

| stage | result |
|---|---|
| corpus | reused v7 corpus, SHA verified (`c5f5c88e…`) |
| training | guard held at every checkpoint except 225; step 300 selected |
| retention guard (≤ 1.10×) | 0.0771 vs 0.0714 (1.08×) — pass |
| reload, fit gate | reload ok; paired diff 0.00066, 95% CI [−0.0022, 0.0032] — pass |
| endpoints (E1, E2) | **supported**, and the positive control held |
| development screen | H10 pass, **H50 fail**, so the confirmatory final set was not run |

## Result

| settled / reached / n | baseline H10 | an1 H10 | baseline H50 | an1 H50 |
|---|---|---|---|---|
| P_A (dev00/04/06) | 0/0/6 | 5/5/6 | 0/0/6 | 0/2/6 |
| P_B (dev01/03) | 0/0/4 | 1/2/4 | 4/4/4 | 2/2/4 |
| P_C (dev02/05/07) | 2/4/6 | 3/4/6 | 4/4/6 | 4/4/6 |
| **pooled** | **2/16** | **9/16** | **8/16** | **6/16** |

"Reached" is counted from the per-step geometric trace.

Per-row H10 difference (an1 − baseline, settled, over 2 runs): dev00 +1, dev01 0,
dev02 +1, dev03 +1, dev04 +2, dev05 −1, dev06 +2, dev07 +1.

**Endpoints.**
- **E1:** in the first H50 chunk, an1 opens only the right gripper in 0/16
  episodes (both grippers in 16/16). The baseline does so in 2/16.
- **E2:** dev03 lift by action 150 is 0.1166 / 0.1163 m for an1 and 0.1169 m
  for the baseline. a2, pb1 and dq1 lifted 0.0085–0.0105 m.

## Findings

These were verified by 4 lenses, each checked by two independent skeptics
(`analysis/diagnosis/diagnosis.json`).

1. **Changing the anchor removed the shared early-H50 shift.**
   - The v2 AWR fine-tunes used the same 16 anchor files, and they show the
     same right-only first chunk on 7, 6 and 7 of 8 H50 rows. The shift is
     older than the recovery corpus.
   - an1 settled dev03 H50 2/2, the first of the v5–v8 matched fine-tunes to
     do so. v2 iter1 and iter3 each settled it once despite the shift, so a
     late lift does not by itself lose dev03.
   - The attribution has three confounds: the anchor's composition, its
     source (`storm_capture` → `storm_capture100`) and the selected step
     (300 here, dq1's 200). They cannot be separated.
2. **dev01 H50 is the whole H50 deficit, and it is a late fold miss.**
   - an1 makes the baseline's bimanual waist grasp at step 75 (2/2; a2, pb1
     and dq1 0/6) and first lifts above 5 cm at action 87 (baseline 88).
   - It folds at actions 426–428 instead of about 548, with its left gripper
     about 3 cm further inboard. Condition c1 peaks at −1.17 / −0.94 cm.
   - The policy then returns to rest (idle 0.57 over actions 501–600) and
     does not retry. It ends with only c1 failing, at −3.07 / −3.10 cm.
   - Just before the fold, margins c1–c3 match the baseline's within about
     0.35 cm, which leans toward fold execution. The cause remains
     inconclusive between fold placement and cloth state, because the
     telemetry logs no landmark positions.
3. **P_A H50 is a trade-off, not a new defect.**
   - Restoring the baseline's early plan also restored the baseline's own
     mid-episode failure. That failure is a right-arm-only close at table
     level with the left arm parked, followed within 45 actions by an inboard
     drag of at least 8 cm and a cloth lift of at least 3 cm.
   - The baseline does this in 8/8 runs at each of dev00 (action 135), dev04
     (173) and dev02 (86) over v5–v8. an1 does it on the same rows (151, 183,
     84) and also at dev06. a2, pb1 and dq1 never do it on those rows, which
     is how they won P_A H50 10/18 against 0/18.
   - an1 gets further than the baseline before the drag. At dev04 it first
     reached all four conditions at action 122 in both runs, held them only
     intermittently, and then the drag undid the fold.
4. **The H10 pass is probably real on these rows, but not robust.**
   - It survives stratification: by row p = 0.012, by pose 0.011, and by
     row × GPU model 0.010 (9/15 vs 1/13). Taking rows as the unit gives
     0.047–0.10.
   - dev04 and dev06 carry +4 of the +7 margin, and dropping either gives
     p = 0.052.
   - dev04 is robust: the baseline settled it 0/10, on both GPU models. dev06
     is not: the baseline fails it only on A40 (0/6, against 4/4 on RTX 8000),
     and all four v8 dev06 episodes ran on A40.
   - 4 of the 9 wins end within 1 cm of the threshold, and dev05 regressed
     (0/2, where a2, pb1 and dq1 settled 8/8).
   - The baseline's 2/16 is its lowest matched H10 score (5, 5, 3, 2 over
     v5–v8). A matched rerun would pass the H10 clause about 35–45% of the
     time if the GPU is ignored, or 18–52% depending on the GPU mix. It would
     pass the full rule 10–20% of the time.
5. **The GPU model is an uncontrolled second quasi-seed.**
   - The noise-RNG digests are identical at every replan for every label in
     v2–v8.
   - Same policy, same seed, same GPU model: the two runs replay identical
     actions for a stretch (median about 70 actions at H10). On different
     models they differ from action 1 (0 of 219 H10 pairs share an action).
   - Within rows, the baseline's H10 outcomes depend on the GPU model
     (permutation p ≤ 0.0014 in two independent tests).
   - v8 ran 51 of 64 episodes on A40, which is why its replicates are tightly
     coupled. The effective n is about 8–13 per arm, not 16.
   - The GPU model is recorded only in `rollout.log`.
6. **dev01 has no supervision to fix it.**
   - an1 ends dev01 H50 with c2 folded, c1 open, and the policy parked. In
     v7's H10 search this state is a dead end: on PS_M1_089 at P_B, 73 of 192
     attempts enter it and 1 settles, against 47 of the 106 that never enter
     it.
   - The corpus holds 3 of 1,279 P_B labels inside such runs, and no
     H50-format labels (6,263 are H10 and 112 have horizon −1).
   - dev01's mesh, PS_049, is the final set's mesh.

**Hidden by the settled counts at dev01 H10.**
- an1 idles through actions 50–200 and makes its first table-level close at
  action 406 (baseline: 107, lift 0.124 m).
- an1 r1 also loses the baseline's c2 fold (−5.91 vs +4.54 cm).

## Corrections to earlier reports

- **v7's diagnosis.** Its suspect, the anchor, is supported. Its premise that
  the dev01 waist grasp needed P_B or PS_049 supervision is wrong: v8
  recovered the grasp on a byte-identical corpus. Its practical conclusion,
  that dev01 H50 cannot be fixed with the current supervision, still stands.
- **"Common random numbers rule out seed luck."** This is withdrawn.
  Identical RNG streams are a property of the runner. GPU-model assignment is
  not controlled (finding 5).
- **"The dev03 recovery is the first by any fine-tune."** This is wrong. It
  is the first among the v5–v8 matched fine-tunes; v2 iter1 and iter3 each
  settled dev03 once.
- **"No fine-tune has beaten the baseline on PS_049."** This is false. pb1
  scored 2/8 vs 1/8. Pooled over v5–v8 the score is 3/32 vs 6/32
  (two-sided p = 0.47), which is underpowered either way.

## Why the final set was not spent

- The preregistration allowed it only after a passing development screen.
- It cannot see the clause that failed: its 8 rows use one PS_049 mesh at
  6 distinct poses (final04 is the P_C development pose), with no P_A or P_B
  row.
- Its joint power for an1 is about 8–30% across transfer models.
- Both checkpoints ship with `n_action_steps = 50` (checked in `config.json`),
  and the challenge loader passes no override. A shipped an1 therefore runs
  at H50, where it scores 6/16 vs 8/16.

## Limitations

- **Evaluation is in-sample and narrow.** Only Pant_Short is evaluated, on
  8 development poses. Nine checkpoints have been scored on these rows, so a
  development pass is only a screen, and the best checkpoint's effect size is
  inflated.
- **Seeds are reused.** Every campaign has used seeds 200–207, so whether a
  gain holds on fresh seeds has never been measured.
- **Hardware is uncontrolled.** The GPU model is neither recorded in
  `request.json` nor balanced across arms.
- **Per-pose cells are descriptive.** Each holds 4–6 episodes, and the
  replicates are coupled.

## Next step (not launched; needs the user's authorization)

**A GPU-controlled, fresh-seed re-measurement (v9).** This is measurement
only: no training, no checkpoint choice, and the final set is not touched.
The baseline stays the deliverable whatever the result.

- **Policies:** the baseline and `an1-step000300`, both pinned by SHA-256.
- **Rows:** the 8 development poses × H10/H50 × 3 seeds never used before
  (not 200–207 and not 960xx–962xx), shared across horizons and policies.
  That is 96 episodes in one interleaved array throttled at 8, about
  6.1 GPU-h.
- **Hardware:** the GPU model pinned by a Slurm constraint. The cluster
  exposes `a40` and `rtx8000` as node features. The GPU model is recorded
  per episode.
- **Primary report:** `matched_verdict` applied unchanged to 24 vs 24 per
  horizon. It is reported, not used as a gate.
- **Secondary reports:**
  - row-stratified and row sign-flip tests at H10;
  - P_B H50 per row;
  - presence of the right-only drag at dev00, dev02 and dev04;
  - presence of the waist grasp;
  - dev06 by GPU model;
  - terminal margins.
- **Stop rule:** one array, with a single outcome-blind resubmission allowed
  only for a task that left no complete rollout.
- **What it cannot show:** generalization to held-out poses, anchor
  causation, or an unbiased effect size.

Targeted training toward dev01 is not supported, for three reasons:
- the needed supervision is nearly absent (finding 6);
- dev01's mesh is the final-set mesh, so a fix could not be verified
  without contaminating the holdout;
- the P_A drag is the baseline's own behaviour (finding 3).

## Budget and provenance

**Cost.** 67 GPU tasks, 4.64 GPU-hours. All 5 stage jobs COMPLETED with no
retries and no nonzero exits. Nothing needs rerunning.

**Provenance.**
- Manifest `d6f5e5d`, frozen at `9adbd6c`: 55 pinned sources, 16 pinned
  anchor files, the reused corpus and the calibrated endpoints.
- Ledger `ledger/slurm-jobs.json`; decision log `ledger/driver_state.json`.
- Candidate pin `audit/candidate_checkpoint.json` (model `0c5e73b6…`).
- Reload and fit-gate reports in `audit/`.
- Pre-launch review `analysis/review/`; diagnosis
  `analysis/diagnosis/diagnosis.json`.
- Checkpoints and rollouts stay local, with hashes recorded.
