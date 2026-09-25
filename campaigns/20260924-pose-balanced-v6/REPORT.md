# Pose-balanced supervision v6 — final report

**Outcome: not improved under the preregistered rule. The untouched baseline
remains the deliverable; the final set stays unspent.** Measured side by
side in the same Slurm arrays (v5's protocol, two runs each), the candidate
`pb1-step000200` settled **4/16 vs 5/16 at H10** (one-sided Fisher p = 0.78)
and **8/16 vs 7/16 at H50**. All 64 episodes were valid.

## What was tried, and why

The per-pose transfer analysis over v4/v5 artifacts (`analysis/pose-transfer.json`)
showed v4's candidate changed where v4's search had supplied supervision:
P_C (53% of the corpus) 1/6 → 6/6 at H10, P_B (19%, two rows) 4/8 → 0/8.
v6 supplied the missing supervision first — a targeted search at the exact
P_B and P_A poses on training-only garments: **P_B 123/384 settled** (v4:
14/96), **P_A 38/384** — passed the preregistered per-pose gate, merged it
with v4's corpus (10,305 samples, pose-balanced to 1/3 each) and trained v4
attempt 2's recipe unchanged. The retention guard held to step 200 (selected);
reload and the fit gate (repair verified, non-inferior) passed.

## Result

| settled / reached / episodes | baseline H10 | baseline H50 | candidate H10 | candidate H50 |
|---|---|---|---|---|
| P_A (dev00/04/06) | 0/0/6 | 0/0/6 | 0/5/6 | **6/6/6** |
| P_B (dev01/03) | 1/1/4 | 3/3/4 | 1/2/4 | 0/0/4 |
| P_C (dev02/05/07) | 4/4/6 | 4/4/6 | 3/6/6 | 2/4/6 |
| **pooled** | **5/5/16** | **7/7/16** | **4/13/16** | **8/10/16** |

"Reached" = the checker passed at some step (latched); "settled" = passed
after the 60-step terminal settle, the preregistered metric. Reached is
descriptive only and carries no claim (`analysis/reach-hold.json`).

## Findings

1. **Dense supervision taught the policy to reach the fold.** At H10 the
   candidate reached it in 13/16 episodes against the baseline's 5/16, and
   early (median policy step 191 vs 461). At H50 it settled every P_A
   episode (6/6 vs 0/6).
2. **At H10 its folds land shallow.** *(Corrected 2026-09-25 after a
   verified diagnosis, `analysis/diagnosis/synthesis.json`; the earlier text
   said it "moves through the fold instead of stopping", which fits only 3 of
   the 10 losses.)* Of its 10 reached-but-unsettled H10 episodes, 3 were
   transient crossings while still carrying the hem, 2 were a released flap
   falling through the fold, 4 were landed folds relaxing open with both
   grippers more than 10 cm away, and 1 (dev04 r2) was reached at step 593
   and lost in the terminal settle. Final breaking condition: condition 2 in
   7 of 9 policy-phase losses, condition 1 in 2. Across all 128 v5+v6
   episodes, a fold released with its closure margin at least 1 cm inside
   the threshold settled 37/41 times; below 1 cm, 13/25 (Fisher p = 0.0008).
   The candidate's H10 folds landed at a median 1.0 cm, against 2.1 cm for
   a2 and 2.45 cm for the baseline. At H50 it reaches later and lands deeper
   (median 1.7 cm), which is why P_A settled 6/6 there.
3. **P_B's H50 loss is a garment-size problem, and rests on about one
   trajectory per row.** Every P_B label ever used came from the two small
   training garments with the P_B pose (Seen_6/8, 8.1 cm closure threshold).
   On dev03's large garment (Seen_3, 9.9 cm class) the candidate grasps
   inboard, where the small garments need it, and never lifts the cloth.
   Same-seed runs are near-replicates, so each P_B H50 cell is about one
   independent trajectory.

## Remaining limitation

**Landing depth, not stopping.** Replan-boundary jumps, joint-space state
shift and differences in label content between corpora were each tested and
refuted. Policies that hold their fold do not stop moving either. The labels
are the baseline's own continuations filtered only by settled success, from a
process that loses about 45% of its in-branch folds. About 65% of them are
post-fold states, but those states are release-and-retract, not stops, and
nothing in the corpus prefers a deep landing. The next attempt filters
supervision by landing depth, which the runner does not yet record.

## Limitations

- Per-pose cells are 4–6 episodes. On identical seeds the matched baseline's
  per-pose H10 counts moved between v5 and v6 (P_A 3/6 → 0/6, P_C 1/6 → 4/6)
  while its pooled counts stayed 5/16 and 7/16. Only pooled numbers carry
  the verdict.
- Robust across both campaigns: the baseline settled P_A at H50 in 0/6
  episodes in v5 and in v6, so the candidate's 6/6 there is a real contrast.
  Not a matched comparison: this candidate's P_C 3/6 against a2's 6/6 in v5
  (different campaigns); do not read it as a regression.
- "Reached" (latched) counts are descriptive; the preregistered metric is
  settled success.

## Budget and provenance

83 GPU tasks, 14.06 GPU-hours, all completed, no retries. Manifest
`9aa80b4` (frozen at launch); ledger `ledger/slurm-jobs.json`; decision log
`ledger/driver_state.json`; corpus provenance and per-pose gate in
`analysis/recovery-dataset-provenance.json` and `audit/recovery_coverage_gate.json`;
candidate pin `audit/candidate_checkpoint.json`. Bulk corpus, checkpoints and
rollouts stay local with their hashes recorded.
