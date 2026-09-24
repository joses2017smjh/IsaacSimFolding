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
2. **It does not hold the fold at H10.** 9 of its 13 H10 folds came undone,
   a median of **6 steps** after they formed (range 1–40): replanning every
   10 steps, the policy moves through the folded state instead of stopping
   and releasing. At H50, where each 50-step chunk runs to completion, the
   folds it reached mostly held.
3. **P_B's H50 loss is not supply-limited.** With ~10× the P_B supervision
   the candidate still settled 0/4 at H50 against the baseline's 3/4.

## Remaining limitation

**Fold holding, not fold reaching.** Settled-success branch labels teach the
motion into the fold, but nothing in them teaches ending it: stopping,
releasing and staying clear once the garment is folded. The next lever is
supervision of the post-fold phase — not more reach supervision, and not
another pose rebalance.

## Budget and provenance

83 GPU tasks, 14.06 GPU-hours, all completed, no retries. Manifest
`9aa80b4` (frozen at launch); ledger `ledger/slurm-jobs.json`; decision log
`ledger/driver_state.json`; corpus provenance and per-pose gate in
`analysis/recovery-dataset-provenance.json` and `audit/recovery_coverage_gate.json`;
candidate pin `audit/candidate_checkpoint.json`. Bulk corpus, checkpoints and
rollouts stay local with their hashes recorded.
