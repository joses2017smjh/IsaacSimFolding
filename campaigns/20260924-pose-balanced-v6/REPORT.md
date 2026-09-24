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
2. **At H10 it moves through the fold instead of stopping in it.** 9 of its
   13 H10 folds were lost during the policy phase, never in the terminal
   settle. The full fold lasted a median 15 steps. In 6 of the 8 lost folds
   with a clean trace, the same condition — condition 2, one fold's closure
   distance — reopened mid-chunk and stayed open to the end (2 others lost
   condition 3). At H50, where each 50-step chunk runs to completion, the
   folds it reached mostly held.
3. **P_B's H50 loss is not supply-limited.** With ~10× the P_B supervision
   the candidate still settled 0/4 at H50 against the baseline's 3/4.

## Remaining limitation

**Stopping at the fold, not reaching it — and not for lack of supervision.**
64% of the training labels (6,614 of 10,305) are post-fold states from
branches that then held their fold. 87% of settled branches entered the full
fold once and stayed in it, and post-fold labels span a median 169 steps
after the fold, covering the regime the candidate now reaches at about step
190. The candidate still passes through the fold at H10. The open question is
why the fine-tune does not reproduce the folding-to-holding transition its
labels contain when executed at H10. It does so at H50 (P_A 6/6). More data
of the same kind is not the lever.

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
