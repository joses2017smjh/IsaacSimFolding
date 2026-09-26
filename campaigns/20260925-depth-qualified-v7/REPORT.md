# Depth-qualified supervision v7 — final report

**Outcome: not improved under the preregistered rule. The untouched baseline
remains the deliverable; the final set stays unspent.** Measured side by side
in the same Slurm arrays (v5's protocol, two runs each), `dq1-step000200`
settled **7/16 vs 3/16 at H10** (margin 4, one-sided Fisher p = 0.126; a pass
needed 9/16 against 3) and **6/16 vs 7/16 at H50** (fails non-regression).
All 64 episodes valid.

## What ran

A fresh 28-row baseline search on the training-only garments (1,344 executed
continuations) with new per-step branch telemetry; a preregistered mechanism
check; a depth-qualified, pose-balanced corpus; v6's recipe unchanged; v5's
matched evaluation. The pre-launch review (27 agents) confirmed and fixed 22
findings before any GPU task, including a gate that would have stopped for
supply at P_A (replaced by a per-pose cut ladder) and a fit-gate crash.

| stage | result |
|---|---|
| smoke | telemetry recorded; off-metadata large-garment P_B spawn valid |
| mechanism check | landed-deep settled 169/219 vs shallow 89/191, stratified p = 5.5e-10 (caveat below) |
| ladder cuts | P_A 1.5 cm (30 qualifying branches), P_B 1.0 (26), P_C 1.5 (80) |
| corpus | 6,375 labels (P_A 1,415 · P_B 1,279 · P_C 3,681), 1/3 of draws per pose |
| training | guard held to step 200 (selected); reload and fit gate passed |
| audit (post hoc) | **pass** — every label satisfies the rule; hash chain intact |

## Result

| settled / reached / n | baseline H10 | baseline H50 | candidate H10 | candidate H50 |
|---|---|---|---|---|
| P_A (dev00/04/06) | 1/1/6 | 0/0/6 | 2/4/6 | 1/1/6 |
| P_B (dev01/03) | 0/0/4 | 3/4/4 | 0/0/4 | 0/0/4 |
| P_C (dev02/05/07) | 2/3/6 | 4/4/6 | 5/6/6 | 5/6/6 |
| **pooled** | **3/16** | **7/16** | **7/16** | **6/16** |

## Findings (verified: 4 lenses, 2 independent skeptics each — `analysis/diagnosis/`)

1. **H10 cannot be passed by depth alone.** Of the candidate's 9 H10 losses,
   6 never reached the fold (dev01, dev03, dev06 ×2 each), 2 landed at the
   threshold (dev00) and 1 landed deep and crept open (dev02 r2). Fixing all 3
   depth-type losses gives 10/16 — a pass only against a baseline ≤ 4.
2. **The whole H50 deficit is P_B, in every campaign.** Baseline P_B H50 is
   3/4 in v5, v6 and v7 (9/12); a2, pb1 and dq1 are each 0/4 (0/12). Each cell
   is about one deterministic trajectory, so the clause rests on ~2 trajectories.
3. **That loss comes from a shift every fine-tune shares, not a P_B data gap.**
   From the identical first observation, all three fine-tunes' first H50 chunk
   opens only the right gripper in 38/48 episodes (baseline 6/48), and their
   first lift comes 150–200 actions later on most rows. The same shift turns
   P_A H50 from 0/18 to 10/18. At dev03 the candidate pinches inboard and never
   lifts, exactly as pb1 did, despite 806 labels (21% of draws) from dev03's
   own mesh at its pose. Its cause is unresolved among: the BC anchor
   (byte-identical in every fine-tune, 12/16 records are tops, every right-only
   frame-0 target is from a top), H10-only labels, and the shared recipe.
4. **Landing depth matters, but less than v6 claimed.** The mechanism check
   passed as preregistered, but mostly separates folds still held at release
   from folds already undone (negative margin at release, 7/66 settled). Among
   folds held at release: 169/219 vs 82/125, stratified p = 0.041 — clear at
   P_B, flat at P_C, reversed at P_A. Whether the depth filter deepened the
   candidate's landings is inconclusive on rows both v6 and v7 reached.

## Corrections to earlier reports

- **v6's motivating statistic.** "Below 1 cm settled 13/25" pooled 6 landings
  whose fold had already undone (1 settled) with 19 held shallow ones (12
  settled). On held folds alone: 12/19 vs 37/41 (one-sided p = 0.017).
  Pooled over v5–v7: 14/24 vs 58/63, carried mostly by H50 (8/15 vs 34/34;
  H10 6/9 vs 24/29, p = 0.28).
- **Reach counts** should come from the per-step geometric trace; the sampled
  official checker misses crossings of 2–17 steps (baseline dev07 r1, a2
  dev00/dev06 r1, pb1 dev04 r1).

## Limitations and statistics

- The rule's per-attempt false-positive rate is ~1% and its power low: on a
  matched rerun a2 would pass 8–19% of the time, dq1 2–9%. Four fine-tunes
  have now been scored on the same in-sample dev rows (6 of 8 dev
  mesh-and-pose pairs occur in the search), so any future dev pass is only a
  screen; a positive claim needs a confirmatory final-set run.
- Per-pose cells are 4–6 episodes, r1/r2 share seeds (effective n about half).
- Minor provenance defects, none affecting the result: training.json's
  `heldout` block carries the retention baseline; student labels store
  `step_index = -1` (they sit at steps 0–550); telemetry lacks gripper xyz.

## Next step

A single-variable **anchor diagnostic** (v8): the v7 corpus unchanged, only
the BC anchor replaced by pants demonstrations of the same size and frame
schedule, judged first by preregistered mechanistic endpoints (chunk-1
gripper opening, early lift at dev03), then by the unchanged rule, with a
confirmatory final-set gate for any positive claim.

## Budget and provenance

96 GPU tasks, 24.2 GPU-hours, all completed. Manifest `c74ce4b`; ledger
`ledger/slurm-jobs.json`; decision log `ledger/driver_state.json`; gate report
`audit/recovery_coverage_gate.json`; corpus provenance
`analysis/recovery-dataset-provenance.json`; candidate pin
`audit/candidate_checkpoint.json`; diagnosis and pre-launch review in
`analysis/diagnosis/`. Search records (with telemetry) are committed; bulk
examples, checkpoints and rollouts stay local with hashes recorded.
