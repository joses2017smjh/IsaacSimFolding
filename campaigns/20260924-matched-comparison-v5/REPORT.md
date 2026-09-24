# Matched comparison v5 — final report

**Outcome: not improved under the preregistered rule. The untouched baseline
checkpoint remains the deliverable; the final set stays unspent.** Measured
side by side, `a2-step000250` folds more often than the baseline at H10
(8/16 vs 5/16) but not by the preregistered margin (3, not 4; one-sided
Fisher p = 0.24), and equals it at H50 (7/16 vs 7/16).

## Result

Both policies ran in the same Slurm arrays on v2's 16 development rows
(8 poses × H10/H50, seeds 200–207), two runs each. All 64 episodes valid.

| Settled terminal success | baseline r1 | r2 | **pooled** | candidate r1 | r2 | **pooled** |
|---|---|---|---|---|---|---|
| H10 | 3/8 | 2/8 | **5/16** | 4/8 | 4/8 | **8/16** |
| H10 mean conditions (of 4) | 3.25 | 2.88 | | 3.38 | 3.13 | |
| H50 | 4/8 | 3/8 | **7/16** | 2/8 | 5/8 | **7/16** |
| H50 mean conditions (of 4) | 3.00 | 2.88 | | 3.00 | 3.50 | |

Rule: H10 margin ≥ 4 ✗ (3) · Fisher p < 0.05 ✗ (0.24) · H50 non-regression ✓
(equal) · retention guard ✓ · reload ✓ · all rows valid ✓.

Per row (S = settled, l = latched then came undone, · = never; digit =
terminal conditions passed):

| row | pose | baseline r1 | r2 | candidate r1 | r2 |
|---|---|---|---|---|---|
| dev00 H10 | P_A | ·3 | S4 | ·3 | l2 |
| dev00 H50 | P_A | ·1 | ·2 | ·2 | ·3 |
| dev01 H10 | P_B | ·3 | ·3 | ·3 | ·3 |
| dev01 H50 | P_B | S4 | ·3 | ·3 | ·3 |
| dev02 H10 | P_C | ·3 | ·2 | S4 | S4 |
| dev02 H50 | P_C | ·2 | ·2 | l3 | S4 |
| dev03 H10 | P_B | S4 | ·3 | ·2 | ·2 |
| dev03 H50 | P_B | S4 | S4 | ·2 | ·2 |
| dev04 H10 | P_A | ·3 | ·2 | S4 | ·2 |
| dev04 H50 | P_A | ·2 | ·1 | S4 | S4 |
| dev05 H10 | P_C | ·2 | ·2 | S4 | S4 |
| dev05 H50 | P_C | S4 | S4 | S4 | S4 |
| dev06 H10 | P_A | S4 | S4 | ·3 | S4 |
| dev06 H50 | P_A | ·3 | ·3 | ·3 | S4 |
| dev07 H10 | P_C | S4 | ·3 | S4 | S4 |
| dev07 H50 | P_C | S4 | S4 | l3 | S4 |

## What v5 settles

1. **The cross-campaign baseline was low.** On the same rows, seeds and
   runner, the baseline scored 5/16 at H10 here against 2/16 pooled over
   v2+v3, and 7/16 at H50 against 4/8 in v2. Most of v4's apparent 8/16-vs-2/16
   gap was baseline variance between campaigns, not a candidate effect.
   Matched, same-array measurement is now the standard for any comparison
   in this repository.
2. **The candidate's H10 advantage is real, pose-specific, and tracks
   supervision density.** It settles every P_C episode at H10 (6/6 over
   dev02/05/07 in both runs; baseline 1/6) and is the more consistent policy
   run to run (4/8, 4/8 vs 3/8, 2/8). It is no better on P_A (2/6 vs 3/6)
   and fails every P_B episode (0/8 over H10 and H50; baseline 4/8), always
   on checker condition 1 (mean margin −3.3 cm vs the baseline's +0.3 cm).
   Grouped by initial pose — `match_pose` identity, never pose key, which
   maps to different poses on different garments — v4's search supplied 53%
   of the training corpus at P_C (1,572 samples; 44 settled branches, 37
   from `Seen_4` key 1, the exact P_C pose), 28% at P_A (838; 23) and 19% at
   P_B (580; 14). The gain sits where supervision was densest and the
   regression where it was thinnest. *Corrected 2026-09-24:* an earlier
   edit compared that search row with the wrong development pose (a
   pose-key error) and called the mechanism unexplained.
3. **H50 is a wash with large run-to-run swing** on identical seeds:
   candidate 2/8 then 5/8, baseline 4/8 then 3/8. Two of the candidate's H50
   misses in run 1 were folds that came undone during the settle (dev02,
   dev07 latched at 3/4).

## Limitations

- 16 episodes per cell. A one-sided test at this N detects roughly a
  doubling of success rate, not a 3-episode edge; the protocol fixed N at
  two runs before any result and no run was added.
- Development poses only (Pant_Short, three pose types). The final set was
  not spent, as preregistered.

## Media

64 episodes recorded; **27 settled successes** (30 ever). `gallery.html`
shows settled successes only with every other episode counted alongside;
`media/INDEX.json` hashes every GIF, MP4 and snapshot with its Slurm task,
seed and rollout digest. Per settled success the final triptych and the
top-camera GIF (102 frames, gif_every 6) are copied under `media/`; the
committed GIF copies are resized to 320 px wide after closure
(`scripts/web_media.py`, not an executed campaign source) with both digests
recorded, and the 640×480 originals stay under `evaluation/`.

## Next step

Denser supervision where it was thinnest — P_B, and P_A, whose folds came
undone before the settle in 3 of the candidate's 6 H10 episodes — with
pose-balanced sampling, measured exactly as v5 measured it. Run as
`campaigns/20260924-pose-balanced-v6/`.

## Budget and provenance

66 GPU tasks (2 smoke + 64 development), all completed, no retries,
**4.75 GPU-hours** (episodes now take ~6 minutes each); the final set (16)
unspent. Wall clock from smoke submission to verdict: 57 minutes. Manifests `3674b9b` (archived, never run) →
`6cdd0d4` (executed). Ledger `ledger/slurm-jobs.json`; decision log
`ledger/driver_state.json`; verdict function `scripts/driver.py::matched_verdict`
with its tests in `tests/test_v5.py`.
