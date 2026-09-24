# Pose-balanced supervision v6 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | targeted search: P_A 38/384, P_B 123/384 settled (gate pass); latest guard-passing checkpoint: step_000200, reload ok, fit gate pass; baseline: H10 3/8+2/8, H50 3/8+4/8; pb1-step000200: H10 1/8+3/8, H50 4/8+4/8; not improved (H10 4/16 vs 5/16, p=0.783; H50 8/16 vs 7/16); FINAL: `baseline` delivered |
| **Limitation / blocker** | the candidate did not meet the preregistered rule |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-24T20:44:11Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-24 — opened

The per-pose transfer analysis over v4/v5 artifacts shows a2's change tracks
per-pose supervision density (README). v6 supplies dense validated
supervision at P_B and P_A, gated per pose, and trains a2's recipe unchanged
on a pose-balanced corpus; evaluation is v5's matched protocol verbatim.

### 2026-09-24 — closed: not improved; baseline retained

Targeted search P_B 123/384, P_A 38/384 settled; per-pose gate passed;
10,305-sample pose-balanced corpus; a2's recipe; guard held to step 200;
reload and fit gate passed. Matched: H10 **4/16 vs 5/16** (p = 0.78), H50
**8/16 vs 7/16**. The candidate reached the fold in 13/16 H10 episodes (vs
5/16) but lost 9 of them a median 6 steps later; at H50 it settled P_A 6/6
(vs 0/6). Remaining limitation: stopping at the fold at H10 — the lost folds
reopen condition 2 mid-chunk — although 64% of the training labels are
post-fold holding states; not a data-supply problem. See `REPORT.md`. 83
tasks, 14.06 GPU-hours.
