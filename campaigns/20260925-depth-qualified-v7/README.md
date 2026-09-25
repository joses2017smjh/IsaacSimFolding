# Depth-qualified supervision v7

One bounded training attempt, chosen by the verified v6 diagnosis
(`../20260924-pose-balanced-v6/analysis/diagnosis/synthesis.json`: six
hypothesis analyses, two independent skeptics each).

**What v6 showed.** Its candidate reached the fold at H10 in 13/16 matched
episodes (baseline 5/16) but settled only 4/16: it **lands the fold shallow**
(median closure margin at release 1.0 cm vs the baseline's 2.45). Across all
128 matched episodes, folds landed >= 1 cm inside the threshold settled 37/41;
below 1 cm, 13/25. Replan-boundary jumps, joint-space state shift and
differences in label content were each tested and refuted. The training
labels were selected only by settled success, from a baseline process that
loses about 45% of its in-branch folds, so shallow survivors were in the
corpus and nothing marked them. Separately, every P_B label ever used came
from the two small garments that list the P_B pose, and the candidate's P_B
H50 failure was on dev03's large garment.

**The attempt: change the label criterion, and nothing downstream.**

```
smoke     one short search row: the new branch telemetry is recorded, the
          off-metadata large-garment P_B placement spawns validly, and the
          compiler's functions run on real attempts
search    28 training-only rows, baseline as student, early roots 30-180,
          8 H10 candidates per root, with per-step branch telemetry
          (closure/width margins, gripper-to-cloth distance, cloth lift):
          P_A 4 garments x 2 seeds, P_C 4 x 2, P_B small (Seen_6/8) x 4,
          P_B large (Seen_4/5 pinned at the P_B pose, off-metadata) x 2
check     MECHANISM: among branches that reached the fold and landed inside
          the branch (landings inherited from an already-folded root are
          excluded), landed >= 1 cm must settle more often than < 1 cm --
          exact conditional test stratified by pose, p < 0.05 -- else
          landing depth is not a lever here; stop untrained
labels    branches that settled with first all-4 by action 450 and terminal
          min(c1, c2) margin >= each pose's cut: the strictest of 1.5, 1.0,
          0.5, 0.0 cm (0.0 = settled-only) at which the pose has >= 24
          qualifying branches from >= 8 (row, root) pairs and >= 3 rows;
          >= 3000 labels in total; no earlier corpus reused
train     v6's recipe verbatim from the untouched baseline (pose-balanced
          awr_weight, lr 3.3e-6, fp32 master weights, latest guard-passing)
evaluate  v5's matched protocol verbatim: baseline and candidate interleaved
          in the same arrays, 2 runs each at H10/H50, same rule; final set
          only if the rule holds
```

**Why a cut ladder** (from the pre-launch review, before any v7 data):
every settled baseline P_A fold ever measured ended below 1.47 cm and P_B
rarely clears 1.5 cm, while the depth effect in the matched data comes mainly
from P_C. A single 1.5 cm cut would have stopped the attempt for supply at
P_A. With the ladder, P_A may train on settled-only labels (v6's criterion)
while the depth filter acts where supply allows; the chosen cut per pose is
recorded.

**Same-mesh caveat.** Garments share meshes in groups (PS_049 = Seen_0/1/2,
PS_050 = Seen_3/4/5, PS_M1_089 = Seen_6/7, PS_Short047 = Seen_8/9), differing
only in texture, with identical checker thresholds. The large-garment P_B rows
(Seen_4/5) are therefore dev03's physical garment at dev03's pose, and 6 of 8
dev rows have their mesh at their exact pose somewhere in the search. The dev
set selects candidates; it is not a generalisation test. The final set
(Seen_1/2, PS_049) shares no mesh with any search row and remains the holdout.

The rule needs about 11/16 settled at H10 against a matched baseline near
5/16, so the attempt must turn v6-level reach into deep landings. Passing is
plausible, not likely; the mechanism check and yield curve settle whether
landing depth is teachable even if the attempt stops early.

`STATUS.md` is the live page; `scripts/driver.py` owns every submission.
