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
check     MECHANISM: among branches that reached the fold, landed >= 1 cm
          must settle more often than landed < 1 cm (one-sided Fisher
          p < 0.05) -- else landing depth is not a lever here; stop untrained
labels    only branches that settled with terminal min(c1, c2) margin
          >= 1.5 cm and folded by step 450; gate >= 24 qualifying branches
          from >= 8 roots and >= 3 rows per pose, >= 3000 labels; no earlier
          corpus reused (its depth cannot be recovered)
train     v6's recipe verbatim from the untouched baseline (pose-balanced
          awr_weight, lr 3.3e-6, fp32 master weights, latest guard-passing)
evaluate  v5's matched protocol verbatim: baseline and candidate interleaved
          in the same arrays, 2 runs each at H10/H50, same rule; final set
          only if the rule holds
```

The rule needs about 11/16 settled at H10 against a matched baseline near
5/16, so the attempt must turn v6-level reach into deep landings. Passing is
plausible, not likely; the mechanism check and yield curve settle whether
landing depth is teachable even if the attempt stops early.

`STATUS.md` is the live page; `scripts/driver.py` owns every submission.
