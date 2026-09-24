# Pose-balanced supervision v6

One bounded training attempt, chosen by the per-pose transfer analysis
(`analysis/pose-transfer.json`, reproducible with `scripts/pose_transfer.py`
from v4/v5 artifacts only).

**Evidence.** Grouped by initial pose (`match_pose` identity — pose keys map to
different poses on different garments), v5's matched candidate changed exactly
where v4's search had supplied supervision:

| pose | v4 supply (settled / attempts, corpus share) | v5 H10 base → cand | v5 H50 base → cand |
|---|---|---|---|
| P_C | 44/96, 53% | 1/6 → **6/6** | 4/6 → 4/6 |
| P_A | 23/192, 28% | 3/6 → 2/6 | 0/6 → 3/6 |
| P_B | 14/96 (2 rows), 19% | 1/4 → **0/4** | 3/4 → **0/4** |

Every candidate P_B failure is checker condition 1 (mean margin −3.3 cm vs
the baseline's +0.3). Under v5's rule the candidate needs about 11/16 at H10
against a matched baseline of 5/16, so P_B and P_A must both improve while
P_C holds.

**The attempt.** Supply the missing supervision first, then train on it:

```
search    16 training-only rows at the EXACT dev poses, baseline as student,
          v4's machinery: P_B = Seen_6 key 2, Seen_8 key 2 (the only exact
          P_B training rows) x 4 fresh seeds, early roots 30-180 (v4: P_B
          roots >= 240 settled 2/48); P_A = 4 rows x 2 seeds, v4's roots
gate      per pose on the new search, before any image is read:
          >= 24 settled, >= 8 roots, >= 3 rows, else stop untrained
corpus    v4's pinned corpus + new examples; awr_weight balances poses to
          1/3 each (the trainer samples proportional to it; unchanged)
train     v4 attempt 2's recipe verbatim, from the untouched baseline;
          latest guard-passing checkpoint; reload; fit gate
evaluate  v5's matched protocol verbatim: baseline and candidate
          interleaved in the same arrays, 2 runs each at H10/H50, same rule;
          the final set only if the rule holds (7 of its 8 rows are poses
          absent from dev and from every search)
```

No GPU smoke: runner, trainer and reload are byte-identical to validated
predecessors; the CPU-side interface changes (per-row roots, the candidate
record, offline-fit provenance columns) are covered by `tests/test_v6.py`.
`STATUS.md` is the live page; `scripts/driver.py` owns every submission.
