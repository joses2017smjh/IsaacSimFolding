# Runtime-horizon screen — October 6, 2026

**Decision: the development gate failed; retain the baseline checkpoint and its shipped H50 configuration.** This compares the same weights at two execution horizons, not a newly trained checkpoint. Evidence was audited and added to the public documentation on October 14.

## Recorded experiment

SmolVLA predicts 50 actions. H10 executes ten before observing and replanning; H50 executes the whole chunk. Both use the immutable raster-adapted baseline, three RGB views, and twelve observed joints. Each episode runs 600 policy actions plus 60 fixed hold-settle physics steps. The terminal checker does not control the policy.

There are **48 matched pairs (96 episodes)**: eight reused development poses, six fresh seeds per pose, with each pair on the same NVIDIA A40 and order counterbalanced. All 48 pairs completed and passed the recorded validity checks.

| Paired terminal outcome | Pairs |
|---|---:|
| H10 succeeds, H50 fails | 14 |
| H50 succeeds, H10 fails | 6 |
| Both succeed | 2 |
| Neither succeeds | 26 |
| **Total** | **48** |

H10 settled **16/48** folds; H50 settled **8/48**. The preregistered rule required all pairs valid, at least five additional settled folds, and a one-sided exact paired McNemar p ≤ 0.05. The 20 discordant pairs give **p = 0.057659149169921875**; the full rule therefore fails despite the numerical difference.

[Unmodified recorded summary](evidence/runtime_horizon_2026-10-06/results.json) · [Manifest excerpt and source/checkpoint hashes](evidence/runtime_horizon_2026-10-06/manifest_summary.json) · [October 14 verification and raw-record hashes](evidence/runtime_horizon_2026-10-06/verification.json). The audit reconciled all 48 status records and 96 rollout records, then independently recomputed the paired table and p-value. No new rollout was run.

## Limits

- These are reused development poses, not held-out garments or geometry. Fresh seeds do not make the poses independent. Six repeats within each pose remain clustered; the 48-pair test is the registered screen, not proof of population generalization.
- Pose 5 accounts for five of the eight net H10 wins. The result does not establish a broad benefit or justify checkpoint promotion.
- The baseline weights are identical between arms. This runtime intervention must not be pooled with the September v8 candidate-versus-baseline fine-tune comparison.
- This is rasterized cloth simulation on one GPU model. It establishes neither real-robot transfer nor a portable simulator installation. External weights, assets, and environment files are not included in this evidence snapshot.

## Historical fine-tune comparison

The separate [September 27 v8 study](../campaigns/20260926-anchor-diagnostic-v8/REPORT.md) scored candidate versus baseline at H10 **9/16 vs 2/16**, but H50 **6/16 vs 8/16** failed its non-regression gate. That earlier result also retained the baseline. The current hero and camera GIFs are historical recordings, not video evidence of the October runtime study.

## Corrected optimizer measurement

The [stored-weight audit](evidence/runtime_horizon_2026-10-06/optimizer_audit.json) compares exact 16-bit values for all 96,607,440 baseline BF16 expert weights against v3 attempt 1, step 300. **11,142,825 changed: 11.5341%**, correcting the older report's 12.9% value. The recorded v4 reload fraction is 59.3614%, but those checkpoints differ in training run and step; this is not an isolated optimizer causal ablation. Weight movement and successful settled folds remain separate measurements.

## Recompute the paired test without the simulator

From the repository root, using Python's standard library:

```bash
python3 - <<'PYCODE'
import json, math
from pathlib import Path
r = json.loads(Path('docs/evidence/runtime_horizon_2026-10-06/results.json').read_text())
wins = sum(p['h10_settled'] and not p['h50_settled'] for p in r['pairs'])
losses = sum(p['h50_settled'] and not p['h10_settled'] for p in r['pairs'])
n = wins + losses
p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
assert (wins, losses) == (14, 6)
assert p == r['exact_paired_one_sided_p']
assert not r['preregistered_screen_passed']
print(f'{len(r["pairs"])} pairs; p={p}; gate failed')
PYCODE
```
