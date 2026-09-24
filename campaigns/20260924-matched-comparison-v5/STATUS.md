# Matched comparison v5 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21404096` run1, `21404097` run2 |
| **Current result** | baseline: H10 0/0+0/0, H50 0/0+0/0; a2-step000250: H10 0/0+0/0, H50 0/0+0/0 |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-24T01:02:02Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-24 — opened

v4 closed at `09db1be` with its candidate `a2-step000250` at H10 8/16 vs
2/16 (p = 0.027) but H50 3/8 vs 4/8, against baseline runs from earlier
campaigns; its report names a same-wave comparison as the next step and the
user asked for it, with the GPU-hour constraint lifted. v5 runs both
policies in the same interleaved Slurm arrays on the same rows, two runs per
policy per horizon, under the unchanged rule (H50 clause made matched:
pooled candidate ≥ pooled baseline, both measured here). N is fixed at two
runs. The final set stays unspent unless the rule holds. Every episode's
media is kept; the gallery shows settled successes only.
