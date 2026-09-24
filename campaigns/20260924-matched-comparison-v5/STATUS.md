# Matched comparison v5 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | baseline: H10 3/8+2/8, H50 4/8+3/8; a2-step000250: H10 4/8+4/8, H50 2/8+5/8; not improved (H10 8/16 vs 5/16, p=0.236; H50 7/16 vs 7/16); FINAL: `baseline` delivered |
| **Limitation / blocker** | none |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-24T01:56:02Z by scripts/driver.py._
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

### 2026-09-24 — closed: not improved under the rule; baseline retained

All 64 development episodes valid. H10: candidate **8/16** vs baseline
**5/16** (margin 3, one-sided Fisher p = 0.24) — short of the margin-4,
p < 0.05 rule. H50: **7/16 vs 7/16** — non-regression holds. Final set not
spent. The matched baseline (5/16, 7/16) is well above the earlier
cross-campaign numbers (2/16, 4/8): most of v4's apparent gap was baseline
variance between campaigns. The candidate's edge is pose-specific (P_C 6/6
at H10 vs 1/6). 27 settled successes recorded; see `REPORT.md` and
`gallery.html`.
