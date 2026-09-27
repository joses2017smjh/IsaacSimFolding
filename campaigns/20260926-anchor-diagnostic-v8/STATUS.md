# Anchor diagnostic v8 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | reused corpus verified; endpoints: right-only chunk-1 rows cand 0 vs base 1; dev03 early lift cand 0.1166 m; anchor hypothesis supported; latest guard-passing checkpoint: step_000300, reload ok, fit gate pass; baseline: H10 1/8+1/8, H50 4/8+4/8; an1-step000300: H10 6/8+3/8, H50 3/8+3/8; dev screen not passed (H10 9/16 vs 2/16, p=0.012; H50 6/16 vs 8/16); FINAL: `baseline` delivered |
| **Limitation / blocker** | the candidate did not pass the development screen |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-27T04:16:53Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-26 — opened

From the verified v7 diagnosis: the H50 deficit (P_B) comes from an early-H50
shift every fine-tune shares; the byte-identical BC anchor (12/16 tops) is
the most concrete shared suspect. v8 changes only the anchor (16 pants),
reuses v7's corpus and recipe, judges the anchor hypothesis by preregistered
mechanistic endpoints first, and requires a confirmatory final-set gate for
any positive claim.

### 2026-09-27 — closed: not improved; diagnosis recorded

**Development screen.** H10 passed at 9/16 vs 2/16 (p = 0.0117). H50 failed
at 6/16 vs 8/16, and the whole deficit is dev01 H50 (2/2 vs 0/2). The
endpoints were SUPPORTED, with the positive control met. All 64 episodes are
valid, all 5 stage jobs COMPLETED, and v8 used 4.64 GPU-h.

**Outcome.** The baseline is delivered and the final set is unspent. Nothing
needs rerunning.

**Diagnosis.** It ran 4 lenses, each checked by 2 skeptics, with a synthesis,
and is saved in `analysis/diagnosis/diagnosis.json`.
- dev01 H50 is a late fold miss: c1 ends 3.1 cm short and the policy parks.
- The P_A H50 result is a trade-off: an1 restored the baseline's own
  right-arm drag.
- The H10 pass is fragile: dev04 and dev06 carry it, and dev06's gain exists
  only on A40.
- The GPU model is a newly identified uncontrolled factor.

**Next step.** The recommended next step is a measurement-only, GPU-pinned,
fresh-seed re-measurement, and it is not launched: it needs the user's
authorization. See `REPORT.md`.
