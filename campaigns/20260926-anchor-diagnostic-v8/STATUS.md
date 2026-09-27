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
