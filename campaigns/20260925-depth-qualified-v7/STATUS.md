# Depth-qualified supervision v7 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | smoke pass; search: qualifying P_A 30/384, P_B 26/576, P_C 80/384 at cuts {'P_A': 1.5, 'P_B': 1.0, 'P_C': 1.5}; landed-deep settled 169/219 vs shallow 89/191 (gate pass); latest guard-passing checkpoint: step_000200, reload ok, fit gate pass; baseline: H10 2/8+1/8, H50 4/8+3/8; dq1-step000200: H10 4/8+3/8, H50 2/8+4/8; not improved (H10 7/16 vs 3/16, p=0.126; H50 6/16 vs 7/16); FINAL: `baseline` delivered |
| **Limitation / blocker** | the candidate did not meet the preregistered rule |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-26T05:57:24Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-25 — opened

From the verified v6 diagnosis: the fine-tune reaches early but lands shallow.
v7 records landing telemetry in a fresh search, keeps only branches that
settled deep and early, gates on a preregistered check that landing depth
predicts settling, adds large-garment P_B rows, and trains v6's recipe
unchanged; evaluation is v5's matched protocol verbatim.

### 2026-09-26 — closed: not improved; baseline retained

Smoke passed; 1,344-attempt search; mechanism check passed (169/219 vs 89/191,
p = 5.5e-10); cuts P_A 1.5, P_B 1.0, P_C 1.5 cm; 6,375 labels; guard held to
step 200; reload and fit gate passed. Matched: H10 **7/16 vs 3/16** (p = 0.126),
H50 **6/16 vs 7/16**. The H50 deficit is P_B (0/4 vs 3/4), from an early-H50
shift every fine-tune shares. Post-hoc audit of the compile: pass. See
`REPORT.md`. 96 tasks, 24.2 GPU-hours.
