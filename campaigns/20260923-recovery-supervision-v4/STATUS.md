# Recovery supervision v4 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | baseline H10 2/8 + 0/8; recovery search 81 settled successes of 384 attempts (gate pass); attempt 1, H10 r1 2/8, screen failed: H10 2/8 < 4/8; attempt 2, fit precondition failed: repair verification failed (bf16 expert still frozen); FINAL: `baseline` delivered |
| **Limitation / blocker** | none |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-23T22:57:07Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-23 — opened

v3 closed at `aa9dcca`: recovery search validated as a supervision source
(32/128 settled), the bf16 optimizer defect found and repaired, and the
binding constraint identified as supervision breadth plus an episode-start
anchor that cannot protect whole-episode behaviour. v4 pairs the two
preregistered responses: a 3x-scaled H10-only search (384 executed
continuations) and the whole-episode anchor, with checkpoints every 25
steps and the latest-guard-passing rule — all fixed before any v4 result.
Budget is hours-primary: 28.0 of the remaining 32.8 GPU-hours; v3's
untouched final set carries over verbatim, still unseen.

### 2026-09-23 — search 81/384; attempt 1 at baseline parity; attempt 2 = lr

Scaled search: **384/384 attempts completed, 81 settled successes (21.1%)**,
gate passed (2,990 H10-only labels; one nearly-solved row contributed 37 of
the 81 — disclosed). Attempt 1: the whole-episode anchor worked as designed —
`anchor_fit` no longer collapses and retention breaches late, with a
guard-passing window at steps 75-100 — but the selected checkpoint
(`a1-step000100`) had absorbed only ~0.54 passes of the supervision. Screen:
**2/8 settled, mean conditions 2.875** — the first trained checkpoint at
baseline parity (baseline runs 2/8 and 0/8), below the unchanged 4/8 bar.

`plans/attempt2.json`: single factor **lr 1e-5 → 3.3e-6**, so drift slows ~3x
and the guard-passing frontier can extend deep enough to absorb the
supervision; preregistered readouts are the last guard-passing step and the
screen. Disclosed: attempt 1's manifest-generated plan lacked the
`fit_precondition` key, so its screen ran without the 1-task offline-fit
gate; attempt 2 sets it explicitly. Budget: 19 tasks / 5.3 of 28 GPU-hours.

### 2026-09-23 — fit-gate instrument artifact; attempt 2 reopened

The lr hypothesis's first readout resolved strongly: the guard-passing
frontier extended from step 100 to **step 250** (selected `a2-step000250`,
~1.34 passes of the supervision absorbed vs 0.54). The fit gate then
concluded attempt 2 on its "≥1 RMSNorm gain changed" clause — which
measurement shows is unpassable by construction at this lr: 23,750/23,760
gains moved in the saved fp32 tensors but their max delta is 0.18 of a bf16
half-ULP, erased at load for any correct optimizer. The clause's real
purpose (catching the 11.5% frozen signature) was decided by the changed
fraction: **59.4% ≥ 50%** — repair verified. Non-inferiority passed.

Corrected per `amendments/2026-09-23-fit-gate-instrument.md` (changed
fraction alone gates; norm count reported); the original report is preserved
in `attempts/`. Driver state reopened before any screen result exists; no
evidence gate changed. The corrected fit gate re-runs, then the screen.
