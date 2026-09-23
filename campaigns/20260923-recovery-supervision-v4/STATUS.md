# Recovery supervision v4 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none |
| **Current result** | baseline H10 2/8 + 0/8; recovery search 81 settled successes of 384 attempts (gate pass); attempt 1, H10 r1 2/8, screen failed: H10 2/8 < 4/8; attempt 2 |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-23T22:03:38Z by scripts/driver.py._
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
