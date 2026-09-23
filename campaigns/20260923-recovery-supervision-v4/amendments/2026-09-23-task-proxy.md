# Amendment: task-count proxy could not fund a two-attempt success path (2026-09-23, ~23:45Z)

Recorded after attempt 2's screen (4/8) and while its second H10 run is in
progress — **before** any confirmation, H50 or final-set result exists.

## Finding

The driver refused `a2.h50` on the task-count proxy: 39 used + 8 + 16 final
reserve = 63 > 60. The cap was inconsistent before any attempt-2 result:

- `plans/attempt2.json` (committed before attempt 2 trained) declared itself
  "within the frozen v4 budget" with 19 used + 11 for attempt 2 + 32 on the
  success path = **62 > 60**;
- `test_budget_is_hours_primary_and_success_path_is_fundable` asserted only a
  **one-attempt** success path (52), while the preregistered loop allows k = 2;
- the manifest itself declares **GPU-hours the binding cap** and the task count
  "an anti-runaway proxy only". Hours used: **6.94 of 28.0**.

Same class as v3's 65 → 78 amendment: a budget that could never fund its own
preregistered success path.

## Change

`budget.gpu_tasks` 60 → **72**: the two-attempt success path (62, plus the
superseded fit re-run = 63) plus one retry wave (8) with one task of slack.
**`gpu_hours` stays 28.0**, the final-set reserve stays 16, and no evidence
gate changes: the confirmation rule (pooled 16 vs 16 against 2/16, margin ≥ 4,
one-sided Fisher p < 0.05 — which requires the second H10 run at ≥ 4/8),
H50 ≥ 4/8, retention guard, reload, and the final-set reservation are exactly
as frozen. The test now asserts the two-attempt path.

## Reopening

The refused `a2.h50` stage (marked `skipped`) is cleared so the driver
submits it; pending chain ticks were cancelled first so a premature
"confirmation skipped: budget" conclusion could not land.
