# Amendment: attempt-2 budget and automation (2026-09-23)

Recorded before `plans/attempt2.json` is committed and before any attempt-2
job runs. Dated rather than numbered: the name "A1" already denotes v2's
double-breach rule referenced in `driver.select_checkpoint`.

## Finding

Under `budget.gpu_tasks: 65` with 16 tasks reserved for the final set,
attempt 2's preregistered confirmation was never fundable. Even with zero
attempt-1 overhead (27 tasks), the H50 confirmation leg is refused
(45 + 8 + 16 = 69 > 65) and the driver concludes "confirmation skipped:
budget". A design inconsistency predating any result: the manifest allowed a
second attempt whose success could never be confirmed.

## Change

- `budget.gpu_tasks`: 65 → 78. `budget.gpu_hours`: 36.9 unchanged, and now
  actually enforced — `within_gpu_hours()` was defined but never called; the
  driver now checks it before every GPU submission.
- The driver's 120-minute attempt-2 plan deadline was a hardcode absent from
  the manifest; it is now manifest-declared
  (`automation.attempt2_plan_deadline_minutes: 480`).

## Derivation

29 tasks used (true ledger sum, reconciled from sacct: 3.65 GPU-h).
Attempt 2: train 1 + reload 1 + offline-fit gate 1 + screen 8 + H10 run-2 8 +
H50 8 = 27 (→ 56). Final untouched set 16 (→ 72). Retry allowance 5, one row
per 8-row rollout stage (→ 77). One manual single-optimizer-step training
smoke of the new fp32/accumulation path before the plan commit, ledgered as
manual (→ 78).

## Restriction

The 13 added tasks fund only attempt 2's preregistered stages, their single
automatic infrastructure retries, the smoke, and the final set. No attempt 3,
no extra evaluation, and no change to any gate, threshold, row or selection
rule. The ≥4/8 screen, the confirmation rules and the final-set reservation
are unchanged.

## Disclosure

v2 + v3 GPU tasks may reach 105 + 78 = 183 against v2's original 170-task
note; GPU-hours stay ≈9 of 36.9 for v3 (≈17 of 45 across v2+v3). This
amendment is made after attempt 1's negative result; its justification is the
preexisting arithmetic above, which is independent of that result.
