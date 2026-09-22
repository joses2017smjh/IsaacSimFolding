# On-policy H50 rescue-oracle diagnostic

This inference-only audit used the untouched original baseline checkpoint and exact validated pose-3 action-5/action-10 snapshots.
Each root was reached by ordinary fresh H10 continuation for +10, +20 and +30 actions, then compared normal H10 continuation with one fresh 50-action open-loop H50 chunk from the identical restored root.

- Actual student-visited roots: `6`.
- Roots with settled fresh-H50 4/4 recovery: `0`.
- H50 branches rescuing a failed H10 continuation: `0`.
- Validated intermediate observation→plan-suffix examples: `0`.
- Reliable all-root H50 recovery gate: `False`.
- Exact root restoration/equivalence: `True`.
- Exact RNG restoration: `True`.
- Zero simulator advancement during prediction: `True`.
- Decision: `stop_self_distillation_h50_insufficient_oracle`.

Fresh H50 did not recover every student-visited failure root reliably. Do not launch self-distillation. Use a true recovery source next: validated simulator DAgger/teleoperation, privileged scripted recovery if available, or a separately preregistered branch-evaluated candidate search.

The root snapshots and observations are in `student-roots.npz`; successful H50 observations and exact unconsumed suffix targets are in `successful-h50-examples.npz`.
