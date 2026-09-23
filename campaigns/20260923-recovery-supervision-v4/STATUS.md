# Recovery supervision v4 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none submitted yet |
| **Current result** | none — campaign being frozen |
| **Limitation / blocker** | none |
| **Next automatic action** | end-to-end smoke, then the 8-row scaled search |
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
