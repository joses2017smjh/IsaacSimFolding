# Fresh-seed re-measurement v9 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none |
| **Current result** | not started |
| **Limitation / blocker** | pre-launch review pending |
| **Next automatic action** | none until launch |
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-28 — opened

**Authorization.** The user authorized the v8 report's recommended next
step ("yes proceed"): a measurement-only, fresh-seed, GPU-pinned
re-measurement of the baseline vs an1-step000300 on the 8 development poses.

**Cluster state at build.**
- Every public A40 node (cn-r-*, cn-s-*, cn-t-1) was drained for
  maintenance. The only A40 up, sail-gpu0, is private (it is otherwise
  reachable only through `preempt`).
- cn-gpu6 and cn-gpu7 (RTX 8000) were up but fully allocated. cn-gpu5 was
  drained.
- Maintenance reservations: 2026-09-29 (cn-gpu6, cn-r-3/4) and 2026-10-01
  (all GPU nodes).

**Design change.** The approved A40 block was kept, and an RTX 8000 block
on the same rows and seeds was added. Each runs when its hardware frees up,
and the pair measures the GPU effect directly. The cost doubles (about
12 GPU-h), within the user's standing lift of the GPU-hour constraint.
