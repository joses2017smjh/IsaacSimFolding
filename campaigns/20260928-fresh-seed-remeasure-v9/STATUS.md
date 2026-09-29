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
- Maintenance reservations, all 08:00–16:00 cluster time:

  | reservation | date | nodes |
  |---|---|---|
  | root_202 | 2026-09-29 | cn-gpu6, cn-r-3/4 and others |
  | root_203 | 2026-09-30 | cn-a/cn-b |
  | root_204 | 2026-10-01 | every GPU node |

- QoS caps: `gpu` 5 concurrent tasks at 8 CPUs, `ampere` 2 GPUs per user.
- The user's own queued A40 jobs from other projects share the ampere cap.
  They are left untouched, per the user's standing instruction.

**Design change.** The approved A40 block stays the **primary**. An RTX
8000 block on the same rows and seeds was added as a **secondary
replication**, because it can start before the A40 maintenance ends. The
added cost is about 7.8 GPU-h, within the user's standing lift of the
GPU-hour constraint.

### 2026-09-28 — pre-launch review

Four lenses, each checked by a skeptic, plus a synthesis
(`analysis/review/prelaunch-review.json`). No blockers.

**Confirmed and fixed before freeze:**
- **Interpretation (5 majors).**
  - The a40 block is named primary.
  - A preregistered platform × policy interaction test now decides any
    claim of hardware dependence; without it, "holds on one block" meant
    nothing.
  - Per-model power is recorded, and the negative label is power-qualified.
  - dev01 H50 has a descriptive rule.
  - The deadline is enforced on each episode's completion time.
- **Robustness (15 minors).**
  - A failing tick keeps the chain alive, and the next tick is secured
    before the old one is cancelled.
  - Every task verifies all 8 checkpoint files, every pinned source and the
    manifest digest. The LeHome checkout and the `lerobot` package are now
    pinned.
  - Dry runs move nothing.
  - External cancellations are logged as deviations.
  - Tests are named by row, with a cluster-stratified test added.
  - Retry states are enumerated.
  - Concurrency and cost statements are corrected.
  - Ticks can also run in `eecs`.
  - The seed scan was broadened to 1,312 files.

**Checked and correct:**
- The exact tests match scipy and exhaustive enumeration.
- The seeds are fresh and correctly shared.
- The checkpoints and sources match their pins.
- A simulation of the driver's retry, deadline and adoption paths behaved
  as specified.
- rtx8000 passes `sbatch --test-only`, and ampere+a40 is accepted and
  queues.
