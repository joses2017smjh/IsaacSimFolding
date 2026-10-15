# Fresh-seed re-measurement v9 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | preflight passed; a40 [primary] (done): H10 6/24 vs 7/24 (p=0.741), H50 4/24 vs 5/24; rtx8000 [secondary] (done): H10 6/24 vs 9/24 (p=0.894), H50 3/24 vs 4/24; a40 (primary): H10 clause fails; H50 clause fails | rtx8000 (secondary): H10 clause fails; H50 clause fails | not replicated at this power (power at p1=0.45: a40 20%, rtx8000 74%); deliverable unchanged: baseline |
| **Limitation / blocker** | none |
| **Next automatic action** | none — campaign complete; see analysis/remeasure/results.json |

_Updated 2026-09-30T13:07:00Z by scripts/driver.py._
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


### 2026-10-06 — closure recorded in existing documents

**Latest measured outcome: v9 closed on 30 September 2026; the v8 candidate did not reproduce its gain.** All 192 fresh-seed episodes were valid and GPU-pinned. On the primary A40 block, candidate versus baseline settled H10 **6/24 vs 7/24** and H50 **4/24 vs 5/24**. On RTX 8000, H10 was **6/24 vs 9/24** and H50 **3/24 vs 4/24**. Neither preregistered improvement rule passed. The baseline remains the deliverable; the final set remains untouched. See [v9 measured results](analysis/remeasure/results.json).

The an1 checkpoint line is closed under the preregistered reading. Runtime execution-horizon tuning is a separate intervention using unchanged baseline weights; its [measured result](../../docs/RUNTIME_HORIZON_2026-10-06.md) does not change the v9 verdict. No new `REPORT.md` was created, following the user's instruction to update only existing Markdown or text documents.
