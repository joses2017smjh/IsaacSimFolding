# Recovery supervision v3 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21401701` baseline.h10, `21401702` search.collect |
| **Current result** | starting |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-23T14:40:43Z by scripts/driver.py._
<!-- driver:status:end -->

## Record

append-only; newest last

### 2026-09-23 — opened

v2 closed as a complete negative result at `857d88e`: no candidate improved
development H10 over the matched baseline, because autonomous rollouts from
this policy contain about one settled success in eight. v3 targets the
missing ingredient — successful actions at states the H10 student actually
visits — with a bounded, simulator-validated recovery search.

Budget: the unused part of v2's allocation, 65 GPU tasks and 36.9 GPU-hours,
reserving 16 tasks for the untouched final set. v2's frozen-test result is
unchanged and not rerun; v3 has its own untouched final set (Pant_Short
garments 1-2 at pose keys 1-4, never trained on or evaluated by any campaign).

### 2026-09-23 — smoke passed; production launched

Smoke `21401700` passed every new component on the first attempt: the
student H10 episode captured a root, both candidate continuations were
restored, executed and settled, and every attempt was recorded with its
per-step condition trace; the compiler correctly refused to train on zero
successes (exit 4); one training step ran with the whole-episode retention
guard (0.0806 vs 0.0808 baseline, four held-out garments) alongside the
anchor-fit metric; the checkpoint reloaded on CUDA. The pre-launch dry run
of every phase also caught and fixed one bug (the recovery phase read its
config as rows).

The driver has submitted the fresh baseline H10 repeat and the 8-row
recovery search.
