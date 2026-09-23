# Recovery supervision v3 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | `21401800` a1.train |
| **Current result** | baseline H10 2/8 + 0/8; recovery search 32 settled successes of 128 attempts (gate pass); attempt 1 |
| **Limitation / blocker** | none |
| **Next automatic action** | driver advances the next stage when a waited job ends |

_Updated 2026-09-23T15:13:23Z by scripts/driver.py._
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

### 2026-09-23 — recovery search: gate passed

Baseline H10 repeat (fresh, this campaign): **0/8** settled (mean conditions
2.50). Preregistered pooled baseline with v2's run: **2/16**.

Recovery search, 8 training-only rows x 4 roots x 4 fixed candidates = **128
attempts, all completed, 32 settled successes (25%)** — 16/64 at H10 and
16/64 at H50 — from **17 of 32 roots across 6 of 8 rows**, including every
P_A garment (the pose the baseline fails even at H50). 16 further attempts
crossed full conditions only transiently and did not settle; they are
recorded and excluded. Two rows (Seen_6 and Seen_8 at key 2, pose P_B)
produced no success from any root. Coverage gate (>= 6 / 4 roots / 3 rows)
passed. Dataset: 1310 labelled samples (1142 from validated recovery
branches, 168 from student episodes that themselves settled).

Attempt 1 training (`21401800`) uses v2's optimisation unchanged; only the
supervision differs.
