# Recovery supervision v3 — live status

<!-- driver:status:begin -->
| | |
|---|---|
| **Active job** | none — campaign complete |
| **Current result** | baseline H10 2/8 + 0/8; recovery search 32 settled successes of 128 attempts (gate pass); attempt 1, H10 r1 1/8, screen failed: H10 1/8 < 4/8; attempt 2, double breach: no guard-passing checkpoint (amendment A1); nothing is evaluated; FINAL: `baseline` delivered |
| **Limitation / blocker** | none |
| **Next automatic action** | none — campaign complete; see REPORT.md |

_Updated 2026-09-23T18:51:35Z by scripts/driver.py._
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
16/64 at H50 — from **15 of 32 roots across 6 of 8 rows** (correction: an
earlier version of this entry and the message of commit 9eccef5 said 17,
which is the number of roots with *zero* successes), including every
P_A garment (the pose the baseline fails even at H50). 16 further attempts
crossed full conditions only transiently and did not settle; they are
recorded and excluded. Two rows (Seen_6 and Seen_8 at key 2, pose P_B)
produced no success from any root. Coverage gate (>= 6 / 4 roots / 3 rows)
passed. Dataset: 1310 labelled samples (1142 from validated recovery
branches, 168 from student episodes that themselves settled).

Attempt 1 training (`21401800`) uses v2's optimisation unchanged; only the
supervision differs.

### 2026-09-23 — orchestration paused for the attempt-2 repair

Attempt 1's screen failed (H10 1/8 < 4/8). An adversarial review (4
independent lenses + synthesis; full record in `analysis/attempt2-review.json`)
overturned my working "noise-dominated optimisation" diagnosis and found the
real defect: **the trainer built AdamW directly on bf16 expert weights with
bf16 optimizer state** — at lr 1e-5, 88.5% of the 96.6M expert weights and
every RMSNorm gain could not move (verified from raw checkpoint bytes: only
12.9% of bf16 weights changed at all in 300 steps, the same share as the
raster fine-tune, whose learning went through its fp32 vision tower). The
offline-fit "regression" was largely a self-sample metric artifact, and label
alignment/data integrity were verified clean.

Pending driver ticks 21401952 and 21401704 are cancelled while the repair is
committed and the manifest refrozen: the driver's 120-minute plan deadline is
a hardcode of this campaign's own scheduler, not a preregistered rule, and
letting it finalize mid-repair would end the campaign on a known bug. The
pause is orchestration only; no gate, threshold, row or selection rule
changes. The tick resumes once `plans/attempt2.json` is committed.

### 2026-09-23 — campaign complete: no improvement; baseline retained

Attempt 2 (fp32 repair + effective batch 32) concluded by preregistered
rule: **double breach** — retention loss 0.084/0.085/0.084 at steps
100/200/300 against a 0.0786 limit, so no candidate existed and nothing was
evaluated. The repair verifiably worked at training level (recovery loss
0.068→0.060, anchor 0.098→0.078 — the first campaign fine-tune to
demonstrably learn its rollout supervision); the model traded held-out
whole-episode behaviour for it within 100 optimizer steps. The driver
finalized on the baseline; the untouched final set was never spent and
remains unseen. Full account, evidence chain and the remaining limitation:
`REPORT.md`. Budget: 31 of 78 GPU tasks, 4.1 of 36.9 GPU-hours.
