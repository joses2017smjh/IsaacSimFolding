# Recovery supervision v3 — final report

**Outcome: performance did not improve. The untouched baseline checkpoint is
retained.** The H10 target of ≥6/8 settled folds was not met; no trained
checkpoint qualified even to be screened against the 4/8 bar in attempt 2.
The untouched final evaluation set was, per preregistration, never spent and
remains unseen. This is a completed negative result with three specific,
verified findings that change what any successor should do.

| | |
|---|---|
| Target | development H10 settled ≥ 6/8 (stretch 8/8), repeated matched runs, no H50 regression |
| Best measured | baseline H10 2/8 (v2 run) and 0/8 (v3 run); attempt 1 screen 1/8; attempt 2 never screened |
| Delivered checkpoint | the baseline, `model.safetensors` `4a37d4dce3d96e39…` (byte-identical throughout) |
| Final untouched set | reserved for a qualifying candidate; none qualified; **not spent, still unseen** |
| Budget | 31 of 78 GPU tasks, 4.1 of 36.9 GPU-hours (`ledger/slurm-jobs.json`, reconciled from sacct) |

## What was measured

**Recovery search (the campaign's new supervision source) worked.** From 32
student-visited roots on 8 training-only Pant_Short rows (4 roots per
episode at steps 50/150/250/350), a fixed panel of 4 candidate continuations
(2 H10, 2 H50, fixed seeds) was executed to episode end plus the unchanged
60-step settle: **128/128 attempts completed, 32 settled successes (25%)**,
evenly spread across root depths (7/9/8/8 by depth) and horizons (16 H10,
16 H50), from 15 distinct roots on 6 of 8 rows. At **12 of 32 roots the
identical simulator state succeeded under one candidate seed and failed
under another** — much of the failure is sampling, not capability: the
policy often contains a successful continuation it does not reliably draw.
Every attempt, including all 96 failures and 16 transient-only condition
crossings, is recorded in `recovery/*/recovery.json`. The coverage gate
(≥6 successes, ≥4 roots, ≥3 rows) passed honestly, yielding 1,310 validated
observation→executed-continuation labels (`datasets/recovery.npz`,
`fefe32a8bf4b1831…`), which the review verified byte-aligned to sources.

**Attempt 1** (supervised training on those labels, v2's optimisation
unchanged): retention guard selected step 200; screen H10 **1/8** < 4/8.
The offline-fit diagnostic then showed the candidate predicted its own
training labels *worse* than the baseline — which triggered the adversarial
review (`analysis/attempt2-review.json`, 4 independent lenses + synthesis).

**The review's central discovery — a real optimizer defect, verified from
raw checkpoint bytes:** the trainer built AdamW directly on bf16 expert
weights with bf16 Adam state and no fp32 master copy. At lr 1e-5 most
updates round to zero: only **12.9% of the 96.6M bf16 expert weights changed
at all** in attempt 1's 300 steps — the same share as the raster fine-tune
that produced the baseline, whose learning went through its fp32 vision
tower — and every RMSNorm gain was untouched. Every fine-tune in v2 and v3
attempt 1 effectively trained ~3.3M of the recorded 99.9M parameters. The
review also showed the offline-fit "regression" was largely a self-sample
artifact (86% of sampled labels are the baseline's own draws at identical
observations) and exonerated the data pipeline entirely.

**Attempt 2** (fp32 master weights — repair, not the factor — plus one
factor: effective batch 4→32 by gradient accumulation; dataset, steps, lr,
seed, anchors byte-identical; preregistered in `plans/attempt2.json` with a
dated budget amendment): the repair verifiably took effect at training time
— recovery loss fell 0.068→0.060 and anchor loss 0.098→0.078 across
checkpoints, the first time any campaign fine-tune demonstrably learned its
rollout supervision. But the whole-episode retention guard (held-out
demonstration episodes on garments disjoint from every training source)
breached at **every** checkpoint: 0.084/0.085/0.084 against a 0.0786 limit
(+18% vs +10% tolerance). Preregistered rule: double breach → no candidate,
nothing evaluated, attempt concluded. Cost of the answer: 2 GPU tasks.

## The finding

The constraint has moved twice and is now clearly identified:

1. **v2:** episode-weighted AWR on autonomous rollouts cannot improve this
   policy — its own rollouts contain almost nothing better to imitate
   (and, retroactively, the optimizer could not have moved the expert anyway).
2. **v3 attempt 1:** validated recovery supervision + the bf16-frozen
   optimizer → nothing learned.
3. **v3 attempt 2:** with the optimizer repaired, learning happens — and
   immediately trades away held-out whole-episode behaviour. 9,600 draws
   over ~1,438 unique frames (83% of recovery labels from rows the baseline
   already solves; the 128-frame episode-start anchor at 37.5 passes)
   overfit 96.6M movable parameters within 100 optimizer steps.

**Remaining limitation (the concrete unavailable dependency):** supervision
breadth. Improving this policy now requires either substantially more
validated successful trajectories than 8 search episodes can produce
(scaled recovery search, teleoperated demonstrations — unavailable in this
headless setup — or a validated within-episode value signal), or a
retention-compatible recipe (whole-episode anchor composition was
preregistered by the review as the next single factor; parameter-efficient
or early-stopped updates with denser checkpointing are alternatives). The
optimizer, the label pipeline, the gates and the orchestration are no
longer suspects: each was either verified clean or repaired with a
regression test.

## Honest accounting

- The budget amendment (65→78 tasks) was made after attempt 1's failure;
  its justification — the original cap could never fund the preregistered
  H50 confirmation (69 minimum at zero overhead) — is arithmetic that
  predates any result. GPU-hours (the binding physical resource) ended at
  4.1 of 36.9. Disclosed in `amendments/2026-09-23-attempt2-budget.md`.
- The driver's 120-minute plan deadline and its finalize path were paused
  once (ticks cancelled, recorded in STATUS) while the repair was
  committed; the deadline is now manifest-declared. No gate, threshold,
  row, or selection rule changed at any point; the fit-gate redesign
  *replaced an unpassable draft that had never run* with a strictly
  meaningful check, before any job used either.
- An earlier STATUS entry and commit message said successes came from "17
  of 32 roots"; the correct figure is 15 (17 is the zero-success count).
  Corrected in place with a note.
- Attempt 2's fit gate and screen never executed (blocked upstream by the
  retention guard), so their thresholds were never tested this campaign.

## Reproduction and provenance

```bash
git clone https://github.com/joses2017smjh/IsaacSimFolding.git && cd IsaacSimFolding
C=$PWD/campaigns/20260923-recovery-supervision-v3
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python     # Python 3.11 + numpy/torch
$PY -m pytest $C/tests -q                                     # 26 CPU tests
$PY $C/scripts/verify_sources.py --campaign $C                # 58 pinned sources + immutable baseline
$PY $C/scripts/driver.py --campaign $C status                 # full resumable state
```

Manifest versions per job are in `ledger/slurm-jobs.json` (`manifest_commit`
per entry); archived manifests in `manifests/`; the frozen protocol is
`manifest.json` + `plans/attempt2.json` + the dated amendment. The
adversarial review that redirected attempt 2 is `analysis/attempt2-review.json`
verbatim. Raw trajectories, checkpoints and media are local with hashes
recorded (`analysis/*-provenance.json`, `training/*/training.json`);
per-attempt search records are committed in full.

## Predecessors

- v2 (`campaigns/20260922-closed-loop-training-v2/REPORT.md`): episode-
  weighted AWR, complete negative result; its frozen-test measurement of
  the baseline (H10 0/8, mean conditions 3.0 on 8 unseen garments) stands
  unchanged.
