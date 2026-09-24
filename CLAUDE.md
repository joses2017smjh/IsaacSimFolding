# CLAUDE.md — orientation for this repository

LeHome garment-folding reproduction: a SmolVLA-450M policy folding cloth in
Isaac Sim 5.1 on a Slurm cluster, scored by the challenge's own
`success_checker_garment_fold`. Written 2026-09-22 against `0e83a7b`.

**The repository is 22 GB and most of it is experiment output, not source.**
`campaigns/` is 18 GB (17 GB of that is `20260921-horizon-pilot`), `results/`
is 3.9 GB, `docs/` is 90 MB. Actual source is `src/` (428 KB) + `scripts/`
(736 KB) + `slurm/` (260 KB).

---

## 1. Read these, in this order, and ignore the rest

Many `.md` files here are **historical records that were true when written**.
`SESSION_STATUS.md` says so itself. Do not treat an old plan as a to-do list.

**In progress (2026-09-24): v5 matched comparison** —
`campaigns/20260924-matched-comparison-v5/STATUS.md` is the live page. Baseline
and v4's `a2-step000250` run side by side in interleaved Slurm arrays on the
same rows, two runs per policy per horizon, unchanged rule; evaluation only.
`campaigns/20260923-recovery-supervision-v4/gallery.html` shows v4's 13
settled successes (candidate that did not clear confirmation) with hashes.

**Previous outcome (2026-09-24): v4 closed, no preregistered improvement.**
3x-scaled recovery search (81/384 settled) + whole-episode anchor + lr
3.3e-6 produced `a2-step000250`: H10 pooled 8/16 vs baseline 2/16 (p = 0.027)
but H50 3/8 vs 4/8 — missed non-regression by one episode. Baseline retained;
final set unspent. Read `campaigns/20260923-recovery-supervision-v4/REPORT.md`
first; its limitation (cross-campaign baseline) sets the next step.

**Previous outcome (2026-09-23, evening).** The recovery-supervision v3
campaign is complete: a bounded simulator-validated recovery search produced
32 settled successes from 128 executed candidate continuations (25%), but
neither training attempt qualified — attempt 1 was blocked by a real
optimizer defect (AdamW ran on bf16 expert weights; ~88% of trainable
parameters could not move, in every v2/v3 fine-tune), and attempt 2, with
the repair in place and demonstrably learning, breached the whole-episode
retention guard at every checkpoint. The **untouched baseline checkpoint
remains the deliverable**. The binding constraint is now supervision
breadth, not the optimizer, labels, or orchestration. Read
`campaigns/20260923-recovery-supervision-v3/REPORT.md` first; v2's report
covers the AWR negative result.

| Live now | File |
|---|---|
| **v5 live status (in progress)** | `campaigns/20260924-matched-comparison-v5/STATUS.md` |
| Final report (v4) — outcome and the step v5 runs | `campaigns/20260923-recovery-supervision-v4/REPORT.md` |
| Final report (v3) | `campaigns/20260923-recovery-supervision-v3/REPORT.md` |
| v3 live status table and record | `campaigns/20260923-recovery-supervision-v3/STATUS.md` |
| v2 final report (AWR negative result; frozen-test baseline measurement) | `campaigns/20260922-closed-loop-training-v2/REPORT.md` |
| v2 status record | `campaigns/20260922-closed-loop-training-v2/STATUS.md` |
| Its frozen protocol (budget, gates, selection rule, targets) | `campaigns/20260922-closed-loop-training-v2/manifest.json` |
| Its order of operations | `campaigns/20260922-closed-loop-training-v2/README.md` |
| Frozen predecessor, closed at `0e83a7b` | `campaigns/20260921-horizon-pilot/STATUS.md` |

`20260922-closed-loop-training-v1` is a **superseded draft**. It was reviewed,
found to have five correctness defects (§6.3-§6.7) and never launched. v2 is
its repair, rebuilt from committed sources. Do not run v1.

Historical — useful as evidence, **not** as instructions:
`SESSION_STATUS.md` (Sep 20), `SESSION_STATUS_2026-09-19.md`, `README.md`
(public-facing, mixes current and superseded results), `docs/PLAN.md`,
`docs/STAGE0.md`, `docs/STORM.md`, `docs/RENDER.md`, `docs/README_long.md`,
and every `campaigns/2026092[01]-*` directory.

**Slurm ledgers.** `SLURM_JOBS.md` at the root is **stale (Sep 8)** — it stops
at job 21214241 while the cluster is now issuing 214xxxxx. The live records are
per-campaign:

```
campaigns/20260922-closed-loop-training-v2/ledger/slurm-jobs.json   # current
campaigns/20260921-horizon-pilot/audit/job_ledger.json              # frozen campaign
campaigns/20260921-horizon-pilot/submissions.jsonl
campaigns/20260921-horizon-pilot/analysis/*/slurm-jobs.json         # one per diagnostic
```

---

## 2. Layout

```
src/lehome_fold/     15 modules, 2.5 kLOC. The paper's ideas. Import-light.
tests/test_pure.py   51 assertions over src/. No pytest, no GPU, no simulator.
scripts/             Stage-era entry points (BC, value head, rollout, Thompson).
slurm/               00_* .. 83_* sbatch, plus _env.sh. Stage-era.
configs/             bc_smolvla.yaml, bc_pi05.yaml.
container/           lehome.def (the .sif itself is gitignored).
external/            git submodule -> lehome-official/lehome-challenge.
docs/                Long-form writeups + demo GIFs/MP4s (90 MB).
results/             383 aggregated JSON/CSV/PNG. Raw .npz/.mp4 are gitignored.
runtime/             Per-job apptainer scratch (home/cache/ov/nv/tmp). Disposable.
campaigns/<date>-<name>/   One self-contained experiment each. See below.
```

### The campaign pattern — this is the important one

Every experiment since 2026-09-19 is a **dated campaign directory that
snapshot-copies the code it runs**, so a completed campaign stays reproducible
while the root tree moves on:

```
campaigns/<date>-<name>/
  STATUS.md          running narrative, appended per experiment
  manifest.json      predeclared rows + SHA-256 of checkpoint AND every source file
  scripts/           a COPY of the scripts, frozen at campaign open
  slurm/             its own sbatch files
  audit/             gates, inventories, job ledger
  analysis/<expt>/   REPORT.md + CSV/JSON provenance, one per diagnostic
  outputs/           per-task request/status/rollout/media  (large, gitignored)
  tests/             campaign-specific pytest files
```

**Root `scripts/` is not the source of truth for anything that currently runs.**
`scripts/render/policy_rollout51.py` is 600 lines; the version that actually
executes rollouts is `campaigns/20260921-horizon-pilot/scripts/render/policy_rollout51.py`
at **3309 lines**. Same story for `rollout_worker.py`, `trainer_loop.py`,
`train_value.py`, `finetune_rasterised.py` — near-duplicates across root and
campaigns, silently divergent. Always check which copy an sbatch invokes.

**`src/` is forked too, and root is not the superset.**
`campaigns/20260921-horizon-pilot/src/lehome_fold/` carries five modules that
**exist nowhere else**: `behavior_telemetry.py`, `folding_geometry.py`,
`policy_media.py`, `strict_observer.py`, `switch_trace.py`. They were never
promoted to root `src/`. Anything importing them needs the *campaign's* `src/`
on `PYTHONPATH`, not the root one.

---

## 3. Key files and how they connect

**Pure half** (`src/lehome_fold/`, numpy/torch only, unit-tested, CI-enforced):

| file | role |
|---|---|
| `awr.py` | `success_residual` → `normalise` → `weights` → `effective_sample_size`. |
| `recap.py` | Advantage-as-text-token conditioning. Imported and used by `scripts/trainer_loop.py:46,182-198` (`binarise`/`balance` + a degeneracy warning); the **gradient update itself is unimplemented**. |
| `labels.py` | Value-head targets, and an explicit account of which ones the released data cannot supply. |
| `calibration.py` | ECE/MCE/Brier; the G2 gate. |
| `ckpt.py` | Atomic checkpoint manifest + staleness guard for async trainer/workers. |
| `thompson.py` | Stage 4 bandit over inference knobs. |
| `splits.py`, `eval_log.py`, `eval_kwargs.py` | Garment holdouts, result logging, evaluator kwarg translation. |

**Glue half** (needs torch + lerobot + Isaac Sim, not unit-testable here):
`storm_obs.py` (builds the USD stage once, re-renders 3 cameras/step, bypasses
the RTX delegate which segfaults on this cluster), `policy_wrap.py` (hooks
`model.embed_prefix` to tap VLA prefix features without a second forward),
`value_head.py`, `storm_eval.py`, `storm_camera.py`.

**The live closed-loop training path** (`campaigns/20260922-closed-loop-training-v2/`):

```
slurm/collect.sbatch   → scripts/run_rollout_task.py
                           └─ subprocess → ../20260921-horizon-pilot/scripts/render/policy_rollout51.py
                              writes rollouts/iteration1/<id>/{rollout.json,trajectory.npz,request.json,status.json}
slurm/compile.sbatch   → scripts/build_rollout_dataset.py
                           reward = terminal conditions_passed/conditions_total
                           advantage = reward − mean(reward)         [lehome_fold.awr.success_residual]
                           weight    = exp(z(advantage)/beta)        [lehome_fold.awr.weights]
                           → datasets/iteration1.npz + analysis/iteration1-dataset-provenance.json
slurm/train.sbatch     → scripts/rollout_weighted_finetune.py
                           each batch: n_rollout drawn ∝ weight, n_anchor from raster BC
                           → training/iteration1/checkpoints/step_NNNNNN/
slurm/benchmark.sbatch → scripts/run_rollout_task.py --phase benchmark   (8 fixed poses)
slurm/boundary.sbatch  → scripts/run_boundary_evaluation.py              (pose-3 action-5/10 roots)
slurm/smoke.sbatch     → all four of the above, once, end to end
```

`manifest.json` holds the predeclared rows: 1 smoke (80 steps), 8 collection
(600 steps, `trajectory_every: 5` → 120 obs/episode → 960 samples ≈ 2.7 GB of
uint8 RGB), 8 benchmark (600 steps). Baseline checkpoint is pinned by SHA-256
and marked `immutable: true`.

---

## 4. Build and test

The pure half runs anywhere. **Verified on 2026-09-22:**

```bash
PYTHONPATH=src /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python tests/test_pure.py
# -> 51 passed, 0 failed   (exit 0)
```

The README's Sep-21 banner says "16 CPU tests pass" — that refers to a
different (campaign) suite or is stale; the root suite is 51. CI runs the same
command on Python 3.11 (`.github/workflows/tests.yml`).

Campaign tests are **pytest**, unlike the root suite's hand-rolled runner. Run
them **inside the container**; on the login node the raw numbers are misleading:

```bash
cd campaigns/20260921-horizon-pilot && /nfs/.../venv/bin/python -m pytest tests/ -q
# login node -> "52 failed, 37 passed, 1 error". Grouped, that is only:
#   4 real failures, all in tests/test_improvement.py  (the 52 are subtests of one test)
#   cause: ModuleNotFoundError: No module named 'pxr'  -- USD, container-only
#   1 ERROR on tests/test_pure.py, which is a SCRIPT, not a pytest module
```

Run the campaign's `test_pure.py` as a script, like the root one. Everything
else passes on the login node:
`pytest tests/rollout_audit_test.py tests/test_behavior_telemetry.py -q` → **11 passed**.

The simulator half has no local run path. It needs `bhl.sif` + the Isaac Sim
5.1 venv, and only runs through `sbatch`.

### Running the campaign — use the driver, never submit stages by hand

The v2 campaign is driven by `scripts/driver.py`, which owns every
submission. It is restartable and duplicate-safe: each stage is keyed in
`ledger/driver_state.json` and never resubmitted while that key holds a job.

```bash
C=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro/campaigns/20260922-closed-loop-training-v2
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
$PY $C/scripts/driver.py --campaign $C tick --dry-run   # what WOULD it submit
$PY $C/scripts/driver.py --campaign $C tick             # advance + schedule next tick
$PY $C/scripts/driver.py --campaign $C status           # resumable state
```

Each tick submits a CPU chain tick that fires when any waited job ends
(Slurm `?` OR-dependency), plus a 2-hour watchdog tick, so progression never
waits on a person. Iterations 2 and 3 run from an explicit committed
`plans/iterationK.json` (build it with `scripts/make_plan.py`, which refuses
rows that break train/evaluation separation); if none exists 90 minutes after
the previous iteration concludes, a preregistered default ladder applies.

**Changing any executed source means refreezing.** `compile.sbatch` and
`train.sbatch` run `verify_sources.py` at start and fail on any drift. The
procedure: commit the source, `git mv manifest.json manifests/manifest_<commit>.json`,
run `build_manifest.py`, commit, push. Never regenerate a manifest in place.

**This section describes v1 and is superseded.** Use
`campaigns/20260922-closed-loop-training-v2/README.md`, whose chain adds a
source-verification preflight, a gate-enforcing compile that exits 4 on a
degenerate signal, and separate development / frozen-test evaluation phases.

<details><summary>superseded v1 chain</summary>

`STATUS.md` names the smoke as the next authorized action. The chain, with the
dependency structure the campaign expects (nothing below has been submitted):

```bash
C=/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro/campaigns/20260922-closed-loop-training-v1

smoke=$(sbatch --parsable        $C/slurm/smoke.sbatch     "$C")
coll=$( sbatch --parsable --array=0-7 --dependency=afterok:$smoke $C/slurm/collect.sbatch "$C")
comp=$( sbatch --parsable --dependency=afterok:$coll  $C/slurm/compile.sbatch "$C")
train=$(sbatch --parsable --dependency=afterok:$comp  $C/slurm/train.sbatch   "$C")
# then, per candidate checkpoint:
sbatch --array=0-7 $C/slurm/benchmark.sbatch "$C" <ckpt> <label>
sbatch             $C/slurm/boundary.sbatch  "$C" <ckpt> <label>
```

Verified 2026-09-22: `run_rollout_task.py --phase smoke --index 0 --dry-run`
exits 0, so the manifest, the pinned baseline SHAs, the runner path, the LeHome
checkout and the asset config all resolve. Nothing above has been submitted.

`afterok:$coll` is correct as written — `compile.sbatch` exits 2 on fewer than 8
trajectories, so a partial collection must never reach the compiler.

</details>

---

## 5. Odd and fragile

**Findings from the 2026-09-22 review are in §6. These are standing hazards.**

- **Three Pythons.** Login-node `python3` is 3.14 and **has no numpy** — any
  ad-hoc analysis must use `/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python`
  (3.11). Stray `__pycache__` carries cpython-311, -312 *and* -314 tags.
- **Four environment routes** in `slurm/_env.sh` (`official`, `train`, `isaac`,
  `source`). Only `isaac` (`bhl.sif` + `$WORKSPACE/venv` + `lehome51-site` on
  `PYTHONPATH`) has ever produced a working rollout. `source` wants a
  `lehome.sif` that was never built. The new campaign's sbatch files bypass
  `_env.sh` entirely and inline their own `apptainer exec`.
- **`apptainer --cleanenv` drops everything.** Any variable an inner script
  needs must be forwarded explicitly (`LH_FORWARD_VARS`, or `--env` in the
  campaign sbatch files). Omitting one has killed jobs in 3 seconds under
  `set -u`.
- **Isaac Sim 5.1's RTX renderer segfaults on this cluster.** Everything goes
  through the Storm rasteriser, which is *why* there is a measured
  renderer-domain gap (action-prediction correlation +0.976 path-traced vs
  −0.038 rasterised on the same checkpoint).
- **Aspirational vs actual.** The README architecture diagram draws one closed
  loop. In reality: RECAP's gradient update is **unimplemented**; the success
  head has never been fit on mixed-class data (the released demonstrations are
  all successes — see `labels.py`'s own docstring); the G2 calibration gate has
  never been reached; `results/stage4_thompson.json` records a Thompson run
  whose earlier numbers were invalidated by a wrong-garment bug. Treat
  `scripts/trainer_loop.py` + `scripts/rollout_worker.py` as never having run a
  full RECAP iteration.
- **Verdicts are latched.** "Official ever-success" fires the first time the
  checker passes and never unfires. Always read the *settled terminal* verdict
  alongside it; a published GIF may show a latched success that later unfolded.
- **~30 `slurm-*.out` files sit untracked at the repo root**, up to 1.1 MB each.
  They are not gitignored.
- **The site `sbatch` word-splits its arguments.** `/apps/slurm/bin/sbatch` is
  `/apps/slurm/current/bin/sbatch --mail-user=$USER $@` with `$@` UNQUOTED, so
  any argument containing a space breaks and any glob character is expanded.
  `--wrap "..."` fails with "Script arguments not permitted". The driver calls
  the real binary and adds `--mail-user` itself.
- **8 GPUs per user** (`QOSMaxGRESPerUser`), shared with every other job you
  run. Array throttles above that are moot; a 16-row evaluation is ~3 waves.
- **Identical seeds do not reproduce outcomes.** Baseline dev00 at H10 scored
  1/4 in the pilot and 3/4 in v2 on the same seed, checkpoint and runner —
  GPU cloth physics is not bitwise deterministic. Compare baseline and
  candidate only when measured in the same campaign, never across campaigns.
- **Checkpoints only load on a GPU.** The saved preprocessor pins
  `device_processor` to `cuda`; on a CPU node `make_pre_post_processors`
  asserts. Reload gates must request a GPU.
- **`save_pretrained` drops the draccus `type` key** from `config.json`, so a
  raw saved SmolVLA checkpoint does not reload. The v2 trainer restores it.
- **The raster retention guard is narrow.** It is the loss on the first 8
  frames of 4 Top_Short demonstrations, and the BC anchor is the first 8 frames
  of 16 demos (2 of them pants). Neither says anything about grasping or
  folding. The matched H50 closed-loop regression check is the real retention
  test.
- **Mixed-class collections confound the advantage.** In v2 iteration 1 every
  pants episode scored 3/4 and every top 2-3/5, so reward differences measured
  garment difficulty, not action quality. The gate checks spread and
  concentration, which that passes.
- **Initial poses are a shared set of ~6.** Pant_Short key 0 on garments 0/7/9
  is the same pose ("P_A"), and the baseline fails it even at H50. The dev
  set's H50 ceiling for the baseline is 4/8.

---

## 6. Code review, 2026-09-22

Reviewed: the uncommitted working tree, `src/lehome_fold/`, and the then-live
`20260922-closed-loop-training-v1` scripts.

**Status: all findings below are now fixed.** §6.2-§6.7 are repaired in
`20260922-closed-loop-training-v2` (commit `d471cd1`), and §6.3 is fixed in
`src/lehome_fold/awr.py` itself (commit `12ea6a5`). §6.1 is **deliberately not
"fixed"**: the horizon pilot's manifest drift is a historical record, and
rewriting it would hide the drift rather than report it. v2 instead pins its
own sources from committed blobs and re-verifies them before every submission.
The findings are kept here as the rationale for what v2 does differently.

### 6.1 Frozen-source hashes have drifted — one runner path is already broken

`campaigns/20260921-horizon-pilot/manifest.json` pins SHA-256 for 142 execution
sources. **Four no longer match:**

```
scripts/finetune_rasterised.py
scripts/render/policy_rollout51.py
tests/rollout_audit_test.py
tests/test_behavior_telemetry.py
```

`scripts/run_improvement_task.py:53-57` raises `ValueError: Frozen source
changed: <rel>` on *any* mismatch, so that entry point **cannot run today**.
`scripts/run_queue_validation_task.py:58-66` was given an explicit exemption for
`policy_rollout51.py` with a comment; `run_improvement_task.py` never got the
same treatment, and neither exempts the other three.

`policy_rollout51.py` is a three-way mismatch:

| | sha256 |
|---|---|
| manifest pin | `40ec556a…` |
| committed at `0e83a7b` | `525d0599…` |
| working tree now | `7e5877e7…` |

So the manifest was **already stale at HEAD** before today's uncommitted edit.
Either regenerate the manifest with `scripts/build_manifest.py` and say so in
`STATUS.md`, or extend the `run_queue_validation_task.py` exemption list — but
do not leave a "frozen, hash-verified" claim next to hashes that do not verify.

**The drift is bookkeeping, not behaviour.** Verified: both drifted test files
pass against the current working-tree runner —
`pytest tests/rollout_audit_test.py tests/test_behavior_telemetry.py -q` →
**11 passed**, including the controller-order regression that reads the live
`policy_rollout51.py`. So the +14 lines are genuinely telemetry-only. The hash
records are what is wrong, not the runner.

### 6.2 The new campaign runs on uncommitted code with no provenance record

`run_rollout_task.py:124-125` executes the horizon-pilot runner, and
`validate_result` **requires** `executed_action_stream` in the npz — a key that
exists only in today's uncommitted `+14` line diff. The campaign therefore
cannot run from a clean checkout of `0e83a7b`.

Worse for a campaign built around provenance: `request.json` records the
checkpoint SHAs and the full command line, but **not the runner's own hash**.
`build_rollout_dataset.py` carries `trajectory_sha256`, `result_sha256`,
`request_sha256` — and no runner hash. Commit the runner change, then add its
digest to `request.json`.

### 6.3 `awr.w_max` is a dead knob, and the docstring justifies it anyway

`awr.weights` subtracts the max before exponentiating:

```python
w = np.exp(np.clip((a - a.max()) / beta, -50.0, 0.0))
return np.clip(w, 0.0, w_max)
```

`exp()` of a non-positive number is in `(0, 1]`, so **any `w_max >= 1` is a
no-op**. Verified — with advantages spread over 0.1, 1.0 and 100.0, and with
`w_max` at 20 or 1e9, `max(w)` is exactly `1.000000` every time. The docstring
("The clip is not cosmetic. Unclipped exponential weights let a single
high-advantage sample dominate a batch") describes a guard that cannot fire,
and the max-subtraction does not change the *ratios* that actually cause
domination.

It is doubly dead downstream: `rollout_weighted_finetune.py:255` does
`probabilities = weights / weights.sum()`, so absolute scale is discarded.
Meanwhile `build_rollout_dataset.py` writes `"w_max": args.w_max` into the
provenance JSON as though it were a recorded hyperparameter.

Fix: either clip before the max-subtraction, or delete `w_max` and stop
recording it. Do not leave a parameter in a provenance file that has no effect.

### 6.4 Nothing gates on a degenerate advantage signal

`normalise` returns all-zeros when `std < 1e-6`, so `weights` returns all-ones
and sampling is uniform. Verified: eight identical rewards → weights
`[1,1,1,1,1,1,1,1]`, ESS 8.0, **no exception**.

That is the case worth guarding. The horizon pilot measured H10 at **0/8**
successes with mean terminal conditions 2.375 — so the eight collection
episodes are likely all failures with a narrow reward spread. If the spread
collapses, this pipeline silently degrades to *uniform imitation of the
policy's own failures*, mixed 50/50 with BC anchors, which is exactly the
outcome the campaign was opened to avoid.

Simulating a plausible eight-episode all-failure collection
(`[2/4,2/4,2/4,1/4,2/5,3/5,2/5,2/5]`) gives a max/min weight ratio of 35× and
an **episode-level ESS of 4.34 of 8** — i.e. half the collection carries almost
no gradient. ESS is computed and written to provenance but nothing reads it.

Add a gate in `build_rollout_dataset.py`: fail closed if `advantages.std()` is
below threshold, or if ESS drops under (say) 3 of 8, before a GPU-hour is spent.

### 6.5 `heldout_loss()` resets the global torch RNG mid-training

`rollout_weighted_finetune.py`:

```python
def heldout_loss() -> float:
    policy.eval()
    torch.manual_seed(args.seed + 1000)
    torch.cuda.manual_seed_all(args.seed + 1000)
```

This is called once before the loop and again at **every checkpoint**. Verified:
SmolVLA's `sample_noise` uses `torch.normal(...)` and `sample_time` uses
`torch.distributions.Beta(...).sample()`, **neither passing a generator**
(`lerobot/policies/smolvla/modeling_smolvla.py:603-617`), so both draw from the
global per-device RNG. Therefore:

- `torch.manual_seed(args.seed)` at setup is overwritten before step 1 ever
  runs, so `--seed 4242` does not drive the flow-matching noise (it still drives
  sample selection, via `np.random.default_rng(args.seed)`);
- with `--steps 300 --checkpoint-every 100`, steps 101-200 and 201-300 both
  begin from the *same* generator state and see a correlated noise sequence.

Deterministic, so it will not look like a bug — it will look like a loss curve.
Save and restore the generator state around the eval, or give the eval its own
`torch.Generator`.

### 6.6 The held-out regression guard is recorded but never enforced

`save()` writes `"heldout_gate": bool(heldout <= initial_holdout * 1.10)` into
each `checkpoint.json` and then does nothing with it. That matches `STATUS.md`
("guard only, not a target"), but it means a candidate that blows past the 10%
regression threshold is saved, and selection must remember to read the field.
Worth at least a loud stderr line at save time.

### 6.7 Smaller items

- **`run_rollout_task.py:180`** — `timeout = 900 if args.phase == "smoke" else 900`.
  Both branches are identical; an intended distinction was lost. The smoke row
  is 80 steps and gets the same 15-minute budget as a 600-step collection row.
  (Sizing itself is fine: the pilot's H10 mean wall time was ~310 s.)
- **`run_boundary_evaluation.py:77`** — `subprocess.run(...)` with **no
  timeout** and no `start_new_session`, unlike `run_rollout_task.py` which
  bounds the child and kills the process group. A hung Isaac child burns the
  full `--time=04:00:00` and leaves `status.json` reading `"state": "running"`.
- **`rollout_weighted_finetune.py` `load_raster_subset`** — the anchor/held-out
  overlap check compares paths from `storm_capture/` against `storm_capture100/`.
  Different directories, so the guard can never fire. Harmless, but it is not
  the protection it reads as.
- **`campaigns/20260922-closed-loop-training-v1/tests/` is empty**, while
  `STATUS.md` says "the campaign scaffold and focused static checks are
  complete." Every other campaign back to `20260919-media` ships tests. The
  compile/AWR path (§6.4) is pure numpy and testable on CPU in seconds.
- **Git-tracked files that are regenerated as a side effect.**
  `campaigns/20260921-folding-pilot-v5/audit/smoke_gate.json` shows
  `passed: false → true` in the working tree. That flip is *legitimate* — it is
  `check_smoke_gate.py`'s output once the five v5 smokes completed (jobs
  21383691/706/707/748/779, the same ids in the horizon-pilot STATUS table).
  The problem is that it is a **generated artifact under version control**, and
  `campaigns/20260921-horizon-pilot/scripts/check_smoke_gate.py:10` re-runs v5's
  checker via `subprocess.run(..., check=True)` and rewrites the file *every
  time* the horizon-pilot gate is verified. So the tree is permanently dirty and
  `git status` can never be used to tell whether anything real changed.
  `results/stage4_thompson.json` (279 → 44 lines) is the same category. Either
  commit these as results, or gitignore them and keep the SHA in the audit JSON.
  `run_improvement_task.py:70` does gate on `smoke_gate.json.passed is True`,
  so the file is load-bearing and should not simply be deleted.

### What is clean

`src/lehome_fold/` is genuinely good: every module's docstring states the
failure mode it exists to prevent, `ckpt.py` gets the atomic-rename and
staleness discipline right, and `labels.py` documents what the released data
*cannot* supply rather than quietly substituting. The pure/glue split is real
and CI-enforced. `build_rollout_dataset.py`'s alignment checks (targets must
begin with the action actually applied after their observation; no padding;
strictly-ascending indices) are the right assertions in the right place.
