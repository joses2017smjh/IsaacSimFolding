# lehome-fold-repro

Reproducing **[Learning to Fold](https://arxiv.org/abs/2606.27163)** (Larchenko,
2026) — 1st of 62 in the LeHome Challenge 2026 simulation round, 2nd in the
real-world final — on the OSU cluster, in Isaac Sim.

Bimanual SO-ARM101, four garment types, binary fold success. The paper's one
structural idea is that **the policy is its own value function**: the same
network that predicts actions also predicts success, progress and a few
task-relevant futures, and those predictions drive advantage estimation, live
failure detection and candidate selection. Everything else is downstream of it.

Separate from `bhl-robustness-ladder` on purpose — different robot, different
learning paradigm, no MuJoCo sim2sim path. What carries over is the
Slurm/apptainer discipline and the build→gate→submit habit, not a line of code.

Sim2real (camera alignment, augmentation, human-in-the-loop collection) is out
of scope. No hardware.

---

## Status

| | |
|---|---|
| Stage 0 — environment available | **resolved, GO** |
| G0-a — RGB on the pinned stack | **FAILS.** Isaac Sim 5.1 segfaults on this cluster |
| Stages 1–4 — implemented | **yes**, 31/31 unit tests pass |
| Stages 1–2 — runnable now | **yes**, via `LH_ROUTE=train` (no simulator needed) |
| Evaluation, Stage 3 rollouts | **blocked** on the renderer |

Everything the reproduction needs is public, ungated and Apache-2.0: the
[environment](https://github.com/lehome-official/lehome-challenge) (pinned at
`a805ad2`), 1,000 demonstration episodes / 265,798 frames, 10 seen + 2 public
unseen garments per type, and the official scorer. Nothing was reimplemented,
so success rate here can mean the number the leaderboard meant. Full evidence,
with the command behind each claim, in [docs/STAGE0.md](docs/STAGE0.md).

### The blocking problem, and it is the inverse of the one predicted

The plan assumed this needed the 6.0 stack. It is exactly backwards:

- LeHome pins **`isaacsim==5.1.0`** on Python 3.11 — the same stack as the
  locked BHL v51 venv.
- LeHome **requires RGB**: three `TiledCamera`s at 640×480. No ray-cast
  shortcut — a Warp mesh query has no colour to return.
- On this cluster, **5.1's RTX renderer segfaults**, which is the documented
  reason `bhl-robustness-ladder` has no RGB in it at all.
- RTX renders on **6.0**, where LeHome is not pinned.

The probe confirmed it (job 21067439, exit 139):

```
omni::usd::UsdManager::createHydraEngine
  -> libomni.hydra.rtx.plugin.so -> libcarb.scenerenderer-rtx.plugin.so
  -> librtx.scenedb.plugin.so :: carbOnPluginStartup
```

`cn-r-2`, **NVIDIA A40**, driver **595.71.05**. Not missing RT cores — an A40
has them. Not node selection either: every GPU driver here is 595 or 610, and
the 6.0 probe that *worked* ran on `cn-r-4`, same node class. **5.1's RTX
plugins do not support this cluster's drivers; 6.0's do.**

Two branches remain. The organizers' Docker image is being tested — low
probability, since `--nv` injects the *host* driver and a bundled userspace
cannot fix a driver incompatibility. Failing that, porting LeHome to the 6.0
stack, as a **stated deviation**: a ported scorer is no longer self-evidently
the leaderboard's scorer, and that has to be said rather than glossed.

**What it blocks:** evaluation, and Stage 3 rollout collection. **What it does
not block:** Stage 1 and Stage 2 *training*, which read the released LeRobot
dataset and never open a simulator. `LH_ROUTE=train` exists for that split and
reuses the already-built `bhl.sif`, so it needs no container build either.

### Four things the plan got wrong

Verified against the paper and the code, not assumed:

- **The base model is π0.5, not SmolVLA.** SigLIP-So400m/14 → Gemma-2B prefix →
  Gemma-300M action expert, 30-step chunk via flow matching. ~2.3B against
  SmolVLA's ~450M. `pi05` is a first-class policy type in lerobot 0.4.3, so the
  paper's base is directly available. SmolVLA stays as the cheap baseline;
  which one produced a number goes in the run name.
- **Physics is CPU-only and there is one environment.** `--device cpu` is the
  only accepted value; the eval loop is hardcoded to a batch of one. There is
  no `NUM_ENVS`. Parallelism is N Slurm tasks.
- **Isaac Lab is a fork**, not PyPI — `lehome-official/IsaacLab`, via
  `isaaclab.sh`.
- **`embed_prefix` is not callable with a batch dict.** Verified against the
  installed lerobot 0.4.3: pi05 takes `(images, img_masks, tokens, masks)` and
  smolvla takes those plus `state`. The first version of the wrapper assumed a
  batch dict and would have failed on the first real forward pass; it now taps
  the method during the policy's own pass instead, which is signature- and
  version-agnostic.
- **The demonstrations contain no failures.** All 40 garments are `Seen`, all
  episodes are successful scripted demos, and there is no `success` column. So
  the success head cannot be calibrated on them and **G2 sits behind the
  renderer** too. Progress and future heads train fine today. See
  [STAGE0 §3a](docs/STAGE0.md#3a--what-the-demonstrations-do-not-contain).

---

## Running it

Two ways to get the environment plus a training-only third, selected with
`LH_ROUTE`:

- **`official`** (default) — the organizers' Docker image converted to
  apptainer, venv baked in. The environment the leaderboard was scored in.
- **`source`** — `uv sync` plus the forked Isaac Lab. The only editable route.
- **`train`** — lerobot + torch, **no isaacsim**. ~8 GB. Unblocks Stages 1–2.

```bash
cd /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro
python tests/test_pure.py          # 31 tests, no GPU, no simulator

# --- blocked on the renderer -------------------------------------------
sbatch slurm/00_rtx_probe.sbatch          # done: segfault, exit 139
sbatch slurm/01_fetch_official_image.sbatch   # the remaining hope
sbatch slurm/02_fetch_data.sbatch             # assets 1.0 GB + demos 18.9 GB

# --- NOT blocked: Stage 1 and Stage 2 training -------------------------
sbatch slurm/06_install_train_venv.sbatch
CONFIG=configs/bc_smolvla.yaml SEED=0 sbatch slurm/10_train_bc.sbatch
POLICY_PATH=<ckpt> sbatch slurm/12_probe_backbone.sbatch    # pin the feature path
POLICY_PATH=<ckpt> FEATURE_PATH=<pinned> HIDDEN_DIM=<pinned> \
    sbatch slurm/20_train_value.sbatch
sbatch slurm/21_g2_calibration.sbatch      # exit 0 pass / 2 fail / 3 undecidable

# --- blocked: everything that opens a simulator ------------------------
POLICY_TYPE=g0_uniform sbatch slurm/05_g0_floor.sbatch
POLICY_TYPE=candidate  sbatch slurm/11_eval_stage.sbatch
sbatch slurm/30_trainer.sbatch ; sbatch slurm/31_rollout_workers.sbatch
sbatch slurm/40_thompson.sbatch
```

Job ledger: [SLURM_JOBS.md](SLURM_JOBS.md).

Storage, measured: ~45 GB for Stage 0–1; 448 GB free, no per-user quota. Stage
3 is the one to watch — the paper's ~4.3M rollout frames would be **~305 GB**
at the released data's bytes-per-frame, so retention has to be designed.

---

## The stages

| | stage | gate | state |
|---|---|---|---|
| S0 | environment + data | **G0** renderer; official scorer emits a verdict | G0-a fails |
| S1 | BC fine-tune, 10 seen garments/type | **G1** above the G0 floor | trainable now |
| S2 | value head — success, progress, futures | **G2** success head calibrated | partly trainable |
| S3 | RECAP + AWR, async trainer + workers | **G3** trainer/workers agree on version | G3 verified dry |
| S4 | Thompson sampling at inference | gain over defaults, same checkpoint | implemented |

**S1 — BC.** The released data is 40 `Seen` garments only, so Stage 1 cannot
leak into the validation split even by accident. Report per garment type, seen
vs unseen separately, ≥2 seeds; `aggregate_results.py` enforces the shape and
flags a single-seed row as not a result.

**S2 — the value head.** Heads over a **frozen** backbone, so Stage 2 vs Stage
1 is a clean single-mechanism ablation on identical action weights. Buys two
things with no RL: live failure detection, and candidate selection (sample N
chunks, score with your own value head, execute the best). G2 is a gate rather
than a plot because an uncalibrated head makes Stage 3's advantages noise
*silently*.

**S3 — RECAP + AWR.** Advantage conditioning feeds a binarised advantage as a
text token (`"Advantage: positive"`), trains supervised on everything including
failures, and conditions on positive at inference — which sidesteps the fact
that flow matching gives no tractable log-likelihood. AWR runs alongside and is
separately switchable, because RECAP's own source reports conditioning beating
AWR on the same data and which one carries the gain is worth measuring.

G3 is enforced, not assumed: every rollout is stamped with the checkpoint
version and digest that produced it, anything past `--max_lag` is dropped and
counted, and the lag histogram prints every cycle. Verified end to end with
synthetic rollouts (`--dry_run 1`): 80 current accepted, 12 stale and 1
unstamped rejected with reasons.

**One necessary deviation.** LeHome's evaluator saves only *successful*
episodes (`else: clear_episode_buffer()`). RECAP trains on failures — that is
the method. So the *recorder* is ours, inside our own policy class. The
**scorer is not**: `success_checker_garment_fold` still decides what counts as
a fold, unmodified. The deviation is in what gets kept, not in what gets
measured.

**S4 — Thompson sampling.** Beta-Bernoulli over inference hyperparameters
(candidates × chunk length × temperature × flow steps). Fold success is binary
per episode, so the Beta posterior is exact rather than approximate. The
default arm gets a **reserved budget** — in a 3,000-pull simulation Thompson
starved it to 10 pulls, and a baseline with 10 episodes behind it cannot anchor
a "gain over defaults" claim. Cost ratio is reported alongside: an arm that
wins by a point at 8× compute is a different result.

---

## Layout

```
src/lehome_fold/          the paper's method, unit-tested, no simulator needed
  splits.py               the 48 garments; the guard that keeps held-out ones out
  labels.py               value-head targets, and what the demos cannot supply
  value_head.py           success / progress / future heads
  policy_wrap.py          attaching them to a LeRobot VLA (verified, method tap)
  awr.py                  success residual, AWR weights, effective sample size
  recap.py                advantage binarisation and the conditioning prompt
  calibration.py          ECE / MCE / Brier and the G2 gate
  thompson.py             Beta-Bernoulli arms, reserved baseline, gain reporting
  ckpt.py                 checkpoint provenance — G3
  eval_log.py             reads results out of the official evaluator's log
scripts/                  entry points; run_eval.py defers to LeHome's evaluator
tests/test_pure.py        31 tests, runs anywhere numpy + torch exist
slurm/00..40              probe, image, data, install, train, gate, rollout, tune
docs/STAGE0.md            the availability dossier — evidence for every claim
```

Nothing under `external/` is modified. `run_eval.py` injects our policies into
LeHome's own registry, hands control to `scripts.eval` unchanged, and removes
the injected files on exit — that property is what keeps the harness
comparable, and it is worth protecting.

**Every number from this repo is on 48 garments, not the leaderboard's 80.**
The 8 private garments per category never shipped.

## Credit

Environment, assets, demonstrations and scorer are the LeHome Challenge
organizers' work (Apache-2.0). The method is Ilia Larchenko's —
[paper](https://arxiv.org/abs/2606.27163),
[write-up](https://ilialarchenko.com/projects/lehome2026/),
[checkpoint](https://huggingface.co/IliaLarchenko/lehome_sim).
