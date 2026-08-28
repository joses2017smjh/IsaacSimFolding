# Stage 0 — is the environment still there?

**Verdict: yes. Everything the reproduction needs is public, ungated and
Apache-2.0. The project is a GO, and the blocking risk moved somewhere else.**

The challenge ran Feb–Apr 2026 and concluded at ICRA in June. Post-challenge
availability was listed as unverified and blocking. It was checked on
2026-08-27, first-hand, by cloning the repo and querying the Hub API rather
than by reading the challenge website. Every claim below has the check that
produced it next to it.

---

## 1 · The four blocking items

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Simulation environment package | **available** | `git clone` succeeded; HEAD `a805ad2`, last commit 2026-04-29 |
| 2 | Released demonstrations, 10 seen per type | **available** | `lehome/dataset_challenge_merged` — 1,000 episodes, 265,798 frames |
| 3 | Garment assets, incl. 2 public unseen per type | **available** | `lehome/asset_challenge` — 10 Seen + 2 Unseen × 4 types = 48 |
| 4 | Official evaluation harness | **available** | `scripts/eval.py` + `lehome/utils/success_checker_chanllege.py` |

Nothing had to be reimplemented, so "success rate" can mean the number the
leaderboard meant. The fallback to DexGarmentLab is not needed and is not being
taken.

### 1.1 The environment package

`github.com/lehome-official/lehome-challenge`, Apache-2.0. Pinned here as a
submodule at `external/lehome-challenge` so the reproduction is anchored to one
commit rather than to whatever `main` becomes.

What it pins, read out of `pyproject.toml` rather than off the badges:

```toml
requires-python = ">=3.11,<3.12"
"isaacsim[all,extscache]==5.1.0"
"lerobot==0.4.3"
"torch==2.7.0"    "torchvision==0.22.0"
"transformers>=4.57.6"
"open3d>=0.19.0"  "pinocchio>=0.4.3"  "transforms3d>=0.4.2"  "num2words>=0.5.14"
override-dependencies = ["packaging==23.0", "numpy==1.26.0"]
```

Two things the badges get wrong or omit, both of which matter:

- The README badge says **LeRobot 0.4.2**; `pyproject.toml` pins **0.4.3**.
  Trust the lockfile.
- Isaac Lab is **not** the PyPI package. `docs/installation.md` clones a
  **fork** — `github.com/lehome-official/IsaacLab` (BSD-3, fork of
  `isaac-sim/IsaacLab`, last pushed 2026-02-04) — into `third_party/` and
  installs it with `isaaclab.sh -i none`. Installing stock `isaaclab==2.3.1`
  from PyPI gets a different environment, and the garment task is presumably
  why the fork exists. Verified the fork is public and not archived.

There is also a **prebuilt image**: `lehome/docker` on the Hub ships
`lehome-challenge.tar.gz`, 26,676,771,349 bytes (26.7 GB), public and ungated,
with the venv baked at `/opt/lehome-challenge/.venv`. Apptainer consumes a
docker archive directly, so this converts in one command and sidesteps every
resolver question at once. **This is the default route** (`LH_ROUTE=official`)
precisely because it is the environment the leaderboard was scored in.

### 1.2 The demonstrations

`lehome/dataset_challenge_merged`, 18.9 GB, LeRobot dataset `v3.0`. Read from
`meta/info.json`:

```
fps               30
total_episodes    1000            (250 per garment type)
total_frames      265798          (~266 frames/episode)
observation.state             float32 [12]
action                        float32 [12]
observation.images.top_rgb    video   [480, 640, 3]
observation.images.left_rgb   video   [480, 640, 3]
observation.images.right_rgb  video   [480, 640, 3]
```

The feature names match the environment's observation keys exactly
(`garment_bi_v2.py::_get_observations`), so no key remapping is needed between
training and evaluation — one less place for the data pipeline to be silently
wrong.

`lehome/dataset_challenge` (24.2 GB) is the same data split per garment and
with a depth column. The paper's policy does not use depth; pull it only if
something downstream wants it.

### 1.3 The garments

`lehome/asset_challenge`, 1.0 GB. Counted from the Hub file listing:

| category | Seen | Unseen |
|---|---|---|
| `Top_Long` | 10 | 2 |
| `Top_Short` | 10 | 2 |
| `Pant_Long` | 10 | 2 |
| `Pant_Short` | 10 | 2 |

Exactly as the work order predicted: 10 training + 2 public held-out per type.
The env config exposes `garment_version: "Release" | "Holdout"`; only `Release`
ships, so the 8 private garments per type that made up the rest of the
leaderboard's 20 are **not** reproducible here. Any number from this repo is on
48 garments, not 80, and must say so.

**The 8 Unseen garments are the validation split and must never be trained on.**
That is a named failure mode with no automatic detector, so it is enforced by
keeping the file list in one place and asserting on it at train time.

### 1.4 The scorer

`source/lehome/lehome/utils/success_checker_chanllege.py` (the misspelling is
upstream's). `success_checker_garment_fold(particle_object, garment_type)`
dispatches to per-category geometric checks over the garment's **particle**
positions — `check_top_sleeve`, `check_pant_long`, `check_pant_short`. It is
called from `garment_bi_v2.py::_check_success` / `_get_success`, and
`scripts/utils/evaluation.py` turns it into the per-episode boolean that is
averaged into a success rate.

This repo calls that code unmodified. `scripts/run_eval.py` registers extra
policies into LeHome's own `PolicyRegistry` and then hands off to
`scripts.eval` — no forked eval loop, no second implementation of success.

---

## 2 · Four places the work order was wrong

Each of these was something the work order flagged as worth checking. All four
changed the plan.

### 2.1 The base model is π0.5, not SmolVLA

The work order named `lerobot/smolvla_base` the strong default and asked for it
to be verified before committing. It does not survive the check. From the
paper:

> SigLIP-So400m/14 image encoder → Gemma-2B prefix transformer → Gemma-300M
> action expert that generates a 30-step action chunk via flow matching

That is a **π0.5-class** model, roughly 2.3B parameters against SmolVLA's ~450M
— a different compute profile, a different action-expert size, and a different
VRAM floor. LeHome does ship `configs/train_smolvla.yaml`, so SmolVLA remains a
legitimate *cheaper baseline*, but it is not what the paper did and a number
from it is not a reproduction of the paper's number. Whichever is used goes in
the run name.

Reference checkpoint: `huggingface.co/IliaLarchenko/lehome_sim`.

### 2.2 The stack is 5.1, not 6.0 — and that is the problem

The work order's compute profile said this project "requires the 6.0 stack" and
that `build60` completing was the prerequisite. That is exactly backwards.
LeHome pins `isaacsim==5.1.0` on **Python 3.11** — the same stack as the locked
`bhl-robustness-ladder` v51 venv, not the 6.0 one.

Which sets up the actual blocking risk, and it is a squeeze:

- LeHome **requires RGB**. Three `TiledCamera`s at 640×480; the policy eats
  images; there is no ray-cast shortcut, because a Warp mesh query has no
  colour to return.
- LeHome **pins 5.1.0**.
- On this cluster, **5.1's RTX renderer segfaults** inside
  `omni.usd.create_hydra_engine`. That is the documented reason
  `bhl-robustness-ladder` has no RGB anywhere in it.
- RTX works on **6.0**, where LeHome is not pinned and the Isaac Lab API has
  moved 2.x → 3.x.

So the pinned version is the broken one, and the working version is unpinned.
**This is now the blocking question, and it replaces "does the environment
exist" as the thing to resolve first.** `slurm/00_rtx_probe.sbatch` exists to
answer it and nothing downstream should start until it has.

**RESULT (2026-08-27, job 21067439): it segfaults.** Exit 139 inside
`AppLauncher()`, before a render product is ever requested:

```
omni::usd::UsdManager::createHydraEngine
  -> libomni.hydra.rtx.plugin.so
  -> libcarb.scenerenderer-rtx.plugin.so
  -> librtx.scenedb.plugin.so :: carbOnPluginStartup
```

Node `cn-r-2`, **NVIDIA A40**, driver **595.71.05**. So it is not missing RT
cores -- an A40 has them. And it is not node selection: every GPU driver on
this cluster is 595.71.05 or 610.43.02, and the successful 6.0 probe ran on
`cn-r-4`, the same node class. **Isaac Sim 5.1's RTX plugins do not support
this cluster's drivers; 6.0's do.** That is cluster-wide and cannot be worked
around by constraining the partition.

The probe has three possible outcomes and all three are actionable:

| outcome | meaning | next move |
|---|---|---|
| renders | the segfault was BHL's configuration, not the cluster's 5.1 | proceed on the official stack, no deviation |
| segfaults on the borrowed venv, renders in the official image | the image's bundled Kit/driver userspace differs | use `LH_ROUTE=official` and say so |
| segfaults everywhere | the official stack cannot run here | port LeHome to the 6.0 stack — a **stated deviation**, because a ported scorer is no longer self-evidently the leaderboard's scorer |

The cheap version costs nothing to run: `PROBE_STACK=borrowed` uses the
**already built** BHL v51 venv and `bhl.sif` read-only, so the question can be
answered before a single byte of the 26.7 GB image is downloaded. The Hydra
engine comes from Kit inside `isaacsim`, not from `isaaclab`, so the forked
Isaac Lab cannot change whether `create_hydra_engine` survives — which is what
makes the borrowed stack a valid instrument for this one question, and only
this one.

### 2.3 Physics is CPU-only, and there is one environment

`--device cpu` is the **only** accepted value for the eval device, stated twice
in the README and enforced in the parser. The garments are PhysX particle
cloth, which is where that comes from.

`scripts/utils/evaluation.py` is hardcoded single-environment — the action is
`unsqueeze(0)`'d into a batch of one on every step. So the work order's "tens of
envs" is not available by turning up `NUM_ENVS`; there is no `NUM_ENVS`.

Corrected compute profile:

| | work order said | actually |
|---|---|---|
| parallelism | tens of envs | **one env per process** |
| bottleneck | RTX rendering + transformer forward | **CPU particle-cloth physics**, then rendering, then the transformer |
| stack | 6.0 (`v60`), `ENABLE_CAMERAS=1` | **5.1, py3.11, `--enable_cameras`, `--device cpu`** |

The consequence lands on Stage 3: parallelism has to come from **N Slurm tasks
each running one environment**, not from a vectorised env. That is the same
shape as the async trainer + rollout-workers design the work order already
budgeted for, so the plan survives — but it is CPU-core-hungry in a way the
original estimate was not, and each worker wants its own GPU slice only for
rendering and the policy forward.

### 2.4 Storage is a Stage 3 problem, with numbers

Measured, not estimated:

| item | size |
|---|---|
| official image (.sif) | ~25 GB (+27 GB tarball, deletable) |
| garment assets | 1.0 GB |
| merged demonstrations | 18.9 GB |
| per-garment + depth (optional) | 24.2 GB |
| **Stage 0–1 steady state** | **~45 GB** |

`/nfs/hpc/share` had **448 GB free** of a shared 1.5 TB at 71% used, and `lfs
quota` reports **no per-user block limit**. So Stage 0–2 is comfortable and the
constraint is the shared filesystem, not an allocation.

Stage 3 is not comfortable. The paper collected ~12,500 rollout episodes
(~4.3M frames). At the released dataset's ~71 KB/frame that is **~305 GB** of
rollout buffer — two thirds of the free space on a filesystem shared with other
people. Rollout retention has to be a designed policy (ring buffer, or drop
frames and keep advantage labels), not an afterthought.

---

## 3 · The control rate, resolved

The work order says G0 must log the control rate, expect 30 Hz, and stop if it
disagrees. It disagrees, and the resolution matters because "control-rate
mismatch" is exactly how G1 says the data pipeline breaks silently.

The numbers, all read first-hand:

| source | value |
|---|---|
| dataset `meta/info.json` | `fps = 30` |
| env config | `dt = 1/90`, `decimation = 1` -> 1/90 s of sim time per step |
| eval loop | exactly one `select_action` per `env.step()` |
| `RateLimiter(step_hz)` | **wall-clock only** -- it sleeps and calls `env.sim.render()` |
| episode 0 metadata | `length = 364`, video `to_timestamp = 12.1333 s` |

That last row settles it. 364 / 12.1333 = **exactly 30.0**, so the recorded
video timestamps are internally consistent with the declared 30 fps: one stored
frame is one recorded step, not three. `RateLimiter` is teleop/GUI machinery
and changes no sim-time semantics.

So the reading is: **`fps` is a frame-rate label, and one action is one
`env.step()` everywhere -- in recording and in evaluation alike.** The pipeline
is self-consistent. A 30-step action chunk is 30 env steps in both, which is
what LeRobot's `delta_timestamps` indexing needs, and it is why
`train_value.py` derives its future horizon from `meta.fps` and never from the
env's `dt`.

The 3x remains real in one place only, and it is worth stating rather than
forgetting: **one second of demonstration video is one third of a second of
simulated time.** Nothing in training depends on that, because nothing in
training measures sim seconds. Anything that reports a duration in seconds does
-- so report durations in FRAMES.

Still to confirm at G0, because metadata cannot: that a released demonstration
replayed through the environment actually scores success. That validates the
harness end to end and is a strictly stronger check than a random policy, which
is why it is G0-d below.

## 3a · What the demonstrations do not contain

Checked against `meta/episodes/chunk-000/file-000.parquet` (1,000 rows) and
`meta/garment_info.json`. This reshapes Stage 2, so it belongs in Stage 0.

**Columns present:** `episode_index`, `tasks`, `length`, video pointers, and
per-episode statistics over `observation.state`, `action`, the three image
streams, `timestamp`, `frame_index`.

**Columns absent:** `success`, `reward`, garment keypoints, particle positions.

Three consequences, in increasing order of how much they change the plan.

1. **The task string carries no garment category.** Every episode's task is the
   same: `"fold the garment on the table"`. That matches the challenge's own
   rule that category labels are withheld at evaluation -- the policy has to
   read the garment from vision. Anything that appends a category to the prompt
   is cheating the benchmark, which is why `recap.BASE_TASK` is a constant.

2. **There are no keypoints.** The paper's value head predicts "keypoint
   distances at t+30". The released data cannot supply that target; it would
   need the simulator to expose particle positions. This repo regresses the
   12-dim joint state at t+H instead and calls it `future_state`, deliberately
   named so no later report can claim the paper's quantity by accident.

3. **There are no failures.** All 40 garments in `garment_info.json` are
   `Seen`, 25 episodes each, and they are successful scripted demonstrations
   with no outcome column. So the work order's "labels are free -- you know how
   each demo ended" holds only in the degenerate direction: every episode ended
   the same way.

   That third point is a gate, not a nuisance. **G2 asks for a CALIBRATED
   success head, and calibration is undefined when every target is 1.** The
   model that predicts 1.0 everywhere is perfectly accurate, perfectly useless,
   and would sail past a careless check. `labels.class_balance` reports the
   degeneracy before the fit and `calibration.gate` refuses it after, so the
   failure is loud in two places.

   What is trainable on demonstrations today: the **progress** head (frame index
   over episode length, well defined everywhere) and the **future** head. What
   needs rollouts with mixed outcomes: the **success** head, and therefore every
   Stage 3 advantage that depends on it.

   `labels.success_targets(mode="terminal")` is the partial way out -- 0 before
   the fold completes, 1 after -- which yields both classes from a purely
   successful demonstration. It answers "has the fold completed", not "will this
   episode succeed", and those are different questions. It is reported
   separately and it is not a substitute for the episode-level head.

**So the dependency chain is:** renderer -> rollouts -> success negatives ->
G2 -> Stage 3 advantages. The renderer is the root, which is the second reason
`00_rtx_probe` is the blocking job rather than merely the first one.

## 4 · G0, restated

Order matters; the first item now gates the rest.

- [ ] **G0-a — the renderer.** `00_rtx_probe.sbatch` with `PROBE_STACK=borrowed`.
      Three `TiledCamera`s at 640×480 under LeHome's own `RenderCfg`, on a lit,
      textured scene. Passes only if RGB has real variance — a black frame is
      the failure that looks like a pass, which is how the first 6.0 probe in
      the sibling repo produced a false negative.
- [ ] **G0-b — one episode, official scorer.** `05_g0_floor.sbatch` runs
      `g0_uniform` and `g0_hold` through `scripts.eval` unmodified and gets a
      success/failure per episode.
- [ ] **G0-c — the numbers agree.** Log and check: observation shapes
      (3 × 480×640×3), action dimension (**12**), control rate (**§3**),
      episode budget (`episode_length_s = 60`, `--max_steps 600`).
- [ ] **G0-d — replay a demonstration.** A known-successful demo replayed
      through the env should score success. If it does not, the harness is
      wrong and no BC result would have been interpretable.

`g0_hold` exists to catch a scorer that rewards the reset state. If a policy
that never moves scores above zero, "success rate" is measuring the garment
spawn distribution and every downstream number is void.

---

## 5 · How this was checked

Reproducible, and worth re-running if any of it is doubted:

```bash
git clone https://github.com/lehome-official/lehome-challenge.git   # a805ad2
curl -s https://huggingface.co/api/datasets/lehome/asset_challenge          # public, ungated
curl -s https://huggingface.co/api/datasets/lehome/dataset_challenge_merged # public, ungated
curl -s https://huggingface.co/api/datasets/lehome/docker                   # public, ungated
curl -sL .../dataset_challenge_merged/resolve/main/four_types_merged/meta/info.json
curl -s https://api.github.com/repos/lehome-official/IsaacLab               # public, BSD-3
```

Paper: [arXiv:2606.27163](https://arxiv.org/abs/2606.27163) ·
[project page](https://ilialarchenko.com/projects/lehome2026/) ·
[challenge](https://lehome-challenge.com/)
