# lehome-fold-repro

Reproducing **[Learning to Fold](https://arxiv.org/abs/2606.27163)** (Larchenko,
2026) — 1st of 62 in the LeHome Challenge 2026 simulation round, 2nd in the
real-world final — on the OSU cluster, in Isaac Sim.

Bimanual SO-ARM101, four garment types, binary fold success. The paper's one
structural idea is that **the policy is its own value function**: the same
network that predicts actions also predicts success, progress and a few
task-relevant futures, and those predictions drive advantage estimation, live
failure detection and candidate selection at inference. Everything else in the
paper is downstream of that.

This is a separate repo from `bhl-robustness-ladder` on purpose — different
robot, different learning paradigm, no MuJoCo sim2sim path. What carries over
is the Slurm/apptainer discipline and the build→gate→submit habit, not a line
of code.

The sim2real half of the paper (camera alignment, heavy augmentation,
DAgger-like human-in-the-loop collection) is out of scope. No hardware.

---

## Status

**Stage 0 is resolved: the environment is available and the project is a GO.**
The full evidence, with the command that produced each claim, is in
[docs/STAGE0.md](docs/STAGE0.md).

| | |
|---|---|
| environment package | `lehome-official/lehome-challenge`, Apache-2.0, pinned at `a805ad2` |
| demonstrations | 1,000 episodes / 265,798 frames, public and ungated |
| garments | 10 seen + 2 public unseen per type (48 of the leaderboard's 80) |
| official scorer | present, called unmodified |
| **blocking risk** | **RGB on Isaac Sim 5.1 — unresolved, see below** |

Nothing had to be reimplemented, so "success rate" here can mean the number the
leaderboard meant. That was the entire reason for reproducing this paper rather
than describing it.

### The blocking risk is not the one that was predicted

The plan going in assumed this project needed the 6.0 stack. It is the exact
opposite, and the inversion is the whole problem:

- LeHome pins **`isaacsim==5.1.0`** on Python 3.11 — the same stack as the
  locked BHL v51 venv.
- LeHome **requires RGB**: three `TiledCamera`s at 640×480. There is no
  ray-cast shortcut here, because a Warp mesh query has no colour to return.
- On this cluster, **5.1's RTX renderer segfaults** inside
  `omni.usd.create_hydra_engine`. That is the documented reason
  `bhl-robustness-ladder` has no RGB anywhere in it.
- RTX renders on **6.0** — where LeHome is not pinned, and where the Isaac Lab
  API has moved 2.x → 3.x.

So the pinned version is the broken one and the working version is unpinned.
`slurm/00_rtx_probe.sbatch` answers it, and **it costs nothing to run**: with
`PROBE_STACK=borrowed` it uses the already-built BHL v51 venv read-only, so the
question is settled before any of the 26.7 GB image is downloaded.

Three outcomes, all actionable — proceed unchanged, proceed on the official
image, or port to 6.0 as a **stated deviation**. The deviation matters: a
ported scorer is no longer self-evidently the leaderboard's scorer, and saying
so is the difference between a reproduction and a number.

### Three other things the plan got wrong

Verified against the paper and the code, not assumed:

- **The base model is π0.5, not SmolVLA.** SigLIP-So400m/14 → Gemma-2B prefix →
  Gemma-300M action expert, 30-step chunk via flow matching. ~2.3B parameters
  against SmolVLA's ~450M. LeHome ships a SmolVLA config, so it stays a
  legitimate cheaper baseline — but a number from it is not the paper's number,
  and which one produced a result goes in the run name.
- **Physics is CPU-only and there is one environment.** `--device cpu` is the
  only accepted value, and the eval loop is hardcoded to a batch of one. There
  is no `NUM_ENVS` to turn up. Parallelism has to come from N Slurm tasks.
- **Isaac Lab comes from a fork**, not PyPI —
  `lehome-official/IsaacLab`, installed via `isaaclab.sh`. Installing stock
  `isaaclab==2.3.1` gets a different environment.

---

## Running it

Two ways to get the environment, selected with `LH_ROUTE`:

- **`official`** (default) — the organizers' own Docker image converted to
  apptainer. The venv is baked in. This is the environment the leaderboard was
  scored in, which removes every resolver question in one step.
- **`source`** — `uv sync` into a venv on Lustre plus the forked Isaac Lab.
  Slower and it can drift, but it is the only route where the environment code
  is editable, which Stage 3 will need.

Start on `official`. Any number produced on `source` gets re-checked against
`official` before it is reported.

```bash
cd /nfs/hpc/share/$USER/Humanoid_Lite/lehome-fold-repro

# 0. THE BLOCKING ONE. Needs nothing built. ~40 min.
sbatch slurm/00_rtx_probe.sbatch

# --- only if 00 says the renderer works ---

sbatch slurm/01_fetch_official_image.sbatch   # 26.7 GB -> ~25 GB .sif
sbatch slurm/02_fetch_data.sbatch             # assets 1.0 GB + demos 18.9 GB

# G0: one episode end to end through the OFFICIAL scorer.
POLICY_TYPE=g0_uniform GARMENT_TYPE=top_long sbatch slurm/05_g0_floor.sbatch
POLICY_TYPE=g0_hold    GARMENT_TYPE=top_long sbatch slurm/05_g0_floor.sbatch
```

The editable route is `03_build_container.sbatch` then
`04_install_source.sbatch`, and is not needed until something in the
environment has to change.

Storage, measured: ~45 GB for Stage 0–1. `/nfs/hpc/share` had 448 GB free and
no per-user block quota. Stage 3 is the one to worry about — the paper's ~4.3M
rollout frames would be **~305 GB** at the released data's bytes-per-frame, so
rollout retention has to be designed rather than discovered.

### Two floor policies, not one

`g0_uniform` is the random floor G0 asks for — a bounded random *walk*, because
independent uniform samples at 1/90 s command a full-range slew every 11 ms and
measure the actuators rather than the task.

`g0_hold` never moves. It exists to catch a scorer that rewards the reset
state: if a policy that does nothing scores above zero, "success rate" is
measuring the garment spawn distribution and every number downstream is void.
Both are expected to score 0.0, and they fail in different, diagnostic ways.

---

## Plan

Four stages, each independently reportable. Gates are the point — they are
where a silent failure gets caught while it is still cheap.

| | stage | gate |
|---|---|---|
| S0 | environment + data available | **G0** renderer works; official scorer emits a verdict |
| S1 | BC fine-tune on the 10 seen garments per type | **G1** above the G0 floor |
| S2 | value head — success, progress, task-relevant futures | **G2** success head is calibrated |
| S3 | RECAP + AWR, async trainer + rollout workers | **G3** workers and trainer agree on checkpoint version |
| S4 | Thompson sampling over inference hyperparameters | gain over fixed defaults, same checkpoint |

Reporting rules that apply from G1 onward:

- **Per garment type, always.** Long pants and long-sleeved tops are harder
  than shorts, and a single averaged number hides that.
- **Seen vs unseen, separately, ≥2 seeds.** Generalisation to unseen garments
  is what the leaderboard actually measured.
- **The 2 public unseen garments per type are never trained on.** There is no
  automatic detector for this failure — it just makes the generalisation number
  meaningless — so the file list lives in one place and is asserted on.
- **48 garments, not 80.** The 8 private garments per type never shipped. Any
  success rate from this repo is on the public half and has to say so.

Stage 3 is where the engineering cost concentrates. **A reproduction that
reaches Stage 2 with clean per-type numbers is a real result**; the RL loop is
not the thing to rush toward.

### Why the value head comes before the RL

It is cheap, its labels are free — every demonstration's outcome is already
known — and it buys two mechanisms with no RL at all: live failure detection,
and candidate selection (sample several action chunks, score them with the
policy's own value head, execute the best). Stage 1 with candidate selection
versus without, on the same checkpoint, is a clean single-mechanism ablation
and is publishable on its own.

G2 is calibration, and it is a gate rather than a diagnostic because an
uncalibrated success head makes Stage 3's advantages noise **silently**. Bin
predicted success probability, check observed frequency tracks it.

---

## What this teaches about Isaac Sim

Worth naming, since it is half the reason for doing it. This project exercises
the parts of the stack the locomotion work never touched:

- **The RTX render path**, `TiledCamera`, and render products — as opposed to
  Warp ray-casting, which is all `bhl-robustness-ladder` could ever use.
  Multi-camera rendering is the cost driver here, not physics stepping.
- **Deformable simulation**: PhysX particle cloth, which is why the environment
  is CPU-only, and why "success" is a geometric predicate over particle
  positions rather than a rigid-body pose check.
- **`DirectRLEnv`** with a hand-written `_get_observations` / `_apply_action`,
  versus the manager-based config style.
- **Reading a simulator someone else configured** — `dt`, `decimation`,
  `render_interval`, `use_fabric`, `RenderCfg` — and working out what the
  control rate actually is when the metadata and the config disagree. That
  disagreement is real here and is written up in
  [§3 of STAGE0](docs/STAGE0.md#3--one-number-that-does-not-reconcile-yet).

---

## Layout

```
container/lehome.def         runtime image for the source route
docs/STAGE0.md               the availability dossier — evidence for every claim
external/lehome-challenge    the official environment, pinned (submodule)
scripts/probe_rtx_tiled.py   the blocking renderer probe
scripts/g0_policies.py       uniform + hold floors
scripts/g0_eval.py           registers them into LeHome's registry, then defers
slurm/_env.sh                paths, routes, container invocation
slurm/00..05                 probe, image, data, build, install, G0
```

Nothing under `external/` is modified. `g0_eval.py` registers into LeHome's own
`PolicyRegistry` and hands control to `scripts.eval` unchanged — that property
is what keeps the harness comparable, and it is worth protecting.

## Credit

The environment, garment assets, demonstrations and scorer are the LeHome
Challenge organizers' work (Apache-2.0). The method being reproduced is Ilia
Larchenko's — [paper](https://arxiv.org/abs/2606.27163),
[write-up](https://ilialarchenko.com/projects/lehome2026/),
[checkpoint](https://huggingface.co/IliaLarchenko/lehome_sim).
