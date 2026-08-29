# The cloth simulates

> **This document previously concluded the cloth state was unreadable and
> hypothesised that the LeHome scorer depends on the render pass to sync it.
> Both were wrong.** The cause was a device string I set myself. Corrected
> below; the diagnostic history is kept because the wrong turns are the
> instructive part.

## The answer: PhysX particle cloth needs GPU dynamics

`cfg.sim.device` must be a **CUDA device**. I had set it to `"cpu"` because
LeHome's README says `--device cpu` — but that flag is **policy inference**, as
its own help text states. On CPU the particle solver silently does nothing
while rigid bodies keep working, which produces exactly the symptom that cost
several runs: arms moving 1.6 rad, cloth frozen to the bit.

With `sim.device = "cuda:0"`:

```
TOTAL max displacement 0.3296 m over 364 steps
top of garment fell 0.1775 m (draping onto the table)
wrote results/cloth51.npz (365, 9774, 3)
official success checker: tensor([False])
VERDICT cloth=SIMULATING
```

Real particle cloth, driven by all 364 steps of episode 0 of the released
demonstrations, with the official geometric checker executing on it.

**What found it:** instrumenting *which* particle reader answered. Both PhysX
paths were failing with the same `'NoneType' object has no attribute 'count'`
and silently falling back to the stale USD `points` attribute — so "the tensor
API does not help" was indistinguishable from "the tensor API was never
reached". One log line separated them, and an identical error across two
independent paths pointed at an uninstantiated particle solver rather than the
sync bug I had been assuming.

## Open-loop replay does not reproduce the fold

Three replays of episode 0, all with the cloth genuinely simulating, none
scoring a success:

| driven by | cloth displacement | official checker |
|---|---|---|
| `observation.state` as targets | 0.330 m | `False` |
| `action` (what the controller commanded) | 0.225 m | `False` |
| `action` + garment pinned to the recorded `object_initial_pose` | 0.325 m | `False` |

Each of those was a real hypothesis. Actions lead states by 0.074 rad on
average (max 1.58), so replaying states is a lagged copy of the trajectory. And
the env randomises the garment on every reset -- `initial_pos_range` in the
garment JSON, with the per-episode pose recorded in `garment_info.json` -- so
an open-loop replay grasps at fixed coordinates while the cloth sits somewhere
else. Pinning the pose fixed that, and the fold still did not complete.

**Why this is unsurprising rather than a bug.** Open-loop replay of a
contact-rich manipulation is fragile in a way an arm trajectory is not. A rigid
`object_initial_pose` does not restore the cloth's *particle configuration*;
solver state, contact history and the exact settle differ; and once a grasp
misses by a centimetre the rest of the trajectory is operating on a garment
that is no longer where the demonstration left it. Reproducing a scored fold
needs a closed-loop policy, which is what Stages 1-3 exist to train -- not a
better replay.

**What this does establish:** the environment, the cloth, the demonstrations
and the official scorer all run end to end on this cluster, and a policy can be
evaluated the moment there is one. That is the gate G0 was asking for.

---

# Historical: the diagnosis before the fix

## What works

The LeHome garment environment **runs on Isaac Sim 5.1** — the stack that still
has PhysX particle cloth:

```
[cloth] 5.1 app launched with cameras OFF -- no hydra engine, no segfault
[cloth] GarmentEnv instantiated -- particle cloth spawned
[cloth] object.post_reset() ok
[cloth] object._get_initial_info() ok
[cloth] env.reset() -- particle system brought up
[cloth] particles: 9774 points, z range [0.571, 0.773]
```

`slurm/65_cloth_sim51.sbatch`. Five blockers cleared to get there, each a
distinct bug:

| # | symptom | cause |
|---|---|---|
| 1 | `'NoneType' object has no attribute 'prim_path'` | `_setup_scene` builds `TiledCamera` unconditionally — stub the **class**, not the config |
| 2 | `KeyError: 'rgb'` | `_get_observations` reads the stub's empty output dict — return zero images |
| 3 | `No module named 'open3d'` | the official particle accessor imports it; it was in the 6.0 site dir, not the 5.1 one |
| 4 | `no attribute 'initial_points_positions'` | `reset()` restores it but only `_get_initial_info()` creates it — call that first |
| 5 | X server error from `pynput` | teleop keyboard imported at package-import time; stubbed |

**Cameras off is the whole trick.** 5.1's crash is in the RTX hydra delegate,
so an app that never requests a render product never reaches it — which is how
`bhl-robustness-ladder` ran thousands of headless 5.1 jobs.

## What does not work, and how it was localised

Physics **runs**. The cloth state **cannot be read**. That distinction took a
dedicated diagnostic, because both look identical from the outside:

```
step 20   max|Δcloth|=0.0000 m   max|Δjoint|=1.5768 rad
step 40   max|Δcloth|=0.0000 m   max|Δjoint|=1.5951 rad
```

The arms swing through 1.6 radians of the real demonstration — physics is
advancing. The cloth reader returns **bit-identical** values, which is the
signature of a static buffer rather than a settled simulation.

Both read paths are unavailable:

- `get_current_mesh_points()` — what the **official success checker** uses —
  reads the USD `points` attribute on CPU. PhysX is not writing particle state
  back to USD in this configuration, so it returns the authored mesh forever.
- `_cloth_prim_view.get_world_positions()` — the direct PhysX route — needs a
  view initialised against a physics sim view. `initialize()` fails with
  `'NoneType' object has no attribute 'count'`; `DirectRLEnv` never sets one up
  for this object.

## The likely reason, and why it matters

This is the **same root cause** that froze the arms in the Storm render: in this
configuration PhysX results do not reach USD. `/physics/updateToUsd` and
`world.step(render=True)` both failed to change it there.

The official evaluator always runs with `--enable_cameras`. The strong
hypothesis is that **the render pass is what drives the physics→USD sync**, and
that the LeHome success checker therefore depends on rendering happening. On
this cluster 5.1 cannot render, so that sync never occurs — the simulation is
correct and unobservable.

If that hypothesis holds it is a real constraint on the whole project, not a
bug in this repo: **the official scorer cannot run on a stack that cannot
render**, regardless of whether the physics is fine.

## Untried, in order of expected value

1. **Read particles through `omni.physx`'s tensor API directly**, bypassing
   both USD and the prim view. This is the one that would settle it without
   needing a renderer.
2. **A node with a driver Isaac Sim 5.1 supports.** Everything downstream
   follows: RTX renders, the sync happens, the official scorer works
   unmodified, and none of the deviations in this repo are needed.
3. Confirm the hypothesis by testing whether an `enable_cameras=True` run on
   the **6.0** stack shows cloth state updating — 6.0 has no particle cloth, so
   this would need a different deformable, and it tests the sync rather than
   the task.

## What must not be claimed from this

No success rate. The official checker reads a buffer that never updates here,
so it would report a fold that never happened — or, more likely, a failure that
is really an unobservable success. Nothing about folding performance can be
measured on this path until the cloth state is readable.
