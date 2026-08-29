# The cloth simulates. Its state is not readable. Those are different problems.

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
