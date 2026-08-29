# Rendering the challenge scene on Isaac Sim 6.0

The task's own stack (5.1) cannot render on this cluster, so the scene is
rendered on 6.0 with raw `isaacsim` + `omni.replicator` — skipping the Isaac
Lab 2.x→3.x port entirely, since the assets are USD and the renderer is Isaac
Sim's own.

`slurm/60_render_scene.sbatch` → `scripts/render/render_scene.py`.

## What is real in the render

- `so101_follower.usd`, both arms, at the exact poses in `garment_bi_cfg_v2.py`
  (`±0.23, −0.25, 0.5`) with the config's 180° yaw.
- A real `Release` garment mesh (9,774 points) with its `BaseColor.jpg` fabric
  texture, at the JSON's `0.45` scale, seated on the table by measured extent.
- The challenge's own overhead camera placement and intrinsics
  (focal 28.7, aperture 38.11, 640×480).
- Joint angles from **episode 0 of the released demonstrations** — 364 frames of
  real 12-dim state pulled out of the parquet.

## Open issue: the joint replay does not reach the renderer

**The arms hold one pose. The motion in the GIF is the camera orbiting.**

This is not a guess. A read-back at each frame shows the articulation holds
exactly what was commanded, and it changes per frame:

```
frame 0 Left_Robot: set=[-1.237, -1.688, 1.492, 1.052, -0.085, -0.012]
frame 0 Left_Robot: got=[-1.237, -1.688, 1.492, 1.052, -0.085, -0.012]
frame 1 Left_Robot: set=[-0.908, -1.701, 1.505, 1.116, -0.175, -0.098]
frame 1 Left_Robot: got=[-0.908, -1.701, 1.505, 1.116, -0.175, -0.098]
```

…while the fixed overhead camera produces **byte-identical frames**
(`meanΔ = 0.00`, 0.0% of pixels differ between first and last). Physics has the
pose; the render product does not.

Three fixes tried, none of which worked:

| attempt | result |
|---|---|
| `world.step(render=False)` | no motion |
| `world.step(render=True)` | no motion |
| `/physics/updateToUsd`, `/physics/updateVelocitiesToUsd`, `fabricUpdateTransformations=False` | no motion |

The likely cause is that `rep.orchestrator.step()` drives its own render path
that does not observe the `World`'s physics writeback. The next thing to try is
dropping replicator's orchestrator and reading the cameras through
`isaacsim.sensors.camera.Camera` while stepping the `World` directly.

`docs/img/isaacsim_topcam.png` is a still from the challenge's own overhead
camera -- the view the policy would actually consume. It is a STILL and not a
GIF for the same reason: all 90 frames of it are byte-identical, and animating
them would be dressing a static image up as motion.

**Until that is resolved, the GIF is labelled as a camera orbit, not a replay.**
The distinction matters: a viewer who thinks they are watching a policy — or
even a demonstration — is being misled by a moving camera.

## Reproducing

```bash
sbatch slurm/60_render_scene.sbatch                 # 90-frame orbit
sbatch slurm/62_defect_gallery.sbatch               # the failure/fix gallery
python scripts/render/make_gif.py --frames <dir> --out docs/img
python scripts/render/make_gallery.py
```

`--defect <name>` reintroduces one known bug; see `DEFECTS` in
`render_scene.py`. Two candidates (`garment_culling`, `no_yaw`) are kept in the
list but produce byte-identical output and are excluded from the published
gallery for that reason.
