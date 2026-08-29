# RGB on Isaac Sim 5.1, without the RTX renderer

**The "no stack can both simulate and render" conclusion in
[STAGE0.md](STAGE0.md) was wrong.** It rested on "5.1 cannot render", and that
is too strong: what segfaults is `librtx.scenedb.plugin.so`, the **RTX hydra
delegate**. 5.1 *physics* is fine — `bhl-robustness-ladder` ran thousands of
headless jobs on it — and the same wheels ship OpenUSD's **Storm** rasteriser,
which never touches those plugins.

Storm renders. `slurm/63_storm51_probe.sbatch` and `slurm/64_storm_scene.sbatch`.

## Why this is the deviation worth taking

| approach | physics | scorer | images |
|---|---|---|---|
| port the garment to 6.0's deformable API | **changed** | official | RTX |
| **Storm on 5.1** | **official** | **official** | rasterised |

The official success checker is **geometric over particle positions**
(`success_checker_garment_fold`). It does not care which rasteriser drew the
pixels. So a Storm pipeline keeps official physics *and* the official scorer,
and deviates only in what the policy sees. Porting the cloth to 6.0 changes the
physics, and the success rate with it — which is the one thing a reproduction
cannot trade away.

## Five things that had to line up

Each was found by a probe, not guessed:

1. **`PXR_PLUGINPATH_NAME`** must point at `omni.usd.libs-*/bin/usd`, or the
   registry reports `No renderer plugins found!`.
2. **`Engine()` asks for a default delegate and gets an empty plugin id**
   (`Couldn't find plugin for id ''`). `Parameters.rendererPluginId` is the
   only overload that accepts one.
3. **HgiGL needs a real OpenGL 4.5 context.** Headless, that means an EGL
   pbuffer bound to `EGL_OPENGL_API` — desktop GL, not GLES.
4. **It must be a COMPATIBILITY profile.** A core 4.5 context reaches
   `Render()` and then throws `GL error: invalid enum` out of
   `HgiGL_ScopedStateHolder::~HgiGL_ScopedStateHolder` — it saves and restores
   fixed-function state a core profile refuses to report.
5. **Readback is `UsdAppUtils.FrameRecorder`, not `glReadPixels`.** Storm draws
   into its own AOV framebuffer, so the default one the process owns stays
   black. `FrameRecorder` is what `usdrecord` uses and it owns the readback.

That last one nearly produced a false negative in the other direction: the
first *correct* frame measured `mean=1.7` and looked like a failure, because
`FrameRecorder` writes RGBA and `.convert("RGB")` turns a transparent
background into pure black. The image was fine. Looking at it settled it.

## The honest limitation: Storm cannot read MDL

Omniverse assets ship **MDL** materials. Storm only evaluates
`UsdPreviewSurface`. So in `storm51_scene.png` the table and the garment — both
given `UsdPreviewSurface`, the garment with its real `BaseColor.jpg` — shade
correctly, and the **robots render as flat white**.

Attempting to rebind them did not work, and the reason is worth recording:
`so101_follower.usd` declares `defaultPrim = /so101_new_calib` while its 47
meshes live under `/visuals`, a **sibling** of the default prim. A reference
composes the default prim, so the meshes are not in the composed stage as
`Mesh` prims and a traversal cannot find them to rebind — yet Storm still draws
them. Binding by path prefix and binding by exclusion both rebound 0 and left
the frame byte-identical.

**What this means for the reproduction, stated plainly:** the images a Storm
pipeline produces differ from the RTX-rendered demonstrations by more than
"rasteriser vs path tracer" — the robot appearance is materially wrong. A
policy trained on the released data would face a real domain gap. That is
measurable, and it must be measured rather than assumed, before anyone reports
a success rate from this path.

## Next steps, in order

1. Fix the robot materials — reference `/visuals` explicitly rather than
   relying on `defaultPrim`, then bind `UsdPreviewSurface`.
2. Drive the cameras from the LeHome env's own poses, stepping 5.1 physics
   (which works) and rendering each step through Storm.
3. Measure the domain gap: BC on released RTX data, evaluated on Storm
   observations, against the same policy evaluated on RTX where possible.
