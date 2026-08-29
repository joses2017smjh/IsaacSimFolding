"""Can Isaac Sim 5.1 produce RGB WITHOUT the RTX renderer that segfaults here?

The pincer in docs/STAGE0.md is: 5.1 has the particle cloth the task needs and
cannot render; 6.0 renders and PhysX dropped the cloth. But those are not the
same subsystem. What crashes on 5.1 is specifically

    librtx.scenedb.plugin.so :: carbOnPluginStartup

-- the RTX hydra delegate. Physics on 5.1 is fine; bhl-robustness-ladder ran
thousands of headless PPO jobs on it.

And the 5.1 wheels ship OpenUSD's OWN rasterizer, hdStorm.so, alongside
omni.usd.libs. Storm is a GL rasteriser with no dependency on the RTX plugins.
If Storm can draw this stage headless, then 5.1 can produce camera images
without ever loading the thing that kills it -- and the deviation becomes
"different renderer" rather than "different physics".

That distinction is the whole point. The official success checker is geometric
over particle positions, so it is untouched by which rasteriser drew the
pixels: success rates would still come from official physics and the official
scorer, and only the images the policy SEES would differ. Rewriting the garment
onto 6.0's deformable API would change the physics, and with it the number.

Deliberately does NOT import isaacsim or launch SimulationApp. Kit is what
loads the RTX delegate; this talks to OpenUSD directly.
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = os.environ.get("PROBE_OUT", "results/storm51_probe.txt")
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)


def emit(line: str) -> None:
    print(line, flush=True)
    with open(OUT, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


open(OUT, "w").close()
emit("STARTED | probe_storm51.py (no isaacsim, no SimulationApp)")

# Headless GL. Storm renders through Hgi/GL, which needs a context; on a
# compute node with no display that has to be EGL.
os.environ.setdefault("PXR_ENABLE_GL_SUPPORT", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    emit("PXR | OpenUSD imported")
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | pxr unavailable -- setup_pxr not applied?")
    sys.exit(3)

try:
    from pxr import UsdImagingGL
    emit("PXR | UsdImagingGL imported")
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | UsdImagingGL unavailable -> Storm cannot be driven from here")
    sys.exit(3)


def build_stage(path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    for i, (colour, x) in enumerate(
        [((0.9, 0.2, 0.2), -0.35), ((0.2, 0.8, 0.3), 0.0), ((0.25, 0.4, 0.95), 0.35)]
    ):
        cube = UsdGeom.Cube.Define(stage, f"/World/cube{i}")
        cube.CreateSizeAttr(0.2)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.1))
        mat = UsdShade.Material.Define(stage, f"/World/m{i}")
        sh = UsdShade.Shader.Define(stage, f"/World/m{i}/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(mat)

    light = UsdLux.DistantLight.Define(stage, "/World/key")
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))
    stage.GetRootLayer().Save()
    return stage


scene = os.environ.get("SCENE_USD", "")
if scene and os.path.exists(scene):
    stage = Usd.Stage.Open(scene)
    emit(f"STAGE | opened {os.path.basename(scene)}")
else:
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "storm_probe.usda")
    if os.path.exists(tmp):
        os.unlink(tmp)
    stage = build_stage(tmp)
    emit("STAGE | built a lit, textured 3-cube scene")

try:
    engine = UsdImagingGL.Engine()
    emit("ENGINE | UsdImagingGL.Engine() constructed")
    plugins = list(engine.GetRendererPlugins())
    emit(f"ENGINE | renderer plugins: {plugins}")

    target = None
    for p in plugins:
        if "Storm" in p:
            target = p
            break
    if target is None:
        emit("VERDICT | no Storm plugin offered -> 5.1 has no non-RTX raster path")
        sys.exit(2)

    ok = engine.SetRendererPlugin(target)
    emit(f"ENGINE | SetRendererPlugin({target}) -> {ok}")

    W, H = 640, 480
    engine.SetRenderViewport(Gf.Vec4d(0, 0, W, H))
    engine.SetRendererAov("color")

    cam = Gf.Matrix4d(1.0)
    cam.SetLookAt(Gf.Vec3d(0.9, -1.3, 0.8), Gf.Vec3d(0, 0, 0.1), Gf.Vec3d(0, 0, 1))
    frustum = Gf.Frustum()
    frustum.SetPositionAndRotationFromMatrix(cam.GetInverse())
    frustum.SetPerspective(45.0, float(W) / H, 0.05, 100.0)
    engine.SetCameraState(frustum.ComputeViewMatrix(),
                          frustum.ComputeProjectionMatrix())

    params = UsdImagingGL.RenderParams()
    params.frame = Usd.TimeCode.Default()
    params.enableLighting = True
    params.clearColor = Gf.Vec4f(0.12, 0.13, 0.15, 1.0)

    engine.Render(stage.GetPseudoRoot(), params)
    emit("RENDER | Render() returned without crashing")

    import numpy as np

    buf = engine.GetAovTexture("color") if hasattr(engine, "GetAovTexture") else None
    arr = None
    if hasattr(engine, "GetRendererAov"):
        try:
            arr = np.asarray(engine.GetRendererAov("color"))
        except Exception:
            arr = None
    if arr is None or arr.size == 0:
        emit("RENDER | no AOV readback API on this build; Render() alone succeeded")
        emit("VERDICT | storm=PARTIAL -> plugin loads and renders, readback path TBD")
        sys.exit(0)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    std = float(arr.std())
    uniq = int(np.unique(arr).size)
    emit(f"RGB | shape={arr.shape} mean={arr.mean():.1f} std={std:.1f} unique={uniq}")
    good = std > 1.0 and uniq > 32
    emit(f"VERDICT | storm={'OK' if good else 'DEGENERATE'} "
         f"-> 5.1 {'CAN' if good else 'cannot'} render without RTX")
    sys.exit(0 if good else 2)

except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | storm=EXCEPTION")
    sys.exit(3)
