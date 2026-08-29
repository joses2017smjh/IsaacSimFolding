"""Can Isaac Sim 5.1 produce RGB WITHOUT the RTX renderer that segfaults here?

The pincer in docs/STAGE0.md only holds if "5.1 cannot render" is true of 5.1
as a whole. It is not. What dies is specifically

    librtx.scenedb.plugin.so :: carbOnPluginStartup

-- the RTX hydra delegate. 5.1 PHYSICS is fine; bhl-robustness-ladder ran
thousands of headless jobs on it. And the same wheels ship OpenUSD's own Storm
rasteriser, which has no dependency on those plugins.

If Storm draws, the deviation becomes "different renderer" rather than
"different physics". That distinction decides whether the numbers survive: the
official success checker is geometric over particle positions, so it does not
care which rasteriser drew the pixels -- physics and scorer stay official, and
only the images the policy SEES differ. Porting the garment to 6.0's deformable
API would change the physics, and the success rate with it.

Deliberately does NOT import isaacsim or launch SimulationApp: Kit is what
loads the delegate that crashes. This talks to OpenUSD directly.

Four things had to line up, each found by a probe rather than guessed:
  1. PXR_PLUGINPATH_NAME must point at .../omni.usd.libs-*/bin/usd
  2. Engine() defaults to an EMPTY plugin id; Parameters.rendererPluginId is
     the only overload that accepts one
  3. HgiGL needs a real OpenGL 4.5+ context -- headless, that means EGL
  4. it must be a COMPATIBILITY profile; a core profile makes HgiGL's state
     holder throw "GL error: invalid enum" on teardown
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = os.environ.get("PROBE_OUT", "results/storm51_probe.txt")
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
W, H = 640, 480


def emit(line: str) -> None:
    print(line, flush=True)
    with open(OUT, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


open(OUT, "w").close()
emit("STARTED | probe_storm51.py (no isaacsim, no SimulationApp)")

os.environ["PYOPENGL_PLATFORM"] = "egl"

# --- headless OpenGL context ----------------------------------------------
try:
    from OpenGL import EGL as E

    dpy = E.eglGetDisplay(E.EGL_DEFAULT_DISPLAY)
    maj, mino = E.EGLint(), E.EGLint()
    if not E.eglInitialize(dpy, maj, mino):
        raise RuntimeError("eglInitialize failed")
    emit(f"EGL | initialised {maj.value}.{mino.value}")

    cfgs = (E.EGLConfig * 1)()
    n = E.EGLint()
    if not E.eglChooseConfig(dpy, [
        E.EGL_SURFACE_TYPE, E.EGL_PBUFFER_BIT,
        E.EGL_RED_SIZE, 8, E.EGL_GREEN_SIZE, 8, E.EGL_BLUE_SIZE, 8,
        E.EGL_ALPHA_SIZE, 8, E.EGL_DEPTH_SIZE, 24,
        E.EGL_RENDERABLE_TYPE, E.EGL_OPENGL_BIT, E.EGL_NONE,
    ], cfgs, 1, n) or n.value == 0:
        raise RuntimeError("no desktop-GL EGL config")

    if not E.eglBindAPI(E.EGL_OPENGL_API):
        raise RuntimeError("eglBindAPI(EGL_OPENGL_API) failed")

    # Compatibility, not core: a core 4.5 context reached Storm's Render() and
    # then threw `GL error: invalid enum` from
    # HgiGL_ScopedStateHolder::~HgiGL_ScopedStateHolder -- it saves and restores
    # fixed-function state a core profile refuses to report. Selectable so the
    # failing configuration stays reproducible instead of being edited out.
    profile = os.environ.get("GL_PROFILE", "compat")
    bit = (E.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT if profile == "core"
           else E.EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT)
    emit(f"EGL | requesting GL 4.5 {profile} profile")

    surf = E.eglCreatePbufferSurface(
        dpy, cfgs[0], [E.EGL_WIDTH, W, E.EGL_HEIGHT, H, E.EGL_NONE])
    ctx = E.eglCreateContext(dpy, cfgs[0], E.EGL_NO_CONTEXT, [
        E.EGL_CONTEXT_MAJOR_VERSION, 4, E.EGL_CONTEXT_MINOR_VERSION, 5,
        E.EGL_CONTEXT_OPENGL_PROFILE_MASK, bit, E.EGL_NONE])
    if ctx == E.EGL_NO_CONTEXT:
        raise RuntimeError("eglCreateContext returned EGL_NO_CONTEXT")
    if not E.eglMakeCurrent(dpy, surf, surf, ctx):
        raise RuntimeError("eglMakeCurrent failed")

    from OpenGL import GL

    emit(f"GL  | {GL.glGetString(GL.GL_VERSION).decode()} | "
         f"{GL.glGetString(GL.GL_RENDERER).decode()}")
except Exception as exc:  # noqa: BLE001
    emit(f"EGL | context creation FAILED: {type(exc).__name__}: {exc}")
    emit("VERDICT | no GL context -> Storm cannot run headless here")
    sys.exit(3)

# --- OpenUSD ---------------------------------------------------------------
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdImagingGL, UsdLux, UsdShade
    emit("PXR | OpenUSD + UsdImagingGL imported")
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | pxr unavailable -- PYTHONPATH/PXR_PLUGINPATH_NAME wrong?")
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
    emit("STAGE | built a lit 3-cube scene")

# --- render via FrameRecorder ---------------------------------------------
# glReadPixels on the default framebuffer came back pure black -- not even the
# clear colour. UsdImagingGL with Storm renders into its OWN AOV framebuffer,
# so the default one this process owns is never written to.
#
# UsdAppUtils.FrameRecorder is the supported way through that: it is what the
# `usdrecord` CLI uses, it owns the AOV readback, and it writes an image file.
try:
    from pxr import UsdAppUtils

    cam = UsdGeom.Camera.Define(stage, "/World/probeCam")
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
    m = Gf.Matrix4d(1.0)
    m.SetLookAt(Gf.Vec3d(0.9, -1.3, 0.8), Gf.Vec3d(0.0, 0.0, 0.1), Gf.Vec3d(0, 0, 1))
    # SetLookAt builds a VIEW matrix; a camera prim wants its inverse.
    UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
    stage.GetRootLayer().Save()

    png = os.environ.get("STORM_PNG", "results/storm51_frame.png")
    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)

    rec = UsdAppUtils.FrameRecorder()
    rec.SetRendererPlugin("HdStormRendererPlugin")
    rec.SetImageWidth(W)
    emit(f"RECORDER | plugin={rec.GetCurrentRendererId()} width={W}")

    ok = rec.Record(stage, cam, Usd.TimeCode.Default(), png)
    emit(f"RECORDER | Record() -> {ok}  ({png})")
    del rec

    if not ok or not os.path.exists(png):
        emit("VERDICT | storm=NO_OUTPUT -> FrameRecorder wrote nothing")
        sys.exit(2)

    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(png).convert("RGB"))
    std, uniq = float(arr.std()), int(np.unique(arr).size)
    emit(f"RGB | shape={arr.shape} mean={arr.mean():.1f} std={std:.1f} unique={uniq}")

    # A uniform frame is the clear colour and nothing else -- the failure that
    # looks like a success.
    good = std > 1.0 and uniq > 32
    emit(f"VERDICT | storm={'OK' if good else 'DEGENERATE'} "
         f"-> 5.1 {'CAN' if good else 'cannot'} render RGB without RTX")
    sys.exit(0 if good else 2)

except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | storm=EXCEPTION")
    sys.exit(3)
