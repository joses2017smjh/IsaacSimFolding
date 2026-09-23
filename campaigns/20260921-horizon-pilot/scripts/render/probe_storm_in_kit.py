"""Can Storm render inside a live Isaac Sim 5.1 process?

This decides whether a trained policy can be evaluated at all on this cluster.

The env hands the policy three RGB images (garment_bi_v2._get_observations).
On 5.1 the RTX TiledCameras segfault, so cloth_sim51.py stubs them and returns
ZEROS -- fine for measuring particle positions, useless for a VLA, which would
be looking at black frames.

Storm renders on 5.1 (docs/STORM.md), but so far only in a process with no Kit
in it. The question here is whether both can coexist: Kit running headless with
cameras off, and an EGL/Storm context created alongside it in the same process.

If yes -> the observation pipeline is: step physics, read particles + link
poses, write them into a USD stage, Storm-render the three challenge cameras,
hand those to the policy. Evaluation becomes possible with official physics and
the official scorer.

If no -> Storm has to run in a separate process and the frames shipped across,
which is a different and much more annoying design.
"""

from __future__ import annotations

import os
import sys
import traceback

OUT = os.environ.get("PROBE_OUT", "results/storm_in_kit.txt")
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)


def emit(line):
    print(line, flush=True)
    with open(OUT, "a") as fh:
        fh.write(line + "\n")


open(OUT, "w").close()
emit("STARTED | Storm-inside-Kit feasibility")

# 1. Kit first, cameras OFF -- the configuration that does not segfault.
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=False).app
emit("KIT | 5.1 app launched, cameras off")

import numpy as np  # noqa: E402

# 2. Does Kit already hold a GL context in this configuration?
os.environ["PYOPENGL_PLATFORM"] = "egl"
try:
    from OpenGL import EGL as E

    cur = E.eglGetCurrentContext()
    emit(f"EGL | context already current in this process: {bool(cur)}")
except Exception as exc:  # noqa: BLE001
    emit(f"EGL | could not query current context: {exc}")

# 3. Try to stand up our own EGL context alongside Kit.
try:
    dpy = E.eglGetDisplay(E.EGL_DEFAULT_DISPLAY)
    maj, mino = E.EGLint(), E.EGLint()
    if not E.eglInitialize(dpy, maj, mino):
        raise RuntimeError("eglInitialize failed")
    cfgs = (E.EGLConfig * 1)()
    n = E.EGLint()
    E.eglChooseConfig(dpy, [
        E.EGL_SURFACE_TYPE, E.EGL_PBUFFER_BIT,
        E.EGL_RED_SIZE, 8, E.EGL_GREEN_SIZE, 8, E.EGL_BLUE_SIZE, 8,
        E.EGL_ALPHA_SIZE, 8, E.EGL_DEPTH_SIZE, 24,
        E.EGL_RENDERABLE_TYPE, E.EGL_OPENGL_BIT, E.EGL_NONE], cfgs, 1, n)
    E.eglBindAPI(E.EGL_OPENGL_API)
    surf = E.eglCreatePbufferSurface(dpy, cfgs[0],
                                     [E.EGL_WIDTH, 640, E.EGL_HEIGHT, 480, E.EGL_NONE])
    ctx = E.eglCreateContext(dpy, cfgs[0], E.EGL_NO_CONTEXT, [
        E.EGL_CONTEXT_MAJOR_VERSION, 4, E.EGL_CONTEXT_MINOR_VERSION, 5,
        E.EGL_CONTEXT_OPENGL_PROFILE_MASK,
        E.EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, E.EGL_NONE])
    if ctx == E.EGL_NO_CONTEXT:
        raise RuntimeError("eglCreateContext returned EGL_NO_CONTEXT")
    if not E.eglMakeCurrent(dpy, surf, surf, ctx):
        raise RuntimeError("eglMakeCurrent failed")
    from OpenGL import GL

    emit(f"GL  | {GL.glGetString(GL.GL_VERSION).decode()} alongside Kit")
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | no GL context alongside Kit -> Storm must be out-of-process")
    app.close()
    sys.exit(2)

# 4. And can Storm actually draw with Kit resident?
try:
    from pxr import Gf, Sdf, Usd, UsdAppUtils, UsdGeom, UsdLux, UsdShade

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "storm_in_kit.usda")
    if os.path.exists(tmp):
        os.unlink(tmp)
    st = Usd.Stage.CreateNew(tmp)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(st, "/World")
    c = UsdGeom.Cube.Define(st, "/World/c")
    c.CreateSizeAttr(0.3)
    m = UsdShade.Material.Define(st, "/World/m")
    sh = UsdShade.Shader.Define(st, "/World/m/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.3, 0.2))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(m)
    UsdLux.DomeLight.Define(st, "/World/d").CreateIntensityAttr(1500.0)
    cam = UsdGeom.Camera.Define(st, "/World/cam")
    cam.CreateFocalLengthAttr(24.0)
    v = Gf.Matrix4d(1.0)
    v.SetLookAt(Gf.Vec3d(0.6, -0.8, 0.5), Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(v.GetInverse())
    st.GetRootLayer().Save()

    png = os.path.join(os.environ.get("TMPDIR", "/tmp"), "storm_in_kit.png")
    rec = UsdAppUtils.FrameRecorder()
    rec.SetRendererPlugin("HdStormRendererPlugin")
    rec.SetImageWidth(640)
    rec.SetCameraLightEnabled(True)
    ok = rec.Record(st, cam, Usd.TimeCode.Default(), png)
    del rec
    emit(f"STORM | Record() -> {ok}")

    from PIL import Image

    a = np.asarray(Image.open(png).convert("RGB"))
    emit(f"RGB | {a.shape} mean={a.mean():.1f} std={a.std():.1f} uniq={np.unique(a).size}")
    good = bool(ok) and a.std() > 1.0
    emit(f"VERDICT | storm_in_kit={'OK' if good else 'DEGENERATE'} "
         f"-> observation pipeline is {'FEASIBLE in-process' if good else 'NOT feasible in-process'}")
    app.close()
    sys.exit(0 if good else 2)
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | storm_in_kit=EXCEPTION")
    app.close()
    sys.exit(3)
