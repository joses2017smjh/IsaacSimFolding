"""Render the real LeHome scene on Isaac Sim 5.1 -- via Storm, without RTX.

This is the payoff of probe_storm51.py. The pincer in docs/STAGE0.md said 5.1
cannot render and 6.0 has no particle cloth, so the task could not both
simulate and render on this cluster. But "5.1 cannot render" was too strong:
what dies is the RTX hydra delegate, and 5.1 also ships OpenUSD's Storm
rasteriser, which does not touch it.

So the scene is authored with plain OpenUSD -- no Kit, no SimulationApp, no
`import isaacsim` -- and drawn by UsdAppUtils.FrameRecorder with Storm.

Why this matters for the reproduction: the official success checker is
GEOMETRIC over particle positions. It does not care which rasteriser drew the
pixels. So a Storm-rendered pipeline keeps the official physics AND the
official scorer, and deviates only in the images the policy sees. Porting the
garment to 6.0's deformable API would change the physics, and the number with
it.

What is still unproven: that a policy trained on RTX-rendered demonstrations
transfers to Storm-rendered observations. Storm is a rasteriser and the
released data is path-traced; that is a real domain gap and it has to be
measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--assets", required=True)
ap.add_argument("--garment", default="Top_Short/Top_Short_Seen_0")
ap.add_argument("--out", default="results/storm_scene.png")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--usd_out", default="")
args = ap.parse_args()

os.environ["PYOPENGL_PLATFORM"] = "egl"


def log(*a):
    print("[storm]", *a, flush=True)


# --- headless GL 4.5 compatibility context --------------------------------
from OpenGL import EGL as E  # noqa: E402

dpy = E.eglGetDisplay(E.EGL_DEFAULT_DISPLAY)
maj, mino = E.EGLint(), E.EGLint()
if not E.eglInitialize(dpy, maj, mino):
    raise SystemExit("eglInitialize failed")
cfgs = (E.EGLConfig * 1)()
n = E.EGLint()
E.eglChooseConfig(dpy, [
    E.EGL_SURFACE_TYPE, E.EGL_PBUFFER_BIT,
    E.EGL_RED_SIZE, 8, E.EGL_GREEN_SIZE, 8, E.EGL_BLUE_SIZE, 8,
    E.EGL_ALPHA_SIZE, 8, E.EGL_DEPTH_SIZE, 24,
    E.EGL_RENDERABLE_TYPE, E.EGL_OPENGL_BIT, E.EGL_NONE], cfgs, 1, n)
E.eglBindAPI(E.EGL_OPENGL_API)
surf = E.eglCreatePbufferSurface(dpy, cfgs[0], [E.EGL_WIDTH, args.width,
                                               E.EGL_HEIGHT, args.width, E.EGL_NONE])
# COMPATIBILITY profile: a core context makes HgiGL's state holder throw
# "GL error: invalid enum" on teardown.
ctx = E.eglCreateContext(dpy, cfgs[0], E.EGL_NO_CONTEXT, [
    E.EGL_CONTEXT_MAJOR_VERSION, 4, E.EGL_CONTEXT_MINOR_VERSION, 5,
    E.EGL_CONTEXT_OPENGL_PROFILE_MASK,
    E.EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, E.EGL_NONE])
E.eglMakeCurrent(dpy, surf, surf, ctx)
from OpenGL import GL  # noqa: E402

log(f"GL {GL.glGetString(GL.GL_VERSION).decode()} | "
    f"{GL.glGetString(GL.GL_RENDERER).decode()}")

from pxr import Gf, Sdf, Usd, UsdAppUtils, UsdGeom, UsdLux, UsdShade  # noqa: E402

TABLE_Z = 0.5
usd_path = args.usd_out or os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                        "storm_scene.usda")
if os.path.exists(usd_path):
    os.unlink(usd_path)
stage = Usd.Stage.CreateNew(usd_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def material(path, colour, tex=None, rough=0.6):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    if tex:
        rd = UsdShade.Shader.Define(stage, path + "/uv")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st = UsdShade.Shader.Define(stage, path + "/tex")
        st.CreateIdAttr("UsdUVTexture")
        st.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
        st.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        st.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            rd.ConnectableAPI(), "result")
        st.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            st.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


# table
table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
tx = UsdGeom.Xformable(table)
tx.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.30, TABLE_Z - 0.4))
tx.AddScaleOp().Set(Gf.Vec3f(1.5, 1.1, 0.8))
UsdShade.MaterialBindingAPI(table.GetPrim()).Bind(
    material("/World/mats/table", (0.66, 0.60, 0.51), rough=0.8))

# robots, at the task config's poses and 180 deg yaw
ROBOT = os.path.join(args.assets, "robots/lerobot/so101_follower.usd")
YAW180 = Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))
for name, pos in (("Left_Robot", (-0.23, -0.25, TABLE_Z)),
                  ("Right_Robot", (0.23, -0.25, TABLE_Z))):
    holder = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}")
    hx = UsdGeom.Xformable(holder)
    hx.AddTranslateOp().Set(Gf.Vec3d(*pos))
    hx.AddOrientOp().Set(YAW180)
    # reference on a CHILD: the robot USD brings its own xform ops
    UsdGeom.Xform.Define(stage, f"/World/Robot/{name}/model") \
        .GetPrim().GetReferences().AddReference(ROBOT)
log(f"referenced 2x {os.path.basename(ROBOT)}")

# Override the robots' materials with UsdPreviewSurface.
#
# Omniverse assets ship MDL materials, and Storm -- OpenUSD's GL rasteriser --
# only evaluates UsdPreviewSurface. Left alone, every robot mesh renders as
# flat blown-out white while the table beside it stays dark, which is exactly
# what the first two Storm frames showed. This is the honest cost of the
# non-RTX path: geometry and PreviewSurface survive, MDL does not.
# Bind by EXCLUSION, not by path prefix.
#
# so101_follower.usd declares defaultPrim = /so101_new_calib while its 47
# meshes live under /visuals -- a SIBLING of the default prim. A reference
# composes the default prim, so a traversal filtered on "/World/Robot" matched
# nothing and the first attempt rebound 0 meshes. Everything that is not the
# table or the garment is robot, so select that way instead.
robot_mat = material("/World/mats/robot", (0.85, 0.72, 0.12), rough=0.45)
_nrobot = 0
for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if not prim.IsA(UsdGeom.Mesh):
        continue
    if path.startswith("/World/Garment") or path.startswith("/World/table"):
        continue
    UsdShade.MaterialBindingAPI(prim).Bind(
        robot_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    _nrobot += 1
log(f"rebound {_nrobot} robot meshes to UsdPreviewSurface (Storm cannot read MDL)")

# garment
gdir = os.path.join(args.assets, "objects/Challenge_Garment/Release", args.garment)
gusd = next((os.path.join(gdir, f) for f in sorted(os.listdir(gdir))
             if f.endswith(".usd")), None)
gjson = next((os.path.join(gdir, f) for f in sorted(os.listdir(gdir))
              if f.endswith(".json")), None)
scale, tex = 0.45, None
if gjson:
    meta = json.load(open(gjson))
    scale = float(meta.get("scale", [0.45])[0])
    import glob as _g
    for v in meta.get("visual_usd_paths", []):
        rel = v.lstrip("/")
        rel = rel[len("Assets/"):] if rel.startswith("Assets/") else rel
        full = os.path.join(args.assets, rel)
        stem = os.path.splitext(os.path.basename(full))[0]
        # ".../Fabric050_1K_JPG.usd" sits beside ".../Fabric050_1K-JPG/BaseColor.jpg"
        d = os.path.join(os.path.dirname(full),
                         stem.rsplit("_", 1)[0] + "-" + stem.rsplit("_", 1)[-1])
        hit = _g.glob(os.path.join(d, "*BaseColor*")) if os.path.isdir(d) else []
        if hit:
            tex = hit[0]
            break

g = UsdGeom.Xform.Define(stage, "/World/Garment")
gx = UsdGeom.Xformable(g)
gt = gx.AddTranslateOp()
gx.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
UsdGeom.Xform.Define(stage, "/World/Garment/mesh") \
    .GetPrim().GetReferences().AddReference(gusd)
gmat = material("/World/mats/garment", (0.8, 0.25, 0.25), tex=tex, rough=0.9)

lo = [1e9] * 3
hi = [-1e9] * 3
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/Garment") and prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI(prim).Bind(gmat)
        UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
        # the garment USD authors xformOp:scale = 0.01; neutralise it so the
        # JSON's scale is the only one
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if op.GetOpName().endswith("scale"):
                op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
        ext = UsdGeom.Mesh(prim).GetExtentAttr().Get()
        if ext:
            for k in range(3):
                lo[k] = min(lo[k], ext[0][k])
                hi[k] = max(hi[k], ext[1][k])
if lo[0] < 1e8:
    gt.Set(Gf.Vec3d(-(lo[0] + hi[0]) / 2 * scale,
                    -0.34 - (lo[1] + hi[1]) / 2 * scale,
                    TABLE_Z + 0.004 - lo[2] * scale))
    log(f"garment {(hi[0]-lo[0])*scale:.2f}x{(hi[1]-lo[1])*scale:.2f} m, "
        f"texture={os.path.basename(tex) if tex else None}")

# lights
# Storm honours these intensities on the usual scale -- dropping them to ~1.0
# produced a near-black frame. The blown-out robots in the first attempt came
# from FrameRecorder's CAMERA light, not from these, so these stay and that
# goes.
UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(1400.0)
key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 35.0))

# camera
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(30.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
view = Gf.Matrix4d(1.0)
view.SetLookAt(Gf.Vec3d(0.75, -1.45, TABLE_Z + 0.62),
               Gf.Vec3d(0.0, -0.36, TABLE_Z + 0.10), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
stage.GetRootLayer().Save()
log(f"scene authored -> {usd_path}")

rec = UsdAppUtils.FrameRecorder()
rec.SetRendererPlugin("HdStormRendererPlugin")
rec.SetImageWidth(args.width)
# Camera light ON. Storm's dome-light support is limited without an
# environment texture -- with the camera light off the table rendered black --
# and now that the robots carry PreviewSurface instead of MDL it no longer
# blows them out.
rec.SetCameraLightEnabled(True)
rec.SetComplexity(1.5)   # a float; "high" is a usdview label, not the API
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
ok = rec.Record(stage, cam, Usd.TimeCode.Default(), args.out)
log(f"Record() -> {ok} ({args.out})")
del rec

if not ok or not os.path.exists(args.out):
    raise SystemExit(2)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

im = Image.open(args.out)
# Composite over grey: FrameRecorder writes RGBA and transparent background
# reads as pure black through .convert("RGB"), which made the first frame look
# like a failure at mean 1.7 when it was actually correct.
if im.mode == "RGBA":
    bg = Image.new("RGBA", im.size, (60, 62, 68, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    im.save(args.out)
else:
    im = im.convert("RGB")
arr = np.asarray(im)
log(f"RGB shape={arr.shape} mean={arr.mean():.1f} std={arr.std():.1f} "
    f"unique={np.unique(arr).size}")
sys.exit(0)
