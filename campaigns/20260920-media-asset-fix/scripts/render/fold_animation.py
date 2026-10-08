"""Render the simulated cloth as an animation.

The pieces this joins:
  - `cloth_sim51.py` produced results/cloth51.npz -- (365, 9774, 3) world-space
    particle positions from REAL PhysX particle cloth on Isaac Sim 5.1, driven
    by all 364 steps of episode 0 of the released demonstrations.
  - `probe_storm51.py` established that 5.1 can rasterise through OpenUSD's
    Storm without ever loading the RTX delegate that segfaults here.

So each frame writes that frame's particle positions into the garment mesh's
`points` attribute and rasterises it. The geometry is simulation output, not an
authored pose -- which is the whole difference from every earlier render in
this repo, where the garment was a static blob because nothing had ever
draped it.

Topology comes from the original garment USD, so the mesh is the challenge's
own mesh with simulated vertices.
"""

from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--particles", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--garment", default="Top_Short/Top_Short_Seen_0")
ap.add_argument("--out", default="frames_fold")
ap.add_argument("--width", type=int, default=800)
ap.add_argument("--stride", type=int, default=3)
args = ap.parse_args()

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.makedirs(args.out, exist_ok=True)


def log(*a):
    print("[fold]", *a, flush=True)


# --- headless GL 4.5 COMPATIBILITY context (core throws in HgiGL) ----------
from OpenGL import EGL as E  # noqa: E402

dpy = E.eglGetDisplay(E.EGL_DEFAULT_DISPLAY)
maj, mino = E.EGLint(), E.EGLint()
E.eglInitialize(dpy, maj, mino)
cfgs = (E.EGLConfig * 1)()
n = E.EGLint()
E.eglChooseConfig(dpy, [
    E.EGL_SURFACE_TYPE, E.EGL_PBUFFER_BIT,
    E.EGL_RED_SIZE, 8, E.EGL_GREEN_SIZE, 8, E.EGL_BLUE_SIZE, 8,
    E.EGL_ALPHA_SIZE, 8, E.EGL_DEPTH_SIZE, 24,
    E.EGL_RENDERABLE_TYPE, E.EGL_OPENGL_BIT, E.EGL_NONE], cfgs, 1, n)
E.eglBindAPI(E.EGL_OPENGL_API)
surf = E.eglCreatePbufferSurface(dpy, cfgs[0],
                                 [E.EGL_WIDTH, args.width, E.EGL_HEIGHT,
                                  args.width, E.EGL_NONE])
ctx = E.eglCreateContext(dpy, cfgs[0], E.EGL_NO_CONTEXT, [
    E.EGL_CONTEXT_MAJOR_VERSION, 4, E.EGL_CONTEXT_MINOR_VERSION, 5,
    E.EGL_CONTEXT_OPENGL_PROFILE_MASK,
    E.EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, E.EGL_NONE])
E.eglMakeCurrent(dpy, surf, surf, ctx)
from OpenGL import GL  # noqa: E402

log(f"GL {GL.glGetString(GL.GL_VERSION).decode()}")

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdAppUtils, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

_npz = np.load(args.particles, allow_pickle=True)
P = _npz["particles"]
log(f"particles {P.shape} from {os.path.basename(args.particles)}")

# Robot link poses, if the simulation recorded them. The animation is meant to
# show the ARMS doing the folding -- a garment moving on its own is not what
# anyone means by "the robot folding a shirt".
# The articulation reports bodies in this order; it was logged by the
# simulation and is stable for so101_follower.usd. Used as a fallback because
# the names were written as a pickled object array that will not read back --
# not worth a 45-second simulator boot to re-record a constant.
DEFAULT_BODIES = ["base", "shoulder", "upper_arm", "lower_arm",
                  "wrist", "gripper", "jaw"]
ARMS = {}
for side in ("left", "right"):
    if f"{side}_pos" not in _npz:
        continue
    pos = _npz[f"{side}_pos"]
    try:
        names = [str(x) for x in _npz[f"{side}_names"]]
    except Exception:  # noqa: BLE001
        names = DEFAULT_BODIES[:pos.shape[1]]
        log(f"{side}: names unreadable, using known body order")
    ARMS[side] = (pos, _npz[f"{side}_quat"], names)
    log(f"{side} arm: {pos.shape} poses, bodies={names}")
if not ARMS:
    log("no arm poses in this npz -- rendering cloth only")

# --- topology + texture from the original garment -------------------------
gdir = os.path.join(args.assets, "objects/Challenge_Garment/Release", args.garment)
gusd = next(os.path.join(gdir, f) for f in sorted(os.listdir(gdir)) if f.endswith(".usd"))
src = Usd.Stage.Open(gusd)
mesh_src = next(p for p in src.Traverse() if p.IsA(UsdGeom.Mesh))
m_src = UsdGeom.Mesh(mesh_src)
counts = m_src.GetFaceVertexCountsAttr().Get()
indices = m_src.GetFaceVertexIndicesAttr().Get()
uvs = None
pv = UsdGeom.PrimvarsAPI(mesh_src).GetPrimvar("st")
if pv:
    uvs = pv.Get()
log(f"topology: {len(counts)} faces, {len(indices)} indices, "
    f"{P.shape[1]} verts, uvs={'yes' if uvs else 'no'}")

import glob as _g  # noqa: E402
import json as _json  # noqa: E402

tex = None
gjson = next((os.path.join(gdir, f) for f in sorted(os.listdir(gdir))
              if f.endswith(".json")), None)
if gjson:
    for v in _json.load(open(gjson)).get("visual_usd_paths", []):
        rel = v.lstrip("/")
        rel = rel[len("Assets/"):] if rel.startswith("Assets/") else rel
        full = os.path.join(args.assets, rel)
        stem = os.path.splitext(os.path.basename(full))[0]
        d = os.path.join(os.path.dirname(full),
                         stem.rsplit("_", 1)[0] + "-" + stem.rsplit("_", 1)[-1])
        hit = _g.glob(os.path.join(d, "*BaseColor*")) if os.path.isdir(d) else []
        if hit:
            tex = hit[0]
            break
log(f"texture: {os.path.basename(tex) if tex else None}")

# --- scene ----------------------------------------------------------------
usd_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fold_anim.usda")
if os.path.exists(usd_path):
    os.unlink(usd_path)
stage = Usd.Stage.CreateNew(usd_path)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def material(path, colour, tex_file=None, rough=0.75):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/s")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    if tex_file:
        rd = UsdShade.Shader.Define(stage, path + "/uv")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st = UsdShade.Shader.Define(stage, path + "/tex")
        st.CreateIdAttr("UsdUVTexture")
        st.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_file)
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


lo = P.reshape(-1, 3).min(axis=0)
hi = P.reshape(-1, 3).max(axis=0)
table_top = float(P[-1][:, 2].min())      # the cloth settles onto it
log(f"cloth bounds x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
    f"z[{lo[2]:.2f},{hi[2]:.2f}]  table_top~{table_top:.3f}")

table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
tx = UsdGeom.Xformable(table)
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
tx.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.30, table_top - 0.2))
tx.AddScaleOp().Set(Gf.Vec3f(1.8, 1.4, 0.4))
UsdShade.MaterialBindingAPI(table.GetPrim()).Bind(
    material("/World/mats/table", (0.72, 0.68, 0.60)))

garment = UsdGeom.Mesh.Define(stage, "/World/cloth")
garment.CreateFaceVertexCountsAttr(counts)
garment.CreateFaceVertexIndicesAttr(indices)
garment.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P[0]))
garment.CreateDoubleSidedAttr(True)
if uvs:
    UsdGeom.PrimvarsAPI(garment).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying if len(uvs) == len(indices) else UsdGeom.Tokens.vertex
    ).Set(uvs)
UsdShade.MaterialBindingAPI(garment.GetPrim()).Bind(
    material("/World/mats/cloth", (0.82, 0.26, 0.26), tex_file=tex, rough=0.9))

# Storm under-lights this scene badly at "sensible" intensities -- the first
# animation came back at a mean of ~20/255 with a correct but nearly invisible
# shirt. Push both hard and let the table read mid-grey.
# --- robot links -----------------------------------------------------------
# One Xform per link, each referencing that link's subtree from the robot USD
# and driven by the recorded world pose. /visuals is a SIBLING of the USD's
# defaultPrim, so the reference names the prim path explicitly -- referencing
# the layer alone composes only the default prim and brings no meshes.
ROBOT_USD = os.path.join(args.assets, "robots/lerobot/so101_follower.usd")
VISUAL_LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
link_ops = {}
robot_mat = material("/World/mats/robot", (0.86, 0.74, 0.16), rough=0.45)
for side, (pos, quat, names) in ARMS.items():
    for bi, bname in enumerate(names):
        key = bname.lower()
        match = next((v for v in VISUAL_LINKS if v == key), None)
        if match is None:
            continue
        path = f"/World/Robot_{side}/{match}"
        xf = UsdGeom.Xform.Define(stage, path)
        child = UsdGeom.Xform.Define(stage, path + "/geo")
        child.GetPrim().GetReferences().AddReference(ROBOT_USD, f"/visuals/{match}")
        op = UsdGeom.Xformable(xf).AddTransformOp()
        link_ops[(side, bi)] = op
log(f"robot link xforms: {len(link_ops)}")
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World/Robot_") and prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI(prim).Bind(
            robot_mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(9000.0)
key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(14000.0)
fill = UsdLux.DistantLight.Define(stage, "/World/fill")
fill.CreateIntensityAttr(6000.0)
UsdGeom.Xformable(fill).AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, -120.0))
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 40.0))

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(35.0)
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 50.0))
# Frame the WORKSPACE, not just the cloth. The arm bases sit at x = +/-0.23 and
# the links reach well above the table, so a camera fitted to the garment's
# bounds clips an arm out of frame -- which it did.
allx = [float(lo[0]), float(hi[0])]
ally = [float(lo[1]), float(hi[1])]
allz = [float(lo[2]), float(hi[2])]
for _side, (_p, _q, _n) in ARMS.items():
    allx += [float(_p[..., 0].min()), float(_p[..., 0].max())]
    ally += [float(_p[..., 1].min()), float(_p[..., 1].max())]
    allz += [float(_p[..., 2].min()), float(_p[..., 2].max())]
cx, cy = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2
span = max(max(allx) - min(allx), max(ally) - min(ally), max(allz) - min(allz), 0.4)
log(f"framing span {span:.2f} m around ({cx:.2f}, {cy:.2f})")
eye = Gf.Vec3d(cx + span * 0.75, cy - span * 1.5, table_top + span * 0.95)
view = Gf.Matrix4d(1.0)
view.SetLookAt(eye, Gf.Vec3d(cx, cy, table_top + span * 0.18), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())
stage.GetRootLayer().Save()

points_attr = garment.GetPointsAttr()
rec = UsdAppUtils.FrameRecorder()
rec.SetRendererPlugin("HdStormRendererPlugin")
rec.SetImageWidth(args.width)
rec.SetCameraLightEnabled(True)
rec.SetComplexity(1.0)

n_out = 0
def _mat_from(p, q):
    """World transform from position and a wxyz quaternion."""
    m = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(float(q[0]),
                                         Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))),
                    Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    return m


for i in range(0, len(P), args.stride):
    points_attr.Set(Vt.Vec3fArray.FromNumpy(P[i]))
    for (side, bi), op in link_ops.items():
        pos, quat, _ = ARMS[side]
        j = min(i, len(pos) - 1)
        op.Set(_mat_from(pos[j][bi], quat[j][bi]))
    out = os.path.join(args.out, f"fold_{n_out:04d}.png")
    if not rec.Record(stage, cam, Usd.TimeCode.Default(), out):
        log(f"Record failed at frame {i}")
        break
    n_out += 1
    if n_out % 20 == 1:
        try:
            from PIL import Image as _I

            _a = np.asarray(_I.open(out).convert("RGB"))
            _m = f" mean={_a.mean():.0f}"
        except Exception:  # noqa: BLE001
            _m = ""
        log(f"frame {i:4d} -> {os.path.basename(out)}  "
            f"cloth z=[{P[i][:,2].min():.3f},{P[i][:,2].max():.3f}]{_m}")
del rec
log(f"wrote {n_out} frames to {args.out}")
sys.exit(0 if n_out > 1 else 2)
