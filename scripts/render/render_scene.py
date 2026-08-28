"""Render the real LeHome challenge scene with Isaac Sim's RTX renderer.

The point of the whole project is Isaac Sim, and the pinned 5.1 stack cannot
render on this cluster (docs/STAGE0.md). 6.0 can -- so this renders on 6.0,
using the challenge's OWN assets: the SO-101 follower robot USD, a Release
garment mesh with its fabric texture, at the exact bimanual poses the task
config specifies, driven by a REAL demonstration trajectory pulled out of the
released dataset.

What this is: proof the assets, the poses and the renderer all work here, and a
look at the scene the policy will see.

What this is NOT: a policy rollout, and not the physics of the task. The
garment is rendered as a static mesh, not as PhysX particle cloth -- so it does
not drape or fold. Folding needs the task environment, which needs 5.1, which
is the thing that is broken. Said plainly rather than cropped out of frame.

Deliberately raw `isaacsim` + `omni.replicator`, not Isaac Lab: 6.0 pairs with
isaaclab 3.0.0b2 whose API moved from 2.x, and the probe that proved RTX works
on this cluster used this same path. One unknown at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--assets", default=os.environ.get("ASSETS", ""))
p.add_argument("--traj", default="")
p.add_argument("--out", default="frames")
p.add_argument("--garment", default="Top_Short/Top_Short_Seen_0")
p.add_argument("--frames", type=int, default=120)
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=540)
p.add_argument("--stride", type=int, default=3)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)


def log(*a):
    print("[render]", *a, flush=True)


from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": args.width,
    "height": args.height,
})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

try:
    from isaacsim.core.version import get_version
    log("isaacsim", get_version()[0])
except Exception as e:  # noqa: BLE001
    log("version unavailable:", e)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
log("stage opened; RTX survived startup")

world = UsdGeom.Xform.Define(stage, "/World")


# ---------------------------------------------------------------- helpers
def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> Gf.Matrix4d:
    """USD camera transform looking from `eye` at `target`.

    USD cameras look down -Z with +Y up, so the basis is built accordingly
    rather than by composing Euler angles, which is where sign errors live.
    """
    e = Gf.Vec3d(*eye)
    t = Gf.Vec3d(*target)
    fwd = (t - e).GetNormalized()
    u = Gf.Vec3d(*up)
    right = Gf.Cross(fwd, u).GetNormalized()
    trueup = Gf.Cross(right, fwd).GetNormalized()
    m = Gf.Matrix4d(1.0)
    m.SetRow3(0, right)
    m.SetRow3(1, trueup)
    m.SetRow3(2, -fwd)
    m.SetTranslateOnly(e)
    return m


def add_material(path, colour, rough=0.6, metal=0.0, texture=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/tex")
        st.CreateIdAttr("UsdUVTexture")
        st.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        st.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        st.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        rd = UsdShade.Shader.Define(stage, path + "/uv")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        st.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            st.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim_path, mat):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


# ------------------------------------------------------------------ scene
# Table: the challenge mounts both arms at z=0.5, so the work surface is there.
TABLE_Z = 0.5
table = UsdGeom.Cube.Define(stage, "/World/table")
table.CreateSizeAttr(1.0)
xf = UsdGeom.Xformable(table)
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.30, TABLE_Z - 0.4))
xf.AddScaleOp().Set(Gf.Vec3f(1.5, 1.1, 0.8))
bind("/World/table", add_material("/World/mats/table", (0.62, 0.55, 0.46), 0.75))

floor = UsdGeom.Cube.Define(stage, "/World/floor")
floor.CreateSizeAttr(1.0)
xf = UsdGeom.Xformable(floor)
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1.0))
xf.AddScaleOp().Set(Gf.Vec3f(12.0, 12.0, 0.1))
bind("/World/floor", add_material("/World/mats/floor", (0.18, 0.19, 0.22), 0.9))

# Robots, at the exact poses from garment_bi_cfg_v2.py
ROBOT_USD = os.path.join(args.assets, "robots/lerobot/so101_follower.usd")
ROBOTS = {"Left_Robot": (-0.23, -0.25, TABLE_Z), "Right_Robot": (0.23, -0.25, TABLE_Z)}
for name, pos in ROBOTS.items():
    x = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}")
    x.GetPrim().GetReferences().AddReference(ROBOT_USD)
    UsdGeom.Xformable(x).AddTranslateOp().Set(Gf.Vec3d(*pos))
    log(f"referenced {name} at {pos}")

# Garment: real Release mesh + its fabric texture, at the config's 0.45 scale
gdir = os.path.join(args.assets, "objects/Challenge_Garment/Release", args.garment)
gusd = gjson = None
for f in sorted(os.listdir(gdir)):
    if f.endswith(".usd"):
        gusd = os.path.join(gdir, f)
    if f.endswith(".json"):
        gjson = os.path.join(gdir, f)
scale = 0.45
tex = None
if gjson:
    meta = json.load(open(gjson))
    scale = float(meta.get("scale", [0.45])[0])
    for v in meta.get("visual_usd_paths", []):
        cand = os.path.join(args.assets, v.lstrip("/").replace("Assets/", "", 1))
        d = os.path.dirname(cand)
        if os.path.isdir(d):
            jpgs = [j for j in sorted(os.listdir(d)) if j.lower().endswith((".jpg", ".png"))
                    and "color" in j.lower()]
            if jpgs:
                tex = os.path.join(d, jpgs[0])
log(f"garment {os.path.basename(gusd or '?')} scale={scale} texture={os.path.basename(tex) if tex else None}")

if gusd:
    g = UsdGeom.Xform.Define(stage, "/World/Garment")
    g.GetPrim().GetReferences().AddReference(gusd)
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.34, TABLE_Z + 0.005))
    gx.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    gmat = add_material("/World/mats/garment", (0.85, 0.32, 0.30), 0.85, texture=tex)
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Garment") and prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(prim).Bind(gmat)

# Lighting: a dome so nothing depends on one direction, plus a key for shape.
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(900.0)
key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3200.0)
key.CreateAngleAttr(1.2)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-52.0, 0.0, 38.0))
fill = UsdLux.SphereLight.Define(stage, "/World/fill")
fill.CreateIntensityAttr(28000.0)
fill.CreateRadiusAttr(0.4)
UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(-1.4, -1.5, 1.9))
log("scene assembled")

# ---------------------------------------------------------------- cameras
hero = UsdGeom.Camera.Define(stage, "/World/hero")
hero.CreateFocalLengthAttr(32.0)
hero.CreateHorizontalApertureAttr(36.0)
hero.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
hero_xf = UsdGeom.Xformable(hero)
hero_op = hero_xf.AddTransformOp()

# The challenge's own overhead view: right arm base (0.23,-0.25,0.5) plus the
# top_camera offset (0.245,-0.44,0.56) from garment_bi_cfg_v2.py.
top = UsdGeom.Camera.Define(stage, "/World/topcam")
top.CreateFocalLengthAttr(28.7)
top.CreateHorizontalApertureAttr(38.11)
top.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
UsdGeom.Xformable(top).AddTransformOp().Set(
    look_at((0.23 + 0.245, -0.25 - 0.44, TABLE_Z + 0.56), (0.0, -0.34, TABLE_Z)))

rp_hero = rep.create.render_product("/World/hero", (args.width, args.height))
rp_top = rep.create.render_product("/World/topcam", (640, 480))
ann_hero = rep.AnnotatorRegistry.get_annotator("rgb")
ann_top = rep.AnnotatorRegistry.get_annotator("rgb")
ann_hero.attach([rp_hero])
ann_top.attach([rp_top])
log("render products created")

# ------------------------------------------------------------ articulation
traj = None
if args.traj and os.path.exists(args.traj):
    traj = np.load(args.traj)
    log(f"trajectory {traj.shape} from {os.path.basename(args.traj)}")

arts = {}
try:
    from isaacsim.core.prims import SingleArticulation
    for _ in range(6):
        app.update()
    for name in ROBOTS:
        a = SingleArticulation(prim_path=f"/World/Robot/{name}")
        a.initialize()
        arts[name] = a
        log(f"{name} articulation: {a.num_dof} dof {a.dof_names}")
except Exception as e:  # noqa: BLE001
    log(f"articulation unavailable ({type(e).__name__}: {e}); rendering static")
    arts = {}

# ------------------------------------------------------------------ render
from PIL import Image  # noqa: E402

n = args.frames
log(f"rendering {n} frames -> {args.out}")
for i in range(n):
    a = 2.0 * math.pi * i / n
    r = 1.75
    hero_op.Set(look_at((r * math.sin(a) * 0.85, -0.34 + -r * math.cos(a) * 0.85,
                         TABLE_Z + 0.55 + 0.16 * math.sin(a * 2)),
                        (0.0, -0.32, TABLE_Z + 0.06)))

    if arts and traj is not None:
        k = min(int(i * args.stride) % len(traj), len(traj) - 1)
        for j, name in enumerate(("Left_Robot", "Right_Robot")):
            art = arts.get(name)
            if art is None:
                continue
            q = traj[k][j * 6:(j + 1) * 6]
            try:
                art.set_joint_positions(np.asarray(q[:art.num_dof], dtype=np.float32))
            except Exception as e:  # noqa: BLE001
                if i == 0:
                    log(f"joint set failed on {name}: {e}")

    rep.orchestrator.step(rt_subframes=8)
    for k_, (ann, tag) in enumerate(((ann_hero, "hero"), (ann_top, "top"))):
        d = ann.get_data()
        arr = np.asarray(d)
        if arr.size == 0:
            if i == 0:
                log(f"WARNING: {tag} annotator returned nothing on frame 0")
            continue
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        Image.fromarray(arr.astype(np.uint8)).save(
            os.path.join(args.out, f"{tag}_{i:04d}.png"))
        if i == 0:
            log(f"{tag} frame0 shape={arr.shape} mean={arr.mean():.1f} "
                f"std={arr.std():.1f} unique={len(np.unique(arr))}")

log("done")
app.close()
