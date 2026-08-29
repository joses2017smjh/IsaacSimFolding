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
p.add_argument("--defect", default="none",
               help="reintroduce one known bug, for the failure/fix gallery")
p.add_argument("--stills", type=int, default=0,
               help="render N single frames instead of an orbit (gallery mode)")
p.add_argument("--tag", default="")
args = p.parse_args()

# Each defect below is a bug this renderer actually shipped, kept reproducible
# so the fix can be shown against it rather than asserted. See docs/RENDER.md.
DEFECTS = {
    "none": "the corrected render",
    "exposure": "lights 4x too bright -- frame mean 217/255, a white-out",
    "framing": "orbit at 0.92 m and 0.34 m of height -- grazes the tabletop",
    "gravity": "physics on with an unanchored base -- the arms topple",
    "garment_scale": "the USD's own xformOp:scale=0.01 left in place -- a 4 mm garment",
    "garment_culling": "single-sided cloth mesh -- every backface culled",
    "no_yaw": "robots not yawed 180 deg -- both arms face away from the garment",
}
if args.defect not in DEFECTS:
    raise SystemExit(f"unknown --defect {args.defect!r}; expected {sorted(DEFECTS)}")
DEFECT = args.defect

os.makedirs(args.out, exist_ok=True)


def log(*a):
    print("[render]", *a, flush=True)


from isaacsim import SimulationApp  # noqa: E402

# PathTracing, not RayTracedLighting. RTL is the real-time mode: extra
# subframes do not accumulate into a converged image, so 8, 24 and 64
# subframes all produced the same salt-and-pepper grain on the diffuse
# tabletop. Path tracing accumulates per subframe and cleans up properly.
app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "samples_per_pixel_per_frame": 32,
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
# The reference goes on a CHILD prim, never on the prim carrying our
# transform. so101_follower.usd already declares
# xformOpOrder = [translate, orient, scale], and AddTranslateOp() on a prim
# that already has one raises rather than replacing it:
#   "The xformOp 'xformOp:translate' already exists in xformOpOrder"
# A wrapper Xform owns the placement, the child owns the asset, and neither
# has to know what the other declared.
for name, pos in ROBOTS.items():
    holder = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}")
    UsdGeom.Xformable(holder).AddTranslateOp().Set(Gf.Vec3d(*pos))
    child = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}/model")
    child.GetPrim().GetReferences().AddReference(ROBOT_USD)
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
    # The JSON names a sibling USD, e.g. ".../Color_Texture/Fabric050_1K_JPG.usd",
    # while the images live in ".../Color_Texture/Fabric050_1K-JPG/BaseColor.jpg".
    # The stem differs from the directory by exactly one character -- the last
    # underscore is a dash -- which is why matching on "color" in the wrong
    # directory found nothing on the first two runs.
    import glob as _glob

    for v in meta.get("visual_usd_paths", []):
        rel = v.lstrip("/")
        rel = rel[len("Assets/"):] if rel.startswith("Assets/") else rel
        full = os.path.join(args.assets, rel)
        stem = os.path.splitext(os.path.basename(full))[0]
        parent = os.path.dirname(full)
        cands = [os.path.join(parent, stem.rsplit("_", 1)[0] + "-" + stem.rsplit("_", 1)[-1])]
        cands += sorted(_glob.glob(os.path.join(parent, stem.split("_")[0] + "*" + os.sep)))
        for d in cands:
            d = d.rstrip(os.sep)
            if not os.path.isdir(d):
                continue
            for want in ("BaseColor", "Albedo", "Diffuse", "Color"):
                hit = _glob.glob(os.path.join(d, f"*{want}*.jpg")) + \
                      _glob.glob(os.path.join(d, f"*{want}*.png"))
                if hit:
                    tex = hit[0]
                    break
            if tex:
                break
        if tex:
            break
log(f"garment {os.path.basename(gusd or '?')} scale={scale} texture={os.path.basename(tex) if tex else None}")

if gusd:
    g = UsdGeom.Xform.Define(stage, "/World/Garment")
    gx = UsdGeom.Xformable(g)
    gx.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.34, TABLE_Z + 0.005))
    gx.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    # Same reason as the robots: the garment USD brings its own xform ops.
    gchild = UsdGeom.Xform.Define(stage, "/World/Garment/mesh")
    gchild.GetPrim().GetReferences().AddReference(gusd)

    # Neutralise the USD's own scale. TCSC_067_obj_exp.usd authors
    #   xformOp:scale = 0.01
    # on /World/mesh, so its 0.95-unit mesh renders at 9.5 mm and the JSON's
    # 0.45 took it down to 4 mm -- a garment the size of a grain of rice, which
    # is why three renders showed a bare table. The challenge's own loader sets
    # scale from the JSON, so the 0.01 is an authoring artefact, not the
    # operative scale. Override it and let `scale` above be the only one.
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith("/World/Garment/mesh"):
            continue
        xf = UsdGeom.Xformable(prim)
        for op in (xf.GetOrderedXformOps() if xf else []):
            if op.GetOpName().endswith("scale"):
                was = op.Get()
                if DEFECT != "garment_scale":
                    op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
                log(f"neutralised inner scale on {prim.GetPath()}: {was} -> (1,1,1)")
    gmat = add_material("/World/mats/garment", (0.82, 0.24, 0.26), 0.88, texture=tex)
    nmesh = 0
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Garment") and prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI(prim).Bind(gmat)
            # A garment is a zero-thickness sheet. Single-sided by default
            # means every backface is culled, and from an overhead camera that
            # is most of the garment -- which is why the first renders showed
            # a bare table where a shirt should be.
            UsdGeom.Mesh(prim).CreateDoubleSidedAttr(DEFECT != "garment_culling")
            UsdGeom.Imageable(prim).MakeVisible()
            nmesh += 1
    log(f"garment meshes bound (double-sided): {nmesh}")

    # Place from the MESH EXTENTS, not BBoxCache.
    # BBoxCache on the wrapper returned a 4 mm box for a garment whose mesh is
    # 9,774 points spanning 0.95 m -- it was measuring the wrapper, not the
    # referenced geometry. The mesh's own extent attribute is authored and
    # correct, so use it and do the arithmetic here.
    lo = [1e9, 1e9, 1e9]
    hi = [-1e9, -1e9, -1e9]
    for prim in stage.Traverse():
        if not prim.GetPath().pathString.startswith("/World/Garment"):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        ext = UsdGeom.Mesh(prim).GetExtentAttr().Get()
        if not ext:
            continue
        for k in range(3):
            lo[k] = min(lo[k], ext[0][k])
            hi[k] = max(hi[k], ext[1][k])
    if lo[0] < 1e8:
        size = [(hi[k] - lo[k]) * scale for k in range(3)]
        log(f"garment mesh extent {tuple(round(v,3) for v in lo)}..{tuple(round(v,3) for v in hi)} "
            f"-> {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m at scale {scale}")
        cx = (lo[0] + hi[0]) / 2 * scale
        cy = (lo[1] + hi[1]) / 2 * scale
        gx.GetOrderedXformOps()[0].Set(
            Gf.Vec3d(-cx, -0.34 - cy, TABLE_Z + 0.004 - lo[2] * scale))
        log(f"garment seated on the table at z={TABLE_Z + 0.004 - lo[2] * scale:.3f}")
        from pxr import UsdGeom as _UG2
        wb = _UG2.BBoxCache(0, [_UG2.Tokens.default_, _UG2.Tokens.render],
                            useExtentsHint=True).ComputeWorldBound(
            stage.GetPrimAtPath("/World/Garment/mesh")).ComputeAlignedRange()
        log(f"garment world bound {tuple(round(v,3) for v in wb.GetMin())}"
            f"..{tuple(round(v,3) for v in wb.GetMax())}"
            + ("  <-- EMPTY, it will not render" if wb.IsEmpty() else ""))
    else:
        log("WARNING: no garment mesh extents found -- the reference did not load")

# Lighting: a dome so nothing depends on one direction, plus a key for shape.
# Intensities are ~4x lower than the first attempt, which came back at mean
# 217/255 -- a white-out, not a render. Exposure is reported per frame now so
# this is a number to tune against rather than something to eyeball.
_EXP = 4.0 if DEFECT == "exposure" else 1.0
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(220.0 * _EXP)
key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(700.0 * _EXP)
key.CreateAngleAttr(2.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-52.0, 0.0, 38.0))
fill = UsdLux.SphereLight.Define(stage, "/World/fill")
fill.CreateIntensityAttr(4000.0 * _EXP)
fill.CreateRadiusAttr(0.5)
UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(-1.0, -1.2, 1.5))
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

# The arms need a running physics scene. The first attempt built a
# SingleArticulation on a stage with no simulation context and got
#   'NoneType' object has no attribute 'create_articulation_view'
# -- the articulation view is created by the physics backend, so the World has
# to exist and be reset before any handle is valid.
world = None
arts = {}
try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
    # Gravity off. The robots are referenced as raw USD, so their bases are not
    # anchored the way the task's ArticulationCfg anchors them, and stepping
    # physics simply tipped them over onto the table -- the previous render is
    # two SO-101s lying on their sides. This is a kinematic replay of recorded
    # joint angles, not a dynamics simulation, so gravity has nothing to
    # contribute and plenty to break.
    if DEFECT != "gravity":
        try:
            world.get_physics_context().set_gravity(0.0)
            log("gravity disabled (kinematic replay)")
        except Exception as e:  # noqa: BLE001
            log(f"could not disable gravity: {e}")
    else:
        log("gravity LEFT ON (defect): the unanchored bases will topple")
    world.reset()
    for _ in range(4):
        world.step(render=False)
    for name in ROBOTS:
        a = SingleArticulation(prim_path=f"/World/Robot/{name}/model", name=name)
        a.initialize()
        arts[name] = a
        log(f"{name} articulation: {a.num_dof} dof {list(a.dof_names)[:8]}")
except Exception as e:  # noqa: BLE001
    log(f"articulation unavailable ({type(e).__name__}: {e}); rendering static")
    arts = {}
    world = None

# ------------------------------------------------------------------ render
from PIL import Image  # noqa: E402

n = args.stills or args.frames
prefix = args.tag or ("defect_" + DEFECT if DEFECT != "none" else "fixed")
log(f"rendering {n} frame(s) [{DEFECT}: {DEFECTS[DEFECT]}] -> {args.out}")
for i in range(n):
    # 1.75 m made the robots a speck; 0.92 m at 0.34 m of height grazed the
    # tabletop and framed nothing but wood. The arms occupy roughly
    # x in [-0.4,0.4], y in [-0.6,-0.1], z in [0.5,0.85], so orbit wider and
    # look DOWN into that volume rather than across it.
    a = 2.0 * math.pi * i / n
    if args.stills:
        a = 2.0 * math.pi * 0.62      # one fixed, flattering angle for stills
    if DEFECT == "framing":
        r, h, tgt = 0.92, TABLE_Z + 0.34, (0.0, -0.33, TABLE_Z + 0.07)
    else:
        r, h, tgt = 1.25, TABLE_Z + 0.62 + 0.10 * math.sin(a * 2), (0.0, -0.36, TABLE_Z + 0.13)
    hero_op.Set(look_at((r * math.sin(a), -0.34 - r * math.cos(a) * 0.95, h), tgt))

    if arts and traj is not None:
        k = min(int(i * args.stride) % len(traj), len(traj) - 1)
        for j, name in enumerate(("Left_Robot", "Right_Robot")):
            art = arts.get(name)
            if art is None:
                continue
            q = np.asarray(traj[k][j * 6:(j + 1) * 6], dtype=np.float32)
            try:
                art.set_joint_positions(q[:art.num_dof])
                # Teleporting alone leaves the drives commanding the old
                # target, which fights the new pose on the next step.
                if hasattr(art, "set_joint_position_targets"):
                    art.set_joint_position_targets(q[:art.num_dof])
            except Exception as e:  # noqa: BLE001
                if i == 0:
                    log(f"joint set failed on {name}: {e}")
        if world is not None:
            # render=True, not False. With render=False the physics state is
            # never flushed into the render pipeline, so every frame draws the
            # pose authored in the USD -- the overhead camera's 90 frames came
            # out BYTE-IDENTICAL and the "moving" arms were a camera orbit
            # around a statue.
            world.step(render=True)

    # 8 subframes left heavy salt-and-pepper noise; 24 still showed grain on
    # the diffuse tabletop, which is the worst case for a path tracer.
    rep.orchestrator.step(rt_subframes=int(os.environ.get("RT_SUBFRAMES", "48")))
    for k_, (ann, tag) in enumerate(((ann_hero, "hero"), (ann_top, "top"))):
        d = ann.get_data()
        arr = np.asarray(d)
        if arr.size == 0:
            if i == 0:
                log(f"WARNING: {tag} annotator returned nothing on frame 0")
            continue
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        name = (f"{prefix}_{tag}.png" if args.stills
                else f"{tag}_{i:04d}.png")
        Image.fromarray(arr.astype(np.uint8)).save(os.path.join(args.out, name))
        if i == 0:
            log(f"{tag} frame0 shape={arr.shape} mean={arr.mean():.1f} "
                f"std={arr.std():.1f} unique={len(np.unique(arr))} "
                f"clipped={(arr >= 250).mean() * 100:.1f}%")

log("done")
app.close()
