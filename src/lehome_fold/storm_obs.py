"""RGB observations from Storm, for a stack whose RTX cameras segfault.

The problem this solves. LeHome's env hands the policy three RGB images
(`garment_bi_v2._get_observations` reads `top_camera.data.output["rgb"]` plus
the two wrist cameras). On Isaac Sim 5.1 the RTX TiledCameras segfault on this
cluster, so `cloth_sim51.py` stubs them and returns ZEROS -- fine for measuring
particle positions, useless for a vision-language-action policy, which would be
looking at black frames.

Isaac Sim 6.0 renders but PhysX there dropped particle cloth, so 5.1 is the only
stack that can simulate this task at all. The way through is OpenUSD's Storm
rasteriser, which never loads the RTX plugins -- and which was measured
rendering INSIDE a live Kit process (slurm/68_storm_in_kit.sbatch), so it can
sit in the same process as the simulation.

What this preserves and what it costs:

  physics    official -- PhysX particle cloth, untouched
  scorer     official -- success_checker_garment_fold is geometric over
                         particle positions and does not care what drew the
                         pixels
  images     DEVIATION -- Storm rasterises where the released demonstrations
                         were path-traced, and Storm cannot evaluate the MDL
                         materials Omniverse assets ship, so surfaces are
                         approximated with UsdPreviewSurface

That last line is the whole caveat and it is not small: a policy trained on
path-traced demonstrations is being shown rasterised frames. The gap is real
and must be measured before any success rate from this path is reported. See
docs/STORM.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

TABLE_Z = 0.5
# Straight out of garment_bi_cfg_v2.py.
ROBOT_BASES = {"left": (-0.23, -0.25, TABLE_Z), "right": (0.23, -0.25, TABLE_Z)}
TOP_CAM_OFFSET = (0.245, -0.44, 0.56)      # relative to the right arm's base
WRIST_CAM_OFFSET = (-0.001, 0.1, -0.04)    # relative to each gripper link
# Isaac quaternion order (w, x, y, z), straight out of garment_bi_cfg_v2.py.
WRIST_CAM_ROT = (-0.404379, -0.912179, -0.0451242, 0.0486914)
WRIST_CAM_PARENT = "gripper"   # the link the challenge parents them to
CAM_W, CAM_H = 640, 480



def _seam_map(mesh_pts: np.ndarray, n_particles: int) -> np.ndarray:
    """Map each render vertex to the particle it duplicates.

    UV seams split one physical vertex into several render vertices at the SAME
    rest position, so the correspondence is recoverable from the rest pose
    alone -- deduplicate positions and the k-th unique vertex is particle k.

    Measured on the released assets, and the correlation is exact:

        Pant_Short_Seen_0  11,573 verts  11,385 unique  188 dupes  -> was blank
        Top_Long_Seen_0    14,746 verts  14,544 unique  202 dupes  -> was blank
        Top_Short_Seen_1    9,774 verts   9,774 unique    0 dupes  -> rendered
        Top_Long_Seen_1    10,410 verts  10,410 unique    0 dupes  -> rendered

    11,385 is exactly the particle count for PS_049.

    An earlier version matched CURRENT particle positions geometrically. That
    cannot work: the rollout settles the garment before the first render, so
    the particles are draped over a table while the USD holds a flat rest pose.
    No rigid alignment relates those two shapes, and the check correctly
    refused at 0.947 normalised units rather than returning a scramble.
    """
    key = np.round(mesh_pts, 6)
    # First-appearance order, not np.unique's sorted order: a welder walks the
    # vertex list in order, so particle k should be the k-th vertex first seen.
    _, first_idx, inverse = np.unique(key, axis=0, return_index=True,
                                      return_inverse=True)
    order = np.argsort(first_idx)                  # unique ids -> appearance rank
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    weld = rank[inverse]
    n_unique = len(first_idx)
    if n_unique != n_particles:
        raise RuntimeError(
            f"garment mesh has {len(mesh_pts)} vertices collapsing to {n_unique} "
            f"unique positions, but the solver reports {n_particles} particles. "
            f"Rendering this would produce an invisible or garbled garment.")
    return weld


def _edge_sanity(points: np.ndarray, face_counts, face_indices) -> float:
    """Largest edge in the reconstructed mesh, for catching a bad weld.

    A scrambled correspondence still produces a full point list, so the mesh
    renders -- as a spray of stretched triangles. Edge length is what separates
    that from a garment, and it is cheap to check once at build.
    """
    fi = np.asarray(face_indices)
    fc = np.asarray(face_counts)
    if len(fc) == 0 or fc[0] < 2:
        return 0.0
    starts = np.concatenate([[0], np.cumsum(fc)[:-1]])
    a = fi[starts]
    b = fi[starts + 1]
    return float(np.linalg.norm(points[a] - points[b], axis=1).max())


@dataclass
class StormObsConfig:
    assets: str
    garment_dir: str
    width: int = CAM_W
    height: int = CAM_H
    scale: float = 0.45
    workdir: str = ""
    link_names: tuple[str, ...] = ("base", "shoulder", "upper_arm",
                                   "lower_arm", "wrist", "gripper", "jaw")
    extra: dict = field(default_factory=dict)


class StormObserver:
    """Builds a USD stage once, then re-renders it as the simulation moves.

    Usage per step:
        obs.update(particles, link_poses)     # from the live env
        imgs = obs.render()                   # {"top_rgb": HxWx3, ...}

    The stage is built ONCE and only its point/transform attributes are
    rewritten each step. Rebuilding it per frame would dominate the cost and
    would also re-resolve the garment reference 600 times an episode.
    """

    def __init__(self, cfg: StormObsConfig):
        self.cfg = cfg
        self._built = False
        self._gl = None
        self._stage = None
        self._points = None
        self._link_ops: dict[tuple[str, int], object] = {}
        self._cams: dict[str, object] = {}
        self._rec = None
        self.workdir = cfg.workdir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "storm_obs")
        os.makedirs(self.workdir, exist_ok=True)

    # -- GL ---------------------------------------------------------------
    def _ensure_gl(self):
        """Headless GL 4.5, COMPATIBILITY profile.

        Core throws `GL error: invalid enum` out of HgiGL's state holder --
        it saves fixed-function state a core profile refuses to report.
        """
        if self._gl is not None:
            return
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        from OpenGL import EGL as E

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
        surf = E.eglCreatePbufferSurface(
            dpy, cfgs[0], [E.EGL_WIDTH, self.cfg.width,
                           E.EGL_HEIGHT, self.cfg.height, E.EGL_NONE])
        ctx = E.eglCreateContext(dpy, cfgs[0], E.EGL_NO_CONTEXT, [
            E.EGL_CONTEXT_MAJOR_VERSION, 4, E.EGL_CONTEXT_MINOR_VERSION, 5,
            E.EGL_CONTEXT_OPENGL_PROFILE_MASK,
            E.EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT, E.EGL_NONE])
        if ctx == E.EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext returned EGL_NO_CONTEXT")
        E.eglMakeCurrent(dpy, surf, surf, ctx)
        self._gl = (dpy, surf, ctx)

    # -- stage ------------------------------------------------------------
    def build(self, particles0: np.ndarray, link_poses0: dict):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

        self._ensure_gl()
        path = os.path.join(self.workdir, "obs_stage.usda")
        if os.path.exists(path):
            os.unlink(path)
        stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        self._stage = stage

        def material(p, colour, tex=None, rough=0.7):
            mat = UsdShade.Material.Define(stage, p)
            sh = UsdShade.Shader.Define(stage, p + "/s")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            if tex:
                rd = UsdShade.Shader.Define(stage, p + "/uv")
                rd.CreateIdAttr("UsdPrimvarReader_float2")
                rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                st = UsdShade.Shader.Define(stage, p + "/tex")
                st.CreateIdAttr("UsdUVTexture")
                st.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex)
                st.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                st.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    rd.ConnectableAPI(), "result")
                st.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                sh.CreateInput("diffuseColor",
                               Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    st.ConnectableAPI(), "rgb")
            else:
                sh.CreateInput("diffuseColor",
                               Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat

        # table
        t = UsdGeom.Cube.Define(stage, "/World/table")
        t.CreateSizeAttr(1.0)
        tx = UsdGeom.Xformable(t)
        tx.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.30, TABLE_Z - 0.2))
        tx.AddScaleOp().Set(Gf.Vec3f(1.8, 1.4, 0.4))
        UsdShade.MaterialBindingAPI(t.GetPrim()).Bind(
            material("/World/m/table", (0.70, 0.66, 0.58)))

        # garment: the challenge's own topology, vertices driven by the sim
        gusd = next(os.path.join(self.cfg.garment_dir, f)
                    for f in sorted(os.listdir(self.cfg.garment_dir))
                    if f.endswith(".usd"))
        src = Usd.Stage.Open(gusd)
        # Pick the mesh whose vertex count MATCHES the simulated particles, not
        # simply the first one in the stage.
        #
        # Several released garments carry more than one mesh (a textured shell,
        # a collision proxy). Taking the first gave face indices describing a
        # different vertex set than the points written each frame, so the mesh
        # was silently invalid and rendered nothing at all -- an empty table
        # under a caption claiming a successful fold. It was invisible as a bug
        # because the CHECKER reads particle positions from physics, not from
        # the render, so those episodes still scored and still passed.
        #
        # Affected 11 of 33 recorded GIFs, and split by garment INSTANCE rather
        # than class: Top_Long_Seen_1 rendered while Top_Long_Seen_0 did not.
        n_particles = int(np.asarray(particles0).shape[0])
        meshes = [UsdGeom.Mesh(p) for p in src.Traverse() if p.IsA(UsdGeom.Mesh)]
        msrc = max(meshes, key=lambda m: len(m.GetPointsAttr().Get() or ()))
        n_verts = len(msrc.GetPointsAttr().Get() or ())
        # The render mesh and the solver disagree on vertex count, and the
        # render mesh always has MORE: PS_049 carries 11,573 against 11,385
        # particles. That gap is UV seams -- the USD duplicates a vertex
        # wherever the texture atlas cuts, the solver keeps one particle. Writing
        # the particle array straight into `points` therefore produced a mesh
        # whose indices ran past its own point list, which draws nothing at all.
        # Top_Short happened to match exactly, which is why this looked like it
        # affected some garments and not others.
        self._weld = None
        if n_verts != n_particles:
            self._weld = _seam_map(
                np.asarray(msrc.GetPointsAttr().Get(), np.float64), n_particles)
        g = UsdGeom.Mesh.Define(stage, "/World/cloth")
        g.CreateFaceVertexCountsAttr(msrc.GetFaceVertexCountsAttr().Get())
        g.CreateFaceVertexIndicesAttr(msrc.GetFaceVertexIndicesAttr().Get())
        p0 = np.asarray(particles0, np.float32)
        pts0 = p0[self._weld] if self._weld is not None else p0
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts0))
        # A scrambled weld still fills the point list, so the mesh renders --
        # as a spray of stretched triangles rather than a garment. The longest
        # edge separates the two. A settled garment spans well under a metre,
        # so an edge approaching that means the correspondence is wrong.
        if self._weld is not None:
            longest = _edge_sanity(pts0, msrc.GetFaceVertexCountsAttr().Get(),
                                   msrc.GetFaceVertexIndicesAttr().Get())
            if longest > 0.25:
                raise RuntimeError(
                    f"weld produced a {longest:.2f} m edge; the garment is under "
                    f"a metre across, so this correspondence is wrong. Refusing "
                    f"to render a garbled garment.")
        g.CreateDoubleSidedAttr(True)
        uv = UsdGeom.PrimvarsAPI(msrc.GetPrim()).GetPrimvar("st")
        if uv:
            vals = uv.Get()
            interp = (UsdGeom.Tokens.faceVarying
                      if len(vals) == len(msrc.GetFaceVertexIndicesAttr().Get())
                      else UsdGeom.Tokens.vertex)
            UsdGeom.PrimvarsAPI(g).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, interp).Set(vals)
        UsdShade.MaterialBindingAPI(g.GetPrim()).Bind(
            material("/World/m/cloth", (0.82, 0.26, 0.26),
                     tex=self.cfg.extra.get("texture"), rough=0.9))
        self._points = g.GetPointsAttr()

        # robot links -- reference /visuals/<link> EXPLICITLY: the robot USD's
        # meshes are a sibling of its defaultPrim, so referencing the layer
        # alone composes no geometry at all.
        robot = os.path.join(self.cfg.assets, "robots/lerobot/so101_follower.usd")
        rmat = material("/World/m/robot", (0.86, 0.74, 0.16), rough=0.45)
        for side, poses in link_poses0.items():
            for i in range(len(poses[0])):
                if i >= len(self.cfg.link_names):
                    break
                link = self.cfg.link_names[i]
                p = f"/World/R_{side}/{link}"
                xf = UsdGeom.Xform.Define(stage, p)
                UsdGeom.Xform.Define(stage, p + "/geo").GetPrim() \
                    .GetReferences().AddReference(robot, f"/visuals/{link}")
                self._link_ops[(side, i)] = UsdGeom.Xformable(xf).AddTransformOp()
        for prim in stage.Traverse():
            if prim.GetPath().pathString.startswith("/World/R_") and prim.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI(prim).Bind(
                    rmat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

        UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(6000.0)
        k = UsdLux.DistantLight.Define(stage, "/World/key")
        k.CreateIntensityAttr(9000.0)
        UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 35.0))

        # cameras at the challenge's own placements
        rb = ROBOT_BASES["right"]
        top_eye = (rb[0] + TOP_CAM_OFFSET[0], rb[1] + TOP_CAM_OFFSET[1],
                   rb[2] + TOP_CAM_OFFSET[2])
        self._cams["top_rgb"] = self._camera(stage, "top", 28.7, 38.11,
                                             top_eye, (0.0, -0.34, TABLE_Z))
        # WRIST cameras ride the gripper. The challenge mounts them at
        # /World/Robot/{Left,Right}_Robot/gripper/{side}_wrist_camera, so they
        # move with the hand and look at whatever it is reaching for.
        #
        # They were previously pinned to fixed world positions aimed at
        # (0, -0.34, TABLE_Z) -- a point ~0.37 m in y from where the garment
        # actually rests -- so both wrist views rendered empty table for the
        # whole episode. That is not merely a cosmetic problem with the GIFs:
        # the policy was being handed two dead frames out of three, and a
        # policy that servos its gripper from wrist views would do exactly
        # what was observed, move to a plausible pose and then hover.
        #
        # Parenting under the gripper Xform means the per-frame link transform
        # carries the camera along; only the constant local offset is set here.
        for side in ("left", "right"):
            self._cams[f"{side}_rgb"] = self._wrist_camera(stage, side)
        stage.GetRootLayer().Save()

        from pxr import UsdAppUtils

        self._rec = UsdAppUtils.FrameRecorder()
        self._rec.SetRendererPlugin("HdStormRendererPlugin")
        self._rec.SetImageWidth(self.cfg.width)
        self._rec.SetCameraLightEnabled(True)
        self._rec.SetComplexity(1.0)
        self._built = True

    def _wrist_camera(self, stage, side):
        """A camera parented to the gripper link, matching the challenge rig."""
        from pxr import Gf, UsdGeom

        parent = f"/World/R_{side}/{WRIST_CAM_PARENT}"
        cam = UsdGeom.Camera.Define(stage, f"{parent}/wrist_cam")
        cam.CreateFocalLengthAttr(36.5)
        cam.CreateHorizontalApertureAttr(36.83)
        cam.CreateVerticalApertureAttr(36.83 * (CAM_H / CAM_W))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50.0))

        # The challenge declares these offsets with convention="ros": forward
        # +Z, up -Y. A USD camera looks down its local -Z with +Y up, so the
        # quaternion cannot be applied raw -- doing that aims the camera
        # backwards out of the scene and every wrist frame renders as flat
        # background. Converting takes one extra 180 degree turn about X,
        # which maps +Z -> -Z and +Y -> -Y.
        w, x, y, z = WRIST_CAM_ROT
        rot = (Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
               * Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0))
        m = Gf.Matrix4d(rot, Gf.Vec3d(*WRIST_CAM_OFFSET))
        UsdGeom.Xformable(cam).AddTransformOp().Set(m)
        return cam

    def _camera(self, stage, name, focal, aperture, eye, target):
        from pxr import Gf, UsdGeom

        cam = UsdGeom.Camera.Define(stage, f"/World/cam_{name}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(aperture)
        # Vertical aperture must be set explicitly or USD keeps its default
        # (15.2908). FrameRecorder takes only a WIDTH and derives height from
        # the camera's aperture ratio, so leaving this alone rendered
        # 640x256 (38.11/15.2908 = 2.49; 640/2.49 = 256) where the policy was
        # trained on 640x480. Deriving it from CAM_H/CAM_W keeps the intended
        # horizontal FOV and fixes the frame shape at the source, rather than
        # resampling a 2.49:1 image up to 4:3 and feeding the policy a
        # vertically stretched scene.
        cam.CreateVerticalApertureAttr(aperture * (CAM_H / CAM_W))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
        v = Gf.Matrix4d(1.0)
        v.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(cam).AddTransformOp().Set(v.GetInverse())
        return cam

    # -- per step ---------------------------------------------------------
    def update(self, particles: np.ndarray, link_poses: dict):
        from pxr import Gf, Vt

        if not self._built:
            self.build(particles, link_poses)
            return
        pts = np.asarray(particles, np.float32)
        if self._weld is not None:
            pts = pts[self._weld]          # duplicate seam vertices back out
        self._points.Set(Vt.Vec3fArray.FromNumpy(pts))
        for (side, i), op in self._link_ops.items():
            pos, quat = link_poses[side]
            q = quat[i]
            op.Set(Gf.Matrix4d(
                Gf.Rotation(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]),
                                                           float(q[3])))),
                Gf.Vec3d(*[float(v) for v in pos[i]])))

    def render(self) -> dict[str, np.ndarray]:
        """Render every camera. Returns HxWx3 uint8 per key."""
        from PIL import Image
        from pxr import Usd

        out = {}
        for key, cam in self._cams.items():
            png = os.path.join(self.workdir, f"{key}.png")
            if not self._rec.Record(self._stage, cam, Usd.TimeCode.Default(), png):
                raise RuntimeError(f"Storm failed to render {key}")
            im = Image.open(png)
            if im.mode == "RGBA":
                # FrameRecorder writes RGBA with a transparent background;
                # .convert("RGB") would turn that into pure black and the
                # policy would see a frame that is mostly nothing.
                bg = Image.new("RGBA", im.size, (150, 150, 150, 255))
                im = Image.alpha_composite(bg, im)
            out[key] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return out

    def close(self):
        self._rec = None
