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
CAM_W, CAM_H = 640, 480


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
        msrc = next(UsdGeom.Mesh(p) for p in src.Traverse() if p.IsA(UsdGeom.Mesh))
        g = UsdGeom.Mesh.Define(stage, "/World/cloth")
        g.CreateFaceVertexCountsAttr(msrc.GetFaceVertexCountsAttr().Get())
        g.CreateFaceVertexIndicesAttr(msrc.GetFaceVertexIndicesAttr().Get())
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(particles0, np.float32)))
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
        for side in ("left", "right"):
            self._cams[f"{side}_rgb"] = self._camera(
                stage, side, 36.5, 36.83,
                (ROBOT_BASES[side][0], ROBOT_BASES[side][1] - 0.25, TABLE_Z + 0.35),
                (0.0, -0.34, TABLE_Z))
        stage.GetRootLayer().Save()

        from pxr import UsdAppUtils

        self._rec = UsdAppUtils.FrameRecorder()
        self._rec.SetRendererPlugin("HdStormRendererPlugin")
        self._rec.SetImageWidth(self.cfg.width)
        self._rec.SetCameraLightEnabled(True)
        self._rec.SetComplexity(1.0)
        self._built = True

    def _camera(self, stage, name, focal, aperture, eye, target):
        from pxr import Gf, UsdGeom

        cam = UsdGeom.Camera.Define(stage, f"/World/cam_{name}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(aperture)
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
        self._points.Set(Vt.Vec3fArray.FromNumpy(np.asarray(particles, np.float32)))
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
