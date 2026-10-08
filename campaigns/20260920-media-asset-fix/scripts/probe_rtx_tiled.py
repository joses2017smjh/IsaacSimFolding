"""Does the RTX renderer come up on Isaac Sim 5.1 on this cluster?

This is the blocking question for the whole project, and it is the exact
inverse of what the work order assumed.

    lehome-challenge pins isaacsim[all,extscache]==5.1.0 on Python 3.11.
    The policy consumes three RGB cameras and there is no ray-cast shortcut,
    because a Warp mesh query has no colour to return.
    bhl-robustness-ladder recorded that 5.1's RTX renderer segfaults on this
    cluster inside `omni.usd.create_hydra_engine`, and that 6.0 survives it.

So RGB is mandatory and the pinned version is the broken one. Either this probe
renders, or the project needs a documented deviation (port LeHome to the 6.0
stack, or run NVIDIA's own 5.1 container) before Stage 1 can start.

The probe deliberately mirrors LeHome's own rendering configuration rather than
using defaults, because the thing under test is that configuration:

    SimulationCfg(dt=1/90, render_interval=1, use_fabric=False,
                  render=RenderCfg(rendering_mode="quality",
                                   antialiasing_mode="FXAA"))
    TiledCameraCfg(width=640, height=480, data_types=["rgb"])  x3

Three cameras, not one: the challenge uses one overhead and one per wrist, and
a single camera would not exercise the tiled path at the width that matters.

The scene is textured and explicitly lit. That is not decoration -- the first
RTX probe in the sibling repo reported "no image" on a frame whose depth was
correct to the centimetre, because a bare cube with a default-oriented light
renders correct geometry and black shading. A probe that cannot tell a broken
renderer from an unlit scene is worse than no probe.

Verdict is written to $PROBE_OUT. A STARTED line is written as early as
possible so that a segfault (file contains only STARTED) is distinguishable
from a Python exception (file contains TRACEBACK) and from success.
"""

from __future__ import annotations

import os
import sys
import traceback

PROBE_OUT = os.environ.get("PROBE_OUT", "results/rtx51_probe.txt")
os.makedirs(os.path.dirname(PROBE_OUT) or ".", exist_ok=True)


def emit(line: str) -> None:
    """Append one line to the verdict file and to stdout, flushing both.

    Flushing matters: if the next call into Kit segfaults, whatever is still
    sitting in a buffer is lost, and the missing line is the evidence.
    """
    print(line, flush=True)
    with open(PROBE_OUT, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


open(PROBE_OUT, "w").close()
emit("STARTED | probe_rtx_tiled.py")

from isaaclab.app import AppLauncher  # noqa: E402

# enable_cameras=True is what pulls in the render product / Hydra path at all.
# Without it the sensors construct and silently return nothing, which would
# look like a pass.
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app
emit("APP_LAUNCHED | headless=True enable_cameras=True")

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> int:
    import importlib.metadata as md

    for pkg in ("isaacsim", "isaaclab", "torch"):
        try:
            emit(f"VERSION | {pkg}={md.version(pkg)}")
        except md.PackageNotFoundError:
            emit(f"VERSION | {pkg}=(absent)")

    # --- LeHome's exact render configuration -------------------------------
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="FXAA")
    sim_cfg = SimulationCfg(
        dt=1 / 90,
        render_interval=1,
        render=render_cfg,
        use_fabric=False,
        device="cpu",  # LeHome is CPU-physics only; the GPU is for rendering.
    )
    sim = SimulationContext(sim_cfg)
    emit("SIM_CONTEXT | created (dt=1/90, use_fabric=False, device=cpu)")

    # --- A scene that is actually lit and actually textured ----------------
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    dome = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    dome.func("/World/domeLight", dome)
    key = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    key.func("/World/keyLight", key, orientation=(0.87, 0.0, 0.5, 0.0))

    # Three differently coloured blocks, so a correct render has real colour
    # variance and a black frame cannot be mistaken for a pass.
    for i, (colour, x) in enumerate(
        [((0.9, 0.15, 0.15), -0.35), ((0.15, 0.8, 0.25), 0.0), ((0.2, 0.35, 0.95), 0.35)]
    ):
        cube = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour, roughness=0.4),
        )
        cube.func(f"/World/cube{i}", cube, translation=(x, 0.0, 0.1))
    emit("SCENE | ground + dome + distant light + 3 coloured cuboids")

    # --- Three tiled cameras, LeHome's resolution --------------------------
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
    )
    poses = {
        "top":   ((0.0, 0.0, 1.10), (0.0, 0.7071, 0.7071, 0.0)),
        "left":  ((-0.55, -0.45, 0.65), (0.7071, 0.0, 0.0, 0.7071)),
        "right": ((0.55, -0.45, 0.65), (0.7071, 0.0, 0.0, 0.7071)),
    }
    cams = {}
    for name, (pos, rot) in poses.items():
        cams[name] = TiledCamera(
            TiledCameraCfg(
                prim_path=f"/World/cam_{name}",
                offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
                data_types=["rgb"],
                spawn=spawn,
                width=640,
                height=480,
            )
        )
    emit(f"CAMERAS | {len(cams)} x TiledCamera 640x480 rgb constructed")

    # sim.reset() is where the render product is created and where 5.1 has been
    # observed to die. Everything above this line is bookkeeping.
    sim.reset()
    emit("SIM_RESET | survived -- hydra engine came up")

    for _ in range(12):
        sim.step()
        for cam in cams.values():
            cam.update(sim_cfg.dt)

    ok = True
    for name, cam in cams.items():
        rgb = cam.data.output["rgb"]
        arr = rgb[0].detach().to(torch.float32).cpu()
        # Drop alpha if the renderer handed back RGBA.
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        mean = float(arr.mean())
        std = float(arr.std())
        uniq = int(torch.unique(arr).numel())
        # A live RTX frame of this scene has real spread. An all-black or
        # single-valued frame is the failure that looks like a success.
        good = std > 1.0 and uniq > 64
        ok = ok and good
        emit(
            f"RGB | {name:5s} shape={tuple(rgb.shape)} mean={mean:.1f} "
            f"std={std:.1f} unique={uniq} -> {'OK' if good else 'DEGENERATE'}"
        )

    emit(
        f"VERDICT | rgb={'OK' if ok else 'DEGENERATE'} -> "
        f"RTX {'RENDERS' if ok else 'DOES NOT RENDER'} ON 5.1"
    )
    return 0 if ok else 2


try:
    rc = main()
except Exception:
    emit("TRACEBACK | " + traceback.format_exc().replace("\n", "\n  "))
    emit("VERDICT | rgb=EXCEPTION -> RTX DOES NOT RENDER ON 5.1")
    rc = 3
finally:
    simulation_app.close()

sys.exit(rc)
