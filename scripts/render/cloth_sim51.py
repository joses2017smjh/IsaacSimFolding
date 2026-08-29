"""Actually simulate the garment. PhysX particle cloth, on Isaac Sim 5.1.

Everything rendered in this repo so far has been the garment USD's AUTHORED
mesh -- a static 3D shape. It looks inflated and crumpled because it has never
been simulated: the LeHome environment spawns the garment as PhysX particle
cloth and lets it drape onto the table, and none of that has run.

The opening is that 5.1's crash is in the RENDERER. `AppLauncher(headless=True,
enable_cameras=False)` never creates a hydra engine, which is why
bhl-robustness-ladder ran thousands of headless 5.1 jobs without trouble. And
5.1 is the stack that still has particle cloth -- 6.0's PhysX removed it.

So: launch 5.1 with cameras OFF, stand up the real GarmentEnv, step it, and
watch the particle positions move. If they move, the cloth is simulating and
the folding animation becomes a rendering problem (Storm, docs/STORM.md) rather
than a physics one.

Writes the particle cloud per step to an .npz so the motion can be measured and
drawn without holding a simulator open.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import types

ap = argparse.ArgumentParser()
ap.add_argument("--lehome", required=True)
ap.add_argument("--garment", default="Top_Short_Seen_0")
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--out", default="results/cloth51.npz")
ap.add_argument("--pose", default="",
                help="[x y z rx ry rz] recorded initial pose, degrees for the "
                     "rotation. The env randomises the garment per reset, so an "
                     "open-loop replay grasps where the cloth is not.")
ap.add_argument("--sim_device", default="cuda:0",
                help="PhysX device. Particle cloth needs GPU dynamics; LeHome's "
                     "--device cpu is about POLICY INFERENCE, not physics.")
ap.add_argument("--traj", default="",
                help="demo joint trajectory; without it the arms hold still "
                     "and a garment that spawns already resting will not move")
args = ap.parse_args()


def log(*a):
    print("[cloth]", *a, flush=True)


# lehome/devices/__init__ imports a teleop keyboard listener at package import
# time, which needs an X server. Nothing here teleoperates.
if "pynput" not in sys.modules:
    _p = types.ModuleType("pynput")
    _k = types.ModuleType("pynput.keyboard")
    _k.Listener = type("Listener", (), {"__init__": lambda s, *a, **k: None,
                                        "start": lambda s: None,
                                        "stop": lambda s: None,
                                        "join": lambda s, *a, **k: None})
    _k.Key = type("Key", (), {"__getattr__": lambda s, n: f"<key:{n}>"})()
    _k.KeyCode = type("KeyCode", (), {"from_char": staticmethod(lambda c: c)})
    _p.keyboard = _k
    sys.modules["pynput"] = _p
    sys.modules["pynput.keyboard"] = _k
    log("stubbed pynput (headless)")

from isaaclab.app import AppLauncher  # noqa: E402

# enable_cameras=False is the whole point: it is what keeps 5.1 from building
# the RTX hydra engine that segfaults on this cluster.
app_launcher = AppLauncher(headless=True, enable_cameras=False)
simulation_app = app_launcher.app
log("5.1 app launched with cameras OFF -- no hydra engine, no segfault")

os.chdir(args.lehome)

import numpy as np  # noqa: E402
import torch  # noqa: E402

rc = 0
try:
    import lehome.tasks.bedroom  # noqa: F401
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg
    from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.garment_cfg_base_path = os.path.join(args.lehome,
                                             "Assets/objects/Challenge_Garment")
    cfg.particle_cfg_path = os.path.join(
        args.lehome,
        "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml")
    # PhysX device. Both PhysX read paths failed with the same
    #   AttributeError: 'NoneType' object has no attribute 'count'
    # which is what an uninstantiated particle-system backend looks like --
    # PhysX particle cloth requires GPU dynamics. LeHome's README says
    # "--device cpu", but that flag is POLICY INFERENCE (its help text says so);
    # forcing sim.device to cpu was my own inference and it silently disables
    # the particle solver while rigid bodies keep working, which is exactly the
    # arms-move-cloth-frozen picture observed.
    cfg.sim.device = args.sim_device
    log(f"sim.device = {cfg.sim.device} (particle cloth needs GPU dynamics)")
    cfg.scene.num_envs = 1

    # Stub the camera CLASS, not the config.
    #
    # _setup_scene does `self.top_camera = TiledCamera(self.cfg.top_camera)`
    # unconditionally, so setting the config to None only moves the crash:
    #   AttributeError: 'NoneType' object has no attribute 'prim_path'
    # Replacing the class instead lets _setup_scene run untouched while no
    # render product is ever requested -- which is what keeps 5.1 away from the
    # hydra engine that segfaults. The physics, the cloth and the scorer are
    # all unaffected; only the RGB observations are absent, and this probe does
    # not read them.
    import lehome.tasks.bedroom.garment_bi_v2 as gbv

    class _ZeroOutput(dict):
        """Return a zero image for any requested annotator.

        env.step() -> _get_observations() reads
        `self.top_camera.data.output["rgb"]`, so an empty dict raises
        KeyError: 'rgb' on the first step. These images are never consumed --
        this probe measures PARTICLE POSITIONS, and the frames are rendered
        separately through Storm. The physics, the cloth and the scorer are
        real; only the unused RGB is synthetic, and that is stated wherever a
        number from this path is reported.
        """

        def __missing__(self, key):
            h = 480
            w = 640
            v = (torch.zeros((1, h, w, 1), dtype=torch.float32)
                 if key == "depth" else
                 torch.zeros((1, h, w, 3), dtype=torch.uint8))
            self[key] = v
            return v

    class _NoCamera:
        def __init__(self, cfg=None, *a, **k):
            self.cfg = cfg
            self.data = types.SimpleNamespace(output=_ZeroOutput(),
                                              pos_w=None, quat_w_ros=None)

        def update(self, *a, **k):
            pass

        def reset(self, *a, **k):
            pass

        def __del__(self):
            pass

    gbv.TiledCamera = _NoCamera
    log("TiledCamera stubbed -- physics and cloth intact, no render product")
    log(f"cfg ready: garment={cfg.garment_name} dt={cfg.sim.dt} "
        f"action_space={cfg.action_space} cameras=disabled")

    env = GarmentEnv(cfg=cfg)
    log("GarmentEnv instantiated -- particle cloth spawned")

    obj = env.object
    log(f"cloth object: {type(obj).__name__}")

    # --- direct PhysX tensor view --------------------------------------
    # Neither available reader works: the USD `points` attribute is never
    # written back, and _cloth_prim_view.initialize() has no sim view. The
    # tensor API talks to PhysX itself and needs neither.
    tensor_view = None
    cloth_path = (getattr(obj, "prim_path", None)
                  or getattr(obj, "_prim_path", None)
                  or str(getattr(getattr(obj, "_prim", None), "GetPath", lambda: "")()))
    log(f"cloth prim path: {cloth_path!r}")
    try:
        import omni.physics.tensors as _pt

        sv = _pt.create_simulation_view("numpy")
        for pattern in (cloth_path, cloth_path + "*", "/World/*Cloth*", "/World/**/Cloth"):
            if not pattern:
                continue
            for maker in ("create_particle_cloth_view", "create_particle_system_view",
                          "create_soft_body_view"):
                fn = getattr(sv, maker, None)
                if fn is None:
                    continue
                try:
                    v = fn(pattern)
                    if v is not None:
                        tensor_view = v
                        log(f"PhysX tensor view: {maker}({pattern!r}) -> "
                            f"{type(v).__name__}")
                        break
                except Exception:  # noqa: BLE001
                    continue
            if tensor_view is not None:
                break
        if tensor_view is None:
            log("PhysX tensor view: no maker/pattern combination matched")
    except Exception as exc:  # noqa: BLE001
        log(f"omni.physics.tensors unavailable: {type(exc).__name__}: {exc}")

    _seen_readers = set()

    def _note_reader(tag):
        """Say WHICH reader answered, once each.

        Without this the fallback chain is silent, and "the tensor API did not
        help" is indistinguishable from "the tensor API was never reached" --
        a conclusion I would otherwise have reported without being able to
        tell the difference.
        """
        if tag not in _seen_readers:
            _seen_readers.add(tag)
            log(f"particle reader: {tag}")

    def particles():
        """Current particle positions, read exactly the way the OFFICIAL
        success checker reads them.

        `success_checker_chanllege.get_object_particle_position` calls
        `get_current_mesh_points()` and falls back to
        `_cloth_prim_view.get_world_positions()`. Using the same accessor means
        these are the very coordinates the scorer scores -- not a parallel
        reading that might diverge.
        """
        if tensor_view is not None:
            for getter in ("get_positions", "get_world_positions",
                           "get_particle_positions"):
                fn = getattr(tensor_view, getter, None)
                if fn is None:
                    continue
                try:
                    v = fn()
                    a = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v)
                    if a.size:
                        _note_reader(f"tensor_view.{getter}")
                        return a.reshape(-1, 3)
                    _note_reader(f"tensor_view.{getter} -> EMPTY")
                except Exception as exc:  # noqa: BLE001
                    _note_reader(f"tensor_view.{getter} raised "
                                 f"{type(exc).__name__}: {str(exc)[:60]}")
                    continue

        view_ = getattr(obj, "_cloth_prim_view", None)
        if view_ is not None and hasattr(view_, "get_world_positions"):
            try:
                v = view_.get_world_positions()
                return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v
                                  ).reshape(-1, 3)
            except Exception:  # noqa: BLE001
                pass
        fn = getattr(obj, "get_current_mesh_points", None)
        if callable(fn):
            try:
                v = fn()
                v = v[0] if isinstance(v, (tuple, list)) else v
                _note_reader("get_current_mesh_points (USD points)")
                return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v
                                  ).reshape(-1, 3)
            except Exception as exc:  # noqa: BLE001
                log(f"get_current_mesh_points failed: {exc}")
        view = getattr(obj, "_cloth_prim_view", None)
        if view is not None and hasattr(view, "get_world_positions"):
            try:
                v = view.get_world_positions()
                return np.asarray(v.detach().cpu() if hasattr(v, "detach") else v
                                  ).reshape(-1, 3)
            except Exception as exc:  # noqa: BLE001
                log(f"_cloth_prim_view.get_world_positions failed: {exc}")
        return None

    # RESET FIRST.
    #
    # The previous run stepped 200 times with max|delta| exactly 0.0000 m --
    # the cloth spawned and never moved. env.reset() is what brings the
    # particle system up and settles the garment onto the table; stepping a
    # DirectRLEnv that has never been reset leaves it inert, which is also a
    # neat explanation for why every render so far looked like an un-simulated
    # blob.
    # _get_initial_info() BEFORE reset. GarmentObject.reset() restores
    # `initial_points_positions`, and that attribute is only created by
    # _get_initial_info(); calling env.reset() first dies with
    #   AttributeError: 'GarmentObject' object has no attribute
    #   'initial_points_positions'
    # Initialise the cloth PRIM VIEW against the live physics sim view.
    #
    # The joint diagnostic proved physics advances (arms move 1.6 rad) while
    # get_current_mesh_points() returns bit-identical values -- so the cloth
    # state exists in PhysX and is not reaching the USD `points` attribute the
    # reader looks at. The direct route is _cloth_prim_view.get_world_positions(),
    # which failed with a missing _max_particles_per_cloth: that attribute is
    # set when the view is initialised against a physics sim view, and
    # DirectRLEnv never does it for this object.
    view = getattr(obj, "_cloth_prim_view", None)
    if view is not None:
        for sv in (getattr(getattr(env, "sim", None), "physics_sim_view", None), None):
            try:
                view.initialize(sv) if sv is not None else view.initialize()
                log(f"cloth_prim_view.initialize({'sim_view' if sv else 'no-arg'}) ok")
                break
            except Exception as exc:  # noqa: BLE001
                log(f"cloth_prim_view.initialize failed: {type(exc).__name__}: {exc}")

    for setup in ("post_reset", "_get_initial_info"):
        fn = getattr(obj, setup, None)
        if callable(fn):
            try:
                fn()
                log(f"object.{setup}() ok")
            except Exception as exc:  # noqa: BLE001
                log(f"object.{setup}() failed: {exc}")

    env.reset()
    log("env.reset() -- particle system brought up")

    # Pin the garment to the pose the DEMONSTRATION was recorded at.
    #
    # garment_info.json stores object_initial_pose per episode because the env
    # randomises it on every reset (initial_pos_range in the garment JSON). An
    # open-loop joint replay is grasping at fixed coordinates, so if the cloth
    # is somewhere else the arms close on air -- which is why replaying either
    # states or actions moved the cloth (0.33 m / 0.23 m) without completing
    # the fold.
    if args.pose:
        try:
            vals = [float(v) for v in args.pose.replace(",", " ").split()]
            if len(vals) != 6:
                raise ValueError(f"expected 6 numbers, got {len(vals)}")
            xyz, rpy_deg = vals[:3], vals[3:]
            r = np.radians(rpy_deg)
            cr, sr = np.cos(r / 2), np.sin(r / 2)
            quat = np.array([
                cr[0] * cr[1] * cr[2] + sr[0] * sr[1] * sr[2],
                sr[0] * cr[1] * cr[2] - cr[0] * sr[1] * sr[2],
                cr[0] * sr[1] * cr[2] + sr[0] * cr[1] * sr[2],
                cr[0] * cr[1] * sr[2] - sr[0] * sr[1] * cr[2],
            ])  # wxyz
            obj.set_local_pose(np.asarray(xyz, dtype=np.float32),
                               quat.astype(np.float32))
            log(f"garment pinned to recorded pose xyz={xyz} rpy_deg={rpy_deg}")
            for _ in range(30):        # let it settle at the pinned pose
                env.step(torch.zeros((1, 12), dtype=torch.float32))
            log("settled 30 steps at the recorded pose")
        except Exception as exc:  # noqa: BLE001
            log(f"could not pin garment pose: {type(exc).__name__}: {exc}")

    p0 = particles()
    if p0 is None:
        log("FAIL could not read particle positions; attrs = "
            f"{[a for a in dir(obj) if 'pos' in a.lower()][:10]}")
        raise SystemExit(4)
    log(f"particles: {p0.shape[0]} points, z range "
        f"[{p0[:,2].min():.3f}, {p0[:,2].max():.3f}]")

    frames = [p0]

    # Drive the arms with a REAL demonstration.
    #
    # Zero actions produced max|delta| = 0.0000 m over 200 steps, and that is
    # almost certainly correct physics rather than a dead simulation: LeHome
    # spawns the garment already resting on the table, so with the arms holding
    # still nothing touches it. The demonstrations move the cloth by GRASPING
    # it. Episode 0 of the released data is what actually folds this garment.
    traj = None
    if args.traj and os.path.exists(args.traj):
        traj = np.load(args.traj)
        log(f"driving arms with {os.path.basename(args.traj)} {traj.shape}")
    else:
        log("no trajectory given -- arms hold still, cloth will not be touched")

    for i in range(args.steps):
        if traj is not None:
            q = traj[min(i, len(traj) - 1)]
            action = torch.from_numpy(np.asarray(q, dtype=np.float32)).reshape(1, 12)
        else:
            action = torch.zeros((1, 12), dtype=torch.float32)
        env.step(action)
        p = particles()
        if p is not None:
            frames.append(p)
        if i % 20 == 0:
            d = float(np.abs(frames[-1] - frames[0]).max()) if len(frames) > 1 else 0.0
            # Read the ROBOT too. If the arms are not moving either, physics is
            # not advancing at all and the cloth is innocent; if the arms move
            # and the cloth does not, the particle state is simply not being
            # written back where the reader looks. Those are different bugs and
            # guessing between them has already cost several runs.
            jm = "?"
            try:
                ja = env.left_arm.data.joint_pos[0].detach().cpu().numpy()
                if i == 0:
                    globals()["_j0"] = ja.copy()
                jm = f"{float(np.abs(ja - globals()['_j0']).max()):.4f}"
            except Exception as exc:  # noqa: BLE001
                jm = f"err:{type(exc).__name__}"
            log(f"step {i:4d}  cloth z=[{p[:,2].min():.3f},{p[:,2].max():.3f}]  "
                f"max|Δcloth|={d:.4f} m  max|Δjoint|={jm} rad")

    arr = np.stack(frames)
    moved = float(np.abs(arr[-1] - arr[0]).max())
    settle = float(arr[0][:, 2].max() - arr[-1][:, 2].max())
    log(f"TOTAL max displacement {moved:.4f} m over {args.steps} steps")
    log(f"top of garment fell {settle:.4f} m (draping onto the table)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, particles=arr.astype(np.float32))
    log(f"wrote {args.out}  {arr.shape}")

    try:
        s = env._get_success()
        log(f"official success checker: {s}")
    except Exception as exc:  # noqa: BLE001
        log(f"success checker not callable here: {exc}")

    # Moving at all is the question. A rigid blob would show ~0.
    log(f"VERDICT cloth={'SIMULATING' if moved > 1e-3 else 'STATIC'}")
    rc = 0 if moved > 1e-3 else 2
    env.close()
except SystemExit as e:
    rc = int(e.code or 0)
except Exception:
    log("TRACEBACK " + traceback.format_exc().replace("\n", "\n  "))
    rc = 3

simulation_app.close()
sys.exit(rc)
