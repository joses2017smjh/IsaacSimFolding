"""Can the LeHome garment task run on Isaac Sim 6.0 / Isaac Lab 3.0.0b2?

The 5.1 stack the challenge pins cannot render on this cluster, so the fallback
in docs/STAGE0.md is porting to 6.0 as a stated deviation. This measures how
big that port actually is, rather than assuming "2.x -> 3.x, expect breakage".

The first evidence was encouraging: all 15 isaaclab symbols the task imports
resolve on 3.0.0b2, and the particle cloth is built from raw PhysX/USD calls
rather than isaaclab, so that half is version-independent by construction.

This goes further and tries to actually stand the environment up, reporting the
first real failure instead of a guess. Four steps, each reported separately so
a failure localises:

    1. import the lehome package
    2. import the task module (registers the gym env)
    3. build the env config, pointed at a real Release garment
    4. instantiate the env -- opens Isaac Sim, builds the scene, spawns cloth

Nothing here is a claim that the port is DONE. Step 4 succeeding would mean the
scene stands up on 6.0; matching the official scorer still needs the evaluator
and a like-for-like comparison, and any number produced this way is a stated
deviation until that is shown.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--lehome", required=True)
ap.add_argument("--garment", default="Top_Short_Seen_0")
ap.add_argument("--headless", type=int, default=1)
args = ap.parse_args()


def step(n, what):
    print(f"\n=== {n}. {what} ===", flush=True)


def ok(msg):
    print(f"  OK   {msg}", flush=True)


def fail(msg, e=None):
    print(f"  FAIL {msg}", flush=True)
    if e is not None:
        print("       " + traceback.format_exc().replace("\n", "\n       "), flush=True)


# Isaac Sim has to be launched before isaaclab is imported, or the extension
# system is not up and every import below fails for the wrong reason.
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=bool(args.headless), enable_cameras=True)
simulation_app = app_launcher.app
print("APP LAUNCHED (RTX did not segfault -- this is 6.0, not 5.1)", flush=True)

# --- enable the deprecated core extensions --------------------------------
# 6.0 moved the core API to isaacsim.core.experimental.* and the previous run
# died on `No module named 'isaacsim.core.utils'`. But the old API is still
# SHIPPED, under extsDeprecated/, as Kit extensions that are simply not enabled
# by default -- including SingleClothPrim and SingleParticleSystem, which ARE
# the garment's particle cloth. So this is an enablement problem, not a
# rewrite: turn them on and LeHome's imports resolve unchanged.
step(0, "enable the deprecated isaacsim.core extensions")
try:
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    import isaacsim
    dep = os.path.join(os.path.dirname(isaacsim.__file__), "extsDeprecated")
    mgr.add_path(dep)
    for ext in ("isaacsim.core.utils", "isaacsim.core.prims", "isaacsim.core.api"):
        try:
            mgr.set_extension_enabled_immediate(ext, True)
            ok(f"enabled {ext}")
        except Exception as e:  # noqa: BLE001
            fail(f"enable {ext}: {e}")
    import importlib
    for m in ("isaacsim.core.utils.prims", "isaacsim.core.prims", "isaacsim.core.api"):
        try:
            importlib.import_module(m)
            ok(f"import {m}")
        except Exception as e:  # noqa: BLE001
            fail(f"import {m}: {type(e).__name__}: {e}")
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem  # noqa: F401
    ok("SingleClothPrim + SingleParticleSystem available (the particle cloth)")

    # --- shim the gap in NVIDIA's own deprecation layer -------------------
    # Every deprecated prim class does
    #     self._backend_utils = SimulationManager._get_backend_utils()
    # and 6.0's SimulationManager no longer has that method, so instantiating
    # ANY deprecated prim fails with
    #     type object 'PhysxManager' has no attribute '_get_backend_utils'
    # It is used only to pick the numpy/torch/warp helper module, and all three
    # still ship under isaacsim.core.utils. So restore the accessor rather than
    # rewriting the garment loader against core.experimental.
    #
    # This patches a vendor internal, which is a real deviation and is recorded
    # as one -- but it is five lines against rewriting the cloth pipeline.
    from isaacsim.core.simulation_manager import SimulationManager

    if not hasattr(SimulationManager, "_get_backend_utils"):
        import isaacsim.core.utils.numpy as _np_utils

        def _backend_utils_for(_cls=None):
            backend = None
            try:
                backend = SimulationManager.get_backend()
            except Exception:  # noqa: BLE001
                pass
            if backend == "torch":
                import isaacsim.core.utils.torch as _t
                return _t
            if backend == "warp":
                import isaacsim.core.utils.warp as _w
                return _w
            return _np_utils

        SimulationManager._get_backend_utils = staticmethod(_backend_utils_for)
        ok("shimmed SimulationManager._get_backend_utils (6.0 deprecation gap)")
    else:
        ok("SimulationManager._get_backend_utils present, no shim needed")
except Exception as e:  # noqa: BLE001
    fail("enable deprecated core extensions", e)

os.chdir(args.lehome)
print(f"cwd = {os.getcwd()}", flush=True)

# --- stub pynput ----------------------------------------------------------
# lehome/devices/__init__.py imports the SO-101 *leader* at package import
# time, which imports pynput.keyboard, which opens an X connection:
#   ImportError: this platform is not supported:
#   'failed to acquire X connection: Bad display name ""'
# On a headless compute node there is no X, and nothing in this probe
# teleoperates -- the import is incidental to reaching the task module. A stub
# is honest here in a way that faking a result never would be: it removes a
# keyboard listener from a batch job, and it changes nothing about the
# simulation. Note this bites on 5.1 too; it is not a 6.0 problem.
import types  # noqa: E402

if "pynput" not in sys.modules:
    _pynput = types.ModuleType("pynput")
    _kb = types.ModuleType("pynput.keyboard")

    class _Listener:  # noqa: D401
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self, *a, **k):
            pass

    class _Key:
        def __getattr__(self, name):
            return f"<key:{name}>"

    _kb.Listener = _Listener
    _kb.Key = _Key()
    _kb.KeyCode = type("KeyCode", (), {"from_char": staticmethod(lambda c: c)})
    _pynput.keyboard = _kb
    sys.modules["pynput"] = _pynput
    sys.modules["pynput.keyboard"] = _kb
    print("stubbed pynput (headless: no X server, and nothing here teleoperates)",
          flush=True)

rc = 0

step(1, "import the lehome package")
try:
    import lehome  # noqa: F401
    ok(f"lehome from {lehome.__file__}")
except Exception as e:  # noqa: BLE001
    fail("import lehome", e)
    rc = 1

step(2, "import the garment task module")
try:
    import lehome.tasks.bedroom as bedroom  # noqa: F401
    ok(f"lehome.tasks.bedroom  ({bedroom.__file__})")
except Exception as e:  # noqa: BLE001
    fail("import lehome.tasks.bedroom", e)
    rc = 2

step(3, "build the environment config")
cfg = None
try:
    from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg

    cfg = GarmentEnvCfg()
    cfg.garment_name = args.garment
    cfg.garment_cfg_base_path = os.path.join(args.lehome, "Assets/objects/Challenge_Garment")
    cfg.particle_cfg_path = os.path.join(
        args.lehome, "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml")
    cfg.sim.device = "cpu"
    ok(f"GarmentEnvCfg  action_space={cfg.action_space} dt={cfg.sim.dt} "
       f"decimation={cfg.decimation} episode_s={cfg.episode_length_s}")
    ok(f"cameras 3x {cfg.top_camera.width}x{cfg.top_camera.height}")
except Exception as e:  # noqa: BLE001
    fail("build GarmentEnvCfg", e)
    rc = 3

step(4, "instantiate the environment (opens the scene, spawns particle cloth)")
if cfg is not None:
    try:
        from lehome.tasks.bedroom.garment_bi_v2 import GarmentEnv

        env = GarmentEnv(cfg=cfg)
        ok(f"env up: obs_space={getattr(env, 'single_observation_space', '?')} "
           f"action_space={getattr(env, 'single_action_space', '?')}")
        obs = env._get_observations()
        for k, v in obs.items():
            shape = getattr(v, "shape", None)
            ok(f"obs[{k}] shape={shape}")
        import torch

        env.step(torch.zeros((1, 12), dtype=torch.float32))
        ok("env.step() survived")
        s = env._get_success()
        ok(f"official success checker returned {s}")
        env.close()
    except Exception as e:  # noqa: BLE001
        fail("instantiate GarmentEnv", e)
        rc = 4

print(f"\nPORT PROBE EXIT {rc}", flush=True)
simulation_app.close()
sys.exit(rc)
