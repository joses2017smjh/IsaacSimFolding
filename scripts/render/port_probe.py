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

os.chdir(args.lehome)
print(f"cwd = {os.getcwd()}", flush=True)

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
