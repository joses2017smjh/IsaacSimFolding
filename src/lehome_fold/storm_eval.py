"""Run LeHome's OWN evaluator on Isaac Sim 5.1, with Storm supplying pixels.

The problem
-----------
`scripts/run_eval.py` hands control to LeHome's `scripts.eval`, which builds
`TiledCamera` render products. On 5.1 the RTX delegate segfaults against this
cluster's driver -- a probe reached Isaac Sim and died with `Segmentation fault
(core dumped)`. Isaac Sim 6.0 renders fine but ships no particle cloth, and
particle cloth is the whole task, so "evaluate on 6.0" is not an option.

Because Stages 3 and 4 both route through that evaluator, I first recorded them
as structurally blocked. That was wrong, and worth correcting rather than
quietly fixing: the environment touches its cameras through four things only --
construction, registration as a scene sensor, `data.output["rgb"]` and
`data.output["depth"]`. Storm already produces those pixels for every rollout
in this repo. Satisfying the interface is a much smaller change than porting
candidate ranking onto the Storm loop, and it keeps success numbers coming from
the challenge's own eval loop rather than a reimplementation of it.

What this does
--------------
1. Replaces `TiledCamera` in the env module with Storm-backed cameras.
2. Wraps `GarmentEnv._get_observations` so Storm re-renders from the CURRENT
   particle positions and link poses immediately before the env reads pixels.

The wrap is where the render is driven from, because that is the one place with
both `self` (particles, arms) and a guarantee of running once per observation.
Hooking the camera's `update()` instead would depend on the scene calling it,
which is a promise the sensor protocol does not actually make.

Known limits, stated rather than discovered later:
  * depth is synthetic (see storm_camera.DEPTH_IS_SYNTHETIC)
  * cameras bind to views by construction order
  * Storm rasterises; the policy was trained on path-traced frames, and that
    gap is measured at ~1.0 in action-prediction skill
"""
from __future__ import annotations

import numpy as np


def _particles(env):
    obj = getattr(env, "object", None)
    fn = getattr(obj, "get_current_mesh_points", None)
    if fn is None:
        return None
    v = fn()
    v = v[0] if isinstance(v, (tuple, list)) else v
    v = v.detach().cpu() if hasattr(v, "detach") else v
    return np.asarray(v).reshape(-1, 3)


def _link_poses(env):
    out = {}
    for side, attr in (("left", "left_arm"), ("right", "right_arm")):
        art = getattr(env, attr, None)
        if art is None:
            return None
        d = art.data
        out[side] = (np.asarray(d.body_pos_w[0].detach().cpu()),
                     np.asarray(d.body_quat_w[0].detach().cpu()))
    return out


def enable_when_imported(module_name, observer, *, device="cuda"):
    """Patch the env module the moment it is first imported, not before.

    The env module pulls in `isaaclab_tasks`, which does not exist until
    Isaac Lab's AppLauncher has run. `run_eval.py` executes before that, so
    importing the module eagerly there fails with
        ModuleNotFoundError: No module named 'isaaclab_tasks'
    and importing it later is not possible either, because LeHome's eval owns
    the launch. A post-import hook is the only point that is both after the app
    exists and before the env is constructed.
    """
    import builtins
    import sys

    real_import = builtins.__import__
    state = {"done": False}

    def hooked(name, *a, **k):
        mod = real_import(name, *a, **k)
        if not state["done"]:
            target = sys.modules.get(module_name)
            if target is not None and hasattr(target, "GarmentEnv"):
                state["done"] = True
                builtins.__import__ = real_import      # unhook first
                enable(target, observer, device=device)
        return mod

    builtins.__import__ = hooked
    return state


def enable(env_module, observer, *, device="cuda", verbose=True):
    """Point `env_module`'s cameras at Storm and drive them from the env.

    Returns the wrapped `_get_observations` so a caller can undo it in tests.
    """
    from lehome_fold.storm_camera import install

    install(env_module, observer, device=device)

    env_cls = getattr(env_module, "GarmentEnv", None)
    if env_cls is None:
        raise RuntimeError("env module has no GarmentEnv to wrap")
    original = env_cls._get_observations
    state = {"built": False, "fails": 0}

    def _get_observations(self):
        pts, links = _particles(self), _link_poses(self)
        if pts is not None and links is not None:
            try:
                observer.update(pts, links)      # builds the stage on first call
                observer.render()                # refreshes observer._last_frames
                state["built"] = True
            except Exception as exc:  # noqa: BLE001
                # Report every failure, and loudly on the first. A camera that
                # silently serves black frames is exactly how an invisible
                # garment survived 11 scored episodes in this repo.
                state["fails"] += 1
                if state["fails"] == 1 or state["fails"] % 50 == 0:
                    print(f"[storm_eval] render failed ({state['fails']}x): "
                          f"{type(exc).__name__}: {exc}", flush=True)
        return original(self)

    env_cls._get_observations = _get_observations
    if verbose:
        print("[storm_eval] TiledCamera -> Storm; _get_observations wrapped",
              flush=True)
    return original
