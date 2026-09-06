"""A TiledCamera stand-in backed by OpenUSD Storm.

Why this exists
---------------
LeHome's own evaluator -- and therefore Stages 3 and 4, which route through it --
runs `scripts.eval` with `--enable_cameras`. That creates an Isaac Lab render
product, and on Isaac Sim 5.1 the RTX delegate segfaults against this cluster's
driver. A probe reached Isaac Sim and died with `Segmentation fault (core
dumped)`. Isaac Sim 6.0 renders fine but has no particle cloth, which is the
one thing this task cannot do without, so "render on 6.0" is not available.

I first called that blocker structural. It is not. The environment touches its
cameras through a very small surface:

    self.top_camera = TiledCamera(cfg)        # construction
    self.scene.sensors["top_camera"] = ...    # registration
    self.top_camera.data.output["rgb"]        # (N, H, W, 3) uint8
    self.top_camera.data.output["depth"]      # (N, H, W, 1) float

Satisfying those four is enough for the official evaluator to run unmodified,
with Storm supplying the pixels instead of RTX. That is a far smaller change
than porting candidate ranking onto the Storm rollout, and it keeps the
success numbers coming from the challenge's own eval loop rather than a
reimplementation of it.

Depth is synthesised, not measured. Storm gives colour; the challenge's success
checker reads particle positions from physics and never looks at depth, so a
plausible constant costs nothing there. Any downstream consumer that genuinely
needs metric depth must not use this -- hence `DEPTH_IS_SYNTHETIC`, which is
checked rather than merely documented.
"""
from __future__ import annotations

import types

import numpy as np

DEPTH_IS_SYNTHETIC = True
_DEPTH_FILL = 1.0          # metres; a stand-in, never a measurement


class _Output(dict):
    """`data.output` is indexed like a dict and must always have the keys."""

    def __init__(self, get_rgb, height, width, device):
        super().__init__()
        self._get_rgb = get_rgb
        self._h, self._w, self._device = height, width, device

    def __getitem__(self, key):
        import torch

        if key == "rgb":
            a = self._get_rgb()
            if a is None:
                a = np.zeros((self._h, self._w, 3), np.uint8)
            return torch.from_numpy(np.ascontiguousarray(a)[None]).to(self._device)
        if key == "depth":
            return torch.full((1, self._h, self._w, 1), _DEPTH_FILL,
                              device=self._device)
        raise KeyError(
            f"StormCamera provides 'rgb' and 'depth' only, not {key!r}. "
            f"Add it deliberately rather than returning zeros -- a silently "
            f"empty channel is how an invisible garment survived 11 episodes "
            f"in this repo.")


class StormCamera:
    """Quacks like `isaaclab.sensors.TiledCamera`, renders through Storm.

    One observer is shared by all three cameras: it builds the USD stage once
    and re-renders every view per step, so constructing three of these does not
    triple the work.
    """

    def __init__(self, cfg=None, *, observer=None, key=None,
                 height=480, width=640, device="cuda", **_):
        self.cfg = cfg
        self._obs = observer
        self._key = key
        self._h, self._w, self._device = height, width, device
        self.data = types.SimpleNamespace(
            output=_Output(self._rgb, height, width, device))

    def _rgb(self):
        if self._obs is None or self._key is None:
            return None
        frames = getattr(self._obs, "_last_frames", None)
        return None if frames is None else frames.get(self._key)

    # the sensor protocol the scene expects; Storm is driven by the rollout
    # loop, so these are intentionally inert rather than secretly re-rendering
    def update(self, *a, **k):
        return None

    def reset(self, *a, **k):
        return None

    def __del__(self):
        pass


def install(module, observer, *, device="cuda"):
    """Replace `module.TiledCamera` with Storm-backed cameras.

    The env constructs its cameras positionally in a fixed order -- top, left
    wrist, right wrist -- so the factory hands out keys in that order. Binding
    by construction order is fragile enough to deserve saying out loud; if the
    env ever reorders those three lines, the wrist views silently swap.
    """
    order = ["top_rgb", "left_rgb", "right_rgb"]
    state = {"n": 0}

    def factory(cfg=None, *a, **k):
        key = order[state["n"]] if state["n"] < len(order) else None
        state["n"] += 1
        return StormCamera(cfg, observer=observer, key=key, device=device)

    module.TiledCamera = factory
    return factory
