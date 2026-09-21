"""Floor policies for Gate G0.

G0 asks one question: does an episode run end to end and does the OFFICIAL
scorer emit a success/failure? It is a plumbing test, not a learning result.
But the number it produces is also the denominator for G1 -- "BC is above the
random floor" is only meaningful if the floor was measured with the same
harness, the same garments and the same episode budget.

Two floors, because one of them is not enough:

  uniform  A bounded random walk in joint-target space. This is the "random
           policy" G0 names. It is a random WALK rather than white noise
           because the environment takes absolute joint position targets at
           1/90 s per step: independent uniform samples at that rate command a
           full-range slew every 11 ms, which is not a policy, it is an
           actuator stress test, and the arms spend the episode saturated
           against their limits rather than moving over the garment.

  hold     Repeat the first observed state forever. The arms do not move.
           This is the floor that catches a scorer which rewards the initial
           garment configuration -- if `hold` scores above zero, "success" is
           being awarded for a garment that started folded, and every number
           downstream is measuring the reset distribution instead of a policy.

Both are expected to score 0.0. `hold` scoring above zero is a red flag about
the benchmark; `uniform` scoring above zero is a red flag about the scorer.

Action layout, read off the environment rather than assumed:
    actions[:, 0:6]  -> left arm  set_joint_position_target
    actions[:, 6:12] -> right arm set_joint_position_target
    action_scale = 1.0, so these are absolute joint targets in radians.
    (source/lehome/lehome/tasks/bedroom/garment_bi_v2.py::_apply_action)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("g0_uniform")
class G0UniformPolicy(BasePolicy):
    """Bounded random walk around the episode's initial joint configuration."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        step_std: float = 0.02,
        box: float = 0.6,
        seed: Optional[int] = None,
        **kwargs,
    ):
        # `seed` is read from the environment when not passed, so a Slurm array
        # can vary it without the harness needing a new command-line flag.
        if seed is None:
            seed = int(os.environ.get("SEED", "0"))
        self._rng = np.random.default_rng(seed)
        self._step_std = float(step_std)
        self._box = float(box)
        self._home: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None
        print(
            f"[g0_uniform] seed={seed} step_std={self._step_std} box={self._box}",
            flush=True,
        )

    def reset(self) -> None:
        # Forget the home pose, not the RNG: each episode re-anchors to its own
        # reset state, but the seed still determines the whole run.
        self._home = None
        self._target = None

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32).reshape(-1)
        if self._home is None:
            self._home = state.copy()
            self._target = state.copy()
        step = self._rng.normal(0.0, self._step_std, size=self._target.shape)
        self._target = np.clip(
            self._target + step, self._home - self._box, self._home + self._box
        )
        return self._target.astype(np.float32)


@PolicyRegistry.register("g0_hold")
class G0HoldPolicy(BasePolicy):
    """Command the initial joint configuration for the whole episode."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", **kwargs):
        self._home: Optional[np.ndarray] = None
        print("[g0_hold] initialised", flush=True)

    def reset(self) -> None:
        self._home = None

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32).reshape(-1)
        if self._home is None:
            self._home = state.copy()
        return self._home.astype(np.float32)
