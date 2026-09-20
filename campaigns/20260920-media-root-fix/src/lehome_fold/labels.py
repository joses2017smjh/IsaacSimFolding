"""Targets for the value heads, and an honest account of which of them exist.

The paper's policy predicts, from the same network that predicts actions:
success, completion (progress), and task-relevant future quantities, plus a
success residual  y - sg(P_success)  that acts as an action-conditional
advantage over the policy's own baseline.

What the RELEASED demonstrations can actually supply, checked against
`meta/episodes` on 2026-08-27:

    columns present   episode_index, tasks, length, video pointers, per-episode
                      stats over observation.state / action / images
    columns ABSENT    success, reward, garment keypoints, particle positions

So:

    progress    AVAILABLE. Frame index over episode length. Well defined for
                every released demonstration.
    future      AVAILABLE, but only over JOINT STATE. The released data has no
                garment keypoints, so "keypoint distances at t+H" cannot be
                reproduced from demonstrations -- it needs the simulator to
                expose particle positions. `future_state` is the available
                stand-in and is NOT the same quantity.
    success     NOT AVAILABLE as a mixed-class label. Every released episode is
                a successful scripted demonstration; there is no outcome column
                and no failures. A success head fit on this data sees one class.

That last one is a gate, not a nuisance. G2 asks for a CALIBRATED success head,
and calibration is undefined when every target is 1 -- the model that predicts
1.0 everywhere is perfectly accurate and perfectly useless, and it would sail
through a careless check. Negatives have to come from rollouts, which means
the simulator, which means G2 sits behind the renderer.

`class_balance` exists so that degeneracy is reported rather than discovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeOutcome:
    """What is known about one episode after it ended.

    `success_frame` is the step at which the official checker first fired.
    None means the episode never succeeded. For a released demonstration the
    outcome is success with an unknown success frame, which is exactly why
    `success_frame=None` and `success=True` is a representable -- and
    deliberately awkward -- combination: it forces the caller to decide what
    the per-frame target means rather than defaulting silently.
    """

    length: int
    success: bool
    success_frame: int | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.success_frame is not None and not 0 <= self.success_frame < self.length:
            raise ValueError(
                f"success_frame {self.success_frame} outside 0..{self.length - 1}"
            )
        if self.success_frame is not None and not self.success:
            raise ValueError("success_frame set on an episode marked unsuccessful")


def progress(length: int) -> np.ndarray:
    """Fraction of the episode elapsed, in [0, 1], one value per frame.

    Linear in frame index. This is the "completion" head's target and it is the
    one head the released demonstrations can train honestly.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    return (np.arange(length, dtype=np.float32) / (length - 1)).astype(np.float32)


def success_targets(outcome: EpisodeOutcome, *, mode: str = "terminal") -> np.ndarray:
    """Per-frame target for the success head.

    Two modes, because "does this episode succeed" and "has it succeeded yet"
    are different questions and the paper uses the network for both:

      episode   Every frame carries the episode's final outcome. This is the
                value-function reading -- P(this rollout ends in success) --
                and it is what advantage estimation in Stage 3 needs.
                On all-successful demonstration data it is degenerate.

      terminal  0 before the success frame, 1 from it onward. This is the
                "has the fold completed" reading. It yields BOTH classes from a
                purely successful demonstration, which makes it the only head
                that can be fit and calibrated before any rollout exists.
                It is not a substitute for the episode-level head; it answers a
                different question and is reported separately.
    """
    if mode == "episode":
        return np.full(outcome.length, float(outcome.success), dtype=np.float32)
    if mode == "terminal":
        y = np.zeros(outcome.length, dtype=np.float32)
        if outcome.success:
            # An episode known to have succeeded but with no recorded success
            # frame is treated as succeeding only at the last frame. That is
            # the most conservative reading and it keeps both classes present.
            start = outcome.length - 1 if outcome.success_frame is None else outcome.success_frame
            y[start:] = 1.0
        return y
    raise ValueError(f"unknown mode {mode!r}, expected 'episode' or 'terminal'")


def future_state(states: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The joint state `horizon` frames ahead, with a validity mask.

    The paper predicts keypoint distances at t+30. The released data has no
    keypoints, so this predicts the 12-dim joint state instead -- a strictly
    weaker target, named differently on purpose so that a later report cannot
    accidentally claim the paper's quantity.

    The last `horizon` frames have no future inside the episode. They are
    clamped to the final state and masked OUT rather than dropped, so that
    every head keeps the same frame indexing and a batch can carry all targets
    together.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"expected (T, D) states, got shape {states.shape}")
    n = states.shape[0]
    idx = np.minimum(np.arange(n) + horizon, n - 1)
    valid = (np.arange(n) + horizon < n).astype(np.float32)
    return states[idx], valid


def class_balance(y: np.ndarray) -> dict[str, float]:
    """Report positive/negative fractions so a one-class target is visible.

    Called from every value-head training entry point and printed. A silent
    single-class success head is the failure G2 exists to catch, and the
    cheapest place to catch it is before the fit, not after.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return {"n": 0.0, "positive": float("nan"), "negative": float("nan"), "degenerate": 1.0}
    pos = float((y > 0.5).mean())
    return {
        "n": float(y.size),
        "positive": pos,
        "negative": 1.0 - pos,
        "degenerate": float(pos in (0.0, 1.0)),
    }
