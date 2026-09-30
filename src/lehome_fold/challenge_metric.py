"""Reconstruct the LeHome challenge's success metric from a per-step success trace.

The failure mode this exists to prevent: reporting a stricter or looser metric
than the one the challenge scores, and calling it the challenge's.

How the challenge scores an episode (verified 2026-09-30 by reading the code
and replaying the project's logs of the real evaluator):

- ``scripts/utils/evaluation.py`` calls ``env._get_success()`` after every
  ``env.step`` until it first returns True. That first True is the episode's
  result: the loop runs 49 more steps, capped at ``max_steps`` (600), and
  never re-checks. The metric is a LATCHED FIRST SAMPLED SUCCESS, not the
  state at the end of the episode.
- ``success_checker_garment_fold`` is wrapped by ``step_interval(50)``. There
  is one call counter per process, shared by every call site, and the checker
  evaluates only on every 50th call (other calls return False).
- The call sites are GarmentEnv._get_rewards and ._get_success:
  - ``stabilize_garment_after_reset`` adds 20 calls per episode (20 env.step);
  - each policy step before the first success makes 3 calls
    (DirectRLEnv.step -> _get_rewards, env._get_success(), the loop's explicit
    env._get_rewards());
  - each step of the 49-step tail makes 2 calls.
  Because 3 and 50 share no factor, exactly one evaluated call in three lands
  on _get_success. Success is therefore sampled once every 50 policy steps,
  at a phase the counter sets. The counter carries across episodes and
  garments.
- A fresh process samples steps 27, 77, ..., 577. Each fully failed episode
  shifts the phase by +10 steps.
- The replay: in the project's logged runs of the real evaluator
  (``campaigns/20260920-folding-pilot-v3/audit/baseline_episodes.json``), every
  first success falls on the step this model predicts.

What this does not model: the challenge loop's own trajectories. That loop
uses a 20-step home-pose stabilization, a random garment pose per reset, an
early stop, RTX cameras and the challenge's IsaacLab fork. These functions
apply the challenge's SAMPLING to success traces recorded by another runner,
such as the campaign runner's per-step ``geometry_trajectory``.

Pure stdlib; unit-tested in tests/test_pure.py.
"""
from __future__ import annotations

from typing import Iterable

PERIOD = 50              # step_interval(interval=50)
STABILIZE_CALLS = 20     # stabilize_garment_after_reset: 20 env.step -> _get_rewards
CALLS_BEFORE_SUCCESS = 3 # step()->_get_rewards, _get_success(), explicit _get_rewards()
CALLS_AFTER_SUCCESS = 2  # step()->_get_rewards, explicit _get_rewards()
TAIL_STEPS = 49          # extra_steps = 50, decremented on the triggering step
MAX_STEPS = 600          # evaluation.py --max_steps default
FRESH_PROCESS_PHASE = 27


def sampled_phase(counter: int = 0) -> int:
    """First policy step (1..50) whose _get_success call is evaluated, for an
    episode that begins with the process counter at ``counter``."""
    c = counter + STABILIZE_CALLS
    # step k's _get_success call is call number c + 3k - 1
    return next(k for k in range(1, PERIOD + 1) if (c + CALLS_BEFORE_SUCCESS * k - 1) % PERIOD == 0)


def first_sampled_success(success_steps: Iterable[int], phase: int, max_steps: int = MAX_STEPS) -> int | None:
    """The first sampled policy step (1-based) at which the trace is successful."""
    hits = set(success_steps)
    return next((k for k in range(phase, max_steps + 1, PERIOD) if k in hits), None)


def phase_average(success_steps: Iterable[int], max_steps: int = MAX_STEPS) -> float:
    """Probability that the challenge's sampler records a success, averaged over
    the 50 equally likely phases."""
    hits = set(success_steps)
    if not hits:
        return 0.0
    return sum(any(k in hits for k in range(p, max_steps + 1, PERIOD))
               for p in range(1, PERIOD + 1)) / PERIOD


def episode_calls(first_success: int | None, max_steps: int = MAX_STEPS) -> int:
    """Checker calls one episode adds to the process counter, stabilization included."""
    if first_success is None:
        return STABILIZE_CALLS + CALLS_BEFORE_SUCCESS * max_steps
    tail = min(TAIL_STEPS, max_steps - first_success)
    return STABILIZE_CALLS + CALLS_BEFORE_SUCCESS * first_success + CALLS_AFTER_SUCCESS * tail


def evaluate_sequence(traces: Iterable[Iterable[int]], max_steps: int = MAX_STEPS,
                      counter: int = 0) -> list[tuple[int, int | None]]:
    """Score episodes in the order one evaluation process would run them.

    ``traces``: one iterable of successful policy steps (1-based) per episode.
    Returns ``(phase, first sampled success or None)`` per episode. The shared
    counter is carried across episodes as the official loop does.
    """
    out = []
    for trace in traces:
        phase = sampled_phase(counter)
        first = first_sampled_success(trace, phase, max_steps)
        out.append((phase, first))
        counter += episode_calls(first, max_steps)
    return out


def trace_from_rollout(rollout: dict) -> list[int]:
    """Successful policy steps from a campaign runner rollout.json record."""
    return [g["step"] for g in rollout.get("geometry_trajectory", [])
            if g.get("phase") == "policy" and g.get("success")]
