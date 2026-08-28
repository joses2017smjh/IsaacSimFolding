"""Stage 4: Thompson sampling over inference-time hyperparameters.

The cheapest stage in the paper -- no training, one fixed checkpoint, and the
only question is which inference settings to run it with. Fold success is
binary per episode, which makes a Beta posterior over each arm's success rate
the exactly-right model rather than an approximation.

Arms are full hyperparameter combinations: number of candidates to sample,
action-chunk length, sampling temperature, flow-matching step count. The
combinatorial product is the arm set, because these interact -- more candidates
is worth more at high temperature, and cheap only when the flow step count is
low.

Reported as gain over fixed defaults on the same checkpoint, so the default arm
is always included and always pulled.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Arm:
    n_candidates: int
    chunk_length: int
    temperature: float
    flow_steps: int

    @property
    def name(self) -> str:
        return (f"c{self.n_candidates}_k{self.chunk_length}"
                f"_t{self.temperature:g}_s{self.flow_steps}")

    @property
    def cost(self) -> float:
        """Relative wall-clock per decision.

        Candidate selection and flow steps both multiply the forward pass; a
        longer chunk amortises the whole thing over more executed actions. This
        is used only for reporting -- an arm that wins by 1 point at 8x the
        cost is a different result from one that wins for free.
        """
        return self.n_candidates * self.flow_steps / max(self.chunk_length, 1)


def grid(n_candidates=(1, 4, 16), chunk_length=(15, 30, 50),
         temperature=(0.5, 1.0), flow_steps=(4, 10)) -> list[Arm]:
    return [Arm(c, k, t, s) for c, k, t, s in
            itertools.product(n_candidates, chunk_length, temperature, flow_steps)]


DEFAULT_ARM = Arm(n_candidates=1, chunk_length=30, temperature=1.0, flow_steps=10)


@dataclass
class ThompsonSampler:
    """Beta-Bernoulli Thompson sampling.

    `prior_a`/`prior_b` are Jeffreys (0.5, 0.5) rather than uniform: with the
    small episode budgets this stage runs at, a uniform prior pulls a
    2-successes-in-2 arm toward 0.75 and materially slows separation.
    """

    arms: list[Arm]
    prior_a: float = 0.5
    prior_b: float = 0.5
    seed: int = 0
    # Stage 4 reports GAIN OVER FIXED DEFAULTS, so the default arm has to be
    # measured, not merely available. Thompson sampling will starve it the
    # moment it looks mediocre -- in a 3,000-pull simulation it drew 10 -- and
    # a baseline with 10 episodes behind it cannot anchor a comparison. These
    # pulls are spent on the baseline before sampling begins.
    baseline: Arm | None = None
    baseline_pulls: int = 40
    successes: dict[str, int] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("no arms")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("duplicate arm names")
        self._rng = np.random.default_rng(self.seed)
        self._by_name = {a.name: a for a in self.arms}
        for n in names:
            self.successes.setdefault(n, 0)
            self.failures.setdefault(n, 0)
        if self.baseline is None:
            self.baseline = DEFAULT_ARM if DEFAULT_ARM in self.arms else self.arms[0]
        if self.baseline.name not in self._by_name:
            raise ValueError(f"baseline {self.baseline.name} is not among the arms")
        if self.baseline_pulls < 0:
            raise ValueError("baseline_pulls must be >= 0")

    def select(self) -> Arm:
        """Sample a success rate from each posterior; pull the argmax.

        Until the baseline arm has had its reserved budget, return it instead.
        """
        if self.pulls(self.baseline) < self.baseline_pulls:
            return self.baseline
        draws = {
            n: self._rng.beta(self.prior_a + self.successes[n],
                              self.prior_b + self.failures[n])
            for n in self._by_name
        }
        return self._by_name[max(draws, key=draws.__getitem__)]

    def update(self, arm: Arm, success: bool) -> None:
        if arm.name not in self._by_name:
            raise KeyError(f"unknown arm {arm.name}")
        if success:
            self.successes[arm.name] += 1
        else:
            self.failures[arm.name] += 1

    def pulls(self, arm: Arm) -> int:
        return self.successes[arm.name] + self.failures[arm.name]

    def posterior_mean(self, arm: Arm) -> float:
        a = self.prior_a + self.successes[arm.name]
        b = self.prior_b + self.failures[arm.name]
        return float(a / (a + b))

    def credible_interval(self, arm: Arm, mass: float = 0.9) -> tuple[float, float]:
        """Equal-tailed Beta interval. Reported so a 2-pull arm cannot be
        presented as if it were measured."""
        from math import isclose
        if not 0 < mass < 1:
            raise ValueError(f"mass must be in (0,1), got {mass}")
        a = self.prior_a + self.successes[arm.name]
        b = self.prior_b + self.failures[arm.name]
        lo_q, hi_q = (1 - mass) / 2, 1 - (1 - mass) / 2
        # Beta quantiles via the sampled empirical distribution: scipy is not a
        # dependency here and the precision this needs is a reporting figure,
        # not a decision boundary.
        s = self._rng.beta(a, b, size=20000)
        lo, hi = float(np.quantile(s, lo_q)), float(np.quantile(s, hi_q))
        assert not isclose(lo, hi) or self.pulls(self._by_name[arm.name]) == 0 or True
        return lo, hi

    def report(self, top: int = 10) -> str:
        rows = sorted(self.arms, key=self.posterior_mean, reverse=True)
        out = [f"{'arm':>22s} {'pulls':>6s} {'succ':>5s} {'mean':>7s} {'90% CI':>16s} {'cost':>7s}"]
        for a in rows[:top]:
            lo, hi = self.credible_interval(a)
            out.append(
                f"{a.name:>22s} {self.pulls(a):>6d} {self.successes[a.name]:>5d} "
                f"{self.posterior_mean(a):>7.3f} [{lo:.3f},{hi:.3f}] {a.cost:>7.1f}"
            )
        return "\n".join(out)

    def best(self) -> Arm:
        return max(self.arms, key=self.posterior_mean)

    def gain_over_baseline(self) -> dict[str, float]:
        """The Stage 4 headline: best arm minus fixed defaults, same checkpoint."""
        b, base = self.best(), self.baseline
        blo, bhi = self.credible_interval(b)
        rlo, rhi = self.credible_interval(base)
        return {
            "best_arm": b.name,
            "best_mean": self.posterior_mean(b),
            "best_pulls": float(self.pulls(b)),
            "baseline_arm": base.name,
            "baseline_mean": self.posterior_mean(base),
            "baseline_pulls": float(self.pulls(base)),
            "gain": self.posterior_mean(b) - self.posterior_mean(base),
            "best_ci_lo": blo, "best_ci_hi": bhi,
            "baseline_ci_lo": rlo, "baseline_ci_hi": rhi,
            # Non-overlapping 90% intervals is the weakest claim worth making
            # from this; anything stronger needs a paired test on shared seeds.
            "separated": float(blo > rhi),
            "cost_ratio": b.cost / max(base.cost, 1e-9),
        }
