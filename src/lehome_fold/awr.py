"""Advantage-weighted regression, and the advantage estimate it consumes.

AWR reweights a supervised behaviour-cloning loss by exp(A / beta), so actions
that did better than the policy's own baseline are fit harder. It needs no
log-likelihood from the policy, which is the property that makes it applicable
to a flow-matching model at all.

The paper runs AWR alongside RECAP advantage conditioning. RECAP's own source
reports conditioning outperforming AWR on the same data, so which of the two
carries the gain is a question to measure rather than assume -- hence both are
implemented separately and either can be switched off.

The advantage here is the paper's success residual:

    A(s, a) = y - sg(P_success(s))

where y is the realised outcome and sg is a stop-gradient on the policy's own
predicted success probability. The baseline is the policy's own value head,
which is what "the policy is its own value function" buys.
"""

from __future__ import annotations

import numpy as np


def success_residual(outcomes, baseline) -> np.ndarray:
    """A = y - sg(P_success). Both arguments are already detached arrays."""
    y = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    b = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if y.shape != b.shape:
        raise ValueError(f"outcomes {y.shape} and baseline {b.shape} disagree")
    if not np.all(np.isfinite(b)):
        raise ValueError("non-finite baseline -- an untrained value head will do this")
    return (y - b).astype(np.float32)


def normalise(advantages, *, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean unit-std advantages.

    Standardising before exponentiating is what keeps beta meaning the same
    thing across batches. Without it, beta has to be retuned whenever the
    success rate moves -- which it does continuously during Stage 3, so the
    bug would present as "AWR stopped helping" rather than as an error.
    """
    a = np.asarray(advantages, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return a.astype(np.float32)
    std = a.std()
    if std < eps:
        # Every sample equally good: uniform weights, not a divide-by-zero.
        return np.zeros_like(a, dtype=np.float32)
    return ((a - a.mean()) / (std + eps)).astype(np.float32)


def weights(advantages, *, beta: float = 1.0, w_max: float = 20.0,
            w_min: float = 1e-6, normalise_first: bool = True) -> np.ndarray:
    """min(exp(A / beta), w_max), floored at w_min.

    The clip is not cosmetic. Unclipped exponential weights let a single
    high-advantage sample dominate a batch, and with a small beta that happens
    routinely; the run then trains on effectively one trajectory and the loss
    curve looks smooth while doing it.

    It is applied in LOG SPACE so that it actually binds. A previous version
    subtracted the batch maximum before exponentiating:

        w = exp(clip((A - A.max()) / beta, -50, 0))

    which bounds every weight to (0, 1], so `np.clip(w, 0, w_max)` could never
    fire for any w_max >= 1. The cap was dead code, and runs recorded `w_max`
    in their provenance as though it had been applied. Clipping the exponent
    instead removes the overflow that motivated the max-subtraction -- exp() is
    never called on anything above log(w_max) -- while leaving the cap real.

    `w_min` floors the result so every weight stays strictly positive and the
    array can be normalised into a sampling distribution. Without it, a very
    negative advantage underflows to exactly 0 and that row can never be drawn.

    `w_max` is only meaningful relative to `beta` AND to `normalise_first`.
    With standardisation on, the exponent is a z-score divided by beta, so the
    cap binds only where z / beta > log(w_max). A batch of 8 has a z-range of
    roughly +/-2, so the default w_max=20 (log 3.0) never binds at beta=1 on a
    small batch. Choose w_max against the batch size actually being used and
    record the pair -- `weight_summary` reports whether it bound.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if w_max <= 0:
        raise ValueError(f"w_max must be positive, got {w_max}")
    if not 0 < w_min <= w_max:
        raise ValueError(f"w_min must lie in (0, w_max], got {w_min}")
    a = normalise(advantages) if normalise_first else np.asarray(advantages, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return a.astype(np.float32)
    log_w = np.clip(a / beta, np.log(w_min), np.log(w_max))
    # Re-clip in linear space so the reported maximum is exactly w_max rather
    # than w_max * (1 + eps) from the exp/log round trip.
    return np.clip(np.exp(log_w), w_min, w_max).astype(np.float32)


def weight_summary(w, *, w_max: float, w_min: float = 1e-6,
                   rtol: float = 1e-6) -> dict[str, float]:
    """How concentrated a weight vector is, and whether the cap/floor bound.

    Reported before training rather than inferred afterwards. `capped` > 0 is
    the evidence that `w_max` was a live parameter for this batch; `capped` of
    exactly 0 means the recorded w_max had no effect on the run.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return {"n": 0.0, "ess": 0.0, "ess_fraction": float("nan"),
                "capped": 0.0, "floored": 0.0, "max": float("nan"),
                "min": float("nan"), "ratio": float("nan")}
    ess = effective_sample_size(w)
    return {
        "n": float(w.size),
        "ess": ess,
        "ess_fraction": ess / float(w.size),
        "capped": float(np.sum(w >= w_max * (1.0 - rtol))),
        "floored": float(np.sum(w <= w_min * (1.0 + rtol))),
        "max": float(w.max()),
        "min": float(w.min()),
        "ratio": float(w.max() / w.min()) if w.min() > 0 else float("inf"),
    }


def effective_sample_size(w) -> float:
    """(sum w)^2 / sum w^2 -- how many samples the weighted batch is worth.

    Report it every log step. A batch of 256 with an ESS of 3 is the failure
    described above, and ESS is the cheapest way to see it happening.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.any(w > 0):
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))
