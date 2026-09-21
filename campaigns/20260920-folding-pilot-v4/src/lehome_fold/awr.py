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
            normalise_first: bool = True) -> np.ndarray:
    """exp(A / beta), clipped at w_max.

    The clip is not cosmetic. Unclipped exponential weights let a single
    high-advantage sample dominate a batch, and with a small beta that happens
    routinely; the run then trains on effectively one trajectory and the loss
    curve looks smooth while doing it.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if w_max <= 0:
        raise ValueError(f"w_max must be positive, got {w_max}")
    a = normalise(advantages) if normalise_first else np.asarray(advantages, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    # Subtract the max before exponentiating: exp(large/beta) overflows to inf
    # and the clip below would then be applied to inf/inf.
    w = np.exp(np.clip((a - a.max()) / beta, -50.0, 0.0))
    return np.clip(w, 0.0, w_max).astype(np.float32)


def effective_sample_size(w) -> float:
    """(sum w)^2 / sum w^2 -- how many samples the weighted batch is worth.

    Report it every log step. A batch of 256 with an ESS of 3 is the failure
    described above, and ESS is the cheapest way to see it happening.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.any(w > 0):
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))
