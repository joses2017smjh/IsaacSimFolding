"""Gate G2: is the success head calibrated?

An uncalibrated success head does not fail loudly. It produces advantages in
Stage 3 that are noise, and the run degrades slowly instead of crashing --
which is the failure shape that costs weeks. So calibration is a gate with a
number attached, not a plot someone glances at.

Definitions used here, so the reported number is unambiguous:

    ECE   expected calibration error, equal-width bins over [0, 1], weighted
          by bin population. The standard 15-bin variant.
    MCE   maximum calibration error -- the worst single populated bin. ECE can
          look fine while one bin is badly wrong, and the bad bin is usually
          the confident one that matters.
    Brier mean squared error against the binary outcome. Proper scoring rule;
          moves when either calibration or discrimination moves, so it is the
          honest headline when only one number is wanted.

A model that predicts the base rate everywhere is perfectly calibrated and
useless, so calibration alone is not a sufficient gate. `evaluate` therefore
also reports the positive rate and the prediction spread, and `gate` fails a
head whose predictions are effectively constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Reliability:
    n_bins: int
    edges: np.ndarray
    counts: np.ndarray
    mean_pred: np.ndarray
    mean_obs: np.ndarray
    ece: float
    mce: float
    brier: float
    base_rate: float
    pred_std: float
    notes: list[str] = field(default_factory=list)

    def table(self) -> str:
        rows = [f"{'bin':>12s} {'n':>7s} {'pred':>8s} {'obs':>8s} {'gap':>8s}"]
        for i in range(self.n_bins):
            if self.counts[i] == 0:
                continue
            lo, hi = self.edges[i], self.edges[i + 1]
            gap = self.mean_obs[i] - self.mean_pred[i]
            rows.append(
                f"  [{lo:.2f},{hi:.2f}) {int(self.counts[i]):>7d} "
                f"{self.mean_pred[i]:>8.3f} {self.mean_obs[i]:>8.3f} {gap:>+8.3f}"
            )
        rows.append(
            f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier:.4f}  "
            f"base_rate={self.base_rate:.3f}  pred_std={self.pred_std:.4f}"
        )
        return "\n".join(rows)


def evaluate(probs, labels, *, n_bins: int = 15) -> Reliability:
    """Bin predicted probability and check observed frequency tracks it."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and labels {y.shape} disagree")
    if p.size == 0:
        raise ValueError("no predictions to calibrate")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite probabilities")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"probabilities outside [0,1]: [{p.min()}, {p.max()}]")
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0.0, 1.0}:
        raise ValueError(f"labels must be binary, saw {sorted(uniq)[:5]}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed on the final bin so p == 1.0 lands in the last bin rather
    # than falling off the end
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    mean_pred = np.zeros(n_bins)
    mean_obs = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            mean_pred[b] = p[m].mean()
            mean_obs[b] = y[m].mean()

    w = counts / counts.sum()
    gaps = np.abs(mean_obs - mean_pred)
    ece = float((w * gaps).sum())
    mce = float(gaps[counts > 0].max()) if (counts > 0).any() else float("nan")
    brier = float(np.mean((p - y) ** 2))

    notes: list[str] = []
    base_rate = float(y.mean())
    pred_std = float(p.std())
    if base_rate in (0.0, 1.0):
        notes.append(
            "labels are single-class: calibration is undefined and this head "
            "cannot pass G2 on this data (see labels.py -- released demos have "
            "no failures)"
        )
    if pred_std < 1e-3:
        notes.append(
            f"predictions are effectively constant (std={pred_std:.2e}): a "
            "base-rate predictor is calibrated and useless"
        )
    if int((counts > 0).sum()) < 3:
        notes.append(f"only {int((counts > 0).sum())} populated bin(s); ECE is not meaningful")

    return Reliability(
        n_bins=n_bins, edges=edges, counts=counts, mean_pred=mean_pred,
        mean_obs=mean_obs, ece=ece, mce=mce, brier=brier,
        base_rate=base_rate, pred_std=pred_std, notes=notes,
    )


def gate(rel: Reliability, *, max_ece: float = 0.10, min_pred_std: float = 1e-2,
         min_populated_bins: int = 3) -> tuple[bool, list[str]]:
    """G2 pass/fail. Thresholds are arguments so the report can state them."""
    reasons: list[str] = []
    if rel.base_rate in (0.0, 1.0):
        reasons.append("single-class labels -- no negatives to calibrate against")
    if rel.pred_std < min_pred_std:
        reasons.append(f"pred_std {rel.pred_std:.4f} < {min_pred_std} (near-constant head)")
    if int((rel.counts > 0).sum()) < min_populated_bins:
        reasons.append(f"{int((rel.counts > 0).sum())} populated bins < {min_populated_bins}")
    if rel.ece > max_ece:
        reasons.append(f"ECE {rel.ece:.4f} > {max_ece}")
    return (not reasons), reasons
