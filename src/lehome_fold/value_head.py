"""Stage 2: the heads that make the policy its own value function.

The paper's central design choice is that ONE network predicts actions and the
quantities used to judge them. That is what makes advantage estimation possible
without a second critic, and it is what live failure detection and candidate
selection are built on.

This module is the head stack only -- it takes a pooled hidden state from the
VLA backbone and produces the auxiliary predictions. Keeping it separate from
the backbone is deliberate: it can be unit-tested with plain torch, it can be
attached to SmolVLA or pi0.5 without change, and its loss can be checked
against hand-computed values.

Heads, and what each is for downstream:

    success    P(this episode ends in success). Feeds the advantage in Stage 3
               and the failure detector. Needs mixed-outcome data -- see
               labels.py; the released demonstrations cannot train it.
    progress   Fraction of the task complete. Trainable on demonstrations today.
    future     The task-relevant future quantity. The paper predicts keypoint
               distances at t+30; the released data has no keypoints, so this
               regresses the 12-dim joint state at t+H and is NAMED
               differently so a report cannot silently claim otherwise.

All three are small MLPs over a shared trunk. The trunk exists so the heads can
share representation without being able to rewrite the backbone: gradients into
the VLA are controlled by `detach_backbone`, and Stage 2 runs detached by
default so that adding heads cannot degrade the Stage 1 policy that is being
measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ValueHeadConfig:
    hidden_dim: int          # width of the incoming pooled backbone feature
    trunk_dim: int = 512
    future_dim: int = 12     # joint state; 12 for bimanual SO-ARM101
    dropout: float = 0.1
    detach_backbone: bool = True


class ValueHeads(nn.Module):
    def __init__(self, cfg: ValueHeadConfig):
        super().__init__()
        if cfg.hidden_dim <= 0 or cfg.trunk_dim <= 0:
            raise ValueError("hidden_dim and trunk_dim must be positive")
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.trunk_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.trunk_dim, cfg.trunk_dim),
            nn.GELU(),
        )
        self.success = nn.Linear(cfg.trunk_dim, 1)
        self.progress = nn.Linear(cfg.trunk_dim, 1)
        self.future = nn.Linear(cfg.trunk_dim, cfg.future_dim)

        # Start the success head at logit 0 -> p=0.5 rather than at whatever a
        # default init gives. A head that begins confidently wrong produces
        # large early advantages, and in Stage 3 those are fed straight into
        # the conditioning signal before the head knows anything.
        nn.init.zeros_(self.success.bias)
        nn.init.zeros_(self.progress.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.dim() != 2:
            raise ValueError(f"expected (B, hidden_dim) features, got {tuple(features.shape)}")
        if features.shape[1] != self.cfg.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch: config says {self.cfg.hidden_dim}, "
                f"features are {features.shape[1]}"
            )
        x = features.detach() if self.cfg.detach_backbone else features
        h = self.trunk(x)
        return {
            "success_logit": self.success(h).squeeze(-1),
            "progress": torch.sigmoid(self.progress(h)).squeeze(-1),
            "future": self.future(h),
        }

    @torch.no_grad()
    def success_prob(self, features: torch.Tensor) -> torch.Tensor:
        """P(success) with no grad -- this is the `sg(.)` in the advantage."""
        return torch.sigmoid(self(features)["success_logit"])


def value_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    future_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    w_success: float = 1.0,
    w_progress: float = 1.0,
    w_future: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined auxiliary loss, with the future head masked at episode ends.

    `future_mask` is 1 where t+H lands inside the episode and 0 where it was
    clamped. Without it the last H frames of every episode teach the model that
    the future equals the present, which is a real bias toward predicting
    stillness -- and it concentrates exactly at the end of successful folds,
    where the arms genuinely do stop.

    `sample_mask` is 1 for frames whose episode actually has a scored outcome.
    Only a subset of the released episodes has been rolled out and labelled, and
    an unlabelled frame carries a success target of 0 by construction -- so
    without this the success head is trained to call every unscored episode a
    failure, which is not a weaker signal but a wrong one.
    """
    losses: dict[str, torch.Tensor] = {}

    if sample_mask is None:
        losses["success"] = F.binary_cross_entropy_with_logits(
            preds["success_logit"], targets["success"].float()
        )
        losses["progress"] = F.mse_loss(preds["progress"], targets["progress"].float())
    else:
        sm = sample_mask.float()
        n = sm.sum()
        bce = F.binary_cross_entropy_with_logits(
            preds["success_logit"], targets["success"].float(), reduction="none")
        pse = (preds["progress"] - targets["progress"].float()) ** 2
        losses["success"] = (bce * sm).sum() / n if n > 0 else bce.sum() * 0.0
        losses["progress"] = (pse * sm).sum() / n if n > 0 else pse.sum() * 0.0

    fut_err = (preds["future"] - targets["future"].float()) ** 2
    if future_mask is None:
        losses["future"] = fut_err.mean()
    else:
        m = future_mask.float().unsqueeze(-1)
        denom = m.sum() * preds["future"].shape[-1]
        # An all-masked batch is possible with short episodes and a long
        # horizon; return a real zero rather than a nan that poisons the sum.
        losses["future"] = (fut_err * m).sum() / denom if denom > 0 else fut_err.sum() * 0.0

    total = (w_success * losses["success"]
             + w_progress * losses["progress"]
             + w_future * losses["future"])
    return total, {k: float(v.detach()) for k, v in losses.items()}
