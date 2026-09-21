"""Attaching the value heads to a LeRobot VLA.

VERIFIED against lerobot 0.4.3 on 2026-08-28. The layout below was read off the
installed package, not guessed:

    pi05      policy.model = PI05Pytorch
              policy.model.paligemma_with_expert : PaliGemmaWithExpertModel
              policy.model.embed_prefix(images, img_masks, tokens, masks)
    smolvla   policy.model = VLAFlowMatching
              policy.model.vlm_with_expert : SmolVLMWithExpertModel
              policy.model.embed_prefix(images, img_masks, lang_tokens,
                                        lang_masks, state=None)

Both return `(embs, pad_masks, att_masks)` -- the prefix embeddings over the
(images, language[, state]) tokens, which is the representation the action
expert conditions on. That is the feature the paper's heads read.

Why a METHOD TAP rather than calling the attribute. The first version of this
file assumed `embed_prefix(batch)`. It is not: the two models take different,
model-specific positional arguments and neither accepts a batch dict, so a
direct call would have failed at the first real forward pass. Reconstructing
those arguments would mean duplicating each model's preprocessing and would
break on the next lerobot release.

So instead the tap wraps the bound method, lets the policy call it with
whatever arguments it likes during its OWN forward pass, and records the
result on the way past. That is signature-agnostic, version-agnostic, needs no
knowledge of preprocessing, and captures exactly the tensor the expert saw.

`pad_masks` is captured alongside, so pooling ignores padding instead of
averaging it in -- the padded region is a large fraction of a short prompt and
mean-pooling over it would wash out the signal the heads depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .value_head import ValueHeadConfig, ValueHeads

# Verified paths, most likely first. Both entries are the same method name on
# different model classes, which is why one default works for both bases.
CANDIDATE_PATHS: tuple[str, ...] = (
    "model.embed_prefix",              # pi05 AND smolvla (lerobot 0.4.3)
    "model.vlm_with_expert",           # smolvla, module-level fallback
    "model.paligemma_with_expert",     # pi05, module-level fallback
)

DEFAULT_FEATURE_PATH = "model.embed_prefix"


def _resolve(obj: Any, dotted: str) -> Any | None:
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _set_attr(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def probe_feature_source(policy: Any) -> dict[str, Any]:
    """Report which candidate paths exist on this policy object."""
    found: dict[str, Any] = {}
    for path in CANDIDATE_PATHS:
        obj = _resolve(policy, path)
        if obj is None:
            continue
        found[path] = {
            "type": type(obj).__name__,
            "callable": callable(obj),
            "is_module": isinstance(obj, nn.Module),
        }
    try:
        found["_modules"] = [n for n, _ in policy.named_children()]
    except Exception:  # noqa: BLE001
        found["_modules"] = []
    return found


class FeatureTap:
    """Record the output of a method or module the policy calls itself.

    Install once, then run the policy normally; `last` holds whatever the tapped
    call produced. `install` is idempotent and `remove` restores the original,
    so a tap cannot silently stack on repeated construction.
    """

    def __init__(self, policy: nn.Module, path: str):
        self.policy = policy
        self.path = path
        self.last: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None
        self._orig: Any = None
        self._handle: Any = None

    def _record(self, out: Any) -> None:
        # embed_prefix returns (embs, pad_masks, att_masks); a module forward
        # may return a tensor or a tuple whose first element is the states.
        if isinstance(out, (tuple, list)) and out:
            self.last = out[0] if isinstance(out[0], torch.Tensor) else None
            if len(out) > 1 and isinstance(out[1], torch.Tensor):
                self.last_mask = out[1]
        elif isinstance(out, torch.Tensor):
            self.last = out
            self.last_mask = None

    def install(self) -> FeatureTap:
        if self._orig is not None or self._handle is not None:
            return self
        target = _resolve(self.policy, self.path)
        if target is None:
            raise AttributeError(
                f"feature path {self.path!r} does not exist on this policy. "
                f"Run scripts/probe_backbone.py and pass the pinned path; do "
                f"not let a training loop search at runtime."
            )
        if isinstance(target, nn.Module):
            self._handle = target.register_forward_hook(
                lambda _m, _i, out: self._record(out)
            )
            return self
        if not callable(target):
            raise TypeError(f"{self.path!r} is neither a module nor callable")

        self._orig = target

        def wrapper(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            self._record(out)
            return out

        _set_attr(self.policy, self.path, wrapper)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._orig is not None:
            _set_attr(self.policy, self.path, self._orig)
            self._orig = None

    def require(self) -> torch.Tensor:
        if self.last is None:
            raise RuntimeError(
                f"nothing captured at {self.path!r}. The policy has not run a "
                f"forward pass since the tap was installed -- call the policy "
                f"first, then read the features."
            )
        return self.last


@dataclass
class WrapConfig:
    feature_path: str = DEFAULT_FEATURE_PATH
    hidden_dim: int = 0          # 0 -> infer from the first captured feature
    pool: str = "mean"           # "mean" | "cls"
    detach_backbone: bool = True
    future_dim: int = 12


class ValueAugmentedPolicy(nn.Module):
    """A LeRobot policy plus auxiliary heads, sharing one backbone.

    Action selection delegates to the wrapped policy untouched, so a checkpoint
    trained here is still a valid LeRobot policy for LeHome's own
    `--policy_type lerobot` path. The heads ride alongside and are saved
    separately, which is what lets Stage 2 be ablated against Stage 1 on the
    SAME action weights.
    """

    def __init__(self, policy: nn.Module, cfg: WrapConfig):
        super().__init__()
        self.policy = policy
        self.cfg = cfg
        self.tap = FeatureTap(policy, cfg.feature_path).install()
        self.heads: ValueHeads | None = None
        if cfg.hidden_dim:
            self._build_heads(cfg.hidden_dim)

    def _build_heads(self, hidden_dim: int,
                     like: torch.Tensor | None = None) -> None:
        """Build the heads, on the device of the feature that will drive them.

        These are constructed lazily, the first time a feature is captured, so
        the enclosing module's earlier `.to(device)` cannot have reached them:
        they did not exist yet. Creating them on the CPU default and then
        immediately calling them with a CUDA feature raises
          RuntimeError: Expected all tensors to be on the same device
        from inside the first Linear. `like` is that feature.
        """
        self.heads = ValueHeads(
            ValueHeadConfig(hidden_dim=hidden_dim, future_dim=self.cfg.future_dim,
                            detach_backbone=self.cfg.detach_backbone)
        )
        if like is not None:
            # Device only, not dtype: the backbone may run in bf16/fp16, and
            # the heads are what actually gets optimised here, so they stay in
            # their default float32 and the feature is cast at the call site.
            self.heads.to(like.device)
        self.cfg.hidden_dim = hidden_dim

    def _pool(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if x.dim() != 3:
            raise ValueError(f"expected (B,T,D) or (B,D) features, got {tuple(x.shape)}")
        if self.cfg.pool == "cls":
            return x[:, 0]
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        # clamp_min guards the all-padding row: it is reachable with a short
        # prompt and would otherwise produce NaNs that only surface in the loss.
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def features(self) -> torch.Tensor:
        """Pooled prefix features from the policy's most recent forward pass."""
        return self._pool(self.tap.require(), self.tap.last_mask)

    def forward(self, _batch: dict | None = None) -> dict[str, torch.Tensor]:
        f = self.features()
        if self.heads is None:
            self._build_heads(f.shape[-1], like=f)
        # Cast the feature to the heads' dtype rather than the reverse, so a
        # half-precision backbone does not silently drag the trained heads
        # down with it.
        param = next(self.heads.parameters(), None)
        if param is not None and f.dtype != param.dtype:
            f = f.to(param.dtype)
        return self.heads(f)

    @torch.no_grad()
    def score(self) -> dict[str, torch.Tensor]:
        """Success probability and progress, for failure detection and
        candidate selection."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward()
            return {
                "success_prob": torch.sigmoid(out["success_logit"]),
                "progress": out["progress"],
            }
        finally:
            if was_training:
                self.train()

    def select_action(self, *args, **kwargs):
        return self.policy.select_action(*args, **kwargs)

    def predict_action_chunk(self, *args, **kwargs):
        return self.policy.predict_action_chunk(*args, **kwargs)


def candidate_selection(
    sample_chunk: Callable[[], Any],
    score_chunk: Callable[[Any], float],
    *,
    n_candidates: int,
) -> tuple[Any, list[float]]:
    """Sample N action chunks, score each with the policy's own value head,
    return the best and every score.

    The mechanism the paper gets for free from having a value head, and worth
    reporting on its own: same checkpoint, selection on versus off, is a clean
    single-variable ablation with no retraining.

    Scores come back alongside the winner so the SPREAD can be logged. If N
    candidates score identically, selection is choosing arbitrarily and any
    measured gain is noise -- silent otherwise.
    """
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    chunks = [sample_chunk() for _ in range(n_candidates)]
    scores = [float(score_chunk(c)) for c in chunks]
    best = max(range(n_candidates), key=scores.__getitem__)
    return chunks[best], scores
