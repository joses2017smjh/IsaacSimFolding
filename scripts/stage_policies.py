"""Evaluation-time policies for Stages 2 and 3.

These subclass LeHome's own `LeRobotPolicy` rather than reimplementing it. That
is the whole design: their adapter already does observation filtering,
normalisation, device placement and action un-normalisation, and every one of
those is a place where a reimplementation would silently diverge from the
harness the leaderboard used.

    candidate   Stage 2. Sample N action chunks, score each with the policy's
                own value head, execute the best. No retraining -- this runs on
                the Stage 1 checkpoint plus a value-head checkpoint, which is
                what makes it a clean single-mechanism ablation.

    recap       Stage 3. Identical action weights, but the task prompt carries
                "Advantage: positive". Optionally also does candidate selection,
                so the two mechanisms can be measured apart.

Both log the per-step success probability, which is the live failure detector
the paper gets for free from having a value head.

NOT YET RUN. lerobot does not install on this cluster until the renderer
question is settled. The value-head half is unit-tested (src/lehome_fold), the
LeHome half is their tested code; what is unverified is the seam between them,
and `predict_action_chunk` is the specific attribute to confirm first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lehome.utils.logger import get_logger

from .eval_policy.lerobot_policy import LeRobotPolicy
from .eval_policy.registry import PolicyRegistry

logger = get_logger(__name__)


def _load_heads(value_path: str, device: torch.device):
    """Load the value heads saved by scripts/train_value.py."""
    from lehome_fold.value_head import ValueHeadConfig, ValueHeads

    p = Path(value_path)
    cfg_file = p / "value_head_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{cfg_file} not found. Stage 2 policies need a value-head "
            f"checkpoint from scripts/train_value.py, not just the BC weights."
        )
    cfg = ValueHeadConfig(**json.loads(cfg_file.read_text()))
    heads = ValueHeads(cfg)
    heads.load_state_dict(torch.load(p / "value_head.pt", map_location=device))
    heads.eval().to(device)
    return heads, cfg


class _ValueScoredPolicy(LeRobotPolicy):
    """Shared machinery: value heads over the base policy's own features."""

    def __init__(self, value_path: str | None = None, feature_path: str = "",
                 n_candidates: int = 1, log_scores: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.n_candidates = int(n_candidates)
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        self.feature_path = feature_path or os.environ.get("FEATURE_PATH", "")
        self.heads = None
        if value_path:
            self.heads, _ = _load_heads(value_path, self.device)
            logger.info(f"value heads loaded from {value_path}")
        elif self.n_candidates > 1:
            raise ValueError(
                "candidate selection needs a value head to score with; "
                "pass value_path or set n_candidates=1"
            )
        self._log_scores = Path(log_scores) if log_scores else None
        self._trace: list[dict] = []
        self._install_tap()

    # -- feature extraction -------------------------------------------------
    # A TAP, not a call. `embed_prefix` takes model-specific positional
    # arguments (pi05: images, img_masks, tokens, masks; smolvla: the same plus
    # state) and neither form accepts a batch dict -- verified against lerobot
    # 0.4.3. So the policy calls it during its own forward pass and the tap
    # records what went past.
    def _install_tap(self):
        from lehome_fold.policy_wrap import DEFAULT_FEATURE_PATH, FeatureTap

        path = self.feature_path or DEFAULT_FEATURE_PATH
        self.tap = FeatureTap(self.policy, path).install()
        logger.info(f"feature tap installed at {path}")

    def _pooled(self):
        feats = self.tap.require()
        mask = self.tap.last_mask
        if feats.dim() == 2:
            return feats
        if mask is None:
            return feats.mean(dim=1)
        m = mask.to(feats.dtype)
        if m.dim() == 2:
            m = m.unsqueeze(-1)
        return (feats * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def _score(self) -> float:
        """P(success) for the most recent forward pass. Call AFTER the policy."""
        if self.heads is None:
            return 0.0
        with torch.inference_mode():
            logit = self.heads(self._pooled())["success_logit"]
        return float(torch.sigmoid(logit).mean())

    # -- chunk sampling -----------------------------------------------------
    def _sample_chunk(self, batch):
        """One action chunk from the flow-matching head.

        `predict_action_chunk` is LeRobot's documented entry point for chunked
        policies. If it is absent on this version, that is a hard error rather
        than a silent fallback to single-step action selection -- a fallback
        would quietly turn candidate selection into a no-op that still reports
        a number.
        """
        fn = getattr(self.policy, "predict_action_chunk", None)
        if fn is None:
            raise AttributeError(
                "policy has no predict_action_chunk; candidate selection "
                "cannot sample chunks on this lerobot version"
            )
        with torch.inference_mode():
            return fn(batch)

    def reset(self):
        super().reset()
        self._flush()

    def _flush(self):
        if self._log_scores and self._trace:
            with open(self._log_scores, "a") as fh:
                for row in self._trace:
                    fh.write(json.dumps(row) + "\n")
        self._trace = []


@PolicyRegistry.register("candidate")
class CandidateSelectionPolicy(_ValueScoredPolicy):
    """Stage 2: sample N chunks, execute the one the value head likes most."""

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)
        batch = self._process_observation(observation)

        if self.n_candidates == 1 or self.heads is None:
            with torch.inference_mode():
                action = self.policy.select_action(batch)
            if self.postprocessor:
                action = self.postprocessor(action)
            self._trace.append({"p_success": self._score(), "n": 1})
            return action.squeeze(0).cpu().numpy()

        chunks, scores = [], []
        for _ in range(self.n_candidates):
            c = self._sample_chunk(batch)
            chunks.append(c)
            # score AFTER the chunk is sampled: the tap now holds the prefix
            # features from that specific forward pass
            scores.append(self._score())

        best = int(np.argmax(scores))
        spread = float(np.max(scores) - np.min(scores))
        # A zero spread means every candidate scored identically, i.e. selection
        # is choosing arbitrarily and any measured gain is noise. Log it so the
        # ablation cannot be read without seeing it.
        self._trace.append({
            "p_success": float(scores[best]), "n": self.n_candidates,
            "spread": spread, "scores": [round(s, 4) for s in scores],
        })

        action = chunks[best]
        action = action[:, 0] if action.dim() == 3 else action
        if self.postprocessor:
            action = self.postprocessor(action)
        return action.squeeze(0).cpu().numpy()


@PolicyRegistry.register("recap")
class RecapPolicy(CandidateSelectionPolicy):
    """Stage 3: same weights, prompt conditioned on positive advantage."""

    def __init__(self, **kwargs):
        from lehome_fold.recap import BASE_TASK, positive_prompt

        task = kwargs.get("task_description") or BASE_TASK
        # Overwrite before super().__init__ so the base adapter stores the
        # conditioned string and every downstream use sees the same prompt.
        # Training and inference MUST format this identically; recap.condition
        # is the single place that decides.
        kwargs["task_description"] = positive_prompt(task)
        super().__init__(**kwargs)
        logger.info(f"RECAP prompt: {kwargs['task_description']!r}")
