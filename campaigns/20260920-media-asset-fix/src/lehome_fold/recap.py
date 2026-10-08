"""RECAP: RL with Experience and Corrections via Advantage-conditioned Policies.

The idea in one line: train supervised on ALL data including failures, but feed
the model an extra input saying how good the action was, then ask for "good" at
inference.

Why it exists. A flow-matching VLA gives no tractable log-likelihood, so policy
gradients do not apply. Advantage conditioning sidesteps that entirely -- it is
plain supervised learning on every transition, with the advantage moved from
the loss into the INPUT. Failures stop being data to discard and become data
labelled "do not do this", which is why the method can use the whole rollout
buffer rather than a filtered top slice.

The conditioning signal is deliberately coarse -- a binarised advantage
rendered as a text token, not a scalar. Two reasons to keep it that way:
the base model already reads language, so the token needs no new parameters or
embedding surgery; and a binary signal is robust to a value head that is only
roughly calibrated, which in Stage 3 it will be.

At inference, condition on positive. `INFERENCE_TOKEN` exists so that no call
site has to remember which string that was.
"""

from __future__ import annotations

import numpy as np

POSITIVE = "Advantage: positive"
NEGATIVE = "Advantage: negative"
INFERENCE_TOKEN = POSITIVE

# The released demonstrations all carry this single task string, and the
# garment category is deliberately NOT in it -- the challenge withholds
# category labels at evaluation, so the policy has to read the garment from
# vision. Anything that appends a category here is cheating the benchmark.
BASE_TASK = "fold the garment on the table"


def binarise(advantages, *, threshold: float = 0.0) -> np.ndarray:
    """Advantage -> {+1, -1}. Ties count as negative.

    Ties go negative on purpose. `A = y - P_success` is exactly zero when the
    value head predicted the outcome perfectly, which carries no evidence that
    the action was good; calling that "positive" would inject the value head's
    own bias into the conditioning signal it is supposed to be independent of.
    """
    a = np.asarray(advantages, dtype=np.float64).reshape(-1)
    return np.where(a > threshold, 1, -1).astype(np.int8)


def tokens(signs) -> list[str]:
    """{+1, -1} -> the text token the model is conditioned on."""
    s = np.asarray(signs).reshape(-1)
    bad = set(np.unique(s).tolist()) - {1, -1}
    if bad:
        raise ValueError(f"signs must be +1/-1, saw {sorted(bad)}")
    return [POSITIVE if int(v) == 1 else NEGATIVE for v in s]


def condition(task: str, token: str) -> str:
    """Compose the conditioned prompt.

    One newline, token last. The format matters only in that it must be
    IDENTICAL between training and inference -- a trailing-space difference
    here is a silent distribution shift, and it would present as "RECAP made
    the policy slightly worse" rather than as a bug.
    """
    if token not in (POSITIVE, NEGATIVE):
        raise ValueError(f"unknown advantage token {token!r}")
    return f"{task.strip()}\n{token}"


def positive_prompt(task: str = BASE_TASK) -> str:
    """What to ask for at inference."""
    return condition(task, INFERENCE_TOKEN)


def balance(signs) -> dict[str, float]:
    """Fraction positive. Report it every epoch.

    A buffer that is 99% negative trains a model that has barely seen the
    conditioning token it will be asked for at inference. That is the RECAP
    failure mode, it does not raise, and it looks like slow progress.
    """
    s = np.asarray(signs).reshape(-1)
    if s.size == 0:
        return {"n": 0.0, "positive": float("nan"), "negative": float("nan")}
    pos = float((s == 1).mean())
    return {"n": float(s.size), "positive": pos, "negative": 1.0 - pos}
