"""Which garments may be trained on, and which may never be.

The leaderboard measured generalisation to garments the policy had not seen.
Training on the public unseen garments does not crash anything -- it quietly
converts the headline number into a lie. The work order lists it as a failure
mode with no detector, so this module IS the detector: one place that knows the
split, and an assertion that is called from every training entry point.

Ground truth, counted from the released assets and the released demonstrations
on 2026-08-27:

    Assets  (lehome/asset_challenge, Release/)  10 Seen + 2 Unseen per category
    Demos   (lehome/dataset_challenge_merged)   40 garments, ALL Seen,
                                                25 episodes each = 1000

The released training set therefore contains zero Unseen garments, and Stage 1
cannot leak even by accident. The exposure is Stage 3, where rollouts are
COLLECTED -- collecting on an Unseen garment and training on the result is the
leak, and it looks exactly like a well-performing run until the eval numbers
are questioned.

The 8 private garments per category (`Holdout`) never shipped. Every number
this repo reports is on 48 garments, not the leaderboard's 80.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORIES: tuple[str, ...] = ("Top_Long", "Top_Short", "Pant_Long", "Pant_Short")

# `--garment_type` on LeHome's evaluator uses lowercase; the asset tree and the
# demonstration metadata use CamelCase. Keeping both in one place stops the
# translation being re-derived (differently) at each call site.
GARMENT_TYPE_ARG: dict[str, str] = {
    "Top_Long": "top_long",
    "Top_Short": "top_short",
    "Pant_Long": "pant_long",
    "Pant_Short": "pant_short",
}
ARG_TO_CATEGORY: dict[str, str] = {v: k for k, v in GARMENT_TYPE_ARG.items()}

N_SEEN_PER_CATEGORY = 10
N_UNSEEN_PER_CATEGORY = 2

_NAME_RE = re.compile(r"^(Top_Long|Top_Short|Pant_Long|Pant_Short)_(Seen|Unseen)_(\d+)$")


class SplitViolation(RuntimeError):
    """Raised when a held-out garment reaches a training path."""


@dataclass(frozen=True)
class Garment:
    category: str
    split: str  # "Seen" | "Unseen"
    index: int

    @property
    def name(self) -> str:
        return f"{self.category}_{self.split}_{self.index}"

    @property
    def is_trainable(self) -> bool:
        return self.split == "Seen"


def parse(name: str) -> Garment:
    """Parse a garment name, rejecting anything that is not one of the 48."""
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a LeHome garment name: {name!r}")
    category, split, index = m.group(1), m.group(2), int(m.group(3))
    limit = N_SEEN_PER_CATEGORY if split == "Seen" else N_UNSEEN_PER_CATEGORY
    if not 0 <= index < limit:
        raise ValueError(f"{name!r}: index {index} outside 0..{limit - 1} for {split}")
    return Garment(category, split, index)


def seen(category: str | None = None) -> list[str]:
    """The garments it is legitimate to train on."""
    cats = CATEGORIES if category is None else (category,)
    return [
        f"{c}_Seen_{i}" for c in cats for i in range(N_SEEN_PER_CATEGORY)
    ]


def unseen(category: str | None = None) -> list[str]:
    """The validation split. Never train on these."""
    cats = CATEGORIES if category is None else (category,)
    return [
        f"{c}_Unseen_{i}" for c in cats for i in range(N_UNSEEN_PER_CATEGORY)
    ]


def all_garments() -> list[str]:
    return seen() + unseen()


def assert_trainable(names, *, context: str = "training") -> None:
    """Fail loudly if any held-out garment appears in a training set.

    Called from every entry point that writes a checkpoint. The cost of the
    check is nothing; the cost of not having it is a generalisation number that
    nobody can defend and that looks fine on the way past.
    """
    bad = []
    for n in names:
        g = parse(n)  # also rejects typos, which would otherwise silently
                      # filter to an empty set and train on nothing
        if not g.is_trainable:
            bad.append(g.name)
    if bad:
        raise SplitViolation(
            f"{context}: {len(bad)} held-out garment(s) in a training set: "
            f"{sorted(set(bad))}. These are the validation split -- the whole "
            f"point of the leaderboard was generalisation to them."
        )


def summarise(names) -> dict[tuple[str, str], int]:
    """Count garments by (category, split). Used in run logs and reports."""
    out: dict[tuple[str, str], int] = {}
    for n in names:
        g = parse(n)
        out[(g.category, g.split)] = out.get((g.category, g.split), 0) + 1
    return out
