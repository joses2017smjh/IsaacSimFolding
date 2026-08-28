"""Read results back out of the official evaluator's log.

LeHome's `scripts/utils/evaluation.py` prints its results and persists nothing
machine-readable -- no CSV, no JSON, just logger lines. Two ways to deal with
that, and only one of them keeps the numbers comparable:

  fork the eval loop to write structured output   -> a different scorer
  parse the lines it already prints               -> the same scorer

So this parses. It is the less elegant option and it is the correct one: the
entire argument for using the official harness is that nothing about the
measurement changed, and editing the loop that produces the measurement would
give that up to save an afternoon.

The two line formats, quoted from evaluation.py:

    f"Episode {i+1}/{n}: Return={r:.2f}, Length={l}, Success={s}"
    f"  {garment_name}: Success Rate = {sr:.2%}, Avg Return = {ar:.2f}"

Parsing is strict -- an unrecognised line is ignored, but a run that yields
ZERO episodes raises rather than reporting an empty success rate of 0.0. A
silent zero is indistinguishable from a policy that never succeeds, and that
distinction is the entire content of the early gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EPISODE_RE = re.compile(
    r"Episode\s+(\d+)/(\d+):\s*Return=([-\d.eE+]+),\s*Length=(\d+),\s*Success=(True|False)"
)
GARMENT_RE = re.compile(
    r"^\s{2}(\w+):\s*Success Rate\s*=\s*([\d.]+)%,\s*Avg Return\s*=\s*([-\d.eE+]+)\s*$"
)


class NoEpisodes(RuntimeError):
    """The log contained no parseable episodes."""


@dataclass(frozen=True)
class Episode:
    index: int
    total: int
    ret: float
    length: int
    success: bool


@dataclass(frozen=True)
class GarmentSummary:
    garment_name: str
    success_rate: float   # fraction, not percent
    avg_return: float


def parse_episodes(text: str) -> list[Episode]:
    out = []
    for m in EPISODE_RE.finditer(text):
        out.append(Episode(
            index=int(m.group(1)), total=int(m.group(2)), ret=float(m.group(3)),
            length=int(m.group(4)), success=m.group(5) == "True",
        ))
    return out


def parse_garments(text: str) -> list[GarmentSummary]:
    out = []
    for line in text.splitlines():
        m = GARMENT_RE.match(line)
        if m:
            out.append(GarmentSummary(
                garment_name=m.group(1),
                success_rate=float(m.group(2)) / 100.0,
                avg_return=float(m.group(3)),
            ))
    return out


def success_rate(text: str) -> tuple[float, int]:
    """Overall success rate and episode count, from the per-episode lines."""
    eps = parse_episodes(text)
    if not eps:
        raise NoEpisodes(
            "no 'Episode i/N: ... Success=' lines in this log. The run "
            "produced no scored episodes -- that is a crashed evaluation, not "
            "a success rate of zero."
        )
    return sum(e.success for e in eps) / len(eps), len(eps)


class Truncated(RuntimeError):
    """The log's own two accounts of the run disagree."""


def cross_check(text: str) -> dict:
    """Reconcile the per-episode lines against the per-garment summary.

    The evaluator prints the same run twice: once per episode, once aggregated
    per garment. If those disagree, the log was truncated, interleaved by
    another writer, or tailed -- and the aggregate is the half that still looks
    plausible while being wrong.

    This exists because reading a number off a tailed log has burned this
    project before. Parsing is unavoidable here (the official evaluator
    persists nothing machine-readable, and forking it would mean a different
    scorer), so the parser is made to prove the log is whole instead.
    """
    eps = parse_episodes(text)
    gar = parse_garments(text)
    if not eps:
        raise NoEpisodes("no episode lines to cross-check")

    # Every episode line carries the run's total; they must all agree, and the
    # number of lines must equal that total.
    totals = {e.total for e in eps}
    complete = len(totals) == 1 and len(eps) == next(iter(totals)) * max(len(gar), 1)
    per_garment_ok = True
    if gar:
        expected = next(iter(totals)) if len(totals) == 1 else None
        if expected:
            from_eps = sum(e.success for e in eps) / len(eps)
            from_gar = sum(g.success_rate for g in gar) / len(gar)
            # Per-garment rates are printed to 2 decimals, so allow rounding.
            per_garment_ok = abs(from_eps - from_gar) <= 0.01 + 0.5 / len(eps)
    return {
        "n_episodes": len(eps),
        "declared_total": sorted(totals),
        "n_garments": len(gar),
        "complete": bool(complete),
        "aggregates_agree": bool(per_garment_ok),
    }


def success_rate_checked(text: str) -> tuple[float, int]:
    """success_rate, but refuses a log whose two accounts disagree."""
    chk = cross_check(text)
    if not chk["aggregates_agree"]:
        raise Truncated(
            f"the per-episode lines and the per-garment summary disagree "
            f"({chk}). This log is truncated or interleaved -- do not report a "
            f"number from it."
        )
    return success_rate(text)


def per_category(summaries: list[GarmentSummary]) -> dict[tuple[str, str], dict]:
    """Group garment summaries by (category, split).

    Per-type is mandatory in every report from G1 onward -- long pants and
    long-sleeved tops are not the same task as shorts, and one averaged number
    hides that.
    """
    from .splits import parse as parse_name

    groups: dict[tuple[str, str], list[GarmentSummary]] = {}
    for s in summaries:
        g = parse_name(s.garment_name)
        groups.setdefault((g.category, g.split), []).append(s)
    return {
        k: {
            "n_garments": len(v),
            "success_rate": sum(x.success_rate for x in v) / len(v),
            "avg_return": sum(x.avg_return for x in v) / len(v),
        }
        for k, v in groups.items()
    }
