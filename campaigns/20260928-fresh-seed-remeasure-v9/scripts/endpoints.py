"""v8's preregistered mechanistic endpoints, from matched evaluation records.

The v7 diagnosis found every fine-tune (a2, pb1, dq1) changes the first
open-loop H50 chunk away from the baseline: from the identical initial
observation it opens ONLY THE RIGHT gripper (38/48 episodes vs the
baseline's 6/48) and lifts the cloth 150-200 actions later; at dev03 it
pinches inboard and never lifts. v8 replaces only the BC anchor to test
whether the anchor drives that shift. These endpoints are read BEFORE any
settled count and decide the diagnostic question:

  E1  right-only chunk-1 opening at each of the 8 H50 development rows: the
      right gripper target exceeds 0.10 rad at some action in 1-50 and the
      left never does (the diagnosed shift; "neither" is not the shift).
  E2  early lift at dev03 H50: maximum particle lift over actions 1-150.

Supported iff, in run r1, the candidate is right-only on <= 3 of 8 H50 rows
AND lifts dev03 above 0.05 m by action 150. Refuted iff that fails with all
rows valid and the positive control met. INDETERMINATE (None) iff any of the
8 candidate or baseline r1 H50 rows is missing/unfinished, or the baseline
(same arrays) does not itself show its reference pattern (right-only on
<= 3 rows and dev03 lift > 0.05 m) -- an infrastructure event or a drifted
baseline is never recorded as a refutation. r2 shares seeds with r1 and is
reported only.

Calibrated on history before launch: the rule classifies the baseline as
supported and a2, pb1, dq1 as refuted in every one of v5, v6, v7.
Reads only status.json and rollout.json.behavior.jsonl; pure and CPU-testable.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

OPEN_RAD = 0.10
CHUNK1 = 50
EARLY = 150
LIFT_M = 0.05
MAX_RIGHT_ONLY = 3
H50_ROWS = tuple(f"dev0{k}" for k in range(8))


def behavior(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chunk1_opening(rows: list[dict]) -> str:
    """'both', 'left', 'right' or 'neither': which grippers open within actions 1-50."""
    left = any(r["gripper_target_rad"]["left"] > OPEN_RAD for r in rows if r["action"] <= CHUNK1)
    right = any(r["gripper_target_rad"]["right"] > OPEN_RAD for r in rows if r["action"] <= CHUNK1)
    return "both" if left and right else "left" if left else "right" if right else "neither"


def early_lift(rows: list[dict]) -> float:
    return max((r["maximum_particle_lift_m"] for r in rows if r["action"] <= EARLY), default=0.0)


def row_valid(d: Path) -> tuple[bool, list[dict] | None]:
    """A row counts only if its task completed and its stream covers action EARLY."""
    sp, bp = d / "status.json", d / "rollout.json.behavior.jsonl"
    if not sp.is_file() or not bp.is_file():
        return False, None
    try:
        if json.loads(sp.read_text()).get("state") != "completed":
            return False, None
        rows = behavior(bp)
    except ValueError:
        return False, None
    if max((r["action"] for r in rows), default=0) < EARLY:
        return False, None
    return True, rows


def block(root: Path, label: str, run: str) -> dict:
    base = root / "evaluation" / f"{label}-{run}" / "benchmark"
    per_row, missing = {}, []
    for key in H50_ROWS:
        dirs = sorted(base.glob(f"{key}_h50_*"))
        ok, rows = row_valid(dirs[0]) if len(dirs) == 1 else (False, None)
        if not ok:
            missing.append(key)
            continue
        per_row[key] = {"chunk1": chunk1_opening(rows), "early_lift_m": round(early_lift(rows), 4)}
    tally = collections.Counter(v["chunk1"] for v in per_row.values())
    return {"rows": per_row, "missing_rows": missing, "tally": dict(tally),
            "right_only_rows": tally.get("right", 0), "both_open_rows": tally.get("both", 0),
            "dev03_early_lift_m": per_row.get("dev03", {}).get("early_lift_m")}


def pattern(b: dict) -> bool | None:
    """The reference pattern (the baseline's): few right-only rows and an early dev03 lift."""
    if b["missing_rows"] or b["dev03_early_lift_m"] is None:
        return None
    return bool(b["right_only_rows"] <= MAX_RIGHT_ONLY and b["dev03_early_lift_m"] > LIFT_M)


def compute(root: Path, labels: dict[str, str], runs=("r1", "r2")) -> dict:
    """labels: {'baseline': 'baseline', 'candidate': '<cand label>'}."""
    out = {"definitions": {"open_rad": OPEN_RAD, "chunk1_actions": CHUNK1, "early_actions": EARLY,
                           "lift_m": LIFT_M, "max_right_only_rows": MAX_RIGHT_ONLY,
                           "prediction": ("candidate r1: right-only chunk-1 opening on <= 3 of 8 H50 rows AND dev03 "
                                          "H50 lift > 0.05 m by action 150; indeterminate unless every r1 H50 row "
                                          "is valid for both policies and the baseline shows the same pattern")}}
    for which, label in labels.items():
        for run in runs:
            out[f"{which}-{run}"] = block(root, label, run)
    control = pattern(out["baseline-r1"]) if "baseline-r1" in out else None
    cand = pattern(out["candidate-r1"]) if "candidate-r1" in out else None
    out["baseline_control_ok"] = control
    out["anchor_hypothesis_supported"] = None if control is not True or cand is None else cand
    out["status"] = ("indeterminate" if out["anchor_hypothesis_supported"] is None
                     else "supported" if out["anchor_hypothesis_supported"] else "refuted")
    return out
