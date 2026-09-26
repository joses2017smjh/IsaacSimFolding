"""v8's preregistered mechanistic endpoints, from matched evaluation records.

The v7 diagnosis found every fine-tune (a2, pb1, dq1) changes the first
open-loop H50 chunk away from the baseline: from the identical initial
observation it opens only the right gripper (38/48 episodes vs the
baseline's 6/48) and lifts the cloth 150-200 actions later; at dev03 it
pinches inboard and never lifts. v8 replaces only the BC anchor (12/16 tops
-> 16 pants) to test whether the anchor drives that shift. These endpoints
are read BEFORE any settled count and decide the diagnostic question:

  E1  chunk-1 gripper opening: at each of the 8 H50 development rows, does
      the policy open BOTH grippers (target > 0.10 rad) within actions 1-50?
  E2  early lift at dev03 H50: maximum particle lift over actions 1-150.

Prediction if the anchor drives the shift: the candidate opens both
grippers in chunk 1 on >= 5 of 8 H50 rows (in run r1), AND lifts dev03 above
0.05 m by action 150 (in r1). Anything else refutes the anchor as the driver.
r2 shares seeds with r1 and is reported, not required.

Reads only rollout.json.behavior.jsonl; pure and CPU-testable.
"""
from __future__ import annotations

import json
from pathlib import Path

OPEN_RAD = 0.10
CHUNK1 = 50
EARLY = 150
LIFT_M = 0.05


def behavior(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chunk1_opening(rows: list[dict]) -> str:
    """'both', 'left', 'right' or 'neither': which grippers open within actions 1-50."""
    left = any(r["gripper_target_rad"]["left"] > OPEN_RAD for r in rows if r["action"] <= CHUNK1)
    right = any(r["gripper_target_rad"]["right"] > OPEN_RAD for r in rows if r["action"] <= CHUNK1)
    return "both" if left and right else "left" if left else "right" if right else "neither"


def early_lift(rows: list[dict]) -> float:
    return max((r["maximum_particle_lift_m"] for r in rows if r["action"] <= EARLY), default=0.0)


def compute(root: Path, labels: dict[str, str], runs=("r1", "r2")) -> dict:
    """labels: {'baseline': 'baseline', 'candidate': '<cand label>'}."""
    out = {"definitions": {"open_rad": OPEN_RAD, "chunk1_actions": CHUNK1, "early_actions": EARLY,
                           "lift_m": LIFT_M,
                           "prediction": ("candidate r1: both grippers open in chunk 1 on >= 5 of 8 H50 rows "
                                          "AND dev03 H50 early lift > 0.05 m")}}
    for which, label in labels.items():
        for run in runs:
            base = root / "evaluation" / f"{label}-{run}" / "benchmark"
            per_row = {}
            for d in sorted(base.glob("dev*_h50_*")):
                f = d / "rollout.json.behavior.jsonl"
                if not f.is_file():
                    continue
                rows = behavior(f)
                per_row[d.name[:5]] = {"chunk1": chunk1_opening(rows), "early_lift_m": round(early_lift(rows), 4)}
            out[f"{which}-{run}"] = {
                "rows": per_row,
                "both_open_rows": sum(v["chunk1"] == "both" for v in per_row.values()),
                "dev03_early_lift_m": per_row.get("dev03", {}).get("early_lift_m"),
            }
    c = out.get("candidate-r1", {})
    both, lift = c.get("both_open_rows"), c.get("dev03_early_lift_m")
    out["anchor_hypothesis_supported"] = bool(both is not None and lift is not None
                                              and both >= 5 and lift > LIFT_M)
    return out
