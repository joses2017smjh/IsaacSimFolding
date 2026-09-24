"""Per-pose transfer analysis over existing v4/v5 artifacts (no rollouts).

    pose_transfer.py --out analysis/pose-transfer.json

For each development pose -- grouped by match_pose identity, never by pose
key -- it joins what v4's recovery search supplied (rows, settled branches,
attempts, corpus samples) with what v5's matched evaluation measured
(settled folds per policy and horizon, and which checker conditions failed).
Reads only small npz members; never the image arrays.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

CAMPAIGNS = Path(__file__).resolve().parents[2]
V4 = CAMPAIGNS / "20260923-recovery-supervision-v4"
V5 = CAMPAIGNS / "20260924-matched-comparison-v5"


def pose_id(match_pose: str) -> tuple:
    return tuple(round(float(x), 4) for x in match_pose.split(":"))


def analyse() -> dict:
    v4 = json.loads((V4 / "manifest.json").read_text())
    dev = {r["id"]: r for r in v4["benchmark"]}
    names = {}
    for rid, name in (("dev00", "P_A"), ("dev01", "P_B"), ("dev02", "P_C")):
        names[pose_id(next(r for i, r in dev.items() if i.startswith(rid + "_h10"))["match_pose"])] = name
    dev_pose = {i[:5]: names[pose_id(r["match_pose"])] for i, r in dev.items()}
    row_pose = {r["id"]: names.get(pose_id(r["match_pose"]), "other") for r in v4["recovery_search"]}

    supply = collections.defaultdict(lambda: {"rows": [], "settled": 0, "attempts": 0, "samples": 0})
    for f in sorted((V4 / "recovery").glob("*/recovery.json")):
        row = f.parent.name
        att = [a for a in json.loads(f.read_text())["attempts"] if a.get("state") == "completed"]
        s = supply[row_pose[row]]
        s["rows"].append({"id": row, "settled": sum(bool(a.get("settled_success")) for a in att),
                          "attempts": len(att)})
        s["settled"] += s["rows"][-1]["settled"]; s["attempts"] += len(att)
    with np.load(V4 / "datasets/recovery.npz", allow_pickle=False) as z:
        for o in np.asarray(z["origin"]).astype(str):
            supply[row_pose[o.split(":", 1)[1]]]["samples"] += 1
    total = sum(s["samples"] for s in supply.values())
    for s in supply.values():
        s["corpus_share"] = round(s["samples"] / total, 3)

    outcome = collections.defaultdict(lambda: {"episodes": 0, "settled": 0, "condition_failures": collections.Counter(),
                                               "margins": collections.defaultdict(list)})
    for f in glob.glob(str(V5 / "evaluation/*/benchmark/*/rollout.json")):
        d = os.path.dirname(f)
        label = Path(d).parents[1].name.rsplit("-", 1)[0]
        policy = "baseline" if label == "baseline" else "candidate"
        row = os.path.basename(d)
        h = "H10" if "_h10_" in row else "H50"
        r = json.loads(Path(f).read_text())
        o = outcome[(dev_pose[row[:5]], policy, h)]
        o["episodes"] += 1
        o["settled"] += int(bool(r["terminal_success"]))
        for c, v in r["terminal_checker"]["details"].items():
            if not v["passed"]:
                o["condition_failures"][c] += 1
            o["margins"][c].append(v["margin_cm"])

    poses = {}
    for p in ("P_A", "P_B", "P_C"):
        ev = {}
        for policy in ("baseline", "candidate"):
            for h in ("H10", "H50"):
                o = outcome[(p, policy, h)]
                ev[f"{policy}_{h}"] = {
                    "settled": f"{o['settled']}/{o['episodes']}",
                    "condition_failures": dict(o["condition_failures"]),
                    "mean_margin_cm": {c: round(float(np.mean(m)), 2) for c, m in sorted(o["margins"].items())}}
        poses[p] = {"dev_rows": sorted(k for k, v in dev_pose.items() if v == p),
                    "v4_supply": supply[p], "v5_matched": ev}
    return {
        "method": ("development poses grouped by match_pose identity (pose keys map to different poses on "
                   "different garments); v4 search rows mapped the same way; v5's 64 matched episodes"),
        "poses": poses,
        "finding": ("the candidate's change tracks per-pose supervision density: P_C (53% of the corpus, 44 "
                    "settled branches) 1/6 -> 6/6 at H10; P_A (28%, 23) mixed; P_B (19%, 14 branches from 2 rows) "
                    "0/8 vs the baseline's 4/8 over H10 and H50, every failure on checker condition 1. The "
                    "runner's 'never_approached_cloth' category is not used: successful folds also stay >= 5 cm "
                    "from the cloth by its last-link metric."),
        "decision": ("supply dense validated supervision at P_B (only 2 exact-pose training rows exist: Seen_6 "
                     "key 2, Seen_8 key 2 -> 4 fresh seeds each, early roots) and P_A (4 rows x 2 seeds), gate it "
                     "per pose, then train a2's recipe unchanged on a pose-balanced corpus"),
        "rule_arithmetic": ("against a matched baseline of 5/16 at H10 the rule needs 11/16 (p = 0.038); a2 "
                            "reached 8/16, so P_A and P_B must both improve while P_C holds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = analyse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=list) + "\n")
    for p, v in result["poses"].items():
        s = v["v4_supply"]
        m = v["v5_matched"]
        print(f"{p}: supply {s['settled']}/{s['attempts']} settled, {s['samples']} samples "
              f"({s['corpus_share']:.0%}) | H10 base {m['baseline_H10']['settled']} cand {m['candidate_H10']['settled']}"
              f" | H50 base {m['baseline_H50']['settled']} cand {m['candidate_H50']['settled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
