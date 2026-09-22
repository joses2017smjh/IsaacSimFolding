#!/usr/bin/env python3
"""Aggregate fixed-RNG causal captures without requiring NumPy."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "fixed-rng-replanning"
RAW = ROOT / "outputs"
POSES = {"dev01": "1", "dev03": "3", "dev05": "5", "dev07": "7"}


def max_abs(a, b):
    return max((abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)


def l2(row):
    return math.sqrt(sum(float(x) * float(x) for x in row))


def metrics(old, new):
    out = {}
    for width in (1, 3, 5, 10, min(40, len(old), len(new))):
        a, b = old[:width], new[:width]
        if not a or not b:
            continue
        delta = [[y - x for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
        out[str(width)] = {
            "mean_abs_rad": sum(abs(x) for row in delta for x in row) / (len(delta) * len(delta[0])),
            "l2_per_action_rad": [l2(row) for row in delta],
            "max_joint_abs_rad": max(abs(x) for row in delta for x in row),
            "left_arm_l2_rad": [l2(row[:6]) for row in delta],
            "right_arm_l2_rad": [l2(row[6:]) for row in delta],
            "gripper_delta_rad": {"left": [row[5] for row in delta], "right": [row[11] for row in delta]},
            "direction_cosine": [sum(x * y for x, y in zip(ra, rb)) / (l2(ra) * l2(rb) + 1e-12) for ra, rb in zip(a, b)],
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, panel = [], {}
    for prefix, pose in POSES.items():
        paths = sorted(RAW.glob(f"causal_{prefix}_*/replan-causality.json"))
        if not paths:
            raise SystemExit(f"missing causal capture for {prefix}")
        data = json.loads(paths[0].read_text())
        audit = data.get("rng_audit", {})
        reconstructed = data.get("reconstructed_original_h50_chunk") or []
        baseline_path = ROOT / "outputs" / (paths[0].parent.name.replace("causal_", "")) / "rollout.json.behavior.jsonl"
        baseline = []
        if baseline_path.is_file():
            baseline = [json.loads(line)["executed_action_rad"] for line in baseline_path.read_text().splitlines() if line.strip()][:50]
        panel[pose] = {"source": str(paths[0].relative_to(ROOT)), "match_pose": data.get("match_pose"), "rng_audit": audit, "reconstructed_original_h50_chunk": reconstructed, "reconstruction_verification": {"baseline_behavior": str(baseline_path.relative_to(ROOT)), "actions_compared": min(len(reconstructed), len(baseline)), "max_abs_action_error": max_abs(reconstructed[:len(baseline)], baseline[:len(reconstructed)]) if reconstructed and baseline else None}, "boundaries": {}}
        for boundary, branch in data.get("branches", {}).items():
            old = branch["old_plan_suffix_first_50"]
            try:
                original_suffix = reconstructed[int(boundary):]
            except (TypeError, ValueError):
                original_suffix = []
            same = branch.get("same_rng_replan", {})
            same_plan = same.get("chunks", [[]])[0] if same.get("chunks") else []
            cached = branch["cached_plan_control"]
            normal = branch["fresh_replan"]
            seed_rows = []
            for entry in branch.get("seed_panel", []):
                m = metrics(old, entry["chunk"])
                seed_rows.append({"seed": entry["seed"], "first_action_l2_rad": m.get("1", {}).get("l2_per_action_rad", [None])[0], "max_joint_abs_rad": m.get("10", {}).get("max_joint_abs_rad"), "metrics": m, "chunk": entry["chunk"]})
            sm = metrics(original_suffix or old, same_plan)
            execution_sm = metrics(old, same_plan)
            rows.append({
                "pose": pose, "boundary_action": boundary, "horizon": branch.get("horizon"),
                "primary_validated_pose": pose in {"1", "3"},
                "cached_best_conditions": cached.get("best_conditions"), "cached_geometric_success": cached.get("geometric_ever_success"),
                "same_rng_best_conditions": same.get("best_conditions"), "same_rng_geometric_success": same.get("geometric_ever_success"),
                "normal_fresh_best_conditions": normal.get("best_conditions"), "normal_fresh_geometric_success": normal.get("geometric_ever_success"),
                "same_rng_action1_l2_rad": sm.get("1", {}).get("l2_per_action_rad", [None])[0],
                "same_rng_action3_l2_rad": sm.get("3", {}).get("l2_per_action_rad", [None])[-1],
                "same_rng_action5_l2_rad": sm.get("5", {}).get("l2_per_action_rad", [None])[-1],
                "same_rng_action10_l2_rad": sm.get("10", {}).get("l2_per_action_rad", [None])[-1],
                "same_rng_max_joint_abs_rad": sm.get("10", {}).get("max_joint_abs_rad"),
                "cached_execution_suffix_action1_l2_rad": execution_sm.get("1", {}).get("l2_per_action_rad", [None])[0],
                "immediate_restore_joint_rms_rad": same.get("restore", {}).get("joint_rms_rad"),
                "immediate_restore_cloth_rms_m": same.get("restore", {}).get("cloth_rms_m"),
                "same_rng_cached_success_separation": bool(cached.get("geometric_ever_success") and not same.get("geometric_ever_success")),
            })
            panel[pose]["boundaries"][boundary] = {"horizon": branch.get("horizon"), "cached": {"best_conditions": cached.get("best_conditions"), "geometric_ever_success": cached.get("geometric_ever_success")}, "same_rng": {"best_conditions": same.get("best_conditions"), "geometric_ever_success": same.get("geometric_ever_success"), "metrics_vs_reconstructed_original_suffix": sm, "metrics_vs_cached_execution_suffix": execution_sm, "first_chunk": same_plan}, "normal_fresh": {"best_conditions": normal.get("best_conditions"), "geometric_ever_success": normal.get("geometric_ever_success")}, "seed_panel": seed_rows}
    with (OUT / "plan-comparisons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "seed-panel.json").write_text(json.dumps(panel, indent=2))


if __name__ == "__main__":
    main()
