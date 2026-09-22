#!/usr/bin/env python3
"""Build compact, reviewable artifacts from GPU causal-capture JSON files."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "replan-causality"
RAW = ROOT / "outputs"
POSES = {"dev01": "1", "dev03": "3", "dev05": "5", "dev07": "7"}


def max_abs(a, b):
    return max((abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)


def l2(row):
    return math.sqrt(sum(float(x) * float(x) for x in row))


def write_svg(pose, data):
    """A dependency-free diagnostic: old/fresh action norms and gripper commands."""
    w, h = 1000, 640
    panels = []
    for col, boundary in enumerate(("5", "10")):
        d = data["branches"][boundary]
        old, fresh = d["old_plan_suffix_first_50"], d["fresh_plan_first_50"]
        x0 = 40 + col * 480
        def poly(vals, y0, height, color):
            if not vals:
                return ""
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-6)
            pts = []
            for i, v in enumerate(vals):
                x = x0 + 10 + 450 * i / max(len(vals) - 1, 1)
                y = y0 + height - (v - lo) / span * height
                pts.append(f"{x:.1f},{y:.1f}")
            return f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{" ".join(pts)}"/>'
        old_norm = [l2(r) for r in old]
        fresh_norm = [l2(r) for r in fresh]
        old_grip = [float(r[5]) for r in old]
        fresh_grip = [float(r[5]) for r in fresh]
        panels.append(f'<text x="{x0}" y="25" font-size="18">boundary {boundary}: old cached suffix vs fresh chunk</text>')
        panels.append(f'<rect x="{x0}" y="40" width="450" height="220" fill="#fafafa" stroke="#bbb"/>')
        panels.append(poly(old_norm, 50, 190, "#1565c0"))
        panels.append(poly(fresh_norm, 50, 190, "#c62828"))
        panels.append(f'<text x="{x0+8}" y="58" font-size="13">action L2 norm</text>')
        panels.append(f'<rect x="{x0}" y="310" width="450" height="220" fill="#fafafa" stroke="#bbb"/>')
        panels.append(poly(old_grip, 320, 190, "#1565c0"))
        panels.append(poly(fresh_grip, 320, 190, "#c62828"))
        panels.append(f'<text x="{x0+8}" y="328" font-size="13">left gripper command (joint 5)</text>')
        panels.append(f'<text x="{x0+260}" y="285" fill="#1565c0" font-size="13">blue old</text><text x="{x0+330}" y="285" fill="#c62828" font-size="13">red fresh</text>')
        panels.append(f'<text x="{x0+8}" y="555" font-size="12">A {d["cached_plan_control"]["best_conditions"]}/4, B {d["fresh_replan"]["best_conditions"]}/4; A geom={d["cached_plan_control"]["geometric_ever_success"]}, B geom={d["fresh_replan"]["geometric_ever_success"]}</text>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640"><rect width="100%" height="100%" fill="white"/>' + "".join(panels) + "</svg>"
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "plots" / f"pose-{pose}.svg").write_text(svg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, comparison = [], {}
    for prefix, pose in POSES.items():
        paths = sorted(RAW.glob(f"causal_{prefix}_*/replan-causality.json"))
        if not paths:
            raise SystemExit(f"missing causal JSON for {prefix}")
        data = json.loads(paths[0].read_text())
        baseline_source = data.get("baseline_source") or "fresh H50 capture (validation run)"
        comparison[pose] = {"source": str(paths[0].relative_to(ROOT)), "match_pose": data.get("match_pose"), "baseline_source": baseline_source, "branches": {}}
        for boundary, branch in data["branches"].items():
            A, B = branch["cached_plan_control"], branch["fresh_replan"]
            det = branch["repeated_prediction"]["deterministic_repeat"]
            varied = branch["repeated_prediction"]["varied_seed"]
            det_max = max((max_abs(det[0], x) for x in det[1:]), default=0.0)
            varied_max = max((max_abs(varied[i], varied[j]) for i in range(len(varied)) for j in range(i)), default=0.0)
            restore = A["restore"]
            replay_errors = A.get("restore_fidelity_errors", [])
            replay_joint_max = max((e.get("joint_rms_rad", 0.0) for e in replay_errors), default=0.0)
            replay_cloth_max = max((e.get("cloth_centroid_error_m", 0.0) for e in replay_errors), default=0.0)
            m = branch["old_vs_fresh_plan_metrics"]
            records.append({
                "pose": pose, "boundary_action": boundary, "horizon": branch["horizon"],
                "baseline_source": baseline_source,
                "cached_best_conditions": A["best_conditions"], "cached_geometric_success": A["geometric_ever_success"],
                "fresh_best_conditions": B["best_conditions"], "fresh_geometric_success": B["geometric_ever_success"],
                "immediate_restore_joint_rms_rad": restore.get("joint_rms_rad", ""),
                "immediate_restore_cloth_rms_m": restore.get("cloth_rms_m", ""),
                "cached_replay_max_joint_rms_rad": replay_joint_max, "cached_replay_max_cloth_centroid_error_m": replay_cloth_max,
                "action1_l2_rad": m["1"]["l2_per_action_rad"][-1], "action3_l2_rad": m["3"]["l2_per_action_rad"][-1],
                "action5_l2_rad": m["5"]["l2_per_action_rad"][-1], "action10_l2_rad": m["10"]["l2_per_action_rad"][-1],
                "action1_direction_cosine": m["1"]["direction_cosine"][0], "deterministic_repeat_max_abs": det_max,
                "varied_seed_max_abs": varied_max,
                "paired_outcome_supported": bool(A["geometric_ever_success"] and not B["geometric_ever_success"]),
            })
            comparison[pose]["branches"][boundary] = {
                "horizon": branch["horizon"], "cached": {"best_conditions": A["best_conditions"], "geometric_ever_success": A["geometric_ever_success"], "restore": A["restore"]},
                "fresh": {"best_conditions": B["best_conditions"], "geometric_ever_success": B["geometric_ever_success"]},
                "metrics": m, "old_plan_suffix_first_50": branch["old_plan_suffix_first_50"], "fresh_plan_first_50": branch["fresh_plan_first_50"],
                "deterministic_repeat_first_50": det, "varied_seed_first_50": varied,
            }
        write_svg(pose, data)
    with (OUT / "branch-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (OUT / "action-plan-comparison.json").write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
