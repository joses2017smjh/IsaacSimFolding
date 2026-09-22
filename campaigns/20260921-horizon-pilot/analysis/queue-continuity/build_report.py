#!/usr/bin/env python3
"""Aggregate the deterministic hard retained-prefix pose-3 diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "queue-continuity"


def l2(row):
    return math.sqrt(sum(float(x) * float(x) for x in row))


def first(path_pattern: str) -> Path:
    paths = sorted(ROOT.glob(path_pattern))
    if not paths:
        raise SystemExit(f"missing queue diagnostic output: {path_pattern}")
    return paths[0]


def divergence_at(trace, local_step):
    row = next((x for x in trace if int(x.get("local_step", -1)) == local_step), None)
    return ("", "") if row is None else (row.get("joint_rms_rad", ""), row.get("cloth_centroid_error_m", ""))


def action_rows(queue_actions, cached_actions, boundary, delay):
    rows = []
    for local, (queued, cached) in enumerate(zip(queue_actions[:10], cached_actions[:10]), start=1):
        delta = [float(a) - float(b) for a, b in zip(queued, cached)]
        rows.append({
            "boundary_action": boundary,
            "delay": delay,
            "local_action": local,
            "global_action": boundary + local,
            "l2_rad": l2(delta),
            "max_abs_rad": max((abs(x) for x in delta), default=0.0),
            "mean_abs_rad": sum(abs(x) for x in delta) / len(delta),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    source = args.input.resolve() if args.input else first("outputs/queue_dev03_h50_Pant_Short_Seen_3/replan-causality.json")
    data = json.loads(source.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    delays = [int(x) for x in data["queue_reference"]["delays"]]
    boundaries = ["5", "10"]
    comparison, condition_rows, action_diff_rows = [], [], []
    pass_by_delay = {}
    timing = data.get("prediction_timing_audit", [])
    timing_failures = [x for x in timing if x.get("sim_step_delta") != 0 or x.get("episode_length_delta") not in (0, None)]

    for boundary in boundaries:
        branch = data["branches"][boundary]
        cached = branch["cached_plan_control"]
        same = branch["same_rng_replan"]
        for row in [{"label": "cached_h50", "record": cached}, {"label": "same_rng_h10", "record": same}]:
            for local, point in enumerate(row["record"]["trace"], start=1):
                condition_rows.append({
                    "condition": row["label"], "boundary_action": int(boundary), "delay": "",
                    "local_action": local, "global_action": point["global_step"],
                    "conditions_passed": point["conditions_passed"],
                    "conditions_total": point["conditions_total"],
                    "geometric_success": point["geometric_success"],
                    "action_source": "cached_h50" if row["label"] == "cached_h50" else "fresh_h10",
                    "period_offset": "", "source_index": "",
                    "joint_rms_rad": "", "cloth_centroid_error_m": "",
                })
        for delay in delays:
            queued = branch["queue_diagnostic"][str(delay)]
            q_trace = queued["divergence_from_h50"]
            d2j, d2c = divergence_at(q_trace, 2)
            d10j, d10c = divergence_at(q_trace, 10)
            d20j, d20c = divergence_at(q_trace, 20)
            for local, point in enumerate(queued["trace"], start=1):
                reference = next((x for x in q_trace if x["local_step"] == local), {})
                condition_rows.append({
                    "condition": "queue", "boundary_action": int(boundary), "delay": delay,
                    "local_action": local, "global_action": point["global_step"],
                    "conditions_passed": point["conditions_passed"],
                    "conditions_total": point["conditions_total"],
                    "geometric_success": point["geometric_success"],
                    "action_source": point["action_source"],
                    "period_offset": point["period_offset"],
                    "source_index": point["source_index"],
                    "joint_rms_rad": reference.get("joint_rms_rad", ""),
                    "cloth_centroid_error_m": reference.get("cloth_centroid_error_m", ""),
                })
            action_diff_rows.extend(action_rows(queued["actions"], cached["actions"], int(boundary), delay))
            settled = queued["settled"]
            restore = queued["restore"]
            rng_invariant = queued.get("rng_reconstruction_invariant", queued["same_rng_first_chunk_exact"])
            passed = bool(settled["settled_4_of_4"] and queued["all_retained_actions_exact"]
                          and queued["all_fresh_prefixes_discarded"]
                          and queued["all_fresh_suffix_indices_exact"]
                          and queued["all_predictions_zero_sim_steps"]
                          and queued["same_rng_first_chunk_exact"]
                          and rng_invariant)
            comparison.append({
                "boundary_action": int(boundary), "delay": delay,
                "cached_best_conditions": cached["best_conditions"],
                "cached_settled_4_of_4": cached["settled"]["settled_4_of_4"],
                "same_rng_best_conditions": same["best_conditions"],
                "same_rng_settled_4_of_4": same["settled"]["settled_4_of_4"],
                "queue_best_conditions": queued["best_conditions"],
                "queue_ever_4_of_4": queued["geometric_ever_success"],
                "queue_terminal_conditions": queued["terminal"]["conditions_passed"],
                "queue_settled_conditions": settled["terminal"]["conditions_passed"],
                "queue_settled_4_of_4": settled["settled_4_of_4"],
                "queue_settled_success": settled["settled_success"],
                "pass": passed,
                "restore_joint_rms_rad": restore["joint_rms_rad"],
                "restore_cloth_rms_m": restore["cloth_rms_m"],
                "first_fresh_same_rng_exact": queued["same_rng_first_chunk_exact"],
                "rng_reconstruction_invariant": rng_invariant,
                "retained_actions_exact": queued["all_retained_actions_exact"],
                "fresh_prefix_discarded": queued["all_fresh_prefixes_discarded"],
                "fresh_suffix_indices_exact": queued["all_fresh_suffix_indices_exact"],
                "prediction_zero_sim_steps": queued["all_predictions_zero_sim_steps"],
                "joint_divergence_action2_rad": d2j,
                "joint_divergence_action10_rad": d10j,
                "joint_divergence_action20_rad": d20j,
                "cloth_divergence_action2_m": d2c,
                "cloth_divergence_action10_m": d10c,
                "cloth_divergence_action20_m": d20c,
                "first10_action_l2_rad": [r["l2_rad"] for r in action_diff_rows[-10:]],
            })
            pass_by_delay.setdefault(str(delay), []).append(passed)

    for delay in delays:
        pass_by_delay[str(delay)] = {
            "boundaries_passed": sum(pass_by_delay[str(delay)]),
            "both_boundaries_pass": all(pass_by_delay[str(delay)]),
        }
    passing = [delay for delay in delays if pass_by_delay[str(delay)]["both_boundaries_pass"]]
    smallest = min(passing) if passing else None
    gate_pass = bool(smallest is not None and not timing_failures)
    provenance = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controller": "scripts/render/policy_rollout51.py",
        "controller_sha256": hashlib.sha256((ROOT / "scripts/render/policy_rollout51.py").read_bytes()).hexdigest(),
        "slurm_jobs": json.loads((OUT / "slurm-jobs.json").read_text()),
        "checkpoint": data.get("checkpoint"),
        "garment": data.get("garment"),
        "seed": data.get("seed"),
        "match_pose": data.get("match_pose"),
        "boundaries": [int(x) for x in boundaries],
        "delays": delays,
        "execution_horizon": 10,
        "baseline_horizon": 50,
        "controls": ["cached H50 suffix", "same-reconstructed-RNG fresh H10"],
        "rtc_guidance_used": False,
        "handoff_rule": data["queue_reference"]["handoff_rule"],
        "prediction_timing_records": len(timing),
        "prediction_timing_failures": timing_failures,
        "queue_gate_pass": gate_pass,
        "pass_by_delay": pass_by_delay,
        "smallest_passing_delay": smallest,
        "validation_policy": "submit only pose slots 1/3/5/7 when both pose-3 boundaries pass; otherwise stop and recommend observation-component ablation",
    }
    write_csv(OUT / "comparison.csv", comparison)
    write_csv(OUT / "condition-trajectories.csv", condition_rows)
    write_csv(OUT / "first10-action-differences.csv", action_diff_rows)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (OUT / "validation-gate.json").write_text(json.dumps(provenance, indent=2) + "\n")

    report = [
        "# Hard retained-prefix queue diagnostic",
        "",
        f"Source: `{source.relative_to(ROOT)}`.",
        "The diagnostic uses the exact pose-3 action-5 and action-10 in-process snapshots, cached H50 and same-reconstructed-RNG fresh H10 controls, and no RTC guidance.",
        "",
        "## Timing and invariance gate",
        "",
        f"Prediction records: {len(timing)}; simulator/episode-counter failures: {len(timing_failures)}.",
        "Every queue prediction must have zero simulator steps and zero episode-length delta; every first fresh chunk must exactly match the same-RNG control. Snapshot restore, retained-action identity, fresh-prefix discard and suffix indexing are fail-closed checks in the raw JSON.",
        "",
        "## Results",
        "",
        "| delay | action-5 settled | action-10 settled | both-boundary gate |",
        "|---:|:---:|:---:|:---:|",
    ]
    for delay in delays:
        rows = [r for r in comparison if r["delay"] == delay]
        report.append(f"| {delay} | {rows[0]['queue_settled_conditions']}/4 | {rows[1]['queue_settled_conditions']}/4 | {'PASS' if pass_by_delay[str(delay)]['both_boundaries_pass'] else 'FAIL'} |")
    report.extend(["", f"Smallest passing delay: **{smallest if smallest is not None else 'none'}**.", ""])
    if gate_pass:
        report.append("Both pose-3 boundaries passed. The conditional validation is authorized only for pose slots 1/3/5/7 and the smallest passing delay; no other rollout is authorized by this report.")
    else:
        report.append("No delay passed both pose-3 boundaries, or a required timing check failed. Stop here and recommend an observation-component ablation rather than another continuity intervention. No larger validation was submitted.")
    report.extend([
        "",
        "Detailed condition trajectories are in `condition-trajectories.csv`; first-ten action differences versus the cached H50 suffix are in `first10-action-differences.csv`; divergence at local actions 2/10/20 and gate fields are in `comparison.csv`.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"source": str(source), "gate_pass": gate_pass, "smallest_passing_delay": smallest, "pass_by_delay": pass_by_delay}, indent=2))


if __name__ == "__main__":
    main()
