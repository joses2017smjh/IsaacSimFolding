#!/usr/bin/env python3
"""Aggregate the pose-3 observation-component causal diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "observation-components"
BOUNDARIES = (5, 10)


def first_source() -> Path:
    paths = sorted(ROOT.glob("outputs/observation_dev03_h50_Pant_Short_Seen_3/replan-causality.json"))
    if not paths:
        raise SystemExit("missing pose-3 observation diagnostic output")
    return paths[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timing_failures(data: dict) -> list[dict]:
    return [row for row in data.get("prediction_timing_audit", [])
            if row.get("sim_step_delta") != 0
            or row.get("episode_length_delta") not in (0, None)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()
    source = args.input.resolve() if args.input else first_source()
    data = json.loads(source.read_text())
    reference = data["observation_component_reference"]
    records = reference["records"]
    comparison, action_rows, condition_rows = [], [], []

    for boundary in BOUNDARIES:
        boundary_record = records[str(boundary)]
        probes = boundary_record["probes"]
        for name, probe in probes.items():
            metrics = probe["action_metrics_vs_cached_h50_suffix"]
            comparison.append({
                "boundary_action": boundary,
                "probe": name,
                "classification": probe["classification"],
                "deployable_controller": probe["deployable_controller"],
                "first_action_rms_rad": metrics["1"]["rms_rad"],
                "first5_rms_rad": metrics["5"]["rms_rad"],
                "first10_rms_rad": metrics["10"]["rms_rad"],
                "first10_mean_abs_rad": metrics["10"]["mean_abs_rad"],
                "first10_max_abs_rad": metrics["10"]["max_abs_rad"],
                "prefix_feature_distance_rms": probe["prefix_feature_distance_to_current_all"]["rms"],
                "same_rng_current_all_first10_exact": probe["same_rng_current_all_first10_exact"],
                "restore_joint_rms_rad": probe["restore"]["joint_rms_rad"],
                "restore_cloth_rms_m": probe["restore"]["cloth_rms_m"],
                "rng_restore_exact": probe["rng_restore_exact"],
                "prediction_zero_sim_steps": not probe["simulator_stepped_during_prediction"],
            })
            for local, delta in enumerate(metrics["10"]["action_l2_rad"], start=1):
                action_rows.append({
                    "boundary_action": boundary,
                    "probe": name,
                    "local_action": local,
                    "global_action": boundary + local,
                    "action_l2_rad": delta,
                    "per_joint_rms_rad": json.dumps(metrics["10"]["per_joint_rms_rad"]),
                })

        # Controls and any selected executions provide condition trajectories.
        branch = data["branches"][str(boundary)]
        for label, record in (("cached_h50", branch["cached_plan_control"]),
                              ("same_rng_h10", branch["same_rng_replan"])):
            for local, point in enumerate(record["trace"], start=1):
                condition_rows.append({
                    "condition": label,
                    "boundary_action": boundary,
                    "local_action": local,
                    "global_action": point["global_step"],
                    "conditions_passed": point["conditions_passed"],
                    "conditions_total": point["conditions_total"],
                    "geometric_success": point["geometric_success"],
                })

    per_joint_rows = []
    for boundary in BOUNDARIES:
        for name, probe in records[str(boundary)]["probes"].items():
            values = probe["action_metrics_vs_cached_h50_suffix"]["10"]["per_joint_rms_rad"]
            for joint, value in enumerate(values):
                per_joint_rows.append({
                    "boundary_action": boundary,
                    "probe": name,
                    "joint_index": joint,
                    "first10_per_joint_rms_rad": value,
                })
    write_csv(OUT / "comparison.csv", comparison)
    write_csv(OUT / "first10-action-differences.csv", action_rows)
    write_csv(OUT / "per-joint-deltas.csv", per_joint_rows)

    selection = reference.get("selection") or {}
    executions = selection.get("executions", {})
    for name, by_boundary in executions.items():
        for boundary in BOUNDARIES:
            record = by_boundary[str(boundary)]
            for local, point in enumerate(record["trace"], start=1):
                condition_rows.append({
                    "condition": f"executed_{name}",
                    "boundary_action": boundary,
                    "local_action": local,
                    "global_action": point["global_step"],
                    "conditions_passed": point["conditions_passed"],
                    "conditions_total": point["conditions_total"],
                    "geometric_success": point["geometric_success"],
                })
    write_csv(OUT / "condition-trajectories.csv", condition_rows)

    timing = data.get("prediction_timing_audit", [])
    failures = timing_failures(data)
    probe_restore_ok = all(
        records[str(boundary)]["all_snapshot_restores_within_tolerance"]
        and records[str(boundary)]["all_predictions_zero_sim_steps"]
        and records[str(boundary)]["all_rng_restores_exact"]
        and records[str(boundary)]["current_all_matches_same_rng_control"]
        for boundary in BOUNDARIES)
    selected = selection.get("selected_at_most_two", [])
    settled = selection.get("candidate_settled_4_of_4_both_boundaries", {})
    provenance = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "controller": "scripts/render/policy_rollout51.py",
        "controller_sha256": hashlib.sha256((ROOT / "scripts/render/policy_rollout51.py").read_bytes()).hexdigest(),
        "launcher": "slurm/observation.sbatch",
        "slurm_jobs": json.loads((OUT / "slurm-jobs.json").read_text()) if (OUT / "slurm-jobs.json").exists() else [],
        "garment": data.get("garment"),
        "seed": data.get("seed"),
        "match_pose": data.get("match_pose"),
        "boundaries": list(BOUNDARIES),
        "active_observation_keys": reference["active_observation_keys"],
        "controls": ["cached H50 suffix", "same-reconstructed-RNG fresh H10"],
        "rtc_guidance_used": False,
        "feature_hook": reference["feature_hook"],
        "prediction_timing_records": len(timing),
        "prediction_timing_failures": failures,
        "probe_restore_and_zero_step_gate": probe_restore_ok,
        "preregistered_meaningful_first10_rms_improvement_rad": selection.get(
            "preregistered_meaningful_first10_rms_improvement_rad"),
        "selected_at_most_two": selected,
        "selected_candidate_settled_4_of_4_both_boundaries": settled,
        "larger_validation_submitted": False,
        "larger_validation_reason": selection.get("larger_validation_reason", "not run"),
        "recommendation": (
            "small boundary-state chunk-consistency/on-policy corrective-data experiment"
            if not any(settled.values()) else
            "do not broaden validation from this inference-only probe without separate authorization"
        ),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    execution_summary = {}
    for name, by_boundary in executions.items():
        execution_summary[name] = {
            str(boundary): {
                "best_conditions": by_boundary[str(boundary)]["best_conditions"],
                "terminal_conditions": by_boundary[str(boundary)]["terminal"]["conditions_passed"],
                "settled_conditions": by_boundary[str(boundary)]["settled"]["terminal"]["conditions_passed"],
                "settled_4_of_4": by_boundary[str(boundary)]["settled"]["settled_4_of_4"],
                "all_snapshot_restores_within_tolerance": by_boundary[str(boundary)]["all_snapshot_restores_within_tolerance"],
                "all_predictions_zero_sim_steps": by_boundary[str(boundary)]["all_predictions_zero_sim_steps"],
                "all_rng_restores_exact": by_boundary[str(boundary)]["all_rng_restores_exact"],
            }
            for boundary in BOUNDARIES
        }
    (OUT / "execution-gate.json").write_text(json.dumps({
        "preregistered_threshold_rad": selection.get("preregistered_meaningful_first10_rms_improvement_rad"),
        "ranking": selection.get("ranking", []),
        "selected_at_most_two": selected,
        "execution_requested": selection.get("execution_requested", False),
        "execution_summary": execution_summary,
        "probe_restore_and_zero_step_gate": probe_restore_ok,
        "larger_validation_submitted": False,
    }, indent=2) + "\n")
    (OUT / "validation-submission.json").write_text(json.dumps({
        "submitted": False,
        "reason": "broad pose validation was explicitly out of scope for this diagnostic",
        "selected_candidates": selected,
        "settled_4_of_4_both_boundaries": settled,
    }, indent=2) + "\n")

    report = [
        "# Observation-component causal diagnostic",
        "",
        f"Source: `{source.relative_to(ROOT)}`.",
        "This pose-3-only diagnostic uses the exact action-5 and action-10 H50 snapshots, cached H50 and same-reconstructed-RNG fresh H10 controls, and no RTC guidance.",
        "",
        "## Timing and restoration gate",
        "",
        f"Prediction records: {len(timing)}; simulator/episode-counter failures: {len(failures)}.",
        f"All component probes restored within tolerance, matched the current-all same-RNG control, and had zero simulator advancement: **{'PASS' if probe_restore_ok else 'FAIL'}**.",
        "The original action-0 H50 observation and unchanged task are retained in the raw provenance JSON. The existing passive prefix-feature hook is reported there when available.",
        "",
        "## Probe ranking",
        "",
        "Stale and hybrid rows are attribution probes only; they are not deployable controllers.",
        f"Meaningful gate: at least {selection.get('preregistered_meaningful_first10_rms_improvement_rad')} rad first-10 RMS improvement at both boundaries relative to same-RNG current-all.",
        "",
        "| probe | action-5 improvement | action-10 improvement | both-boundary gate |",
        "|---|---:|---:|:---:|",
    ]
    for row in selection.get("ranking", []):
        scores = {x["boundary"]: x for x in row["boundary_scores"]}
        report.append(
            f"| {row['name']} | {scores[5]['improvement_rad']:.6f} | {scores[10]['improvement_rad']:.6f} | {'PASS' if row['passes_both_boundary_improvement_gate'] else 'FAIL'} |"
        )
    report.extend(["", f"Selected for execution (at most two): **{', '.join(selected) if selected else 'none'}**.", ""])
    if selected:
        for name in selected:
            report.append(f"- `{name}` settled 4/4 at both boundaries: **{bool(settled.get(name, False))}**.")
    if any(settled.values()):
        report.extend(["", "A selected probe met the two-boundary settling criterion, but no larger validation was submitted because broad pose validation is explicitly out of scope."])
    else:
        report.extend(["", "No executed candidate recovered settled 4/4 at both boundaries. Stop inference interventions and recommend a small boundary-state chunk-consistency/on-policy corrective-data experiment. No larger validation was submitted."])
    report.extend(["", "Condition trajectories are in `condition-trajectories.csv`; first-ten action differences and per-joint RMS deltas are in `first10-action-differences.csv` and `per-joint-deltas.csv`; machine-readable gates and provenance are in `execution-gate.json` and `provenance.json`."])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"source": str(source), "probe_restore_and_zero_step_gate": probe_restore_ok,
                      "selected": selected, "settled": settled}, indent=2))


if __name__ == "__main__":
    main()
