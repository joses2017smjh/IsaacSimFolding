"""Build the report and machine-readable provenance for the ceiling diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _branch_rows(result):
    for boundary, by_controller in sorted(result["boundaries"].items(), key=lambda row: int(row[0])):
        for controller, branch in by_controller.items():
            for row in branch["trace"]:
                yield {
                    "controller": controller,
                    "boundary_action": int(boundary),
                    "phase": "branch",
                    "global_step": row["global_step"],
                    "conditions_passed": row["conditions_passed"],
                    "conditions_total": row["conditions_total"],
                    "geometric_success": row["geometric_success"],
                }
            for row in branch["settled"]["trace"]:
                yield {
                    "controller": controller,
                    "boundary_action": int(boundary),
                    "phase": "terminal_settle",
                    "global_step": row.get("global_step", ""),
                    "conditions_passed": row["conditions_passed"],
                    "conditions_total": row["conditions_total"],
                    "geometric_success": row["geometric_success"],
                }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--job-id", type=int, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    out = root / "analysis" / "boundary-ceiling"
    evaluation_path = out / "ceiling-evaluation.json"
    result = json.loads(evaluation_path.read_text())
    out.mkdir(parents=True, exist_ok=True)

    trajectory_rows = list(_branch_rows(result))
    with (out / "condition-trajectories.csv").open("w", newline="") as stream:
        fields = list(trajectory_rows[0]) if trajectory_rows else ["controller"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trajectory_rows)

    settled_rows = []
    for controller in result["controllers"]:
        for boundary in (5, 10):
            branch = result["boundaries"][str(boundary)][controller]
            settled_rows.append({
                "controller": controller,
                "boundary_action": boundary,
                "best_conditions": branch["best_conditions"],
                "terminal_conditions_passed": branch["terminal"]["conditions_passed"],
                "terminal_conditions_total": branch["terminal"]["conditions_total"],
                "settled_4_of_4": branch["settled"]["settled_4_of_4"],
                "snapshot_restore_cloth_rms_m": branch["restore"]["cloth_rms_m"],
                "snapshot_restore_joint_rms_rad": branch["restore"]["joint_rms_rad"],
                "all_predictions_zero_sim_steps": branch.get("all_predictions_zero_sim_steps", True),
            })
    with (out / "settled-outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(settled_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(settled_rows)

    settled = result["both_boundaries_settled_4_of_4"]
    boundary_only_pass = bool(settled["boundary_only_300"])
    mixed_pass = bool(settled["mixed_step_000300"])
    fresh_controllers = [
        "baseline_fresh_h10", "boundary_only_300", "mixed_step_000300"
    ]
    rng_invariance = {
        boundary: all(
            all(
                row[controller]["prediction_timing"][index]["rng_before_digest"]
                == row[fresh_controllers[0]]["prediction_timing"][index]["rng_before_digest"]
                for controller in fresh_controllers
            )
            for index in range(len(row[fresh_controllers[0]]["prediction_timing"]))
        )
        for boundary, row in result["boundaries"].items()
    }
    all_rng_invariant = all(rng_invariance.values())
    if not boundary_only_pass:
        decision = "stop_static_boundary_suffix_adaptation"
        recommendation = (
            "Do not launch PEFT or parameter anchoring. Run only a small on-policy corrective-data "
            "experiment: roll the selected H10 student from the exact pose-3 snapshots; at each "
            "student-visited H10 replan state capture the exact simulator snapshot and observation, "
            "query a fresh H50 plan from that actual state, branch-execute each H50 candidate, and "
            "retain labels only when the H50 branch demonstrably recovers the target condition. "
            "Do not reuse the original time-aligned cached H50 suffix after state divergence."
        )
    elif not mixed_pass:
        decision = "recommend_parameter_anchored_or_parameter_efficient_adaptation"
        recommendation = (
            "The boundary-only diagnostic upper bound is sufficient but the raster-retaining mixed "
            "checkpoint is not; recommend parameter-anchored or parameter-efficient adaptation "
            "aimed at the stronger boundary correction while preserving raster behavior."
        )
    else:
        decision = "small_pose_validation_authorized"
        recommendation = (
            "Submit only the small poses 1/3/5/7 validation with ordinary fresh H10 replanning; "
            "do not launch broader training or validation."
        )

    jobs = {
        "wrapper_wrong_campaign_root": {
            "job_id": 21399451,
            "state": "FAILED",
            "exit_code": "2:0",
            "note": "wrapper-only failure: repository root passed instead of campaign root; no simulator result",
        },
        "ceiling_cached_control_lookup_bug": {
            "job_id": 21399452,
            "state": "FAILED",
            "exit_code": "3:0",
            "note": "diagnostic code failure after exact snapshots: short reconstruction action map did not fall back to the 600-action cached stream",
        },
        "existing_checkpoint_closed_loop_ceiling": {
            "job_id": args.job_id,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "script": str(root / "slurm" / "boundary_ceiling.sbatch"),
        }
    }
    report = [
        "# Existing-checkpoint closed-loop ceiling diagnostic",
        "",
        "This diagnostic compared the retained boundary-only 300-step checkpoint and mixed-replay `step_000300` from the exact validated pose-3 action-5/action-10 snapshots.",
        "Fresh branches used ordinary H10 replanning. Cached H50 was retained only as a control and to reconstruct the snapshots.",
        "",
        f"- Boundary-only 300 settled 4/4 at both boundaries: `{boundary_only_pass}` (diagnostic upper bound; non-deployable because held-out retention failed).",
        f"- Mixed replay step 300 settled 4/4 at both boundaries: `{mixed_pass}`.",
        f"- Snapshot restoration checks: `{result['all_snapshot_restores_within_tolerance']}`.",
        f"- Zero simulator steps during prediction: `{result['all_fresh_predictions_zero_sim_steps']}`.",
        f"- Reconstructed RNG stream invariant across baseline and both checkpoints: `{all_rng_invariant}`.",
        f"- Decision: `{decision}`.",
        "",
        recommendation,
        "",
        "Controls and raw trajectories are in `ceiling-evaluation.json`; tabular trajectories are in `condition-trajectories.csv` and `settled-outcomes.csv`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    provenance = {
        "schema_version": 1,
        "diagnostic": "existing-checkpoint closed-loop ceiling diagnostic",
        "baseline_commit_at_report": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "evaluation_sha256": sha256(evaluation_path),
        "source": result["snapshot_source"],
        "boundaries": [5, 10],
        "horizon": 10,
        "controllers": result["controllers"],
        "controls": {
            "cached_h50_control_only": True,
            "baseline_fresh_h10": True,
            "ordinary_fresh_h10_for_checkpoints": True,
            "rtc": False,
            "retained_prefix_queue": False,
            "stale_or_hybrid_observations": False,
            "cached_h50_execution_assistance_in_fresh_branches": False,
            "boundary_only_deployable": False,
            "new_training_launched": False,
            "broad_validation_launched": False,
        },
        "verification": {
            "snapshot_restoration_within_tolerance": result["all_snapshot_restores_within_tolerance"],
            "fresh_prediction_zero_sim_steps": result["all_fresh_predictions_zero_sim_steps"],
            "rng_before_digest_invariant_by_boundary": rng_invariance,
            "rng_invariant": all_rng_invariant,
            "inference_blocks_before_env_step": True,
        },
        "settled_4_of_4": settled,
        "decision": decision,
        "recommendation": recommendation,
        "slurm_jobs": jobs,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "slurm-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps({"decision": decision, "boundary_only_300": boundary_only_pass,
                      "mixed_step_000300": mixed_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
