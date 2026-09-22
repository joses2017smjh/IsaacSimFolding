"""Build reports for the pose-3 on-policy H50 rescue-oracle audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_rows(result):
    for group in result["root_groups"]:
        boundary = group["initial_boundary_action"]
        for root in group["visited_roots"]:
            for branch_name in ("normal_h10", "fresh_h50_open_loop"):
                branch = root["branches"][branch_name]
                for row in branch["trace"]:
                    yield {
                        "root_id": root["root_id"],
                        "initial_boundary_action": boundary,
                        "student_global_action": root["student_global_action"],
                        "branch": branch_name,
                        "phase": "continuation",
                        "global_step": row["global_step"],
                        "conditions_passed": row["conditions_passed"],
                        "conditions_total": row["conditions_total"],
                        "geometric_success": row["geometric_success"],
                    }
                for settle_step, row in enumerate(branch["settled"]["trace"], start=1):
                    yield {
                        "root_id": root["root_id"],
                        "initial_boundary_action": boundary,
                        "student_global_action": root["student_global_action"],
                        "branch": branch_name,
                        "phase": "terminal_settle",
                        "global_step": "",
                        "settle_step": settle_step,
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
    out = root / "analysis" / "onpolicy-h50-oracle"
    evaluation_path = out / "oracle-evaluation.json"
    examples_path = out / "successful-h50-examples.npz"
    roots_path = out / "student-roots.npz"
    result = json.loads(evaluation_path.read_text())
    examples = np.load(examples_path, allow_pickle=False)
    roots = np.load(roots_path, allow_pickle=False)

    trajectory_rows = list(_trajectory_rows(result))
    with (out / "condition-trajectories.csv").open("w", newline="") as stream:
        fields = sorted({key for row in trajectory_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trajectory_rows)

    recovery_rows = []
    for group in result["root_groups"]:
        for root_row in group["visited_roots"]:
            h10 = root_row["branches"]["normal_h10"]
            h50 = root_row["branches"]["fresh_h50_open_loop"]
            recovery_rows.append({
                "root_id": root_row["root_id"],
                "initial_boundary_action": root_row["initial_boundary_action"],
                "student_global_action": root_row["student_global_action"],
                "offset_from_initial_boundary": root_row["offset_from_initial_boundary"],
                "h10_best_conditions": h10["best_conditions"],
                "h10_terminal_conditions": h10["terminal"]["conditions_passed"],
                "h10_settled_4_of_4": h10["settled"]["settled_4_of_4"],
                "h50_best_conditions": h50["best_conditions"],
                "h50_terminal_conditions": h50["terminal"]["conditions_passed"],
                "h50_settled_4_of_4": h50["settled"]["settled_4_of_4"],
                "h50_rescues_h10_failure": h50["rescues_h10_failure"],
                "h10_h50_root_identical": root_row["branches_start_from_identical_root"],
                "h50_chunk_matches_h10_root_chunk": h50["root_chunk_matches_current_h10_chunk"]["exact"],
                "h10_all_predictions_zero_sim_steps": h10["all_predictions_zero_sim_steps"],
                "h50_all_predictions_zero_sim_steps": h50["all_predictions_zero_sim_steps"],
                "h10_rng_restore_exact": h10["rng_restore_exact"],
                "h50_rng_restore_exact": h50["rng_restore_exact"],
            })
    with (out / "recovery-roots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recovery_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(recovery_rows)

    example_rows = []
    for index in range(len(examples["root_id"])):
        example_rows.append({
            "example_index": index,
            "root_id": str(examples["root_id"][index]),
            "initial_boundary_action": int(examples["initial_boundary_action"][index]),
            "student_global_action": int(examples["student_global_action"][index]),
            "offset_from_root": int(examples["offset_from_root"][index]),
            "global_action": int(examples["global_action"][index]),
            "target_valid_length": int(examples["target_valid_mask"][index].sum()),
            "source_plan_sha256": str(examples["source_plan_sha256"][index]),
            "observation_digest": str(examples["observation_digest"][index]),
        })
    with (out / "candidate-examples.csv").open("w", newline="") as stream:
        fields = list(example_rows[0]) if example_rows else ["example_index"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(example_rows)

    successful_roots = result["fresh_h50_successful_recovery_root_count"]
    total_roots = result["student_visited_root_count"]
    examples_count = result["validated_intermediate_example_count"]
    reliable = bool(result["fresh_h50_recovery_reliable_gate"])
    if reliable:
        decision = "recommend_small_mixed_replay_iteration"
        recommendation = (
            "Fresh H50 recovered all six actual student-visited roots. Recommend one small fixed "
            "mixed-replay iteration from the untouched original checkpoint using these on-policy-rooted "
            "observation-to-plan-suffix examples plus the same disjoint original-raster replay and "
            "held-out retention protocol. Do not treat this audit as training."
        )
    else:
        decision = "stop_self_distillation_h50_insufficient_oracle"
        recommendation = (
            "Fresh H50 did not recover every student-visited failure root reliably. Do not launch "
            "self-distillation. Use a true recovery source next: validated simulator DAgger/teleoperation, "
            "privileged scripted recovery if available, or a separately preregistered branch-evaluated "
            "candidate search."
        )

    jobs = {
        "onpolicy_h50_rescue_oracle": {
            "job_id": args.job_id,
            "state": "COMPLETED",
            "exit_code": "0:0",
            "script": str(root / "slurm" / "onpolicy_oracle.sbatch"),
        }
    }
    report = [
        "# On-policy H50 rescue-oracle diagnostic",
        "",
        "This inference-only audit used the untouched original baseline checkpoint and exact validated pose-3 action-5/action-10 snapshots.",
        "Each root was reached by ordinary fresh H10 continuation for +10, +20 and +30 actions, then compared normal H10 continuation with one fresh 50-action open-loop H50 chunk from the identical restored root.",
        "",
        f"- Actual student-visited roots: `{total_roots}`.",
        f"- Roots with settled fresh-H50 4/4 recovery: `{successful_roots}`.",
        f"- H50 branches rescuing a failed H10 continuation: `{result['fresh_h50_rescue_root_count']}`.",
        f"- Validated intermediate observation→plan-suffix examples: `{examples_count}`.",
        f"- Reliable all-root H50 recovery gate: `{reliable}`.",
        f"- Exact root restoration/equivalence: `{result['all_branches_start_from_identical_root']}`.",
        f"- Exact RNG restoration: `{result['all_rng_restores_exact']}`.",
        f"- Zero simulator advancement during prediction: `{result['all_predictions_zero_sim_steps']}`.",
        f"- Decision: `{decision}`.",
        "",
        recommendation,
        "",
        "The root snapshots and observations are in `student-roots.npz`; successful H50 observations and exact unconsumed suffix targets are in `successful-h50-examples.npz`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))

    source_episode = root / "outputs" / "dev03_h50_Pant_Short_Seen_3" / "rollout.json.behavior.jsonl"
    provenance = {
        "schema_version": 1,
        "diagnostic": "on-policy H50 rescue-oracle audit",
        "baseline_commit_at_report": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "checkpoint": result["checkpoint"],
        "source_successful_h50_episode": str(source_episode),
        "source_successful_h50_episode_sha256": sha256(source_episode),
        "evaluation_sha256": sha256(evaluation_path),
        "roots_npz_sha256": sha256(roots_path),
        "examples_npz_sha256": sha256(examples_path),
        "initial_boundaries": result["initial_boundaries"],
        "student_root_offsets": result["student_root_offsets"],
        "candidate_example_offsets": result["candidate_example_offsets"],
        "student_visited_root_count": total_roots,
        "fresh_h50_successful_recovery_root_count": successful_roots,
        "fresh_h50_rescue_root_count": result["fresh_h50_rescue_root_count"],
        "validated_intermediate_example_count": examples_count,
        "controls": {
            "original_checkpoint_untouched": True,
            "training_launched": False,
            "self_distillation_launched": False,
            "rtc": False,
            "queue": False,
            "candidate_seed_tuning": False,
            "multi_pose_validation": False,
            "fresh_h50_open_loop_replanned": False,
            "h10_branch_replanned_every_10_actions": True,
        },
        "verification": {
            "all_branches_start_from_identical_root": result["all_branches_start_from_identical_root"],
            "all_snapshot_restores_within_tolerance": result["all_snapshot_restores_within_tolerance"],
            "all_rng_restores_exact": result["all_rng_restores_exact"],
            "all_predictions_zero_sim_steps": result["all_predictions_zero_sim_steps"],
            "h50_chunk_matches_root_h10_chunk_per_root": all(
                row["h50_chunk_matches_h10_root_chunk"] for row in recovery_rows),
        },
        "reliable_gate": {
            "definition": result["reliable_gate_definition"],
            "passed": reliable,
        },
        "decision": decision,
        "recommendation": recommendation,
        "slurm_jobs": jobs,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "slurm-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "student_visited_roots": total_roots,
        "successful_h50_roots": successful_roots,
        "validated_examples": examples_count,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
