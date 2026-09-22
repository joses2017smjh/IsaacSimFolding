"""Build compact machine-readable outputs for the boundary replay diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


JOBS = {
    "capture_bind_failure": {"job_id": 21398508, "state": "FAILED", "exit_code": "1:0"},
    "capture_off_by_one_failure": {"job_id": 21398509, "state": "FAILED", "exit_code": "3:0"},
    "capture_wrong_campaign_root_failure": {"job_id": 21398510, "state": "FAILED", "exit_code": "2:0"},
    "boundary_capture": {"job_id": 21398513, "state": "COMPLETED", "exit_code": "0:0"},
    "boundary_micro_finetune": {"job_id": 21398555, "state": "COMPLETED", "exit_code": "0:0"},
    "boundary_eval_loader_failure": {"job_id": 21398565, "state": "FAILED", "exit_code": "3:0"},
    "boundary_eval_loader_failure_rerun": {"job_id": 21398572, "state": "FAILED", "exit_code": "3:0"},
    "boundary_snapshot_inference_gate": {"job_id": 21399086, "state": "COMPLETED", "exit_code": "0:0"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    out = root / "analysis" / "boundary-replay"
    out.mkdir(parents=True, exist_ok=True)
    capture = out / "pose3-boundary-replay.npz"
    capture_meta = json.loads(capture.with_suffix(".json").read_text())
    training = json.loads((out / "trained_checkpoint" / "finetune.json").read_text())
    evaluation = json.loads((out / "inference-evaluation.json").read_text())
    source_result_path = root / "outputs" / "dev03_h50_Pant_Short_Seen_3" / "rollout.json"
    source_result = json.loads(source_result_path.read_text())
    source_actions = []
    with Path(capture_meta["source_successful_h50_episode"]).open() as stream:
        for line in stream:
            if line.strip():
                source_actions.append(json.loads(line)["executed_action_rad"])

    rows = []
    action_differences = []
    for boundary, boundary_row in evaluation["boundaries"].items():
        for label, model_row in boundary_row["models"].items():
            metrics = model_row["deviation_from_cached_h50_suffix"]
            rows.append({
                "boundary_action": int(boundary),
                "model": label,
                "first_action_rms_rad": metrics["1"]["rms_rad"],
                "first_action_max_abs_rad": metrics["1"]["max_abs_rad"],
                "first10_rms_rad": metrics["10"]["rms_rad"],
                "first10_max_abs_rad": metrics["10"]["max_abs_rad"],
                "snapshot_cloth_rms_m": model_row["snapshot_restore"]["cloth_rms_m"],
                "snapshot_joint_rms_rad": model_row["snapshot_restore"]["joint_rms_rad"],
                "rng_restore_exact": model_row["rng_restore_exact"],
                "simulator_stepped_during_prediction": model_row["simulator_stepped_during_prediction"],
            })
            chunk = model_row["chunk_first_10"]
            target = source_actions[int(boundary):int(boundary) + 10]
            for action_index, action in enumerate(chunk):
                for joint, value in enumerate(action):
                    action_differences.append({
                        "boundary_action": int(boundary),
                        "model": label,
                        "action_index_in_first10": action_index,
                        "joint": joint,
                        "delta_rad": value - target[action_index][joint],
                        "predicted_action_rad": value,
                    })

    reductions = {}
    for boundary in ("5", "10"):
        base = evaluation["boundaries"][boundary]["models"]["baseline"]
        trained = evaluation["boundaries"][boundary]["models"]["trained"]
        base_rms = base["deviation_from_cached_h50_suffix"]["10"]["rms_rad"]
        trained_rms = trained["deviation_from_cached_h50_suffix"]["10"]["rms_rad"]
        reductions[boundary] = {
            "baseline_first10_rms_rad": base_rms,
            "trained_first10_rms_rad": trained_rms,
            "relative_reduction": 1.0 - trained_rms / base_rms,
            "passes_50_percent_inference_gate": bool(1.0 - trained_rms / base_rms >= 0.5),
        }
    inference_gate = all(x["passes_50_percent_inference_gate"] for x in reductions.values())
    heldout_gate = bool(training["heldout_gate"])

    with (out / "inference-deviations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "first10-action-differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(action_differences[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(action_differences)

    condition_rows = []
    for row in source_result["geometry_trajectory"]:
        condition_rows.append({
            "condition_source": "validated_cached_H50_source_episode",
            "closed_loop_candidate": "none",
            "step": row["step"],
            "phase": row["phase"],
            "conditions_passed": row["conditions_passed"],
            "conditions_total": row["conditions_total"],
            "success": row["success"],
            "note": "trained closed-loop execution was not authorized because the held-out inference gate failed",
        })
    with (out / "condition-trajectories.csv").open("w", newline="") as stream:
        fields = list(condition_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(condition_rows)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    provenance = {
        "schema_version": 1,
        "diagnostic": "boundary-state successful-H50 replay micro-finetune",
        "baseline_commit_at_run": commit,
        "garment": capture_meta["garment"],
        "match_pose": capture_meta["match_pose"],
        "task": capture_meta["task"],
        "boundaries": [5, 10],
        "source_successful_h50_episode": capture_meta["source_successful_h50_episode"],
        "source_successful_h50_episode_sha256": capture_meta["source_successful_h50_episode_sha256"],
        "capture_npz_sha256": sha256(capture),
        "capture_metadata": capture_meta,
        "checkpoint_controls": evaluation["checkpoints"],
        "training": training,
        "evaluation": evaluation,
        "gate": {
            "preregistered_inference_relative_first10_rms_reduction": 0.5,
            "preregistered_heldout_max_regression_fraction": 0.10,
            "boundary_reductions": reductions,
            "inference_gate_passed": inference_gate,
            "heldout_gate_passed": heldout_gate,
            "overall_gate_passed": inference_gate and heldout_gate,
        },
        "controls": {
            "cached_h50": True,
            "same_rng_fresh_replanning": True,
            "rtc": False,
            "retained_prefix_queue": False,
            "observation_substitution": False,
            "closed_loop_candidate_execution": False,
            "larger_pose_boundary_dataset": False,
            "larger_validation_submitted": False,
        },
        "diagnosis": {
            "primary": "optimization/generalization overfit to two corrective examples",
            "dataset_plumbing": "passed exact two-example schema and source alignment assertions",
            "target_alignment": "passed cached H50 suffix equality at both boundaries",
            "normalization": "passed unchanged checkpoint pre/postprocessor provenance",
            "trainable_capacity": "not indicated as primary; action/state path reduced both boundary RMS values substantially",
            "reason_for_stop": "held-out fixed raster gate failed despite both boundary inference reductions",
            "recommendation": "stop this intervention; next diagnostic should be a small boundary-state chunk-consistency/on-policy corrective-data experiment with a held-out-preserving objective",
        },
        "slurm_jobs": JOBS,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "slurm-jobs.json").write_text(json.dumps(JOBS, indent=2) + "\n")
    print(json.dumps({"out": str(out), "inference_gate": inference_gate,
                      "heldout_gate": heldout_gate, "overall_gate": inference_gate and heldout_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
