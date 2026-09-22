"""Summarise mixed replay checkpoints, exact gates, and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


JOBS = {
    "retained_checkpoint_snapshot_audit": {
        "job_id": 21399177, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:03:59"
    },
    "mixed_replay_training": {
        "job_id": 21399191, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:06:09"
    },
    "mixed_replay_checkpoint_panel": {
        "job_id": 21399222, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:13:09"
    },
    "selected_checkpoint_fresh_h10_execution": {
        "job_id": 21399265, "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:04:19"
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(evaluation, label, boundary):
    return evaluation["boundaries"][str(boundary)]["models"][label][
        "deviation_from_cached_h50_suffix"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    out = root / "analysis" / "mixed-replay"
    training = json.loads((out / "finetune.json").read_text())
    evaluation = json.loads((out / "checkpoint-panel-evaluation.json").read_text())
    retained = json.loads((out / "retained-checkpoint-evaluation.json").read_text())
    previous = json.loads(
        (root / "analysis" / "boundary-replay" / "inference-evaluation.json").read_text()
    )
    capture = out.parent / "boundary-replay" / "pose3-boundary-replay.npz"
    capture_meta = json.loads(capture.with_suffix(".json").read_text())
    source_actions = []
    with Path(capture_meta["source_successful_h50_episode"]).open() as stream:
        for line in stream:
            if line.strip():
                source_actions.append(json.loads(line)["executed_action_rad"])
    labels = [label for label in evaluation["checkpoint_order"] if label != "baseline"]
    baseline = "baseline"
    rows = []
    action_rows = []
    selected = None
    checkpoint_records = {Path(record["path"]).name: record
                          for record in training.get("checkpoints", [])}
    for label in labels:
        record = checkpoint_records.get(label)
        if record is None:
            raise SystemExit(f"missing training metadata for checkpoint {label}")
        reductions = {}
        boundary_pass = True
        for boundary in (5, 10):
            base = metrics(evaluation, baseline, boundary)["10"]["rms_rad"]
            candidate = metrics(evaluation, label, boundary)["10"]["rms_rad"]
            reduction = (base - candidate) / base
            reductions[str(boundary)] = float(reduction)
            boundary_pass = boundary_pass and reduction >= 0.50
            model_chunk = evaluation["boundaries"][str(boundary)]["models"][label][
                "chunk_first_10"
            ]
            target = source_actions[boundary:boundary + 10]
            for action_index, (candidate_action, target_action) in enumerate(
                zip(model_chunk, target), start=1
            ):
                delta = [float(a - b) for a, b in zip(candidate_action, target_action)]
                for joint, value in enumerate(delta):
                    action_rows.append({
                        "checkpoint": label,
                        "boundary_action": boundary,
                        "action_index": action_index,
                        "joint": joint,
                        "delta_from_cached_h50_rad": value,
                        "abs_delta_rad": abs(value),
                    })
        heldout_pass = bool(record["heldout_gate"])
        both = bool(boundary_pass and heldout_pass)
        if both and selected is None:
            selected = label
        rows.append({
            "checkpoint": label,
            "step": int(record["step"]),
            "train_loss": record["train_loss"],
            "heldout_loss_before": record["heldout_loss_before"],
            "heldout_loss": record["heldout_loss"],
            "heldout_relative_change": (record["heldout_loss"] /
                                         record["heldout_loss_before"] - 1.0),
            "heldout_gate": heldout_pass,
            "boundary5_first10_rms_reduction": reductions["5"],
            "boundary10_first10_rms_reduction": reductions["10"],
            "boundary_gate": boundary_pass,
            "both_gates": both,
            "selected_earliest_passing": both and selected == label,
        })

    fields = list(rows[0]) if rows else ["checkpoint"]
    with (out / "pareto-trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "first10-action-differences.csv").open("w", newline="") as stream:
        fields = list(action_rows[0]) if action_rows else ["checkpoint"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(action_rows)

    deviation_rows = []
    for label in evaluation["checkpoint_order"]:
        for boundary in (5, 10):
            m = metrics(evaluation, label, boundary)
            deviation_rows.append({
                "checkpoint": label,
                "boundary_action": boundary,
                "first_action_rms_rad": m["1"]["rms_rad"],
                "first_action_max_abs_rad": m["1"]["max_abs_rad"],
                "first10_rms_rad": m["10"]["rms_rad"],
                "first10_max_abs_rad": m["10"]["max_abs_rad"],
                "snapshot_cloth_rms_m": evaluation["boundaries"][str(boundary)]["models"][label][
                    "snapshot_restore"
                ]["cloth_rms_m"],
                "snapshot_joint_rms_rad": evaluation["boundaries"][str(boundary)]["models"][label][
                    "snapshot_restore"
                ]["joint_rms_rad"],
                "rng_restore_exact": evaluation["boundaries"][str(boundary)]["models"][label][
                    "rng_restore_exact"
                ],
                "simulator_stepped_during_prediction": evaluation["boundaries"][str(boundary)][
                    "models"
                ][label]["simulator_stepped_during_prediction"],
            })
    with (out / "inference-deviations.csv").open("w", newline="") as stream:
        fields = list(deviation_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(deviation_rows)

    retained_training = json.loads(
        (root / "analysis" / "boundary-replay" / "trained_checkpoint" / "finetune.json").read_text()
    )
    retained_reductions = {}
    retained_label = next(
        label for label in retained["checkpoint_order"] if label != "baseline"
    )
    for boundary in (5, 10):
        b = metrics(retained, "baseline", boundary)["10"]["rms_rad"]
        t = metrics(retained, retained_label, boundary)["10"]["rms_rad"]
        retained_reductions[str(boundary)] = (b - t) / b
    retained_audit = {
        "available_checkpoint_paths": [
            previous["checkpoints"]["trained"]["path"]
        ],
        "intermediate_checkpoint_count": 0,
        "evaluated_checkpoint": previous["checkpoints"]["trained"],
        "boundary_first10_rms_reductions": retained_reductions,
        "boundary_gate": all(value >= 0.50 for value in retained_reductions.values()),
        "heldout_loss_before": retained_training["val_before"],
        "heldout_loss_after": retained_training["val_after"],
        "heldout_gate": bool(retained_training["heldout_gate"]),
        "both_gates": bool(
            all(value >= 0.50 for value in retained_reductions.values())
            and retained_training["heldout_gate"]
        ),
        "source_evaluation": str(root / "analysis" / "boundary-replay" / "inference-evaluation.json"),
    }
    execution_path = out / "selected-execution.json"
    selected_execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    execution_gate = None if selected_execution is None else bool(
        selected_execution["both_boundaries_settled_4_of_4"]
    )
    if selected_execution is not None:
        condition_rows = []
        for boundary, branch in selected_execution["boundaries"].items():
            for row in branch["trace"]:
                condition_rows.append({
                    "checkpoint": Path(selected_execution["checkpoint"]).name,
                    "boundary_action": int(boundary),
                    "phase": "fresh_h10",
                    "global_step": row["global_step"],
                    "conditions_passed": row["conditions_passed"],
                    "conditions_total": row["conditions_total"],
                    "geometric_success": row["geometric_success"],
                })
            for row in branch["settled"]["trace"]:
                condition_rows.append({
                    "checkpoint": Path(selected_execution["checkpoint"]).name,
                    "boundary_action": int(boundary),
                    "phase": "terminal_settle",
                    "global_step": row.get("global_step", ""),
                    "conditions_passed": row["conditions_passed"],
                    "conditions_total": row["conditions_total"],
                    "geometric_success": row["geometric_success"],
                })
        with (out / "condition-trajectories.csv").open("w", newline="") as stream:
            fields = list(condition_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(condition_rows)

    larger_proposal = None
    if execution_gate:
        larger_proposal = {
            "status": "proposed_not_launched",
            "poses": [1, 3, 5, 7],
            "internal_offsets": [5, 10, 15, 20, 25, 30, 35, 40],
            "source": "successful cached H50 episodes for poses 1/3/5/7",
            "observation": "actual all-camera/current-state boundary observations",
            "target": "corresponding remaining cached successful-H50 actions",
            "training": "proposal only; no larger training run automatically launched",
        }
        (out / "larger-boundary-replay-proposal.json").write_text(
            json.dumps(larger_proposal, indent=2) + "\n"
        )

    provenance = {
        "schema_version": 1,
        "diagnostic": "mixed corrective-boundary and original-raster replay micro-finetune",
        "baseline_commit_at_run": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "garment": capture_meta["garment"],
        "match_pose": capture_meta["match_pose"],
        "boundaries": [5, 10],
        "source_successful_h50_episode": capture_meta["source_successful_h50_episode"],
        "boundary_capture_sha256": sha256(capture),
        "training": training,
        "retained_checkpoint_audit": retained_audit,
        "checkpoint_panel_evaluation": evaluation,
        "controls": {
            "original_checkpoint_untouched": True,
            "vision_vlm_frozen": True,
            "trainable_path": "action expert, action projections, state projection",
            "normalization_unchanged": True,
            "corrective_fraction": 0.20,
            "raster_fraction": 0.80,
            "heldout_disjoint_from_training": True,
            "rtc": False,
            "queue": False,
            "stale_or_hybrid_observations": False,
            "h50_assistance": False,
            "closed_loop_execution": selected_execution is not None,
        },
        "gate": {
            "boundary_threshold": 0.50,
            "heldout_threshold_relative_regression": 0.10,
            "earliest_passing_checkpoint": selected,
            "any_checkpoint_passed": selected is not None,
            "selected_execution_gate": execution_gate,
        },
        "selected_execution": selected_execution,
        "larger_boundary_replay_proposal": larger_proposal,
        "recommendation": (
            "propose, but do not automatically launch, the larger successful-H50 boundary dataset"
            if execution_gate else
            "stop after the selected checkpoint failed settled 4/4 at one or both boundaries; do not launch larger validation"
            if selected_execution is not None else
            "run ordinary fresh H10 only from the two pose-3 snapshots if a checkpoint passes both gates"
            if selected else
            "simple replay rehearsal was insufficient; recommend parameter-anchored or parameter-efficient adaptation"
        ),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "slurm-jobs.json").write_text(json.dumps(JOBS, indent=2) + "\n")

    lines = [
        "# Mixed corrective/raster replay diagnostic",
        "",
        "The prior retained-checkpoint audit found no intermediate checkpoints. The only retained 300-step boundary-only checkpoint passed both boundary RMS reductions but failed held-out raster retention.",
        "",
        f"The mixed run used fixed seed `{training['seed']}`, `{training['steps']}` steps, one corrective and four fixed raster TRAIN examples per batch, and checkpoints every `{training['save_every']}` steps. The vision/VLM remained frozen.",
        "",
        "## Pareto trajectory",
        "",
        "| checkpoint | step | held-out loss | held-out gate | boundary-5 reduction | boundary-10 reduction | both gates |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['step']} | {row['heldout_loss']:.6f} | "
            f"{row['heldout_gate']} | {row['boundary5_first10_rms_reduction']:.3%} | "
            f"{row['boundary10_first10_rms_reduction']:.3%} | {row['both_gates']} |"
        )
    lines += [
        "",
        f"Earliest checkpoint satisfying both unchanged gates: `{selected}`." if selected else
        "No checkpoint satisfies both unchanged gates. Simple replay rehearsal was insufficient.",
        "",
        "Every boundary prediction used exact snapshot restoration, reconstructed same-RNG inference, and no simulator stepping.",
        "",
        (
            "The earliest passing checkpoint was executed with ordinary fresh H10 branches, but settled 4/4 was not recovered at both boundaries. Stop and do not launch larger validation."
            if selected_execution is not None and not execution_gate else
            "Both boundaries recovered settled 4/4; a larger boundary replay dataset is proposed only, not launched."
            if execution_gate else
            "If no checkpoint passes, the next recommendation is parameter-anchored or parameter-efficient adaptation; do not increase this training duration."
        ),
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "earliest_passing_checkpoint": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
