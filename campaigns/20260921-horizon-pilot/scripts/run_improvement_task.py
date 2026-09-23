"""Run one manifest row in a fresh Isaac process with a bounded lifetime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_result(result, row):
    for key, value in {"garment": row["garment"], "seed": row["seed"], "steps": row["steps"],
                       "effective_n_action_steps": row["horizon"], "prediction_chunk_size": 50}.items():
        if result.get(key) != value:
            raise ValueError(f"Result {key} mismatch")
    if not result.get("landmark_mapping", {}).get("runtime_correspondence_verified"):
        raise ValueError("Landmark correspondence is unverified")
    if not result.get("physics_finite") or not result.get("robot_finite"):
        raise ValueError("Invalid physics/robot state")
    if result["terminal_checker"]["success"] != result["terminal_success"]:
        raise ValueError("Terminal success does not match fresh geometry")
    if result.get("terminal_settle_steps") != 60:
        raise ValueError("Terminal settling not completed")
    acquisition = result.get("render_integrity", {}).get("acquisitions", [])
    if len(acquisition) < row["steps"] + 1 or any(a["fresh_camera_count"] != 3 or a["top_garment_pixels"] < 16 for a in acquisition):
        raise ValueError("Insufficient fresh visible garment observations")
    if result["replanning_count"] != (row["steps"] + row["horizon"] - 1) // row["horizon"]:
        raise ValueError("Execution horizon did not produce the expected replanning count")
    telemetry = result.get("behavior_telemetry", {})
    if telemetry.get("steps") != row["steps"] or not telemetry.get("passive"):
        raise ValueError("Passive behavioral telemetry incomplete")
    if len(telemetry["replan_boundary_jumps"]) != result["replanning_count"] - 1:
        raise ValueError("Missing replan boundary metrics")
    if len(Path(telemetry["path"]).read_text().splitlines()) != row["steps"]:
        raise ValueError("Behavior timeline has missing actions")
    for path in [*result["gif_views"].values(), result["media"]["mp4"]["path"]]:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f"Missing policy evidence: {path}")


def run(root, phase, index, dry_run=False):
    manifest = json.loads((root / "manifest.json").read_text())
    row = manifest[phase][index]
    checkpoint = manifest["checkpoint"]
    for rel, expected in manifest["source_sha256"].items():
        if file_hash(root / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    for rel, expected in checkpoint["sha256"].items():
        if file_hash(Path(checkpoint["path"]) / rel) != expected:
            raise ValueError(f"Checkpoint changed: {rel}")
    inventory = json.loads((root / "audit/garment_inventory.json").read_text())
    asset = next(r for r in inventory["garments"] if r["garment_id"] == row["garment"])
    for field in ("mesh", "config"):
        key = "mesh_path" if field == "mesh" else "config"
        if file_hash(asset[key]) != asset[field + "_sha256"]:
            raise ValueError(f"Garment {field} changed")
    if phase == "pilot_tasks":
        gate = json.loads((root / "audit/smoke_gate.json").read_text())
        if gate.get("passed") is not True:
            raise ValueError("Experiment-level smoke gate has not passed")
    certification = json.loads((root / "audit/v5_certification.json").read_text())
    if certification.get("passed") is not True or len(certification["rows"]) != 5:
        raise ValueError("Original v5 certification incomplete")
    for certified in certification["rows"]:
        if file_hash(certified["result_path"]) != certified["result_sha256"]:
            raise ValueError("Certified v5 evidence changed")
    dest = root / "outputs" / row["id"]
    command = [sys.executable, "-u", str(root / "scripts/render/policy_rollout51.py"),
        "--lehome", str(root / "external/lehome-challenge"),
        "--policy_path", checkpoint["path"], "--garment", row["garment"],
        "--garment_dir", str(Path(asset["config"]).parent), "--assets", manifest["assets"],
        "--steps", str(row["steps"]), "--frames_out", str(dest / "frames"),
        "--match_pose=" + row["match_pose"], "--match_scale", str(row["match_scale"]),
        "--settle_steps", "60", "--terminal_settle_steps", "60",
        "--seed", str(row["seed"]), "--n_action_steps", str(row["horizon"]),
        "--policy_variant", "historical_raster_fixed", "--task", manifest["task_prompt"],
        "--gif_every", "6", "--result_out", str(dest / "rollout.json")]
    if row.get("switch_to"):
        command += ["--switch_to", row["switch_to"]]
    if dry_run:
        print(json.dumps({"row": row, "command": command}, indent=2))
        return 0
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "frames").mkdir()
    (dest / "request.json").write_text(json.dumps({"row": row, "command": command,
        "checkpoint": checkpoint, "source_sha256": manifest["source_sha256"]}, indent=2) + "\n")
    started = time.monotonic()
    status = {"task": row["id"], "state": "running", "success": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "process_isolated": True, "infrastructure_error": None}
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    with open(dest / "rollout.log", "w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status["child_pid"] = proc.pid
        try:
            proc.wait(timeout=480 if phase == "smoke_tasks" else 780)
            status["returncode"] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f"Isaac child exited {proc.returncode}")
            result = json.loads((dest / "rollout.json").read_text())
            validate_result(result, row)
            status.update(state="completed", success=result["success"], terminal_success=result["terminal_success"])
        except subprocess.TimeoutExpired:
            # This is exclusively the process group created for this row.
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status.update(state="infrastructure_timeout", infrastructure_error="bounded Isaac process timeout", success=None)
        except Exception as exc:
            status.update(state="infrastructure_error", infrastructure_error=repr(exc), success=None)
    status["wall_seconds"] = time.monotonic() - started
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0 if status["state"] == "completed" else 3


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--phase", choices=["smoke_tasks", "pilot_tasks"], required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    raise SystemExit(run(args.campaign.resolve(), args.phase, args.index, args.dry_run))
