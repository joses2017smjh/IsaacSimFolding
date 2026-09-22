"""Run one conditional hard-queue validation pose in an isolated Isaac process."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SLOTS = (1, 3, 5, 7)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_result(result, row, delay):
    expected = {"garment": row["garment"], "seed": row["seed"], "steps": row["steps"],
                "effective_n_action_steps": 10, "prediction_chunk_size": 50,
                "hard_retained_prefix_delay": delay}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Result {key} mismatch: {result.get(key)!r} != {value!r}")
    hard = result.get("hard_queue") or {}
    if not hard.get("enabled") or hard.get("delay") != delay:
        raise ValueError("hard queue metadata missing or mismatched")
    if result.get("replanning_count") != 60:
        raise ValueError("hard H10 validation did not produce 60 predictions")
    if result.get("terminal_settle_steps") != 60:
        raise ValueError("terminal settling not completed")
    for timing in result.get("prediction_timing_audit", []):
        if timing.get("sim_step_delta") != 0 or timing.get("episode_length_delta") not in (0, None):
            raise ValueError("prediction advanced simulator state")
    for path in [*result["gif_views"].values(), result["media"]["mp4"]["path"]]:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f"Missing policy evidence: {path}")


def run(root, index, delay):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = sorted((r for r in manifest["pilot_tasks"] if int(r.get("horizon", -1)) == 10
                   and int(r.get("development_pose_slot", -1)) in SLOTS),
                  key=lambda r: int(r["development_pose_slot"]))
    if len(rows) != 4 or not 0 <= index < len(rows):
        raise ValueError("conditional validation must contain exactly pose slots 1/3/5/7")
    row = rows[index]
    checkpoint = manifest["checkpoint"]
    source_hashes = {}
    for rel, expected in manifest["source_sha256"].items():
        if rel == "scripts/render/policy_rollout51.py":
            # This is the deliberately changed diagnostic controller. Its
            # actual hash is recorded below rather than compared to the frozen
            # historical pilot source hash.
            source_hashes[rel] = file_hash(root / rel)
            continue
        if file_hash(root / rel) != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    for rel, expected in checkpoint["sha256"].items():
        if file_hash(Path(checkpoint["path"]) / rel) != expected:
            raise ValueError(f"Checkpoint changed: {rel}")
    inventory = json.loads((root / "audit/garment_inventory.json").read_text())
    asset = next(r for r in inventory["garments"] if r["garment_id"] == row["garment"])
    dest = root / "outputs" / f"queue_validation_d{delay}_slot{row['development_pose_slot']}_{row['garment']}"
    command = [sys.executable, "-u", str(root / "scripts/render/policy_rollout51.py"),
               "--lehome", str(root / "external/lehome-challenge"),
               "--policy_path", checkpoint["path"], "--garment", row["garment"],
               "--garment_dir", str(Path(asset["config"]).parent), "--assets", manifest["assets"],
               "--steps", str(row["steps"]), "--frames_out", str(dest / "frames"),
               "--match_pose=" + row["match_pose"], "--match_scale", str(row["match_scale"]),
               "--settle_steps", "60", "--terminal_settle_steps", "60",
               "--seed", str(row["seed"]), "--n_action_steps", "50",
               "--hard_queue_delay", str(delay),
               "--policy_variant", f"hard_queue_validation_d{delay}",
               "--task", manifest["task_prompt"], "--gif_every", "6",
               "--result_out", str(dest / "rollout.json")]
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "frames").mkdir()
    (dest / "request.json").write_text(json.dumps({"row": row, "delay": delay, "command": command,
                                                     "source_hashes": source_hashes}, indent=2) + "\n")
    status = {"task": row["id"], "validation_pose_slot": row["development_pose_slot"],
              "delay": delay, "state": "running", "success": None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "process_isolated": True, "infrastructure_error": None}
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    with (dest / "rollout.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=900)
            status["returncode"] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f"Isaac child exited {proc.returncode}")
            result = json.loads((dest / "rollout.json").read_text())
            validate_result(result, row, delay)
            status.update(state="completed", success=result["success"], terminal_success=result["terminal_success"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status.update(state="infrastructure_timeout", infrastructure_error="bounded Isaac process timeout")
        except Exception as exc:
            status.update(state="infrastructure_error", infrastructure_error=repr(exc))
    status["wall_seconds"] = time.monotonic() - started
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0 if status["state"] == "completed" else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--delay", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.campaign.resolve(), args.index, args.delay))
