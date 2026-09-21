"""Execute one immutable, single-garment media episode from the campaign manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_result(result, row, checkpoint, directory):
    expected = {"mode": "policy", "steps": row["steps"], "completed_steps": row["steps"],
                "policy": str(checkpoint), "garment": row["garment"], "seed": row["seed"],
                "policy_variant": row["policy_variant"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"result {key}={result.get(key)!r}, expected {value!r}")
    if type(result.get("success")) is not bool or type(result.get("terminal_success")) is not bool:
        raise ValueError("checker verdicts must be booleans")
    views = result.get("gif_views", {})
    if set(views) != {"top", "left_wrist", "right_wrist", "triptych"}:
        raise ValueError("all four camera GIFs are required before releasing more jobs")
    for value in views.values():
        path = Path(value)
        path = path if path.is_absolute() else directory / path
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or external camera GIF: {path}")
    if result.get("n_rendered", 0) < row["steps"] + 1 or not result.get("render_integrity"):
        raise ValueError("camera integrity record or final rendered observation missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not 0 <= args.index < len(manifest["tasks"]):
        parser.error("index is outside the manifest")
    row = manifest["tasks"][args.index]
    checkpoint = manifest["checkpoints"][row["policy_variant"]]
    ckpt = Path(checkpoint["path"])
    dest = root / "outputs" / row["id"]
    cmd = [sys.executable, "-u", str(root / "scripts/render/policy_rollout51.py"),
           "--lehome", str(root / "external/lehome-challenge"),
           "--policy_path", str(ckpt), "--garment", row["garment"],
           "--garment_dir", row["garment_dir"], "--assets", manifest["assets"],
           "--steps", str(row["steps"]), "--frames_out", str(dest / "frames"),
           "--match_pose=" + row["match_pose"], "--match_scale", "0.45",
           "--settle_steps", "60", "--sim_device", "cuda:0",
           "--seed", str(row["seed"]), "--n_action_steps", "0",
           "--policy_variant", row["policy_variant"], "--task", "fold the garment",
           "--gif_every", "6", "--result_out", str(dest / "rollout.json")]
    if args.dry_run:
        for name, expected in checkpoint["sha256"].items():
            if file_hash(ckpt / name) != expected:
                raise RuntimeError(f"checkpoint changed since submission: {ckpt / name}")
        print(json.dumps({"task": row, "command": cmd}, indent=2))
        return 0
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "frames").mkdir()
    (dest / "request.json").write_text(json.dumps(row, indent=2) + "\n")
    # A simulator crash is an infrastructure error, never a failed fold.
    status = {"task": row["id"], "returncode": None,
              "state": "infrastructure_error", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
              "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID")}
    try:
        for name, expected in checkpoint["sha256"].items():
            if file_hash(ckpt / name) != expected:
                raise RuntimeError(f"checkpoint changed since submission: {ckpt / name}")
        with open(dest / "rollout.log", "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        status["returncode"] = proc.returncode
        if proc.returncode:
            raise RuntimeError(f"simulator exited {proc.returncode}; read rollout.log")
        result = json.loads((dest / "rollout.json").read_text())
        validate_result(result, row, ckpt, dest)
        status.update(state="completed", success=result["success"])
    except Exception as exc:
        status["error"] = repr(exc)
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0 if status["state"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
