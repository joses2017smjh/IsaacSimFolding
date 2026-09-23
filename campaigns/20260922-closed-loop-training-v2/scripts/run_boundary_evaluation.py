"""Evaluate one checkpoint at the two validated pose-3 H10 replanning roots.

This delegates exact snapshot reconstruction and its existing restoration/RNG/
zero-step assertions to the prior validated controller.  Candidate branches
use ordinary fresh H10 replanning; the cached H50 stream is only the frozen
source needed to reconstruct and control the two roots.
"""
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


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    candidate = args.checkpoint.resolve()
    if not (candidate / "model.safetensors").is_file():
        raise SystemExit("candidate checkpoint has no model.safetensors")
    label = args.label.replace("/", "_")
    output = root / "evaluation" / label / "boundary"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    pilot = root.parent / "20260921-horizon-pilot"
    behavior = pilot / "outputs" / "dev03_h50_Pant_Short_Seen_3" / "rollout.json.behavior.jsonl"
    runner = pilot / "scripts" / "render" / "policy_rollout51.py"
    asset = Path(manifest["assets"]) / "objects/Challenge_Garment/Release/Pant_Short/Pant_Short_Seen_3/PS_050_obj_exp.json"
    if not behavior.is_file() or not runner.is_file() or not asset.is_file():
        raise SystemExit("validated pose-3 source artifacts are unavailable")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    command = [
        sys.executable, "-u", str(runner),
        "--lehome", manifest["lehome"],
        "--policy_path", manifest["baseline_checkpoint"]["path"],
        "--garment", "Pant_Short_Seen_3", "--garment_dir", str(asset.parent),
        "--assets", manifest["assets"], "--steps", "10", "--frames_out", str(output / "frames"),
        "--match_pose=-0.02527160197:0.02281407639:0.6700000167:0.5693775415:4.805557728:90",
        "--match_scale", "0.45", "--settle_steps", "60", "--terminal_settle_steps", "60",
        "--seed", "203", "--n_action_steps", "50", "--policy_variant", f"closed_loop_boundary_{label}",
        "--task", manifest["task_prompt"], "--gif_every", "12", "--result_out", str(output / "source-root.json"),
        "--causal_out", str(output / "replan-causality.json"), "--causal_branch_steps", "120",
        "--causal_action_jsonl", str(behavior),
        "--boundary_execute_trained_path", str(candidate),
        "--boundary_execute_out", str(output / "boundary-execution.json"),
    ]
    request = {
        "candidate": str(candidate), "candidate_model_sha256": digest(candidate / "model.safetensors"),
        "source_cached_h50": str(behavior), "source_cached_h50_sha256": digest(behavior),
        "semantics": "ordinary fresh H10 in candidate branches; cached H50 used only for source-root reconstruction/control",
        "command": command, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    status = {"state": "running", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "created_utc": request["created_utc"]}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    # Bound the child explicitly. An unbounded subprocess.run here would let a
    # hung Isaac process consume the whole Slurm allocation and then be killed
    # from outside, leaving status.json permanently reading "running".
    timeout = float(manifest["budget"]["rollout_timeout_seconds"]["boundary"])
    with (output / "boundary.log").open("w") as stream:
        proc = subprocess.Popen(command, cwd=manifest["lehome"], stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        status["child_pid"] = proc.pid
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status.update(state="infrastructure_timeout",
                          error=f"boundary child exceeded {timeout:.0f}s")
    if status["state"] == "infrastructure_timeout":
        pass
    elif proc.returncode:
        status.update(state="error", returncode=proc.returncode)
    elif not (output / "boundary-execution.json").is_file():
        status.update(state="error", error="controller did not write boundary execution output")
    else:
        status.update(state="completed", returncode=0,
                      output_sha256=digest(output / "boundary-execution.json"))
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))
    return 0 if status["state"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
