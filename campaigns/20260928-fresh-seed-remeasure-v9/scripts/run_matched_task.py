"""Run one predeclared matched-comparison row in an isolated Isaac process.

v9 copy of v8's matched runner, with three changes:

1. The only phase is "remeasure": 48 fresh-seed development rows.
2. Both policies are pinned in the manifest (the candidate is v8's
   an1-step000300, byte for byte). Nothing unpinned can run.
3. Each task belongs to one GPU-model block (--gpu-model, pinned by the
   driver's Slurm --constraint). The allocated GPU is checked before Isaac
   starts, and the GPU Isaac reports in rollout.log is checked after it ends.
   A task on the wrong model is an infrastructure error, never a policy
   failure; the driver retries it once, outcome-blind.

--matched-index N is the single source of truth for the interleaving that
makes a run "same wave": even N runs the baseline, odd N the candidate, on
row N // 2, so ONE Slurm array holds both policies and each row's two
episodes start adjacently in the same queue and node pool. (Concurrency is
set by the partition QoS: at most 5 tasks at 8 CPUs on gpu, 2 GPUs on
ampere.) The driver reads results through the same function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from analysis import GPU_NAMES, gpu_model

PHASES = ("remeasure",)
POLICIES = ("baseline", "candidate")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_record(path: Path) -> dict[str, str]:
    required = ("config.json", "model.safetensors", "policy_preprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint {path} lacks {missing}")
    return {name: digest(path / name) for name in required}


def matched_index(index: int) -> tuple[str, int]:
    """Even array indices run the baseline, odd the candidate, on row index // 2."""
    if index < 0:
        raise ValueError("matched index must be non-negative")
    return POLICIES[index % 2], index // 2


def checkpoint_block(root: Path, manifest: dict, which: str) -> dict:
    """The manifest's pinned record for one policy (both are pinned at freeze)."""
    block = manifest["baseline_checkpoint" if which == "baseline" else "candidate_checkpoint"]
    if not {"path", "label", "sha256"} <= set(block):
        raise SystemExit(f"{which} record lacks path, label or sha256")
    return block


def allocated_gpu_names() -> list[str] | None:
    """Names nvidia-smi reports for the GPUs this task can see; None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def destination(root: Path, phase: str, row: dict, label: str) -> Path:
    if phase == "smoke":
        # Both policies smoke the same row; the label keeps them apart.
        return root / "smoke" / label / row["id"]
    return root / "evaluation" / label / phase / row["id"]


def validate_result(result: dict, row: dict, *, capture: bool, raw_path: Path | None) -> None:
    expected = {
        "garment": row["garment"],
        "seed": row["seed"],
        "steps": row["steps"],
        # Development rows run at both horizons; the row says which.
        "effective_n_action_steps": int(row.get("horizon", 10)),
        "prediction_chunk_size": 50,
        "terminal_settle_steps": 60,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"result {key}={result.get(key)!r}, expected {value!r}")
    if not result.get("physics_finite") or not result.get("robot_finite"):
        raise ValueError("non-finite simulator state")
    terminal = result.get("terminal_checker", {})
    if terminal.get("success") != result.get("terminal_success"):
        raise ValueError("terminal checker and terminal success disagree")
    if not capture:
        return
    if raw_path is None or not raw_path.is_file():
        raise ValueError("smoke did not persist trajectory data")
    with np.load(raw_path, allow_pickle=False) as raw:
        required = {"step_index", "images", "state", "executed_action",
                    "executed_action_stream", "camera_keys", "success",
                    "terminal_success", "seed"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"trajectory missing {sorted(missing)}")
        steps = np.asarray(raw["step_index"], dtype=np.int64)
        stream = np.asarray(raw["executed_action_stream"], dtype=np.float32)
        if (steps.ndim != 1 or len(steps) == 0 or np.any(np.diff(steps) <= 0)
                or steps[0] != 0):
            raise ValueError("trajectory observation indices are not increasing from zero")
        if stream.shape != (row["steps"], 12) or not np.isfinite(stream).all():
            raise ValueError(f"invalid full executed action stream {stream.shape}")
        if np.asarray(raw["images"]).shape[1:] != (3, 480, 640, 3):
            raise ValueError("trajectory camera shape changed")
        if np.asarray(raw["state"]).shape[1:] != (12,):
            raise ValueError("trajectory state shape changed")
        if int(np.asarray(raw["seed"]).item()) != row["seed"]:
            raise ValueError("trajectory seed differs from manifest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--matched-index", type=int, required=True,
                    help="array index: even = baseline, odd = candidate, row = index // 2")
    ap.add_argument("--run", required=True,
                    help="run label appended to the checkpoint label, e.g. r1")
    ap.add_argument("--gpu-model", required=True, choices=sorted(GPU_NAMES),
                    help="the block's pinned GPU model; a task on any other model is an infrastructure error")
    ap.add_argument("--manifest-sha256", required=True,
                    help="digest of the manifest the driver submitted under; the task refuses any other")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.campaign.resolve()
    manifest_sha = digest(root / "manifest.json")
    if manifest_sha != args.manifest_sha256:
        raise SystemExit(f"manifest.json is {manifest_sha[:12]}, the driver submitted under "
                         f"{args.manifest_sha256[:12]}; refusing to run under a different protocol")
    manifest = json.loads((root / "manifest.json").read_text())
    # Every pinned source is re-verified at task start, not only at the
    # driver's preflight: the a40 block may start days after launch.
    repo = root.parents[1]
    drift = sorted(rel for rel, sha in manifest["executed_sources"].items()
                   if not (repo / rel).is_file() or digest(repo / rel) != sha)
    if drift:
        raise SystemExit(f"executed sources differ from the manifest: {drift}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", args.run):
        raise SystemExit(f"invalid run label {args.run!r}")
    rows = manifest[args.phase]
    which, row_index = matched_index(args.matched_index)
    if not 0 <= row_index < len(rows):
        raise SystemExit(f"{args.phase} matched index {args.matched_index} -> row {row_index} "
                         f"is out of range for {len(rows)} rows")
    row = rows[row_index]
    block = checkpoint_block(root, manifest, which)
    label = f"{block['label']}-{args.run}"
    checkpoint = Path(block["path"]).resolve()
    checkpoint_record(checkpoint)   # the four files the runner cannot start without
    # Both checkpoints are declared immutable: EVERY file the manifest pinned is
    # checked (all eight, including the postprocessor the runner loads), because a
    # changed normalizer or processor silently shifts the action space measured here.
    pinned = block["sha256"]
    missing = sorted(name for name in pinned if not (checkpoint / name).is_file())
    ckpt = {name: digest(checkpoint / name) for name in pinned if name not in missing}
    drifted = sorted(name for name, sha in ckpt.items() if sha != pinned[name])
    if drifted or missing:
        raise SystemExit(f"immutable {which} checkpoint changed: altered={drifted} missing={missing}")

    dest = destination(root, args.phase, row, label)
    if dest.exists() and not args.dry_run:
        # A dry run writes nothing, so an existing output is information for
        # the caller (recorded below), not grounds to refuse an inspection.
        raise SystemExit(f"refusing to overwrite retained output {dest}")
    runner = root / "scripts" / "render" / "policy_rollout51.py"
    lehome = Path(manifest["lehome"])
    if not runner.is_file() or not lehome.is_dir() or not Path(row["asset_config"]).is_file():
        raise SystemExit("validated runner, LeHome checkout, or garment config is unavailable")
    expected_runner = manifest["executed_sources"].get(manifest["runner_key"])
    if expected_runner and digest(runner) != expected_runner:
        raise SystemExit(
            f"runner {runner} does not match the manifest-pinned source hash; "
            f"the campaign snapshot and the executed code have diverged")

    capture = args.phase == "smoke"
    raw = dest / "trajectory.npz" if capture else None
    # gif_every only selects which already-rendered frames are KEPT for the
    # GIF/MP4 (runner: `if i % gif_every == 0: keep_frame`); the policy
    # observes every rendered frame regardless, so density is cosmetic.
    gif_every = int(manifest.get("media", {}).get("gif_every", 12))
    command = [
        sys.executable, "-u", str(runner),
        "--lehome", str(lehome),
        "--policy_path", str(checkpoint),
        "--garment", row["garment"],
        "--garment_dir", str(Path(row["asset_config"]).parent),
        "--assets", manifest["assets"],
        "--steps", str(row["steps"]),
        "--frames_out", str(dest / "frames"),
        "--match_pose=" + row["match_pose"],
        "--match_scale", str(row["match_scale"]),
        "--settle_steps", "60", "--terminal_settle_steps", "60",
        "--seed", str(row["seed"]), "--n_action_steps", str(row.get("horizon", 10)),
        "--policy_variant", f"matched_{args.phase}_{label}",
        "--task", manifest["task_prompt"], "--gif_every", str(gif_every),
        "--result_out", str(dest / "rollout.json"),
    ]
    if capture:
        command.extend(["--trajectory_out", str(raw), "--trajectory_every",
                        str(row["trajectory_every"])])
    request = {
        "phase": args.phase,
        "policy": which,
        "matched_index": args.matched_index,
        "run": args.run,
        "row": row,
        "checkpoint": {"path": str(checkpoint), "label": block["label"], "sha256": ckpt},
        "runner": {"path": str(runner), "sha256": digest(runner)},
        "campaign_commit": manifest["git"]["commit"],
        "manifest_sha256": manifest_sha,
        "host": socket.gethostname(),
        "task_script_sha256": digest(Path(__file__).resolve()),
        "command": command,
        "capture": capture,
        "gpu": {"block": args.gpu_model, "expected_name": GPU_NAMES[args.gpu_model],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.dry_run:
        request["destination_exists"] = dest.exists()
        print(json.dumps(request, indent=2))
        return 0

    dest.mkdir(parents=True)
    (dest / "frames").mkdir()
    (dest / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    status = {
        "state": "running", "phase": args.phase, "task": row["id"], "policy": which, "run": args.run,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        "started_utc": request["created_utc"],
    }
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    expected_gpu = GPU_NAMES[args.gpu_model]
    names = allocated_gpu_names()
    request["gpu"]["nvidia_smi"] = names
    (dest / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    if names is not None and set(names) != {expected_gpu}:
        status.update(state="infrastructure_error", error=f"allocated GPU {names}, block pins {expected_gpu!r}",
                      completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        return 3
    started = time.monotonic()
    with (dest / "rollout.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=lehome, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        status["child_pid"] = proc.pid
        try:
            timeout = float(manifest["budget"]["rollout_timeout_seconds"][args.phase])
            proc.wait(timeout=timeout)
            status["returncode"] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f"Isaac process exited {proc.returncode}")
            result = json.loads((dest / "rollout.json").read_text())
            validate_result(result, row, capture=capture, raw_path=raw)
            status["gpu_model"] = gpu_model(dest / "rollout.log")
            if status["gpu_model"] != expected_gpu:
                raise RuntimeError(f"Isaac ran on {status['gpu_model']!r}; the block pins {expected_gpu!r}")
            status.update(state="completed", success=result["success"],
                          terminal_success=result["terminal_success"],
                          terminal_conditions=result["terminal_checker"]["conditions_passed"],
                          terminal_conditions_total=result["terminal_checker"]["conditions_total"])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status.update(state="infrastructure_timeout", error="Isaac child exceeded bounded timeout")
        except Exception as exc:  # persist a useful per-row failure record
            status.update(state="infrastructure_error", error=repr(exc))
    status["wall_seconds"] = time.monotonic() - started
    status["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0 if status["state"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
