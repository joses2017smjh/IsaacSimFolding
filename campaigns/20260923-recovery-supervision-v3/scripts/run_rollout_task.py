"""Run one predeclared closed-loop campaign row in an isolated Isaac process.

The simulator/controller remains the validated horizon-pilot runner.  This
adapter deliberately only supplies a new manifest row, checkpoint and output
location; it neither changes inference semantics nor synthesizes corrective
actions.  Collection requests persist sampled observations plus the full
executed H10 action stream so a later compiler can form real future-action
chunks without tiling a single action.
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

import numpy as np


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


def destination(root: Path, phase: str, row: dict, label: str,
                plan: dict | None = None) -> Path:
    if plan is not None:
        # Iteration >= 2 collection. Rows come from a resolved, hashed plan and
        # land in their own group so no compile glob can mix iterations.
        k = int(plan["iteration"])
        group = f"iteration{k}" if phase == "collection" else f"iteration{k}-expansion"
        return root / "rollouts" / group / row["id"]
    if phase == "recovery":
        return root / "recovery" / row["id"]
    if phase == "recovery_smoke":
        return root / "smoke" / "recovery" / row["id"]
    if phase == "smoke":
        return root / "smoke" / "rollouts" / row["id"]
    if phase == "collection":
        return root / "rollouts" / "iteration1" / row["id"]
    if phase == "collection_expansion":
        return root / "rollouts" / "expansion" / row["id"]
    return root / "evaluation" / label / phase / row["id"]


def validate_result(result: dict, row: dict, *, capture: bool, raw_path: Path | None) -> None:
    expected = {
        "garment": row["garment"],
        "seed": row["seed"],
        "steps": row["steps"],
        # Development rows are run at both horizons so the comparison against
        # the pilot baseline is matched; everything else is ordinary H10.
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
        raise ValueError("collection did not persist trajectory data")
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
    ap.add_argument("--phase", required=True, choices=(
        "smoke", "collection", "collection_expansion", "benchmark", "frozen_test",
        "recovery", "recovery_smoke", "final_test"))
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--checkpoint", default="", help="defaults to the immutable baseline")
    ap.add_argument("--label", default="baseline", help="safe output/checkpoint label")
    ap.add_argument("--plan", type=Path, default=None,
                    help="resolved iteration plan supplying collection rows (iteration >= 2)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if plan is not None and args.phase not in ("collection", "collection_expansion"):
        raise SystemExit("--plan only supplies collection rows")
    # "recovery" names the search CONFIG in the manifest; its rows live apart.
    rows_key = {"recovery": "recovery_search"}.get(args.phase, args.phase)
    rows = plan[args.phase] if plan is not None else manifest[rows_key]
    if not 0 <= args.index < len(rows):
        raise SystemExit(f"{args.phase} index {args.index} out of range")
    row = rows[args.index]
    label = args.label.replace("/", "_")
    if not label or label in {".", ".."}:
        raise SystemExit("invalid checkpoint label")
    checkpoint = Path(args.checkpoint or manifest["baseline_checkpoint"]["path"]).resolve()
    ckpt = checkpoint_record(checkpoint)
    baseline = Path(manifest["baseline_checkpoint"]["path"]).resolve()
    if checkpoint == baseline:
        # The baseline is declared immutable, so check every file the manifest
        # pinned rather than only the weights: a changed normalizer or
        # preprocessor silently shifts the action space this campaign measures.
        pinned = manifest["baseline_checkpoint"]["sha256"]
        drifted = sorted(name for name, sha in ckpt.items()
                         if name in pinned and sha != pinned[name])
        missing = sorted(name for name in ckpt if name not in pinned)
        if drifted or missing:
            raise SystemExit(
                f"immutable baseline checkpoint changed: altered={drifted} unpinned={missing}")

    dest = destination(root, args.phase, row, label, plan)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite retained output {dest}")
    # v3 executes its own snapshot of the runner (with the recovery-search
    # mode); the v2 and pilot copies stay byte-identical to what they ran.
    runner = root / "scripts" / "render" / "policy_rollout51.py"
    lehome = Path(manifest["lehome"])
    if not runner.is_file() or not lehome.is_dir() or not Path(row["asset_config"]).is_file():
        raise SystemExit("validated runner, LeHome checkout, or garment config is unavailable")
    expected_runner = manifest["executed_sources"].get(manifest["runner_key"])
    if expected_runner and digest(runner) != expected_runner:
        raise SystemExit(
            f"runner {runner} does not match the manifest-pinned source hash; "
            f"the campaign snapshot and the executed code have diverged")

    capture = args.phase in {"smoke", "collection", "collection_expansion", "recovery", "recovery_smoke"}
    raw = dest / "trajectory.npz" if capture else None
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
        "--policy_variant", f"closed_loop_{args.phase}_{label}",
        "--task", manifest["task_prompt"], "--gif_every", "12",
        "--result_out", str(dest / "rollout.json"),
    ]
    if capture:
        command.extend(["--trajectory_out", str(raw), "--trajectory_every",
                        str(row["trajectory_every"])])
    if args.phase in ("recovery", "recovery_smoke"):
        cfg = manifest["recovery"] if args.phase == "recovery" else manifest["recovery_smoke_config"]
        command.extend([
            "--recovery_out", str(dest / "recovery.json"),
            "--recovery_examples_out", str(dest / "recovery_examples.npz"),
            "--recovery_roots", ",".join(str(r) for r in cfg["roots"]),
            "--recovery_candidates", ",".join(f"{c['horizon']}:{c['seed']}" for c in cfg["candidates"]),
            "--recovery_obs_every", str(cfg["obs_every"]),
            "--recovery_max_seconds", str(cfg["max_seconds"]),
        ])
    # The runner is another campaign's file. Record what was actually executed
    # rather than only the path, so a result can be traced to a source revision
    # even if that file later changes.
    request = {
        "phase": args.phase,
        "row": row,
        "checkpoint": {"path": str(checkpoint), "sha256": ckpt},
        "runner": {"path": str(runner), "sha256": digest(runner)},
        "campaign_commit": manifest["git"]["commit"],
        "task_script_sha256": digest(Path(__file__).resolve()),
        "plan": ({"path": str(args.plan.resolve()), "sha256": digest(args.plan.resolve()),
                  "iteration": plan["iteration"]} if plan is not None else None),
        "command": command,
        "capture": capture,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return 0

    dest.mkdir(parents=True)
    (dest / "frames").mkdir()
    (dest / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    status = {
        "state": "running", "phase": args.phase, "task": row["id"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_utc": request["created_utc"],
    }
    (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    with (dest / "rollout.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=lehome, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        status["child_pid"] = proc.pid
        try:
            timeout_key = {"collection_expansion": "collection", "frozen_test": "benchmark",
                           "final_test": "benchmark", "recovery_smoke": "smoke"}.get(args.phase, args.phase)
            timeout = float(manifest["budget"]["rollout_timeout_seconds"][timeout_key])
            proc.wait(timeout=timeout)
            status["returncode"] = proc.returncode
            if proc.returncode:
                raise RuntimeError(f"Isaac process exited {proc.returncode}")
            result = json.loads((dest / "rollout.json").read_text())
            validate_result(result, row, capture=capture, raw_path=raw)
            if args.phase in ("recovery", "recovery_smoke"):
                for name in ("recovery.json", "recovery_examples.npz"):
                    if not (dest / name).is_file():
                        raise ValueError(f"recovery search did not write {name}")
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
