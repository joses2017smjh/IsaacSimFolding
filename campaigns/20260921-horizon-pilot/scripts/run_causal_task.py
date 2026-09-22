"""Run one matched H50 episode with in-process replan branches."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--index", type=int, required=True,
                    help="0..3 for development pose slots 1,3,5,7")
    ap.add_argument("--branch-steps", type=int, default=120)
    ap.add_argument("--causal-action-jsonl", default="",
                    help="completed pilot H50 behavior stream used as control")
    ap.add_argument("--output-prefix", default="causal",
                    help="output directory prefix, e.g. rtc")
    ap.add_argument("--rtc-guidance", action="store_true",
                    help="enable the installed LeRobot SmolVLA RTC branch")
    ap.add_argument("--queue-diagnostic", action="store_true",
                    help="run the hard retained-prefix diagnostic; pose slot 3 only")
    ap.add_argument("--queue-delays", default="2,5,10")
    ap.add_argument("--observation-diagnostic", action="store_true",
                    help="run the observation-component diagnostic; pose slot 3 only")
    ap.add_argument("--observation-execute-best", action="store_true",
                    help="execute at most two candidates after the preregistered gate")
    ap.add_argument("--boundary-capture", action="store_true",
                    help="replay cached pose-3 H50 and export exact action-5/10 observations")
    ap.add_argument("--boundary-eval-trained-path", default="",
                    help="snapshot-only compare of this trained checkpoint to the baseline")
    ap.add_argument("--boundary-eval-trained-paths", nargs="+", default=[],
                    help="snapshot-only compare of multiple checkpoints to the baseline")
    ap.add_argument("--boundary-eval-out", default="",
                    help="JSON output for the snapshot-only baseline/trained comparison")
    ap.add_argument("--boundary-execute-trained-path", default="",
                    help="selected checkpoint for ordinary fresh H10 snapshot branches")
    ap.add_argument("--boundary-execute-out", default="",
                    help="JSON output for the selected checkpoint branches")
    ap.add_argument("--boundary-ceiling-trained-paths", nargs="+", default=[],
                    help="existing checkpoints for the closed-loop ceiling panel")
    ap.add_argument("--boundary-ceiling-out", default="",
                    help="JSON output for the cached/fresh/checkpoint ceiling panel")
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    wanted = {1, 3, 5, 7}
    rows = [r for r in manifest["pilot_tasks"]
            if int(r["horizon"]) == 50 and int(r["development_pose_slot"]) in wanted]
    rows.sort(key=lambda r: int(r["development_pose_slot"]))
    row = rows[args.index]
    if args.queue_diagnostic:
        if int(row["development_pose_slot"]) != 3:
            raise SystemExit("--queue-diagnostic is restricted to exact development pose slot 3")
        if args.rtc_guidance:
            raise SystemExit("--queue-diagnostic cannot use RTC guidance")
    if args.observation_diagnostic:
        if int(row["development_pose_slot"]) != 3:
            raise SystemExit("--observation-diagnostic is restricted to exact development pose slot 3")
        if args.rtc_guidance or args.queue_diagnostic:
            raise SystemExit("--observation-diagnostic cannot use RTC or queue diagnostics")
    if args.observation_execute_best and not args.observation_diagnostic:
        raise SystemExit("--observation-execute-best requires --observation-diagnostic")
    if args.boundary_capture:
        if int(row["development_pose_slot"]) != 3 or int(row["horizon"]) != 50:
            raise SystemExit("--boundary-capture is restricted to pose-3 H50")
        if args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic:
            raise SystemExit("--boundary-capture cannot use inference diagnostics")
    if args.boundary_eval_trained_path and args.boundary_eval_trained_paths:
        raise SystemExit("use only one boundary trained path option")
    if args.boundary_execute_trained_path and (args.boundary_eval_trained_path or args.boundary_eval_trained_paths):
        raise SystemExit("boundary execution cannot be combined with boundary evaluation")
    if args.boundary_ceiling_trained_paths and (
        args.boundary_eval_trained_path or args.boundary_eval_trained_paths
        or args.boundary_execute_trained_path
    ):
        raise SystemExit("boundary ceiling cannot be combined with boundary evaluation or execution")
    if bool(args.boundary_execute_trained_path) != bool(args.boundary_execute_out):
        raise SystemExit("boundary execution requires both trained checkpoint and output path")
    if bool(args.boundary_ceiling_trained_paths) != bool(args.boundary_ceiling_out):
        raise SystemExit("boundary ceiling requires checkpoints and output path")
    boundary_eval_paths = ([args.boundary_eval_trained_path]
                           if args.boundary_eval_trained_path
                           else list(args.boundary_eval_trained_paths))
    if bool(boundary_eval_paths) != bool(args.boundary_eval_out):
        raise SystemExit("boundary evaluation requires both trained checkpoint and output path")
    if boundary_eval_paths:
        if int(row["development_pose_slot"]) != 3 or int(row["horizon"]) != 50:
            raise SystemExit("--boundary-eval is restricted to pose-3 H50")
        if args.boundary_capture or args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic:
            raise SystemExit("--boundary-eval cannot use capture, RTC, queue, or observation diagnostics")
    if args.boundary_execute_trained_path:
        if int(row["development_pose_slot"]) != 3 or int(row["horizon"]) != 50:
            raise SystemExit("--boundary-execute is restricted to pose-3 H50")
        if args.boundary_capture or args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic:
            raise SystemExit("--boundary-execute cannot use capture, RTC, queue, or observation diagnostics")
    if args.boundary_ceiling_trained_paths:
        if int(row["development_pose_slot"]) != 3 or int(row["horizon"]) != 50:
            raise SystemExit("--boundary-ceiling is restricted to pose-3 H50")
        if args.boundary_capture or args.rtc_guidance or args.queue_diagnostic or args.observation_diagnostic:
            raise SystemExit("--boundary-ceiling cannot use capture, RTC, queue, or observation diagnostics")
    inventory = json.loads((root / "audit/garment_inventory.json").read_text())
    asset = next(r for r in inventory["garments"] if r["garment_id"] == row["garment"])
    checkpoint = manifest["checkpoint"]
    dest = root / "outputs" / (args.output_prefix + "_" + row["id"])
    dest.mkdir(parents=True, exist_ok=True)
    frames = dest / "frames"
    frames.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(root / "scripts/render/policy_rollout51.py"),
        "--lehome", str(root / "external/lehome-challenge"),
        "--policy_path", checkpoint["path"], "--garment", row["garment"],
        "--garment_dir", str(Path(asset["config"]).parent), "--assets", manifest["assets"],
        "--steps", "10" if (
            boundary_eval_paths or args.boundary_execute_trained_path
            or args.boundary_ceiling_trained_paths
        ) else "600",
        "--frames_out", str(frames),
        "--match_pose=" + row["match_pose"], "--match_scale", str(row["match_scale"]),
        "--settle_steps", "60", "--terminal_settle_steps", "60",
        "--seed", str(row["seed"]), "--n_action_steps", "50",
        "--policy_variant", ("boundary_capture" if args.boundary_capture else
                              "boundary_eval" if boundary_eval_paths else
                              "boundary_execute" if args.boundary_execute_trained_path else
                              "boundary_ceiling" if args.boundary_ceiling_trained_paths else
                              "causal_h50_capture"),
        "--task", manifest["task_prompt"], "--gif_every", "12",
        "--result_out", str(dest / "rollout.json")]
    if not args.boundary_capture:
        command += ["--causal_out", str(dest / "replan-causality.json"),
                    "--causal_branch_steps", str(args.branch_steps)]
    if args.rtc_guidance:
        command.append("--rtc_guidance")
    if args.queue_diagnostic:
        command += ["--queue_diagnostic", "--queue_delays", args.queue_delays]
    if args.observation_diagnostic:
        command.append("--observation_diagnostic")
    if args.observation_execute_best:
        command.append("--observation_execute_best")
    if args.boundary_capture:
        command += ["--boundary_capture_out",
                    str(root / "analysis/boundary-replay/pose3-boundary-replay.npz")]
    if boundary_eval_paths:
        command += ["--boundary_eval_trained_paths", *boundary_eval_paths,
                    "--boundary_eval_out", args.boundary_eval_out]
    if args.boundary_execute_trained_path:
        command += ["--boundary_execute_trained_path", args.boundary_execute_trained_path,
                    "--boundary_execute_out", args.boundary_execute_out]
    if args.boundary_ceiling_trained_paths:
        command += ["--boundary_ceiling_trained_paths", *args.boundary_ceiling_trained_paths,
                    "--boundary_ceiling_out", args.boundary_ceiling_out]
    cached = args.causal_action_jsonl or str(root / "outputs" / row["id"] / "rollout.json.behavior.jsonl")
    if Path(cached).is_file():
        command += ["--causal_action_jsonl", cached]
    (dest / "request.json").write_text(json.dumps({"row": row, "command": command}, indent=2) + "\n")
    env = dict(os.environ)
    proc = subprocess.run(command, cwd=root / "external/lehome-challenge", env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
