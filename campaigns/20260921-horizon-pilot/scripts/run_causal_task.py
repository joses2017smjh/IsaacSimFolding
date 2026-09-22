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
        "--steps", "600", "--frames_out", str(frames),
        "--match_pose=" + row["match_pose"], "--match_scale", str(row["match_scale"]),
        "--settle_steps", "60", "--terminal_settle_steps", "60",
        "--seed", str(row["seed"]), "--n_action_steps", "50",
        "--policy_variant", "causal_h50_capture", "--task", manifest["task_prompt"],
        "--gif_every", "12", "--result_out", str(dest / "rollout.json"),
        "--causal_out", str(dest / "replan-causality.json"),
        "--causal_branch_steps", str(args.branch_steps)]
    if args.rtc_guidance:
        command.append("--rtc_guidance")
    if args.queue_diagnostic:
        command += ["--queue_diagnostic", "--queue_delays", args.queue_delays]
    cached = args.causal_action_jsonl or str(root / "outputs" / row["id"] / "rollout.json.behavior.jsonl")
    if Path(cached).is_file():
        command += ["--causal_action_jsonl", cached]
    (dest / "request.json").write_text(json.dumps({"row": row, "command": command}, indent=2) + "\n")
    env = dict(os.environ)
    proc = subprocess.run(command, cwd=root / "external/lehome-challenge", env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
