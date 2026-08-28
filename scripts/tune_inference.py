"""Stage 4: Thompson sampling over inference-time hyperparameters.

No training. One fixed checkpoint. The only question is which inference
settings to run it with -- number of candidates, action-chunk length, sampling
temperature, flow-matching step count -- and fold success is binary per
episode, which makes a Beta posterior per arm exactly the right model rather
than an approximation.

The headline is GAIN OVER FIXED DEFAULTS on the same checkpoint, so the default
arm gets a reserved budget before sampling starts. Thompson sampling starves a
mediocre-looking arm within a few dozen pulls, and a baseline with ten episodes
behind it cannot anchor a comparison -- in simulation it drew 10 of 3,000.

Cost is reported alongside success rate. An arm that wins by a point at eight
times the compute is a different result from one that wins for free, and the
success rate alone cannot tell them apart.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lehome_fold import eval_log as E  # noqa: E402
from lehome_fold import thompson as T  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--value_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--lehome", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--policy_type", default="candidate")
    ap.add_argument("--garment_type", default="top_long")
    ap.add_argument("--budget", type=int, default=400, help="total episodes")
    ap.add_argument("--baseline_pulls", type=int, default=40)
    ap.add_argument("--episodes_per_pull", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--feature_path", default="")
    ap.add_argument("--out", default="results/stage4_thompson.json")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def run_arm(args, arm: T.Arm) -> list[bool]:
    """One pull: run episodes at this arm's settings, return the outcomes."""
    cmd = [
        sys.executable, f"{args.repo}/scripts/run_eval.py",
        "--policy_type", args.policy_type,
        "--policy_path", args.policy_path,
        "--dataset_root", args.dataset_root,
        "--garment_type", args.garment_type,
        "--num_episodes", str(args.episodes_per_pull),
        "--max_steps", str(args.max_steps),
        "--device", "cpu", "--enable_cameras", "--headless",
    ]
    env = dict(os.environ)
    env.update({
        "VALUE_PATH": args.value_path,
        "FEATURE_PATH": args.feature_path,
        "N_CANDIDATES": str(arm.n_candidates),
        "CHUNK_LENGTH": str(arm.chunk_length),
        "TEMPERATURE": str(arm.temperature),
        "FLOW_STEPS": str(arm.flow_steps),
    })
    proc = subprocess.run(cmd, cwd=args.lehome, env=env,
                          capture_output=True, text=True, check=False)
    try:
        return [e.success for e in E.parse_episodes(proc.stdout + proc.stderr)]
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    args = parse_args()
    arms = T.grid()
    ts = T.ThompsonSampler(arms, seed=args.seed, baseline=T.DEFAULT_ARM,
                           baseline_pulls=args.baseline_pulls)
    print(f"[stage4] {len(arms)} arms, budget {args.budget} episodes, "
          f"{args.baseline_pulls} reserved for {T.DEFAULT_ARM.name}", flush=True)

    spent = 0
    while spent < args.budget:
        arm = ts.select()
        outcomes = run_arm(args, arm)
        if not outcomes:
            # A pull that produced no scored episode is a crashed evaluation,
            # not a failure. Charging it to the arm would teach the sampler
            # that a broken configuration is a bad one, which is a different
            # and wrong conclusion.
            print(f"[stage4] {arm.name}: no episodes parsed -- not charged", flush=True)
            spent += 1
            continue
        for ok in outcomes:
            ts.update(arm, ok)
            spent += 1
        if spent % 25 < len(outcomes):
            print(f"[stage4] {spent}/{args.budget}  last={arm.name} "
                  f"{sum(outcomes)}/{len(outcomes)}", flush=True)

    print()
    print(ts.report(top=10))
    gain = ts.gain_over_baseline()
    print()
    for k, v in gain.items():
        print(f"  {k:16s} {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gain": gain,
        "arms": [{"name": a.name, "pulls": ts.pulls(a),
                  "successes": ts.successes[a.name],
                  "posterior_mean": ts.posterior_mean(a),
                  "cost": a.cost} for a in arms],
    }, indent=2))
    print(f"\n[stage4] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
