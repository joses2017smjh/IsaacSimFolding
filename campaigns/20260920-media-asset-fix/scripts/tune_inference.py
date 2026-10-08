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
    ap.add_argument("--only_implemented_dims", type=int, default=1,
                    help="pin the dimensions the policy does not "
                         "apply to their defaults instead of "
                         "sweeping them for nothing")
    ap.add_argument("--max_dead_pulls", type=int, default=3,
                    help="abort after this many consecutive pulls that "
                         "score no episode; a broken evaluator otherwise "
                         "spends the whole budget looking like slow progress")
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
        # SIMULATION device. LeHome's README calls --device the inference
        # device and says only cpu; evaluation.py feeds it to parse_env_cfg,
        # which sets the sim device, and PhysX particle cloth does not
        # simulate on CPU. Measured: cpu gives dist 14.54 every episode
        # forever, cuda:0 gives 13.66 then 13.63. Frozen vs moving cloth.
        "--device", os.environ.get("SIM_DEVICE", "cuda:0"), "--headless",
    ]
    # No --enable_cameras. AppLauncher builds the Isaac Lab render product at
    # LAUNCH when that flag is set, and 5.1's RTX delegate segfaults against
    # this driver -- at 586 ms, before any camera exists. Storm supplies the
    # pixels through a TiledCamera-shaped shim instead, so the flag is not
    # merely unnecessary, it is fatal.
    env = dict(os.environ)
    env.update({
        # Route cameras through Storm; see scripts/run_eval.py.
        "LH_STORM_EVAL": "1",
        "LH_STORM_ASSETS": os.environ.get("LH_STORM_ASSETS", ""),
        "LH_STORM_GARMENT_DIR": os.environ.get("LH_STORM_GARMENT_DIR", ""),
        "LH_STORM_WORKDIR": os.environ.get("LH_STORM_WORKDIR", ""),
        "LH_STORM_DEVICE": os.environ.get("LH_STORM_DEVICE", "cuda"),
        "HF_HUB_OFFLINE": "1",
        "VALUE_PATH": args.value_path,
        "FEATURE_PATH": args.feature_path,
        "N_CANDIDATES": str(arm.n_candidates),
        "CHUNK_LENGTH": str(arm.chunk_length),
        "TEMPERATURE": str(arm.temperature),
        "FLOW_STEPS": str(arm.flow_steps),
    })
    proc = subprocess.run(cmd, cwd=args.lehome, env=env,
                          capture_output=True, text=True, check=False)
    blob = proc.stdout + proc.stderr
    try:
        outcomes = [e.success for e in E.parse_episodes(blob)]
    except Exception as exc:  # noqa: BLE001
        outcomes = []
        print(f"[stage4] {arm.name}: parse raised {exc!r}", flush=True)
    if not outcomes:
        # An arm that yields nothing is not charged against the budget, so a
        # child that crashes every time loops until walltime and writes no
        # result. Say WHY, or the run looks like slow progress rather than a
        # dead subprocess.
        lines = [ln for ln in blob.splitlines() if ln.strip()]
        # The cause is often nowhere near the end. run_eval.py reports a
        # policy that failed to register and then carries on, so the fatal
        # "policy type not found" lands hundreds of lines later while the real
        # explanation scrolled past long before. Pull those out by name.
        keys = ("[run_eval]", "did not register", "not found in registry",
                "Error during evaluation")
        flagged = [ln for ln in lines if any(k in ln for k in keys)]
        print(f"[stage4] {arm.name}: child rc={proc.returncode}, no episodes "
              f"parsed.", flush=True)
        if flagged:
            print("  registration/eval lines:", flush=True)
            for ln in flagged[:12]:
                print(f"    ! {ln[:220]}", flush=True)
        tail = lines[-20:]
        print(f"  last {len(tail)} lines:", flush=True)
        for ln in tail:
            print(f"    | {ln[:200]}", flush=True)
    return outcomes


def main() -> int:
    args = parse_args()
    arms = T.grid(**({"chunk_length": (T.DEFAULT_ARM.chunk_length,),
                      "temperature": (T.DEFAULT_ARM.temperature,),
                      "flow_steps": (T.DEFAULT_ARM.flow_steps,)}
                     if args.only_implemented_dims else {}))
    # Fails loudly rather than sweeping a dimension nothing applies.
    T.assert_dims_implemented(arms)
    ts = T.ThompsonSampler(arms, seed=args.seed, baseline=T.DEFAULT_ARM,
                           baseline_pulls=args.baseline_pulls)
    print(f"[stage4] {len(arms)} arms, budget {args.budget} episodes, "
          f"{args.baseline_pulls} reserved for {T.DEFAULT_ARM.name}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def snapshot() -> dict:
        return {
            "gain": ts.gain_over_baseline(),
            "arms": [{"name": a.name, "pulls": ts.pulls(a),
                      "successes": ts.successes[a.name],
                      "posterior_mean": ts.posterior_mean(a),
                      "cost": a.cost} for a in arms],
        }

    spent = 0
    dead = 0
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
            dead += 1
            if dead >= args.max_dead_pulls:
                # Every pull so far has crashed. Spending the rest of the
                # budget relaunching Isaac Sim to crash the same way buys
                # nothing and writes a result file that looks like a measured
                # zero rather than a broken setup.
                print(f"\n[stage4] ABORT: {dead} consecutive pulls produced no "
                      f"scored episode. The evaluator is not running -- see the "
                      f"child output above. No tuning result written.", flush=True)
                return 2
            continue
        dead = 0
        for ok in outcomes:
            ts.update(arm, ok)
            spent += 1
        # Written every cycle: at budget 400 this job outlives its walltime,
        # and an end-only write would leave nothing behind when it does.
        out.write_text(json.dumps(snapshot(), indent=2))
        if spent % 25 < len(outcomes):
            print(f"[stage4] {spent}/{args.budget}  last={arm.name} "
                  f"{sum(outcomes)}/{len(outcomes)}", flush=True)

    print()
    print(ts.report(top=10))
    gain = ts.gain_over_baseline()
    print()
    for k, v in gain.items():
        print(f"  {k:16s} {v}")

    out.write_text(json.dumps(snapshot(), indent=2))
    print(f"\n[stage4] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
