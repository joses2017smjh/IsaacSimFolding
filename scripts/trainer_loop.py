"""Stage 3: the persistent trainer.

One of these runs while N rollout workers feed a shared directory. Each cycle:

    consume new rollouts -> VERIFY PROVENANCE (G3) -> advantage from the value
    head -> RECAP conditioning + AWR weights -> train -> publish a new
    checkpoint version

G3 is enforced here, not assumed. Every rollout carries the checkpoint version
that produced it; anything more than `--max_lag` versions behind is dropped and
counted, and the lag histogram is printed every cycle. An unbounded lag is the
silent-degradation failure this gate exists for, and the histogram is what
makes it visible before it costs a run.

--- One necessary deviation, stated plainly -------------------------------

LeHome's evaluator records only SUCCESSFUL episodes:

    if success_flag:  eval_dataset.save_episode()
    else:             eval_dataset.clear_episode_buffer()

RECAP trains on all data INCLUDING failures -- that is the entire method. So
the recording path cannot be theirs. It is ours, inside our own policy class
(`--record_dir` on the Stage 3 policies), and it writes every episode whatever
the outcome.

What is NOT ours is the SCORER. `success_checker_garment_fold` still decides
what counts as a fold, unmodified, through the official eval loop. The
deviation is in what gets kept, not in what gets measured, and success rate
stays comparable to the leaderboard's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from lehome_fold import awr, ckpt as K, recap  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_dir", required=True)
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--policy_path", required=True, help="Stage 2 checkpoint to start from")
    ap.add_argument("--value_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_lag", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0, help="AWR temperature")
    ap.add_argument("--w_max", type=float, default=20.0)
    ap.add_argument("--use_awr", type=int, default=1)
    ap.add_argument("--use_recap", type=int, default=1)
    ap.add_argument("--min_new_episodes", type=int, default=64)
    ap.add_argument("--steps_per_cycle", type=int, default=500)
    ap.add_argument("--max_cycles", type=int, default=10**9)
    ap.add_argument("--poll_seconds", type=float, default=60.0)
    ap.add_argument("--dry_run", type=int, default=0,
                    help="do everything except the gradient step; used to exercise G3")
    return ap.parse_args()


def consume(rollout_dir: Path, seen: set[str], ref: K.CheckpointRef, max_lag: int):
    """Read unconsumed rollout files and verify each record's provenance."""
    records, lags, dropped = [], Counter(), Counter()
    for f in sorted(rollout_dir.glob("*.jsonl")):
        if f.name in seen:
            continue
        seen.add(f.name)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                lag = K.verify_stamp(rec, ref, max_lag=max_lag)
            except K.StaleCheckpoint as exc:
                dropped[first_line(exc)] += 1
                continue
            lags[lag] += 1
            records.append(rec)
    return records, lags, dropped


def first_line(exc: Exception) -> str:
    return str(exc).split(" -- ")[0].split(":")[0][:70]


def main() -> int:
    args = parse_args()
    shared, rollouts, out = Path(args.shared_dir), Path(args.rollout_dir), Path(args.out)
    shared.mkdir(parents=True, exist_ok=True)
    rollouts.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # Publish v0 so the workers have something to load. Until this exists they
    # poll and log; they never guess a path.
    version = 0
    ref = K.publish(shared, Path(args.policy_path), version=version, step=0)
    print(f"[trainer] published v0 from {args.policy_path} (digest {ref.digest})", flush=True)

    seen_files: set[str] = set()
    pending: list[dict] = []

    for cycle in range(args.max_cycles):
        new, lags, dropped = consume(rollouts, seen_files, ref, args.max_lag)
        pending.extend(new)

        if lags or dropped:
            hist = " ".join(f"lag{k}={v}" for k, v in sorted(lags.items()))
            print(f"[trainer] cycle {cycle}: +{len(new)} episodes  {hist}", flush=True)
            for reason, n in dropped.items():
                print(f"[trainer]   G3 DROPPED {n}: {reason}", flush=True)

        if len(pending) < args.min_new_episodes:
            time.sleep(args.poll_seconds)
            continue

        # -- advantage, from the policy's own value head --------------------
        outcomes = np.array([float(r["success"]) for r in pending], dtype=np.float64)
        baseline = np.array([float(r.get("p_success", outcomes.mean())) for r in pending])
        adv = awr.success_residual(outcomes, baseline)

        signs = recap.binarise(adv)
        bal = recap.balance(signs)
        weights = awr.weights(adv, beta=args.beta, w_max=args.w_max)
        ess = awr.effective_sample_size(weights)

        print(
            f"[trainer] cycle {cycle}: n={len(pending)} "
            f"success={outcomes.mean():.3f} "
            f"adv[{adv.min():+.3f},{adv.max():+.3f}] "
            f"recap_positive={bal['positive']:.3f} "
            f"AWR_ESS={ess:.1f}/{len(weights)}",
            flush=True,
        )
        if bal["positive"] < 0.02 or bal["positive"] > 0.98:
            # A buffer that is essentially one class trains a model that has
            # barely seen the token it will be conditioned on at inference.
            print("[trainer]   WARNING: RECAP conditioning is near-degenerate", flush=True)
        if ess < 0.05 * len(weights):
            print(f"[trainer]   WARNING: AWR effective sample size is {ess:.1f} -- "
                  f"a handful of episodes dominate this batch; raise --beta", flush=True)

        if not args.dry_run:
            train_cycle(args, pending, weights, signs)

        version += 1
        ckpt_dir = out / f"v{version:05d}"
        if not args.dry_run:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if ckpt_dir.exists():
            ref = K.publish(shared, ckpt_dir, version=version,
                            step=version * args.steps_per_cycle)
            print(f"[trainer] published v{version} ({ref.digest})", flush=True)
        pending.clear()
    return 0


def train_cycle(args, records, weights, signs) -> None:
    """One block of supervised steps on the conditioned, weighted buffer.

    Needs lerobot and the recorded rollout frames; see the deviation note at
    the top of this file for why the recorder is ours and the scorer is not.
    """
    raise NotImplementedError(
        "train_cycle needs the recorded rollout frames and a live lerobot. "
        "The provenance, advantage, conditioning and weighting logic above is "
        "complete and tested; run with --dry_run 1 to exercise G3 end to end "
        "without a gradient step."
    )


if __name__ == "__main__":
    raise SystemExit(main())
