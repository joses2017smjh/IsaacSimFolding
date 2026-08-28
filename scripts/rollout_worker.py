"""Stage 3: one rollout worker.

N of these run as independent Slurm tasks against a shared directory while the
trainer publishes new checkpoints into it. Each worker loops:

    read the manifest -> roll out K episodes on the CURRENT checkpoint ->
    stamp every episode with the checkpoint version that produced it ->
    append to the shared buffer -> repeat

Two things this does that a naive worker does not, and both are load-bearing:

1. **Stamps provenance.** Every rollout carries the checkpoint version and
   digest it came from. The trainer refuses data that is too far behind. That
   is gate G3, and it exists because a stale worker produces advantage labels
   computed against the wrong baseline -- which degrades the run slowly rather
   than crashing it.

2. **Refuses held-out garments.** Rollouts are collected on Seen garments only.
   Collecting on the public Unseen garments and training on the result is the
   leak that makes the generalisation number meaningless, and nothing else in
   the pipeline would notice.

Parallelism is one environment per PROCESS. LeHome's evaluator is hardcoded to
a batch of one and its physics is CPU-only, so there is no vectorised env to
turn up -- N workers is the only axis. See docs/STAGE0.md section 2.3.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lehome_fold import ckpt as K  # noqa: E402
from lehome_fold import eval_log as E  # noqa: E402
from lehome_fold import splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_dir", required=True, help="where the trainer publishes the manifest")
    ap.add_argument("--rollout_dir", required=True, help="where rollouts are appended")
    ap.add_argument("--lehome", required=True, help="the checkout root; eval runs with this as cwd")
    ap.add_argument("--repo", required=True, help="this repo, for run_eval.py")
    ap.add_argument("--worker_id", type=int, required=True)
    ap.add_argument("--garment_types", default="top_long,top_short,pant_long,pant_short")
    ap.add_argument("--episodes_per_batch", type=int, default=4)
    ap.add_argument("--policy_type", default="recap")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--value_path", default=None)
    ap.add_argument("--feature_path", default="")
    ap.add_argument("--n_candidates", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--max_iters", type=int, default=10**9)
    ap.add_argument("--poll_seconds", type=float, default=30.0)
    return ap.parse_args()


def guard_garments(types: list[str]) -> None:
    """Every garment this worker can touch must be trainable.

    `--garment_type top_long` makes the evaluator sweep that category's
    Release list, which contains BOTH Seen and Unseen entries. So the guard
    cannot be a check on the category name -- it has to be a check on the
    garment list the evaluator will actually load, which is what
    Release_test_list.txt controls.
    """
    for t in types:
        if t not in splits.ARG_TO_CATEGORY:
            raise SystemExit(f"unknown garment type {t!r}; expected one of "
                             f"{sorted(splits.ARG_TO_CATEGORY)}")


def write_test_list(lehome: Path, garments: list[str]) -> Path:
    """Pin the evaluator to an explicit, Seen-only garment list.

    `--garment_type custom` reads Release_test_list.txt. Writing it here and
    asserting on it first is what keeps an Unseen garment out of the training
    loop; the alternative -- trusting a category sweep -- would silently
    include the two public held-out garments per type.
    """
    splits.assert_trainable(garments, context="rollout collection")
    p = lehome / "Assets/objects/Challenge_Garment/Release/Release_test_list.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(garments) + "\n")
    return p


def run_batch(args, ref: K.CheckpointRef) -> str:
    cmd = [
        sys.executable, f"{args.repo}/scripts/run_eval.py",
        "--policy_type", args.policy_type,
        "--policy_path", ref.path,
        "--dataset_root", args.dataset_root,
        "--garment_type", "custom",
        "--num_episodes", str(args.episodes_per_batch),
        "--max_steps", str(args.max_steps),
        "--device", "cpu",
        "--enable_cameras", "--headless",
    ]
    env = dict(os.environ)
    if args.value_path:
        env["VALUE_PATH"] = args.value_path
    if args.feature_path:
        env["FEATURE_PATH"] = args.feature_path
    proc = subprocess.run(cmd, cwd=args.lehome, env=env,
                          capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"[worker {args.worker_id}] eval exited {proc.returncode}", flush=True)
    return out


def main() -> int:
    args = parse_args()
    types = [t.strip() for t in args.garment_types.split(",") if t.strip()]
    guard_garments(types)

    garments = [g for t in types for g in splits.seen(splits.ARG_TO_CATEGORY[t])]
    write_test_list(Path(args.lehome), garments)
    print(f"[worker {args.worker_id}] {len(garments)} Seen garments, "
          f"0 held-out (guard passed)", flush=True)

    rollout_dir = Path(args.rollout_dir)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    last_version = -1
    for it in range(args.max_iters):
        try:
            ref = K.read(args.shared_dir)
        except RuntimeError as exc:
            print(f"[worker {args.worker_id}] no manifest yet ({exc}); waiting", flush=True)
            time.sleep(args.poll_seconds)
            continue

        if ref.version != last_version:
            print(f"[worker {args.worker_id}] switching to checkpoint v{ref.version} "
                  f"(step {ref.step}, digest {ref.digest})", flush=True)
            last_version = ref.version

        log = run_batch(args, ref)
        try:
            episodes = E.parse_episodes(log)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {args.worker_id}] could not parse eval output: {exc!r}", flush=True)
            episodes = []

        if not episodes:
            # A worker that silently produces nothing looks identical to a
            # worker that is merely slow. Say so, and back off rather than
            # spinning on a broken environment.
            print(f"[worker {args.worker_id}] iter {it}: ZERO episodes parsed -- "
                  f"the evaluation crashed rather than failing", flush=True)
            time.sleep(args.poll_seconds)
            continue

        out = rollout_dir / f"w{args.worker_id:03d}_v{ref.version:05d}_{it:06d}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            for e in episodes:
                rec = K.stamp({
                    "worker": args.worker_id, "iter": it,
                    "length": e.length, "success": e.success, "return": e.ret,
                    "success_frame": None,
                }, ref)
                fh.write(json.dumps(rec) + "\n")
        # Rename last: the trainer globs this directory, and a half-written
        # file that it can already see is a race it should never have to
        # handle.
        os.replace(tmp, out)

        n_ok = sum(e.success for e in episodes)
        print(f"[worker {args.worker_id}] iter {it}: {n_ok}/{len(episodes)} success "
              f"@v{ref.version} -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
