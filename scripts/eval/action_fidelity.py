"""Does the policy reproduce demonstrated actions on IN-DISTRIBUTION frames?

Two explanations for the policy hovering 12 cm above the cloth have already
died: the rasterised-vs-path-traced domain gap, and the wrist cameras
rendering empty table. What survives is that the policy barely responds to its
vision at all -- which is what a half-trained BC checkpoint looks like when it
has learned a mean trajectory instead of visual servoing.

This separates the remaining stories with one measurement. It feeds the policy
the ORIGINAL path-traced dataset frames -- the exact images it trained on, no
simulator and no Storm anywhere in the loop -- and compares its predicted
action against the action actually recorded at that frame.

The comparison that matters is against a constant baseline that always
predicts the dataset's mean action:

  policy error << mean-action error   -> it did learn the mapping, so failure
                                         in sim is a domain/rollout problem
  policy error ~= mean-action error   -> it has learned little beyond the
                                         average pose, and no amount of
                                         observation-pipeline work will help

Reporting one error without the baseline would be uninterpretable: a small MSE
means nothing when the action range itself is small.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--repo_id", default="lehome_four_types")
    ap.add_argument("--episodes", default="25,26,2,250,500,750")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pcfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    pcfg.pretrained_path = args.policy_path
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).eval().to(dev)
    pre, post = make_pre_post_processors(policy_cfg=pcfg,
                                         pretrained_path=args.policy_path)
    print(f"[fid] policy on {dev}", flush=True)

    eps = [int(x) for x in args.episodes.split(",")]
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=eps)
    print(f"[fid] frames={ds.num_frames} episodes={eps}", flush=True)

    step = max(1, ds.num_frames // args.samples)
    idxs = list(range(0, ds.num_frames, step))[: args.samples]

    acts_true, acts_pred = [], []
    for n, i in enumerate(idxs):
        s = ds[i]
        obs = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)).unsqueeze(0).to(dev)
               for k, v in s.items() if k.startswith("observation.")}
        obs["task"] = s.get("task", "fold the garment")
        policy.reset()                       # fresh action chunk per sample
        with torch.inference_mode():
            a = policy.select_action(pre(obs) if pre else obs)
        if post:
            a = post(a)
        acts_pred.append(np.asarray(a.squeeze(0).detach().cpu()).reshape(-1)[:12])
        acts_true.append(np.asarray(s["action"]).reshape(-1)[:12])
        if n % 25 == 0:
            print(f"[fid] {n}/{len(idxs)}", flush=True)

    T, P = np.stack(acts_true), np.stack(acts_pred)
    mean_act = T.mean(axis=0, keepdims=True)

    mse_policy = float(((P - T) ** 2).mean())
    mse_mean = float(((mean_act - T) ** 2).mean())
    skill = 1.0 - mse_policy / mse_mean if mse_mean else float("nan")
    var_policy, var_true = float(P.var(axis=0).mean()), float(T.var(axis=0).mean())

    print("\n=== action fidelity on IN-DISTRIBUTION (path-traced) frames ===")
    print(f"  samples                : {len(idxs)}")
    print(f"  policy MSE             : {mse_policy:.5f}   MAE {float(np.abs(P - T).mean()):.5f}")
    print(f"  mean-action MSE        : {mse_mean:.5f}   MAE {float(np.abs(mean_act - T).mean()):.5f}")
    print(f"  skill vs mean baseline : {skill:+.3f}   (1 = perfect, 0 = no better than the mean)")
    print(f"  output variance        : policy {var_policy:.5f} vs demos {var_true:.5f}"
          f"   (ratio {var_policy / var_true if var_true else float('nan'):.3f})")
    verdict = ("LEARNED the mapping" if skill > 0.5 else
               "PARTIAL" if skill > 0.15 else
               "NO BETTER THAN PREDICTING THE MEAN ACTION")
    print(f"  verdict                : {verdict}", flush=True)

    if args.out:
        json.dump({"samples": len(idxs), "mse_policy": mse_policy,
                   "mse_mean_baseline": mse_mean, "skill_vs_mean": skill,
                   "var_policy": var_policy, "var_demos": var_true,
                   "verdict": verdict, "episodes": eps, "policy": args.policy_path,
                   "note": "in-distribution dataset frames; no simulator, no Storm"},
                  open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
