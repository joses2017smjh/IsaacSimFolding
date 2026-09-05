"""Fine-tune the BC policy on RASTERISED frames.

The measured problem, at two checkpoints:

    checkpoint   path-traced   Storm     gap
    15K steps      +0.966      +0.063   0.902
    30K converged  +0.976      -0.038   1.014

The policy learned the task and cannot see the renderer it is deployed on.
Doubling the training schedule improved in-distribution skill and pushed
Storm-frame skill BELOW a mean-action baseline -- more training deepens the
overfit to path-traced appearance. Every lever inside training has been
measured and ruled out; the renderer is what is left.

This trains on (Storm image, recorded action) pairs captured by replaying
demonstrations in sim. The action labels are the demonstrations' own, so this
is still behaviour cloning -- only the observation distribution changes, which
is the single variable the 1.014 gap isolates.

Only the vision pathway is unfrozen by default. The action decoder already
maps features to good actions; what fails is producing the right features from
a rasterised image, so leaving the decoder alone keeps what works and spends
the gradient where the deficit was measured.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def _copy_processors(src: str, dst: str) -> None:
    """Carry the policy processors across to the fine-tuned checkpoint.

    save_pretrained writes config.json and model.safetensors only, so a
    fine-tuned directory cannot be loaded by anything that calls
    make_pre_post_processors -- which is every rollout in this repo. Copying is
    correct rather than a workaround: this fine-tune moves vision weights, and
    the normaliser statistics it would otherwise be missing are unchanged.
    """
    import shutil

    for f in ("policy_preprocessor.json", "policy_postprocessor.json",
              "policy_preprocessor_step_5_normalizer_processor.safetensors",
              "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
              "train_config.json"):
        a, b = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(a) and not os.path.exists(b):
            shutil.copy2(a, b)


def load_capture(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no capture files matching {pattern}")
    imgs, states, acts, meta = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        imgs.append(d["images"])
        states.append(d["state"])
        acts.append(d["action"])
        meta.append((str(d["garment"]), int(d["episode"]), bool(d["success"])))
    X = np.concatenate(imgs)
    S = np.concatenate(states)
    A = np.concatenate(acts)
    print(f"[ft] {len(files)} episodes, {len(X)} frames, images {X.shape[1:]}", flush=True)
    return X, S, A, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--capture_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--unfreeze", default="vision",
                    choices=["vision", "all"],
                    help="vision: only the image pathway, where the deficit is")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    cfg.pretrained_path = args.policy_path
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).to(dev)
    pre, _post = make_pre_post_processors(policy_cfg=cfg,
                                          pretrained_path=args.policy_path)

    X, S, A, meta = load_capture(args.capture_glob)
    n_val = max(1, int(len(X) * args.val_fraction))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[ft] train {len(train_idx)} / val {len(val_idx)}", flush=True)

    # Freeze everything, then re-enable only the vision pathway. The action
    # decoder already maps features to good actions -- the 1.014 gap is in
    # producing features from a rasterised image, not in what happens after.
    for p in policy.parameters():
        p.requires_grad_(False)
    trainable = []
    for name, p in policy.named_parameters():
        want = args.unfreeze == "all" or any(
            k in name.lower() for k in ("vision", "image", "patch", "visual"))
        if want:
            p.requires_grad_(True)
            trainable.append(p)
    n_train = sum(p.numel() for p in trainable)
    print(f"[ft] unfreeze={args.unfreeze}: {n_train/1e6:.1f}M trainable of "
          f"{sum(p.numel() for p in policy.parameters())/1e6:.0f}M", flush=True)
    if not trainable:
        raise SystemExit("nothing unfrozen -- refusing to run a no-op fine-tune")

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    chunk = int(getattr(cfg, "chunk_size", 50) or 50)
    keys = ["observation.images.top_rgb", "observation.images.left_rgb",
            "observation.images.right_rgb"]

    def batch_from(idx):
        b = {}
        im = torch.from_numpy(X[idx]).to(dev).float() / 255.0    # (B,3,H,W,C)
        for k, j in zip(keys, range(3)):
            b[k] = im[:, j].permute(0, 3, 1, 2).contiguous()
        b["observation.state"] = torch.from_numpy(S[idx]).to(dev).float()
        a = torch.from_numpy(A[idx]).to(dev).float()
        if a.dim() == 2:
            raise SystemExit(
                "capture holds one action per frame, not a chunk. Tiling it "
                "trains the policy to emit a constant trajectory: the first "
                "attempt cut training loss 82% and moved Storm-frame skill "
                "from -0.038 to -0.201. Re-capture with --capture_chunk.")
        if a.shape[1] != chunk:
            raise SystemExit(f"capture chunk {a.shape[1]} != policy chunk {chunk}")
        b["action"] = a
        b["task"] = ["fold the garment"] * len(idx)
        return pre(b) if pre else b

    def val_loss():
        policy.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_idx), args.batch_size):
                idx = val_idx[i:i + args.batch_size]
                if len(idx) < 2:
                    continue
                out = policy.forward(batch_from(idx))
                loss = out[0] if isinstance(out, tuple) else out["loss"]
                tot += float(loss) * len(idx)
                n += len(idx)
        policy.train()
        return tot / max(n, 1)

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    v0 = val_loss()
    print(f"[ft] val loss before any training: {v0:.4f}", flush=True)

    policy.train()
    step = 0
    while step < args.steps:
        rng.shuffle(train_idx)
        for i in range(0, len(train_idx) - args.batch_size, args.batch_size):
            idx = train_idx[i:i + args.batch_size]
            out = policy.forward(batch_from(idx))
            loss = out[0] if isinstance(out, tuple) else out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                v = val_loss()
                flag = ""
                if v < best:
                    best = v
                    policy.save_pretrained(args.out)
                    _copy_processors(args.policy_path, args.out)
                    flag = "  <- saved"
                print(f"[ft] step {step}/{args.steps} train={float(loss):.4f} "
                      f"val={v:.4f}{flag}", flush=True)
            if step >= args.steps:
                break

    json.dump({"steps": step, "val_before": v0, "val_best": best,
               "frames": int(len(X)), "episodes": len(meta),
               "unfreeze": args.unfreeze, "lr": args.lr,
               "base": args.policy_path,
               "note": "trained on Storm-rasterised frames with demonstration "
                       "actions; only the observation distribution differs from "
                       "the original BC run"},
              open(os.path.join(args.out, "finetune.json"), "w"), indent=2)
    print(f"[ft] done. val {v0:.4f} -> {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
