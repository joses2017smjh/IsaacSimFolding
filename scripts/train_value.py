"""Stage 2: fit the value heads on a frozen Stage 1 backbone.

The heads read the same representation the action expert reads, which is what
"the policy is its own value function" means in practice. The backbone is
frozen by default (`--detach_backbone`), for one reason: Stage 2 is measured as
an ablation AGAINST Stage 1, and that comparison is only clean if the action
weights are byte-identical between the two arms.

Data sources, and they are not interchangeable:

  --dataset_root    the released demonstrations. Trains `progress` and
                    `future` honestly. Its success labels are single-class,
                    because every released episode is a successful scripted
                    demonstration -- see src/lehome_fold/labels.py.
  --rollout_dir     rollouts collected from the Stage 1 policy, containing
                    FAILURES. This is the only source of success negatives and
                    therefore the only way G2 can pass.

Running without --rollout_dir is legitimate and useful -- it trains the progress
head and validates the whole pipeline -- but the success head it produces
cannot pass G2, and this script says so rather than letting the gate discover
it later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True, help="Stage 1 BC checkpoint")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--rollout_dir", default=None,
                    help="rollouts with mixed outcomes; required for a G2-capable success head")
    ap.add_argument("--feature_path", default="model.embed_prefix",
                    help="verified default for pi05 and smolvla on lerobot 0.4.3")
    ap.add_argument("--hidden_dim", type=int, default=0,
                    help="0 infers the width from the first captured feature")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon", type=int, default=30, help="future head lookahead, in frames")
    ap.add_argument("--success_mode", choices=["episode", "terminal"], default="episode")
    ap.add_argument("--detach_backbone", type=int, default=1)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--dump_val_predictions", default=None,
                    help="npz for check_calibration.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    from lehome_fold import labels as L
    from lehome_fold.policy_wrap import ValueAugmentedPolicy, WrapConfig

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- backbone, frozen --------------------------------------------------
    meta = LeRobotDatasetMetadata(repo_id="lehome", root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    cfg.pretrained_path = args.policy_path
    policy = make_policy(cfg, ds_meta=meta).eval().to(device)
    for p in policy.parameters():
        p.requires_grad_(False)

    wrap = ValueAugmentedPolicy(
        policy,
        WrapConfig(feature_path=args.feature_path, hidden_dim=args.hidden_dim,
                   detach_backbone=bool(args.detach_backbone)),
    ).to(device)

    # -- data --------------------------------------------------------------
    # delta_timestamps asks LeRobot for the frame `horizon` steps ahead in the
    # same sample, which is what the future head regresses onto. fps comes from
    # the dataset metadata (30) and NOT from the env's 1/90 dt -- see
    # docs/STAGE0.md section 3; the two disagree and the dataset's own label is
    # the one that indexes its own frames.
    fps = meta.fps
    delta = {"observation.state": [0.0, args.horizon / fps]}
    ds = LeRobotDataset(repo_id="lehome", root=args.dataset_root, delta_timestamps=delta)

    outcomes = load_outcomes(args.rollout_dir, ds)
    y_all = np.concatenate([
        L.success_targets(o, mode=args.success_mode) for o in outcomes.values()
    ]) if outcomes else np.zeros(0)

    bal = L.class_balance(y_all)
    print(f"[stage2] success targets: n={bal['n']:.0f} positive={bal['positive']:.3f}")
    if bal["degenerate"]:
        print(
            "[stage2] WARNING: success labels are SINGLE-CLASS.\n"
            "         The success head will predict the base rate and CANNOT pass G2.\n"
            "         Progress and future heads are still trained and still useful.\n"
            "         Collect rollouts with failures (scripts/rollout_worker.py)\n"
            "         and pass --rollout_dir to fix this.",
            flush=True,
        )

    n_val = max(1, int(len(ds) * args.val_fraction))
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # Materialise the heads BEFORE the optimiser exists.
    #
    # With --hidden_dim 0 the width is inferred from the first captured
    # feature, so `wrap.heads` is None until something has run. Building the
    # optimiser first would hand it `None.parameters()`. One warmup batch also
    # proves the tap actually fires on this checkpoint -- far better to learn
    # that in the first ten seconds than after the dataloader has warmed up.
    warmup = next(iter(train_dl))
    with torch.no_grad():
        policy.forward(to_device(warmup, device))
    wrap()                      # builds the heads at the captured width
    wrap.heads.to(device)       # they were created after the initial .to()
    print(f"[stage2] value heads: hidden_dim={wrap.cfg.hidden_dim} "
          f"params={sum(p.numel() for p in wrap.heads.parameters()) / 1e6:.2f}M",
          flush=True)

    from lehome_fold.value_head import value_loss

    opt = torch.optim.AdamW(wrap.heads.parameters(), lr=args.lr)
    step = 0
    wrap.heads.train()
    while step < args.steps:
        for batch in train_dl:
            batch = to_device(batch, device)
            targets = build_targets(batch, outcomes, args, L)
            # Run the frozen backbone so the tap captures this batch's prefix
            # features. no_grad because the backbone is frozen and the heads
            # detach anyway -- this saves the activation memory that would
            # otherwise be held for a graph nothing backpropagates through.
            with torch.no_grad():
                policy.forward(batch)
            preds = wrap()
            loss, parts = value_loss(preds, targets,
                                     future_mask=targets.get("future_mask"),
                                     sample_mask=targets.get("labelled"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 500 == 0:
                print(f"[stage2] step {step}/{args.steps} loss={float(loss):.4f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in parts.items()), flush=True)
            if step >= args.steps:
                break

    # -- save --------------------------------------------------------------
    torch.save(wrap.heads.state_dict(), out / "value_head.pt")
    (out / "value_head_config.json").write_text(json.dumps({
        "hidden_dim": args.hidden_dim, "trunk_dim": wrap.heads.cfg.trunk_dim,
        "future_dim": wrap.heads.cfg.future_dim, "dropout": wrap.heads.cfg.dropout,
        "detach_backbone": bool(args.detach_backbone),
    }, indent=2))
    (out / "run.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"[stage2] wrote {out}/value_head.pt", flush=True)

    # -- validation predictions for G2 -------------------------------------
    if args.dump_val_predictions:
        probs, labs = [], []
        wrap.heads.eval()
        with torch.no_grad():
            for batch in val_dl:
                batch = to_device(batch, device)
                t = build_targets(batch, outcomes, args, L)
                policy.forward(batch)
                probs.append(torch.sigmoid(wrap()["success_logit"]).cpu().numpy())
                labs.append(t["success"].cpu().numpy())
        p = Path(args.dump_val_predictions)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, probs=np.concatenate(probs), labels=np.concatenate(labs))
        print(f"[stage2] wrote {p} -- now run scripts/check_calibration.py (G2)", flush=True)
    return 0


def to_device(batch, device):
    import torch
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def load_outcomes(rollout_dir, ds):
    """Episode outcomes keyed by episode_index, from rollouts else the demos.

    Always a dict, never a list. Downstream code indexes these BY EPISODE
    INDEX, and the scored rollouts are a sparse subset (1, 2, ... 250, ... 752),
    so a positional list would silently attach episode 250's outcome to episode
    12. Returning one shape everywhere also avoids the half-applied refactor
    that broke the class-balance line here: iterating a dict yields its keys,
    so `for o in outcomes` handed an int to success_targets.

    A released demonstration is a success with no recorded success frame, which
    is exactly the awkward combination EpisodeOutcome is built to represent --
    it forces `--success_mode` to be a decision rather than a default.
    """
    from lehome_fold.labels import EpisodeOutcome

    if rollout_dir:
        rows = []
        for f in sorted(Path(rollout_dir).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        # Key by episode_index when the rows carry one. Returning a bare list
        # here would be indexed BY POSITION downstream, so a sparse set of
        # scored episodes (1, 2, ... 250, ... 752) would silently attach
        # episode 250's outcome to episode 12 -- mislabelled training data
        # that fails no assertion and produces a plausible-looking curve.
        return {int(r.get("episode_index", i)):
                EpisodeOutcome(length=int(r["length"]), success=bool(r["success"]),
                               success_frame=r.get("success_frame"))
                for i, r in enumerate(rows)}
    lengths = getattr(ds.meta, "episodes", None)
    if lengths is None:
        return {}
    return {i: EpisodeOutcome(length=int(n), success=True)
            for i, n in enumerate(lengths["length"])}


def build_targets(batch, outcomes, args, L):
    """Per-frame targets aligned to the batch's episode/frame indices."""
    import torch

    ep = batch["episode_index"].long().cpu().numpy().reshape(-1)
    fr = batch["frame_index"].long().cpu().numpy().reshape(-1)
    dev = batch["observation.state"].device

    succ = np.zeros(len(ep), dtype=np.float32)
    prog = np.zeros(len(ep), dtype=np.float32)
    # Frames whose episode has no scored outcome must not contribute to the
    # loss. Defaulting them to zero would teach the success head that every
    # unlabelled episode failed.
    labelled = np.zeros(len(ep), dtype=np.float32)
    for i, (e, f) in enumerate(zip(ep, fr)):
        o = outcomes.get(int(e))
        if o is None:
            continue
        labelled[i] = 1.0
        succ[i] = L.success_targets(o, mode=args.success_mode)[min(f, o.length - 1)]
        prog[i] = L.progress(o.length)[min(f, o.length - 1)]

    state = batch["observation.state"]
    # LeRobot returns (B, n_delta, D) when delta_timestamps is set: index 0 is
    # the present, index 1 is t+horizon.
    if state.dim() == 3 and state.shape[1] >= 2:
        future, present = state[:, 1], state[:, 0]
        valid = torch.ones(len(ep), device=dev)
    else:
        future, present = state, state
        valid = torch.zeros(len(ep), device=dev)
    batch["observation.state"] = present

    return {
        "success": torch.from_numpy(succ).to(dev),
        "progress": torch.from_numpy(prog).to(dev),
        "future": future,
        "future_mask": valid,
        "labelled": torch.from_numpy(labelled).to(dev),
    }


if __name__ == "__main__":
    raise SystemExit(main())
