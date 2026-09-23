"""Load a saved policy as the evaluator does and run one finite inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline checkpoint whose config.json the boundary runner substitutes")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    checkpoint = args.checkpoint.resolve()
    with np.load(args.dataset, allow_pickle=False) as data:
        images = np.asarray(data["images"][0:1], dtype=np.uint8)
        state = np.asarray(data["state"][0:1], dtype=np.float32)
    if images.shape != (1, 3, 480, 640, 3) or state.shape != (1, 12):
        raise SystemExit("dataset sample shape is incompatible with evaluator input")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint), cli_overrides={})
    cfg.pretrained_path = str(checkpoint)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))
    batch = {"observation.state": torch.from_numpy(state).to(device).float(),
             "task": ["fold the garment on the table"]}
    im = torch.from_numpy(images).to(device).float() / 255.0
    for camera_index, key in enumerate(("observation.images.top_rgb", "observation.images.left_rgb",
                                        "observation.images.right_rgb")):
        batch[key] = im[:, camera_index].permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        if hasattr(policy, "reset"):
            policy.reset()
        action = policy.select_action(pre(batch) if pre else batch)
    if post:
        action = post(action)
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (1, 12) or not np.isfinite(action).all():
        raise SystemExit(f"evaluation checkpoint produced invalid action {action.shape}")
    # policy_rollout51.py's boundary path symlinks every candidate file EXCEPT
    # config.json and copies the baseline's in its place. That is only safe
    # while the two configs agree, so check it here rather than discovering a
    # silently substituted architecture during evaluation.
    config_matches_baseline = None
    if args.baseline is not None:
        config_matches_baseline = (digest(checkpoint / "config.json")
                                   == digest(args.baseline.resolve() / "config.json"))
        if not config_matches_baseline:
            raise SystemExit(
                "candidate config.json differs from the baseline's; the boundary "
                "evaluator substitutes the baseline config and would load a "
                "different architecture than the one that was trained")
    output = {
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": digest(checkpoint / "model.safetensors"),
        "config_matches_baseline": config_matches_baseline,
        "dataset": str(args.dataset.resolve()),
        "action_shape": list(action.shape),
        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
        "finite": True,
        "device": device,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
