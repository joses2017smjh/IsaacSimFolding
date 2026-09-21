"""Find, once, where the pooled backbone feature lives on this policy.

The layout was VERIFIED against lerobot 0.4.3 on 2026-08-28 -- `model.embed_prefix`
exists on both `PI05Pytorch` and `VLAFlowMatching` and both return
`(embs, pad_masks, att_masks)`. So the default usually just works.

This script is the check for when it does not: a different lerobot version, a
different policy type, or a checkpoint whose config renames something. It
reports which candidate paths resolve, what each one is, and prints the module
tree -- because when none match, the tree is what the next candidate is written
from. Run it once, pin the answer, and never let a training loop search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth", type=int, default=3, help="module-tree depth to print")
    args = ap.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    from lehome_fold.policy_wrap import CANDIDATE_PATHS, probe_feature_source

    meta = LeRobotDatasetMetadata(repo_id="lehome", root=args.dataset_root)
    cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides={})
    cfg.pretrained_path = args.policy_path
    policy = make_policy(cfg, ds_meta=meta).eval().to(torch.device(args.device))

    print("=== candidate feature paths ===")
    found = probe_feature_source(policy)
    for path in CANDIDATE_PATHS:
        mark = "HIT " if path in found else "miss"
        print(f"  [{mark}] {path}")
        if path in found:
            print(f"           {json.dumps(found[path])}")

    print()
    print(f"=== module tree (depth {args.depth}) ===")

    def walk(mod, prefix="", depth=0):
        if depth > args.depth:
            return
        for name, child in mod.named_children():
            n_params = sum(p.numel() for p in child.parameters())
            print(f"  {'  ' * depth}{prefix}{name}  [{type(child).__name__}] "
                  f"{n_params / 1e6:.1f}M")
            walk(child, "", depth + 1)

    walk(policy)

    hits = [p for p in CANDIDATE_PATHS if p in found]
    print()
    if hits:
        print(f"PIN THIS:  --feature_path {hits[0]}")
        print("hidden_dim can be left at 0 -- the wrapper infers the width from")
        print("the first captured feature, so it cannot disagree with the model.")
        return 0
    print("NO CANDIDATE MATCHED. Read the module tree above and add the right")
    print("path to policy_wrap.CANDIDATE_PATHS -- do not let the training loop")
    print("search at runtime.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
