"""Did a checkpoint learn its supervision? Offline, on the validated labels.

Closed-loop success did not move after attempt 1. That has two opposite
explanations -- the recovery labels were learned but do not transfer, or they
were barely learned (300 steps x 2 recovery samples per batch visits 1310
samples ~0.46 times) -- and they call for opposite next attempts. This
measures which, without a rollout: for each checkpoint, predict the action
chunk exactly as the runner does (make_observation -> preprocessor ->
_get_action_chunk -> postprocessor, fixed seed) and compare it with the
executed continuation that demonstrably reached settled success. Also
measured on whole held-out demonstration episodes (the retention set).
"""
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
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--checkpoints", required=True, help="label=path,label=path,...")
    ap.add_argument("--per-origin", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    device = "cuda"

    data = np.load(root / "datasets/recovery.npz", allow_pickle=False)
    origin = np.asarray(data["origin"]).astype(str)
    kind = np.array([o.split(":")[0] for o in origin])
    # branch examples: split by the candidate horizon that produced them
    prov = json.loads((root / "analysis/recovery-dataset-provenance.json").read_text())
    rng = np.random.default_rng(1234)
    pick = {}
    for k in sorted(set(kind)):
        idx = np.flatnonzero(kind == k)
        pick[k] = np.sort(rng.choice(idx, size=min(args.per_origin, len(idx)), replace=False))
    images = data["images"]; state = data["state"]; action = data["action"]

    ret_x, ret_s, ret_a = [], [], []
    for f in manifest["training"]["retention"]["files"]:
        with np.load(f, allow_pickle=True) as d:
            n = len(d["action"])
            idx = np.unique(np.linspace(0, n - 1, 16).round().astype(int))
            ret_x.append(np.asarray(d["images"])[idx]); ret_s.append(np.asarray(d["state"])[idx])
            ret_a.append(np.asarray(d["action"])[idx])
    ret_x, ret_s, ret_a = np.concatenate(ret_x), np.concatenate(ret_s), np.concatenate(ret_a)

    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def load(path):
        cfg = PreTrainedConfig.from_pretrained(path, cli_overrides={})
        cfg.pretrained_path = path
        pol = SmolVLAPolicy.from_pretrained(path).to(device).eval()
        pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path)
        return pol, pre, post

    def predict(pol, pre, post, img, st, seed):
        obs = {"observation.state": torch.from_numpy(st.astype(np.float32)).reshape(1, -1).to(device),
               "task": manifest["task_prompt"]}
        for ci, key in enumerate(("top_rgb", "left_rgb", "right_rgb")):
            x = torch.from_numpy(np.ascontiguousarray(img[ci])).permute(2, 0, 1)
            obs[f"observation.images.{key}"] = (x.float() / 255.0).unsqueeze(0).to(device)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            chunk = pol._get_action_chunk(pre(obs))
        if post:
            chunk = post(chunk)
        chunk = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
        return np.asarray(chunk)[0].astype(np.float32)

    def errors(pol, pre, post, xs, ss, as_):
        e10, e50 = [], []
        for i in range(len(xs)):
            p = predict(pol, pre, post, xs[i], ss[i], 5000 + i)
            d = p - as_[i]
            e10.append(float(np.sqrt(np.mean(d[:10] ** 2)))); e50.append(float(np.sqrt(np.mean(d ** 2))))
        return {"n": len(xs), "rms_first10_rad": float(np.mean(e10)), "rms_chunk50_rad": float(np.mean(e50)),
                "per_example_rms_first10": e10}

    results = {}
    for item in args.checkpoints.split(","):
        label, path = item.split("=", 1)
        pol, pre, post = load(path)
        block = {"checkpoint": path, "model_sha256": digest(Path(path) / "model.safetensors")}
        for k, idx in pick.items():
            block[f"recovery_{k}"] = errors(pol, pre, post, images[idx], state[idx], action[idx])
        block["retention_demos"] = errors(pol, pre, post, ret_x, ret_s, ret_a)
        results[label] = block
        print(label, {k: round(v["rms_first10_rad"], 4) for k, v in block.items() if isinstance(v, dict)}, flush=True)
        del pol
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": 1, "script_sha256": digest(Path(__file__)),
                                    "dataset_sha256": prov["dataset_sha256"],
                                    "sampled_per_origin": {k: int(len(v)) for k, v in pick.items()},
                                    "results": results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
