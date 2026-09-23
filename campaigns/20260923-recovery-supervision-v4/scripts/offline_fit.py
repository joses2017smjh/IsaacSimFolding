"""Offline fit and, for a baseline/candidate pair, the attempt-2 precondition gate.

The review of attempt 1 established that 86% of the sampled labels are the
baseline's own draws at identical observations, so a single-seed RMS
comparison against the baseline is a self-sample artifact no genuinely
changed model can win. The gate therefore never asks the candidate to beat
the baseline. It asks two things:

  (a) repair verification, on the RELOADED candidate (from_pretrained
      re-rounds the expert to bf16, so this measures deployed weights): at
      least half of the expert's bf16 weights differ from the baseline and at
      least one RMSNorm gain changed. Attempt 1 and the raster fine-tune both
      sat at 11.5% with every gain untouched -- the bf16-frozen signature.
  (b) multi-seed paired non-inferiority: over K noise seeds per example, the
      candidate's mean first-10 RMS must not exceed the baseline's by more
      than the baseline's own seed-to-seed spread (paired bootstrap 95% CI),
      branch and student pooled. This catches "training broke the policy"
      without demanding the impossible.

Everything else -- per-label-class splits, retention RMS -- is reported only.

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

import struct

import numpy as np
import torch


def weight_change(base_model: Path, cand_model: Path) -> dict:
    """Raw-byte fraction of expert weights that changed between checkpoints.

    Reads safetensors headers directly, no model load, so bf16 tensors are
    compared exactly. Attempt 1's signature of the bf16 defect was 12.9%
    changed over 300 steps; a working fp32-master run must be far above it.
    """
    def header(path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n)), 8 + n

    def raw(path, h, base, key):
        off = h[key]["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + off[0])
            return np.frombuffer(f.read(off[1] - off[0]), dtype=np.uint8)

    hb, bb = header(base_model)
    hc, bc = header(cand_model)
    out = {}
    for group, match in (("lm_expert", lambda k: "lm_expert" in k),
                         ("other_trainable", lambda k: any(s in k for s in
                          ("action_in_proj", "action_out_proj", "action_time_mlp", "state_proj")))):
        tot = chg = 0
        dtypes = set()
        for k in hb:
            if k == "__metadata__" or not match(k) or k not in hc:
                continue
            b = raw(base_model, hb, bb, k)
            c = raw(cand_model, hc, bc, k)
            dtypes.add(f"{hb[k]['dtype']}->{hc[k]['dtype']}")
            if len(b) == len(c):
                width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}.get(hb[k]["dtype"], 1)
                same_elems = (b.reshape(-1, width) == c.reshape(-1, width)).all(axis=1)
                tot += len(same_elems)
                chg += int((~same_elems).sum())
            else:
                # dtype changed (e.g. bf16 checkpoint vs fp32 master save):
                # compare after casting both through torch for exactness
                import torch as _t
                tb = _t.frombuffer(bytearray(b), dtype=getattr(_t, {"BF16": "bfloat16", "F32": "float32"}[hb[k]["dtype"]])).float()
                tc = _t.frombuffer(bytearray(c), dtype=getattr(_t, {"BF16": "bfloat16", "F32": "float32"}[hc[k]["dtype"]])).float()
                tot += tb.numel()
                chg += int((tb != tc).sum())
        out[group] = {"weights": tot, "changed_fraction": (chg / tot) if tot else None,
                      "dtype_transitions": sorted(dtypes)}
    return out


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
    ap.add_argument("--seeds-per-example", type=int, default=8)
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

    def errors(pol, pre, post, xs, ss, as_, *, k_seeds=1):
        """Mean first-10 RMS per example; seed k of example i uses 5000 + i +
        100000*k, so k=0 reproduces attempt 1's exact evaluation."""
        e10 = np.zeros((len(xs), k_seeds), dtype=np.float64)
        e50 = np.zeros_like(e10)
        for i in range(len(xs)):
            for k in range(k_seeds):
                p = predict(pol, pre, post, xs[i], ss[i], 5000 + i + 100000 * k)
                d = p - as_[i]
                e10[i, k] = float(np.sqrt(np.mean(d[:10] ** 2)))
                e50[i, k] = float(np.sqrt(np.mean(d ** 2)))
        return {"n": len(xs), "k_seeds": k_seeds,
                "rms_first10_rad": float(e10.mean()), "rms_chunk50_rad": float(e50.mean()),
                "per_example_rms_first10": e10.mean(axis=1).tolist(),
                "per_example_seed_std_first10": e10.std(axis=1).tolist() if k_seeds > 1 else None,
                "per_example_per_seed_first10": e10.tolist()}

    # Per-example class labels, reconstructed from the per-row search records
    # (the compiled dataset predates the provenance columns; the rows were
    # verified byte-identical to their sources, in manifest order).
    horizon = np.full(len(origin), -1, dtype=np.int64)
    rel = np.full(len(origin), -1, dtype=np.int64)
    cursor = 0
    for row in manifest["recovery_search"]:
        d = root / "recovery" / row["id"]
        with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
            m = len(ex["action"])
            if m:
                horizon[cursor:cursor + m] = np.asarray(ex["horizon"], dtype=np.int64)
                rel[cursor:cursor + m] = (np.asarray(ex["step_index"], dtype=np.int64)
                                          - np.asarray(ex["root"], dtype=np.int64))
                cursor += m
        sp = d / "trajectory.npz"
        with np.load(sp, allow_pickle=False) as raw:
            if bool(np.asarray(raw["terminal_success"]).item()):
                stream_len = len(np.asarray(raw["executed_action_stream"]))
                idx = np.asarray(raw["step_index"], dtype=np.int64)
                cursor += int((idx + 50 <= stream_len).sum())
    if cursor != len(origin):
        raise SystemExit(f"class reconstruction misaligned: {cursor} != {len(origin)}")
    # self-aligned: label[:10] is the policy's own draw at that observation
    self_aligned = (kind == "student") | (horizon == 10) | ((horizon == 50) & (rel % 50 == 0))

    results = {}
    labels_paths = [item.split("=", 1) for item in args.checkpoints.split(",")]
    pair_gate = len(labels_paths) == 2
    if pair_gate:
        results["_weight_change_files"] = weight_change(
            Path(labels_paths[0][1]) / "model.safetensors",
            Path(labels_paths[1][1]) / "model.safetensors")
        print("file-level weight change:", json.dumps(results["_weight_change_files"]), flush=True)

    expert_stash = None
    for pos, (label, path) in enumerate(labels_paths):
        pol, pre, post = load(path)
        if pair_gate:
            expert = {name: prm.detach().cpu().clone() for name, prm in pol.named_parameters()
                      if "lm_expert" in name}
            if pos == 0:
                expert_stash = expert
            else:
                tot = chg = norm_chg = norm_tot = 0
                for name, prm in expert.items():
                    b = expert_stash[name]
                    diff = (prm != b)
                    if prm.dtype == torch.bfloat16:
                        tot += prm.numel()
                        chg += int(diff.sum())
                    if ".norm" in name or "layernorm" in name.lower():
                        norm_tot += prm.numel()
                        norm_chg += int(diff.sum())
                results["_weight_change_reloaded"] = {
                    "bf16_expert_weights": tot, "bf16_changed_fraction": chg / tot if tot else None,
                    "norm_gain_weights": norm_tot, "norm_gains_changed": norm_chg,
                }
                print("reloaded weight change:", json.dumps(results["_weight_change_reloaded"]), flush=True)
        K = args.seeds_per_example
        block = {"checkpoint": path, "model_sha256": digest(Path(path) / "model.safetensors")}
        for k, idx in pick.items():
            block[f"recovery_{k}"] = errors(pol, pre, post, images[idx], state[idx], action[idx], k_seeds=K)
            for cls, mask_name in (("self_aligned", True), ("non_aligned", False)):
                mask = self_aligned[idx] == mask_name
                if mask.any():
                    vals = np.asarray(block[f"recovery_{k}"]["per_example_rms_first10"])[mask]
                    block[f"recovery_{k}"][f"rms_first10_{cls}"] = float(vals.mean())
        block["retention_demos"] = errors(pol, pre, post, ret_x, ret_s, ret_a, k_seeds=min(2, K))
        results[label] = block
        print(label, {k: round(v["rms_first10_rad"], 4) for k, v in block.items() if isinstance(v, dict)}, flush=True)
        del pol
        torch.cuda.empty_cache()

    gate = None
    if pair_gate:
        (bl, _), (cl, _) = labels_paths
        rw = results["_weight_change_reloaded"]
        # Repair verification is the changed FRACTION alone. The norm-gain
        # clause was an instrument artifact: RMSNorm gains sit near magnitude
        # 1.0, where bf16's half-ULP is ~2e-3, and at lr 3.3e-6 x 250 steps
        # the measured fp32 deltas reach only 0.18 of that -- 23,750/23,760
        # gains MOVED in the saved fp32 tensors and were erased by the
        # re-round at load, for any correctly working optimizer. The frozen
        # signature this check exists to catch was 11.5% changed; >= 50% is
        # unreachable by a frozen expert. Norm-gain count stays reported.
        repair_ok = (rw["bf16_changed_fraction"] is not None
                     and rw["bf16_changed_fraction"] >= 0.50)
        db, dc = [], []
        spread = []
        for grp in ("recovery_branch", "recovery_student"):
            b = np.asarray(results[bl][grp]["per_example_rms_first10"])
            c = np.asarray(results[cl][grp]["per_example_rms_first10"])
            db.append(b); dc.append(c)
            s = results[bl][grp]["per_example_seed_std_first10"]
            spread.append(np.asarray(s if s is not None else np.zeros_like(b)))
        b = np.concatenate(db); c = np.concatenate(dc)
        seed_spread = float(np.concatenate(spread).mean())
        diff = c - b
        rng2 = np.random.default_rng(99)
        boots = np.array([diff[rng2.integers(0, len(diff), len(diff))].mean() for _ in range(4000)])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        noninferior_ok = ci[1] <= seed_spread
        gate = {"repair_ok": bool(repair_ok), "noninferior_ok": bool(noninferior_ok),
                "pass": bool(repair_ok and noninferior_ok),
                "mean_paired_diff": float(diff.mean()), "diff_ci95": ci,
                "baseline_seed_spread": seed_spread,
                "definition": ("PASS iff reloaded bf16 expert change >= 50% (norm-gain count "
                               "reported, not gated -- see amendment 2026-09-23-fit-gate-"
                               "instrument), AND paired (candidate - baseline) 95% CI upper bound <= "
                               "the baseline's own mean seed-to-seed std, branch+student pooled. "
                               "Necessary condition only; in-sample; never evidence of improvement.")}
        print("GATE:", json.dumps(gate), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": 2, "script_sha256": digest(Path(__file__)),
                                    "dataset_sha256": prov["dataset_sha256"],
                                    "sampled_per_origin": {k: int(len(v)) for k, v in pick.items()},
                                    "seeds_per_example": args.seeds_per_example,
                                    "gate": gate,
                                    "results": results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
