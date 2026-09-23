"""Bounded AWR-resampled mixed fine-tuning from the untouched BC checkpoint.

SmolVLA's flow-matching ``forward`` exposes one scalar batch loss, not a
per-example likelihood.  The equivalent supported AWR implementation here is
therefore importance *resampling*: rollout samples are drawn with probability
proportional to ``lehome_fold.awr.weights`` and use the unchanged supervised
loss.  Every batch also contains a fixed fraction of original raster BC
examples.  This is intentionally a small, fixed one-iteration protocol, not a
new RL algorithm or a hyperparameter sweep.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


CAMERAS = ("observation.images.top_rgb", "observation.images.left_rgb",
           "observation.images.right_rgb")
REPO = Path(__file__).resolve().parents[3]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_hashes(path: Path) -> dict[str, str]:
    names = ("config.json", "model.safetensors", "policy_preprocessor.json",
             "policy_preprocessor_step_5_normalizer_processor.safetensors",
             "policy_postprocessor.json",
             "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    return {name: digest(path / name) for name in names if (path / name).is_file()}


def copy_processors(src: Path, dst: Path) -> None:
    for name in ("policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                 "train_config.json"):
        source = src / name
        target = dst / name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)


def restore_config_discriminator(base_config: Path, saved_config: Path) -> dict:
    """Put back the ``type`` key that ``save_pretrained`` drops.

    SmolVLAPolicy.save_pretrained writes 48 of the baseline's 49 config keys
    and omits ``type``. That key is draccus's choice-class discriminator, so
    PreTrainedConfig.from_pretrained raises

        ParsingError: Expected a dict with a 'type' key for PreTrainedConfig

    and EVERY checkpoint this trainer saves is unloadable by the evaluator
    path. Fine-tuning changes weights, not architecture, and the remaining 48
    keys were verified byte-identical to the baseline's, so the baseline
    config is restored wholesale rather than patched. That also makes the
    boundary evaluator's baseline-config substitution a provable no-op
    instead of something that has to be argued about.
    """
    base = json.loads(base_config.read_text())
    saved = json.loads(saved_config.read_text()) if saved_config.is_file() else {}
    drifted = sorted(k for k in saved if k in base and saved[k] != base[k])
    if drifted:
        raise ValueError(
            f"saved config disagrees with the baseline on {drifted}; the "
            f"architecture changed and restoring the baseline config would "
            f"misdescribe this checkpoint")
    if "type" not in base:
        raise ValueError(f"{base_config} has no 'type' discriminator to restore")
    shutil.copy2(base_config, saved_config)
    return {"restored_type": base["type"], "keys_saved": len(saved), "keys_baseline": len(base)}


def require_arrays(data: Any, *, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"images", "state", "action"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{source} lacks {sorted(missing)}")
    images = np.asarray(data["images"], dtype=np.uint8)
    state = np.asarray(data["state"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    if (images.ndim != 5 or images.shape[1:] != (3, 480, 640, 3)
            or state.shape != (len(images), 12) or action.shape != (len(images), 50, 12)):
        raise ValueError(f"{source} has incompatible images/state/action shapes "
                         f"{images.shape}/{state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{source} has non-finite state/action values")
    return images, state, action


def load_rollouts(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        images, state, action = require_arrays(data, source=str(path))
        for name in ("awr_weight", "advantage", "reward", "chunk", "camera_keys", "task"):
            if name not in data.files:
                raise ValueError(f"rollout dataset {path} lacks {name}")
        weights = np.asarray(data["awr_weight"], dtype=np.float32)
        advantage = np.asarray(data["advantage"], dtype=np.float32)
        reward = np.asarray(data["reward"], dtype=np.float32)
        if (weights.shape != (len(images),) or advantage.shape != (len(images),)
                or reward.shape != (len(images),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError("invalid rollout AWR arrays")
        if int(np.asarray(data["chunk"]).item()) != 50:
            raise ValueError("rollout dataset target chunk is not 50")
        cameras = tuple(str(x) for x in np.asarray(data["camera_keys"]).tolist())
        if cameras != ("top_rgb", "left_rgb", "right_rgb"):
            raise ValueError(f"rollout dataset camera ordering changed: {cameras!r}")
        if str(np.asarray(data["task"]).item()) != "fold the garment on the table":
            raise ValueError("rollout dataset task changed")
    meta = {
        "dataset": str(path), "dataset_sha256": digest(path), "samples": int(len(images)),
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "reward_min": float(reward.min()), "reward_max": float(reward.max()),
    }
    return images, state, action, weights, advantage, meta


def load_raster_subset(pattern: str, files_requested: int, frames_per_file: int,
                       *, heldout: set[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    files = [Path(name).resolve() for name in sorted(glob.glob(pattern))]
    if files_requested < 1 or files_requested > len(files):
        raise ValueError(f"requested {files_requested} raster files but only {len(files)} are available")
    selected = files[:files_requested]
    # Compare by CONTENT, not by path. The anchor and held-out globs point at
    # different directories, so a path-identity test can never fire however
    # much the two corpora overlap -- and the same episode appearing in both
    # turns the regression guard into a training metric.
    heldout_digests = {digest(path): path for path in sorted(heldout)}
    overlap = sorted(f"{path} == {heldout_digests[digest(path)]}"
                     for path in selected if digest(path) in heldout_digests)
    if overlap:
        raise ValueError(f"BC anchor duplicates fixed held-out content: {overlap}")
    images, state, action, records = [], [], [], []
    for path in selected:
        with np.load(path, allow_pickle=True) as data:
            x, s, a = require_arrays(data, source=str(path))
            if len(x) < frames_per_file:
                raise ValueError(f"{path} has only {len(x)} frames")
            garment = str(data["garment"]) if "garment" in data.files else None
            episode = int(data["episode"]) if "episode" in data.files else None
        images.append(x[:frames_per_file])
        state.append(s[:frames_per_file])
        action.append(a[:frames_per_file])
        records.append({"path": str(path), "sha256": digest(path), "frames_used": frames_per_file,
                        "garment": garment, "episode": episode})
    return np.concatenate(images), np.concatenate(state), np.concatenate(action), records


def load_whole_episodes(paths: list[Path], frames: int, *, anchor: list[dict]):
    """Frames spread across ENTIRE demonstration episodes -- approach, grasp,
    fold and release -- from episodes and garments the anchor never sees.

    The v2 guard read the first 8 frames of 4 demonstrations, the same
    episode-start distribution as the anchor, so training on the anchor
    lowered it every time and it never came close to binding.
    """
    anchor_garments = {r["garment"] for r in anchor}
    anchor_digests = {r["sha256"] for r in anchor}
    images, state, action, records = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            x, s, a = require_arrays(data, source=str(path))
            garment = str(data["garment"]) if "garment" in data.files else None
        if garment in anchor_garments:
            raise ValueError(f"retention episode {path} shares garment {garment} with the anchor")
        sha = digest(path)
        if sha in anchor_digests:
            raise ValueError(f"retention episode {path} duplicates an anchor episode")
        idx = np.unique(np.linspace(0, len(x) - 1, frames).round().astype(np.int64))
        images.append(x[idx]); state.append(s[idx]); action.append(a[idx])
        records.append({"path": str(path), "sha256": sha, "garment": garment,
                        "frames_used": idx.tolist(), "episode_length": int(len(x))})
    return np.concatenate(images), np.concatenate(state), np.concatenate(action), records


def parameter_selection(policy: torch.nn.Module, mode: str) -> tuple[list[torch.nn.Parameter], list[str]]:
    vision = ("vision", "image", "patch", "visual")
    action_prefix = ("model.vlm_with_expert.lm_expert", "model.action_in_proj",
                     "model.action_out_proj", "model.action_time_mlp", "model.state_proj")

    def wanted(name: str) -> bool:
        if mode == "all":
            return True
        if mode == "boundary":
            return name.startswith(action_prefix)
        if any(token in name.lower() for token in vision):
            return True
        return mode == "vision+action" and name.startswith(action_prefix)

    for param in policy.parameters():
        param.requires_grad_(False)
    params, names = [], []
    for name, param in policy.named_parameters():
        if wanted(name):
            param.requires_grad_(True)
            params.append(param)
            names.append(name)
    if not params:
        raise ValueError("no trainable parameters selected")
    return params, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", type=Path, required=True)
    ap.add_argument("--rollout-dataset", type=Path, required=True)
    ap.add_argument("--anchor-glob", required=True)
    ap.add_argument("--anchor-files", type=int, required=True)
    ap.add_argument("--anchor-frames", type=int, required=True)
    ap.add_argument("--heldout-glob", required=True)
    ap.add_argument("--heldout-files", type=int, default=4)
    ap.add_argument("--heldout-frames", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rollout-fraction", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--unfreeze", choices=("vision", "vision+action", "boundary", "all"),
                    default="boundary")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--task", default="fold the garment on the table")
    ap.add_argument("--heldout-tolerance", type=float, default=1.10)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step; the effective "
                         "batch is batch_size x grad_accum with the same rollout/anchor mix")
    ap.add_argument("--retention-files", required=True,
                    help="comma-separated demonstration episodes held out by complete episode AND garment")
    ap.add_argument("--retention-frames", type=int, default=16,
                    help="frames spread evenly across each whole retention episode")
    args = ap.parse_args()
    args.policy_path = args.policy_path.resolve()
    args.rollout_dataset = args.rollout_dataset.resolve()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite training output {args.out}")
    if args.steps < 1 or args.steps > 300:
        raise SystemExit("this campaign permits 1..300 optimizer steps")
    if args.batch_size < 2 or not 0 < args.rollout_fraction < 1:
        raise SystemExit("batch must contain both rollout and anchor examples")
    n_rollout = round(args.batch_size * args.rollout_fraction)
    if not 1 <= n_rollout < args.batch_size:
        raise SystemExit("rollout fraction does not yield both sources in each batch")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint interval must be positive")
    if args.grad_accum < 1:
        raise SystemExit("grad accumulation must be positive")

    heldout_paths = [Path(name).resolve() for name in sorted(glob.glob(args.heldout_glob))]
    if len(heldout_paths) < args.heldout_files:
        raise SystemExit("not enough fixed held-out raster files")
    heldout_paths = heldout_paths[:args.heldout_files]
    rollout_x, rollout_s, rollout_a, weights, rollout_advantage, rollout_meta = load_rollouts(args.rollout_dataset)
    anchor_x, anchor_s, anchor_a, anchor_records = load_raster_subset(
        args.anchor_glob, args.anchor_files, args.anchor_frames, heldout=set(heldout_paths))
    retention_paths = [Path(p).resolve() for p in args.retention_files.split(",") if p]
    retention_x, retention_s, retention_a, retention_records = load_whole_episodes(
        retention_paths, args.retention_frames, anchor=anchor_records)
    heldout_x, heldout_s, heldout_a, heldout_records = load_raster_subset(
        args.heldout_glob, args.heldout_files, args.heldout_frames, heldout=set())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(str(args.policy_path), cli_overrides={})
    cfg.pretrained_path = str(args.policy_path)
    policy = SmolVLAPolicy.from_pretrained(str(args.policy_path)).to(device)
    pre, _post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(args.policy_path))
    trainable, trainable_names = parameter_selection(policy, args.unfreeze)
    # fp32 master weights. The action expert loads in bf16 (96.6M of the 99.9M
    # boundary-unfreeze parameters), and AdamW at lr 1e-5 with bf16 weights and
    # bf16 optimizer state rounds most updates to zero: measured on attempt 1,
    # only 12.9% of the expert's bf16 weights changed AT ALL over 300 steps,
    # while every fp32 tensor changed 100% of its entries. The raster
    # adaptation that produced the baseline only worked because its vision
    # tower is fp32. Casting the trainable parameters up gives fp32 weights,
    # gradients and Adam state; the forward casts activations per layer, and
    # save_pretrained then stores these tensors in fp32 (reloading quantises
    # to bf16, keeping every delta above bf16 resolution -- standard
    # mixed-precision practice: fp32 master, reduced-precision deploy).
    dtype_before = {}
    for name, param in policy.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            dtype_before[name] = str(param.dtype)
            param.data = param.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    n_params = sum(param.numel() for param in trainable)
    n_upcast = sum(policy.get_parameter(name).numel() for name in dtype_before)

    def batch(x: np.ndarray, s: np.ndarray, a: np.ndarray, indices: np.ndarray):
        output: dict[str, Any] = {}
        im = torch.from_numpy(x[indices]).to(device).float() / 255.0
        for camera_index, key in enumerate(CAMERAS):
            output[key] = im[:, camera_index].permute(0, 3, 1, 2).contiguous()
        output["observation.state"] = torch.from_numpy(s[indices]).to(device).float()
        output["action"] = torch.from_numpy(a[indices]).to(device).float()
        output["task"] = [args.task] * len(indices)
        return pre(output) if pre else output

    def loss_on(x: np.ndarray, s: np.ndarray, a: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        out = policy.forward(batch(x, s, a, indices))
        return out[0] if isinstance(out, tuple) else out["loss"]

    def heldout_loss(xs=None, ss=None, as_=None) -> float:
        """Evaluate the raster guard WITHOUT disturbing the training RNG.

        SmolVLA draws its flow-matching noise and timestep from the global
        generators (``sample_noise`` uses ``torch.normal`` and ``sample_time``
        a ``Beta`` sample, neither taking an explicit generator). Reseeding
        them here -- which is what this function used to do -- would restart
        the training noise sequence from a fixed state at every checkpoint, so
        each checkpoint interval would see the same noise as the last. The
        effect is deterministic and silent: it looks like a loss curve, not a
        bug. Snapshot the generator state, use a fixed seed so the guard is
        comparable between checkpoints, then put the training state back.
        """
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        policy.eval()
        torch.manual_seed(args.seed + 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 1000)
        xs, ss, as_ = (heldout_x, heldout_s, heldout_a) if xs is None else (xs, ss, as_)
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(xs), 2):
                index = np.arange(start, min(start + 2, len(xs)), dtype=np.int64)
                if len(index) < 2:
                    continue
                loss = loss_on(xs, ss, as_, index)
                total += float(loss) * len(index)
                count += len(index)
        policy.train()
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if count == 0:
            raise ValueError("held-out loss received no full batch")
        return total / count

    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    initial_holdout = heldout_loss(retention_x, retention_s, retention_a)
    initial_anchor_fit = heldout_loss()
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    sampled_weight: list[float] = []
    sampled_advantage: list[float] = []
    # Per-source interval means. Attempt 1 recorded only the final micro-step's
    # combined loss, which cannot show whether the anchor or the recovery data
    # carried the gradient -- the review's key open question.
    interval_losses: dict[str, list[float]] = {"rollout": [], "anchor": []}

    def save(step: int, train_loss: float) -> None:
        checkpoint = args.out / "checkpoints" / f"step_{step:06d}"
        if checkpoint.exists():
            raise ValueError(f"checkpoint already exists: {checkpoint}")
        checkpoint.mkdir()
        policy.save_pretrained(str(checkpoint))
        copy_processors(args.policy_path, checkpoint)
        config_fix = restore_config_discriminator(
            args.policy_path / "config.json", checkpoint / "config.json")
        heldout = heldout_loss(retention_x, retention_s, retention_a)
        anchor_fit = heldout_loss()
        record = {
            "step": step,
            "path": str(checkpoint),
            "train_loss": train_loss,
            # heldout_* is now the RETENTION guard: whole episodes, disjoint garments.
            "heldout_loss": heldout,
            "heldout_loss_baseline": initial_holdout,
            "guard": "retention: whole held-out demonstration episodes, garments disjoint from the anchor",
            "anchor_fit_loss": anchor_fit,
            "anchor_fit_loss_baseline": initial_anchor_fit,
            "anchor_fit_note": "the v2 guard (first 8 frames of 4 demos); reported only, not a gate",
            "heldout_gate": bool(heldout <= initial_holdout * args.heldout_tolerance),
            "heldout_tolerance": args.heldout_tolerance,
            "checkpoint_sha256": checkpoint_hashes(checkpoint),
            "config_discriminator": config_fix,
            "mean_sampled_awr_weight_since_last_checkpoint": float(np.mean(sampled_weight)),
            "mean_sampled_advantage_since_last_checkpoint": float(np.mean(sampled_advantage)),
            "mean_rollout_loss_since_last_checkpoint": float(np.mean(interval_losses["rollout"])),
            "mean_anchor_loss_since_last_checkpoint": float(np.mean(interval_losses["anchor"])),
        }
        if not record["heldout_gate"]:
            # A regression guard nothing ever reads is not a guard. Selection
            # still happens downstream on closed-loop success, but a breach
            # must be visible in the job log, not only in the artefact.
            print(f"[raster-guard] BREACH at step {step}: held-out loss {heldout:.6f} > "
                  f"{args.heldout_tolerance:.2f} x baseline {initial_holdout:.6f}",
                  file=sys.stderr, flush=True)
        (checkpoint / "checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        sampled_weight.clear()
        sampled_advantage.clear()
        interval_losses["rollout"].clear()
        interval_losses["anchor"].clear()
        print(json.dumps({"checkpoint": str(checkpoint), **record}, sort_keys=True), flush=True)

    policy.train()
    n_anchor = args.batch_size - n_rollout
    for step in range(1, args.steps + 1):
        # One optimizer step averages grad_accum micro-batches, each with the
        # same rollout/anchor mix, so the gradient is an average over
        # batch_size x grad_accum samples instead of batch_size. Attempt 1's
        # effective batch of 4 (2 supervised samples per source) was far below
        # the 32 the original BC used and the 8 the raster adaptation used.
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.grad_accum):
            rollout_index = rng.choice(len(rollout_x), size=n_rollout, replace=True, p=probabilities)
            anchor_index = rng.choice(len(anchor_x), size=n_anchor, replace=True)
            rollout_loss = loss_on(rollout_x, rollout_s, rollout_a, rollout_index)
            anchor_loss = loss_on(anchor_x, anchor_s, anchor_a, anchor_index)
            loss = (rollout_loss * n_rollout + anchor_loss * n_anchor) / (args.batch_size * args.grad_accum)
            loss.backward()
            micro_losses.append(float(loss) * args.grad_accum)
            interval_losses["rollout"].append(float(rollout_loss))
            interval_losses["anchor"].append(float(anchor_loss))
            sampled_weight.extend(weights[rollout_index].tolist())
            sampled_advantage.extend(rollout_advantage[rollout_index].tolist())
        loss = sum(micro_losses) / len(micro_losses)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save(step, float(loss))

    summary = {
        "schema_version": 1,
        "base_checkpoint": str(args.policy_path),
        "base_checkpoint_sha256": checkpoint_hashes(args.policy_path),
        "rollout": rollout_meta,
        "anchor": {"sampling_fraction": n_anchor / args.batch_size, "records": anchor_records,
                   "samples": int(len(anchor_x))},
        "retention": {"records": retention_records, "samples": int(len(retention_x)),
                      "loss_baseline": initial_holdout, "gate": f"<= {args.heldout_tolerance} * baseline"},
        "anchor_fit_metric": {"records": heldout_records, "samples": int(len(heldout_x)),
                              "loss_baseline": initial_anchor_fit, "role": "reported only"},
        "heldout": {"records": heldout_records, "samples": int(len(heldout_x)),
                    "loss_baseline": initial_holdout,
                    "gate": f"<= {args.heldout_tolerance} * baseline",
                    "overlap_check": "sha256 content comparison against the anchor corpus"},
        "rng": "training generator state is saved and restored around every "
               "held-out evaluation so checkpoint intervals stay independent",
        "objective": "AWR importance resampling of rollout frames plus unchanged BC anchor loss",
        "rollout_sampling_fraction": n_rollout / args.batch_size,
        "rollout_sampling": "with replacement, probability proportional to compiled lehome_fold.awr.weights",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "optimizer_hyperparameters": {
            "lr": args.lr,
            "betas": list(optimizer.defaults["betas"]),
            "weight_decay": optimizer.defaults["weight_decay"],
            "eps": optimizer.defaults["eps"],
            "grad_clip_norm": 1.0,
            "note": ("PyTorch AdamW defaults, as every campaign fine-tune used; recorded "
                     "explicitly because the original BC used weight_decay 1e-10 and "
                     "betas (0.9, 0.95) -- a known, recorded difference, not a change"),
        },
        "master_weights": {
            "upcast_parameters": int(n_upcast),
            "original_dtypes": sorted(set(dtype_before.values())),
            "why": "bf16 weights + bf16 Adam state at lr 1e-5 round most updates to zero",
        },
        "seed": args.seed,
        "unfreeze": args.unfreeze,
        "trainable_parameter_count": n_params,
        "trainable_parameter_names": trainable_names,
        "checkpoints": records,
    }
    (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
