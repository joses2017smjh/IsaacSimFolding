"""Compile persisted autonomous trajectories into aligned AWR training samples.

The input is intentionally the policy's *executed* H10 stream, never an H50
branch, stale observation, or interpolated target.  For a sampled observation
at action t, the target is stream[t:t+50]; samples without a complete 50-row
suffix are dropped rather than padded.  This keeps the policy's action-chunk
semantics intact while retaining every failure trajectory for AWR resampling.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from lehome_fold import awr  # noqa: E402


CAMERAS = ("top_rgb", "left_rgb", "right_rgb")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def terminal_reward(result: dict) -> tuple[float, int, int, bool]:
    terminal = result.get("terminal_checker", {})
    passed = int(terminal.get("conditions_passed", -1))
    total = int(terminal.get("conditions_total", 0))
    success = bool(result.get("terminal_success"))
    if total <= 0 or not 0 <= passed <= total:
        raise ValueError("invalid terminal geometric condition count")
    if success != bool(terminal.get("success")):
        raise ValueError("terminal success/checker mismatch")
    return passed / total, passed, total, success


def load_episode(path: Path, chunk: int) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    result_path = path.with_name("rollout.json")
    request_path = path.with_name("request.json")
    if not result_path.is_file() or not request_path.is_file():
        raise ValueError(f"{path} lacks rollout.json or request.json")
    result = json.loads(result_path.read_text())
    request = json.loads(request_path.read_text())
    with np.load(path, allow_pickle=False) as raw:
        required = {"step_index", "images", "state", "executed_action",
                    "executed_action_stream", "camera_keys", "seed"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        indices = np.asarray(raw["step_index"], dtype=np.int64)
        images = np.asarray(raw["images"], dtype=np.uint8)
        state = np.asarray(raw["state"], dtype=np.float32)
        stream = np.asarray(raw["executed_action_stream"], dtype=np.float32)
        cameras = tuple(str(x) for x in np.asarray(raw["camera_keys"]).tolist())
        seed = int(np.asarray(raw["seed"]).item())
    if cameras != CAMERAS:
        raise ValueError(f"{path} camera ordering {cameras!r} changed")
    if images.shape[0] != len(indices) or state.shape != (len(indices), 12):
        raise ValueError(f"{path} observation/state alignment is invalid")
    if images.shape[1:] != (3, 480, 640, 3):
        raise ValueError(f"{path} image shape {images.shape}")
    if stream.shape != (int(result["steps"]), 12) or not np.isfinite(stream).all():
        raise ValueError(f"{path} invalid full action stream {stream.shape}")
    if (indices.ndim != 1 or len(indices) == 0 or indices[0] != 0
            or np.any(np.diff(indices) <= 0) or indices[-1] >= len(stream)):
        raise ValueError(f"{path} invalid sampled action indices")
    row = request["row"]
    if seed != int(row["seed"]) or result["garment"] != row["garment"]:
        raise ValueError(f"{path} trajectory identity differs from request")
    if result.get("effective_n_action_steps") != 10 or result.get("prediction_chunk_size") != chunk:
        raise ValueError(f"{path} does not use normal H10 / predicted H50 semantics")
    valid = indices[indices + chunk <= len(stream)]
    if len(valid) == 0:
        raise ValueError(f"{path} has no observation with a full {chunk}-action suffix")
    # Valid entries are a prefix because indices are strictly ascending.
    n = len(valid)
    targets = np.stack([stream[t:t + chunk] for t in valid]).astype(np.float32)
    if targets.shape != (n, chunk, 12) or not np.isfinite(targets).all():
        raise ValueError(f"{path} target chunks are invalid")
    if not np.array_equal(targets[:, 0], stream[valid]):
        raise ValueError(f"{path} targets do not begin with action applied after their observation")
    source = {
        "trajectory_path": str(path),
        "trajectory_sha256": digest(path),
        "result_path": str(result_path),
        "result_sha256": digest(result_path),
        "request_path": str(request_path),
        "request_sha256": digest(request_path),
        "id": row["id"],
        "garment": row["garment"],
        "pose_key": int(row["pose_key"]),
        "seed": seed,
        "checkpoint": request["checkpoint"],
        "raw_observations": int(len(indices)),
        "usable_samples": int(n),
    }
    return source, result, images[:n], state[:n], targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=20.0)
    args = ap.parse_args()
    if args.chunk != 50:
        raise SystemExit("this campaign is preregistered for the checkpoint's 50-action chunks")
    paths = [Path(x).resolve() for x in sorted(glob.glob(args.trajectory_glob))]
    if not paths:
        raise SystemExit(f"no trajectories match {args.trajectory_glob}")
    if args.out.exists() or args.provenance.exists() or args.csv.exists():
        raise SystemExit("refusing to overwrite an existing compiled dataset/provenance")

    episodes, image_rows, state_rows, action_rows, starts = [], [], [], [], []
    for path in paths:
        source, result, images, state, actions = load_episode(path, args.chunk)
        reward, passed, total, success = terminal_reward(result)
        source.update({
            "terminal_reward": reward,
            "terminal_conditions_passed": passed,
            "terminal_conditions_total": total,
            "terminal_success": success,
        })
        episodes.append(source)
        image_rows.append(images)
        state_rows.append(state)
        action_rows.append(actions)
        with np.load(path, allow_pickle=False) as raw:
            all_starts = np.asarray(raw["step_index"], dtype=np.int32)
        starts.append(all_starts[all_starts + args.chunk <= int(result["steps"])])

    rewards = np.asarray([row["terminal_reward"] for row in episodes], dtype=np.float32)
    baseline = np.full_like(rewards, rewards.mean())
    advantages = awr.success_residual(rewards, baseline)
    weights = awr.weights(advantages, beta=args.beta, w_max=args.w_max)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise SystemExit("AWR produced non-positive or non-finite sampling weights")
    for ep, advantage, weight in zip(episodes, advantages.tolist(), weights.tolist()):
        ep["advantage"] = float(advantage)
        ep["awr_weight"] = float(weight)

    images = np.concatenate(image_rows).astype(np.uint8)
    state = np.concatenate(state_rows).astype(np.float32)
    action = np.concatenate(action_rows).astype(np.float32)
    start_step = np.concatenate(starts).astype(np.int32)
    episode_index = np.concatenate([
        np.full(ep["usable_samples"], index, dtype=np.int32)
        for index, ep in enumerate(episodes)
    ])
    sample_weight = weights[episode_index].astype(np.float32)
    sample_reward = rewards[episode_index].astype(np.float32)
    sample_advantage = advantages[episode_index].astype(np.float32)
    if (images.shape[0] != len(state) or len(state) != len(action) or
            len(action) != len(start_step) or len(start_step) != len(episode_index)):
        raise SystemExit("compiled rollout arrays lost alignment")
    # ``load_episode`` has already checked each target's row zero against the
    # full action stream at its own recorded start index.  Preserve a global
    # shape/finite assertion here without reopening the multi-gigabyte sources.
    if action.shape[1:] != (args.chunk, 12) or not np.isfinite(action).all():
        raise SystemExit("compiled action targets are invalid")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        images=images,
        state=state,
        action=action,
        target_start_step=start_step,
        episode_index=episode_index,
        reward=sample_reward,
        advantage=sample_advantage,
        awr_weight=sample_weight,
        chunk=np.asarray(args.chunk, dtype=np.int32),
        task=np.asarray("fold the garment on the table"),
        camera_keys=np.asarray(CAMERAS),
    )
    provenance = {
        "schema_version": 1,
        "dataset": str(args.out),
        "dataset_sha256": digest(args.out),
        "dataset_samples": int(len(images)),
        "chunk": args.chunk,
        "source": "autonomous ordinary-H10 executed action streams; no H50 branches or labels",
        "reward_definition": "terminal official geometric conditions_passed / conditions_total",
        "advantage_definition": "terminal reward minus mean terminal reward across this fixed collection",
        "advantage_implementation": "lehome_fold.awr.success_residual",
        "awr_weight_implementation": "lehome_fold.awr.weights; weights are used as rollout resampling probabilities",
        "beta": args.beta,
        "w_max": args.w_max,
        "episodes": episodes,
        "summary": {
            "episode_count": len(episodes),
            "terminal_reward_mean": float(rewards.mean()),
            "terminal_reward_min": float(rewards.min()),
            "terminal_reward_max": float(rewards.max()),
            "positive_advantage_episodes": int((advantages > 0).sum()),
            "effective_sample_size": awr.effective_sample_size(weights),
        },
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    fields = ["id", "garment", "pose_key", "seed", "usable_samples", "terminal_reward",
              "terminal_conditions_passed", "terminal_conditions_total", "terminal_success",
              "advantage", "awr_weight", "trajectory_sha256", "result_sha256"]
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in episodes])
    print(json.dumps({"dataset": str(args.out), "samples": len(images),
                      "episodes": len(episodes), "ess": provenance["summary"]["effective_sample_size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
