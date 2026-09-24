"""Compile v6's pose-balanced corpus, gated per pose on the NEW search.

v5 showed the fine-tuned candidate's gains and losses track per-pose
supervision density (P_C 53% of v4's corpus -> 6/6 at H10; P_B 19%, 14
branches -> 0/8 over H10 and H50). This compiler is the one data factor v6
changes:

  1. The targeted search (training-only garments at the P_B and P_A poses)
     is gated PER POSE with v4's own coverage() and the manifest's
     thresholds, before any image is read. A pose whose new supervision is
     too thin fails closed (exit 4): training on the same inadequate signal
     is exactly what v6 exists to avoid.
  2. v4's validated corpus is reused verbatim, pinned by SHA-256, each
     sample's pose read from its origin row's match_pose in v4's manifest.
  3. New examples are compiled exactly as v4 compiles them (same helpers:
     settled-success branches plus any student episode that settled).
  4. awr_weight = N / (P * n_pose): each of the P poses supplies 1/P of the
     trainer's rollout draws (it samples proportional to awr_weight),
     uniform within a pose, mean weight 1. The trainer is unchanged.

Pose is always `match_pose` identity, never pose key: keys map to
different poses on different garments.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_recovery_dataset import CAMERAS, coverage, digest, student_examples  # noqa: E402

TRAINER_MEMBERS = ("images", "state", "action", "awr_weight", "advantage", "reward",
                   "chunk", "task", "camera_keys")
PROV_COLS = ("root", "candidate", "horizon", "candidate_seed", "step_index")


def pose_key(match_pose: str) -> tuple:
    return tuple(round(float(x), 4) for x in match_pose.split(":"))


def pose_of(match_pose: str, clusters: dict[str, str]) -> str:
    """Name of the development pose whose match_pose is identical, else raise."""
    k = pose_key(match_pose)
    names = [name for name, mp in clusters.items() if pose_key(mp) == k]
    if len(names) != 1:
        raise SystemExit(f"match_pose {match_pose} is not exactly one development pose ({names})")
    return names[0]


def balanced_weights(poses: np.ndarray) -> tuple[np.ndarray, dict]:
    """awr_weight so every pose present supplies an equal share of draws."""
    counts = collections.Counter(poses.tolist())
    n, p = len(poses), len(counts)
    w = np.array([n / (p * counts[x]) for x in poses.tolist()], dtype=np.float32)
    share = {k: float(w[poses == k].sum() / w.sum()) for k in counts}
    return w, {"samples": dict(counts), "draw_share": share,
               "multiplier_vs_uniform": {k: n / (p * c) for k, c in counts.items()}}


def pose_gate(attempts_by_row: dict[str, list[dict]], row_pose: dict[str, str],
              gates: dict[str, dict]) -> dict:
    """v4's coverage() applied to the rows of each targeted pose separately."""
    per_pose = {}
    for pose, gate in sorted(gates.items()):
        rows = {r: a for r, a in attempts_by_row.items() if row_pose[r] == pose}
        per_pose[pose] = coverage(rows, gate)
    unmet = [f"{pose}:{req}" for pose, g in per_pose.items() for req in g["unmet_requirements"]]
    return {"passed": not unmet, "unmet_requirements": unmet, "per_pose": per_pose,
            "successful_branches": sum(g["successful_branches"] for g in per_pose.values()),
            "attempts_completed": sum(g["attempts_completed"] for g in per_pose.values()),
            "gates": gates}


def row_counts(d: Path) -> tuple[int, int]:
    """(branch examples, student examples) without reading any image."""
    with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
        n_branch = len(ex["action"])
    with np.load(d / "trajectory.npz", allow_pickle=False) as raw:
        if not bool(np.asarray(raw["terminal_success"]).item()):
            return n_branch, 0
        idx = np.asarray(raw["step_index"], dtype=np.int64)
        return n_branch, int((idx + 50 <= len(np.asarray(raw["executed_action_stream"]))).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--gate-out", type=Path, required=True)
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    clusters = manifest["pose_clusters"]
    rows = manifest["recovery_search"]
    reuse = manifest["corpus"]["reuse"]

    # ---- 1. the per-pose gate, from attempt records only
    attempts_by_row, row_pose = {}, {}
    for row in rows:
        pose = pose_of(row["match_pose"], clusters)
        if pose != row["pose"]:
            raise SystemExit(f"{row['id']}: declared pose {row['pose']} but match_pose is {pose}")
        row_pose[row["id"]] = pose
        attempts_by_row[row["id"]] = json.loads((root / "recovery" / row["id"] / "recovery.json").read_text())["attempts"]
    gate = pose_gate(attempts_by_row, row_pose, manifest["recovery"]["coverage_gate"])
    args.gate_out.parent.mkdir(parents=True, exist_ok=True)
    args.gate_out.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"passed": gate["passed"], "unmet": gate["unmet_requirements"],
                      **{p: g["successful_branches"] for p, g in gate["per_pose"].items()}}), flush=True)
    if not gate["passed"]:
        return 4

    # ---- 2. the reused v4 corpus: pinned, and its poses from its own manifest
    old_path = Path(reuse["path"])
    if digest(old_path) != reuse["sha256"]:
        raise SystemExit("reused v4 corpus changed since the manifest pinned it")
    old_rows = {r["id"]: r for r in json.loads(Path(reuse["manifest"]).read_text())["recovery_search"]}
    with np.load(old_path, allow_pickle=False) as old:
        old_origin = np.asarray(old["origin"]).astype(str)
    old_pose = np.array([pose_of(old_rows[o.split(":", 1)[1]]["match_pose"], clusters) for o in old_origin])

    # ---- 3. sizes first, so images are written once into one allocation
    new_counts = {row["id"]: row_counts(root / "recovery" / row["id"]) for row in rows}
    n_old = len(old_origin)
    n = n_old + sum(b + s for b, s in new_counts.values())
    images = np.empty((n, 3, 480, 640, 3), dtype=np.uint8)
    state = np.empty((n, 12), dtype=np.float32)
    action = np.empty((n, 50, 12), dtype=np.float32)
    prov = {k: np.empty(n, dtype=np.int64) for k in PROV_COLS}
    origin, pose, source = list(old_origin), list(old_pose), ["v4"] * n_old
    with np.load(old_path, allow_pickle=False) as old:
        if tuple(str(x) for x in old["camera_keys"].tolist()) != CAMERAS:
            raise SystemExit("reused corpus camera order changed")
        images[:n_old] = old["images"]
        state[:n_old] = old["state"]; action[:n_old] = old["action"]
        for k in PROV_COLS:
            prov[k][:n_old] = old[k]
    cursor, sources = n_old, []
    for row in rows:
        d = root / "recovery" / row["id"]
        with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
            if tuple(str(x) for x in ex["camera_keys"].tolist()) != CAMERAS:
                raise SystemExit(f"{d}: camera order changed")
            m = len(ex["action"])
            if m:
                if ex["images"].shape[1:] != (3, 480, 640, 3) or ex["action"].shape[1:] != (50, 12):
                    raise SystemExit(f"{d}: example shapes invalid")
                images[cursor:cursor + m] = ex["images"]
                state[cursor:cursor + m] = ex["state"]; action[cursor:cursor + m] = ex["action"]
                for k in PROV_COLS:
                    prov[k][cursor:cursor + m] = np.asarray(ex[k], dtype=np.int64)
                origin += [f"branch:{row['id']}"] * m
                cursor += m
        student = student_examples(d / "trajectory.npz", 50)
        s = 0 if student is None else len(student[2])
        if s:
            images[cursor:cursor + s] = student[0]
            state[cursor:cursor + s] = student[1]; action[cursor:cursor + s] = student[2]
            for k in PROV_COLS:        # -1 marks a student example (no branch attempt)
                prov[k][cursor:cursor + s] = -1
            origin += [f"student:{row['id']}"] * s
            cursor += s
        if (m, s) != new_counts[row["id"]]:
            raise SystemExit(f"{row['id']}: counted {new_counts[row['id']]} but read {(m, s)}")
        pose += [row_pose[row["id"]]] * (m + s)
        source += ["v6"] * (m + s)
        sources.append({"id": row["id"], "pose": row_pose[row["id"]], "garment": row["garment"],
                        "pose_key": row["pose_key"], "seed": row["seed"], "roots": row.get("recovery_roots"),
                        "branch_examples": m, "student_examples": s,
                        "recovery_json_sha256": digest(d / "recovery.json"),
                        "examples_sha256": digest(d / "recovery_examples.npz")})
    if cursor != n or len(origin) != n or len(pose) != n or not np.isfinite(action).all():
        raise SystemExit("merged corpus lost alignment")

    # ---- 4. pose-balanced weights
    pose = np.asarray(pose)
    weights, balance = balanced_weights(pose)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, images=images, state=state, action=action, awr_weight=weights,
        advantage=np.zeros(n, np.float32), reward=np.ones(n, np.float32),
        origin=np.asarray(origin), pose=pose, source=np.asarray(source), **prov,
        chunk=np.asarray(50, np.int32), task=np.asarray(manifest["task_prompt"]),
        camera_keys=np.asarray(CAMERAS))
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        "dataset": str(args.out), "dataset_sha256": digest(args.out), "samples": n,
        "origin_counts": dict(collections.Counter(o.split(":")[0] for o in origin)),
        "source_counts": dict(collections.Counter(source)),
        "pose_balance": balance,
        "samples_by_pose_and_source": {f"{p}/{s}": c for (p, s), c in
                                       sorted(collections.Counter(zip(pose.tolist(), source)).items())},
        "label_definition": ("observation rendered at a state paired with the next 50 actions actually "
                             "executed from it on a continuation that reached SETTLED success"),
        "weighting": ("pose-balanced: awr_weight = N / (P * n_pose), so each pose supplies 1/P of the "
                      "rollout draws; uniform within a pose; the trainer samples proportional to awr_weight"),
        "reused_corpus": {"path": str(old_path), "sha256": reuse["sha256"], "samples": n_old},
        "coverage_gate": gate, "sources": sources,
    }, indent=2) + "\n")
    print(json.dumps({"samples": n, **balance["samples"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
