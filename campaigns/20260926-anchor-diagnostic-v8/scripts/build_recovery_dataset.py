"""Compile validated recovery examples, gated on coverage.

Input is every recovery row's recovery.json (all attempts) and
recovery_examples.npz (examples from SETTLED-successful branches only), plus any
student episode that itself settled successfully. Every example pairs an
observation rendered at a state with the actions actually executed from that
state on a continuation that then reached settled success.

This is supervised data, not episode-weighted AWR: every example has weight 1.
The coverage gate fails closed (exit 4) when successes are too few or too
concentrated to call the search a teacher -- a couple of lucky branches from
one root is not coverage. Every attempt, including failures, goes into the
provenance so the search's actual success rate is reported, not just its wins.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CAMERAS = ("top_rgb", "left_rgb", "right_rgb")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def coverage(attempts_by_row: dict[str, list[dict]], gate: dict) -> dict:
    """Pure: summarise every attempt and apply the preregistered gate."""
    completed = [(row, a) for row, attempts in attempts_by_row.items()
                 for a in attempts if a.get("state") == "completed"]
    wins = [(row, a) for row, a in completed if a.get("settled_success")]
    roots = {(row, a["root"]) for row, a in wins}
    rows = {row for row, _ in wins}
    by_root = collections.defaultdict(lambda: [0, 0])
    by_horizon = collections.defaultdict(lambda: [0, 0])
    for row, a in completed:
        by_root[f"{row}@{a['root']}"][1] += 1
        by_root[f"{row}@{a['root']}"][0] += int(bool(a.get("settled_success")))
        by_horizon[str(a["horizon"])][1] += 1
        by_horizon[str(a["horizon"])][0] += int(bool(a.get("settled_success")))
    transient = sum(1 for _, a in completed
                    if a.get("full_entries", 0) > 0 and not a.get("settled_success"))
    checks = [
        {"name": "successful_branches", "observed": len(wins),
         "threshold": gate["min_successful_branches"], "passed": len(wins) >= gate["min_successful_branches"]},
        {"name": "distinct_successful_roots", "observed": len(roots),
         "threshold": gate["min_distinct_roots"], "passed": len(roots) >= gate["min_distinct_roots"]},
        {"name": "distinct_successful_rows", "observed": len(rows),
         "threshold": gate["min_rows"], "passed": len(rows) >= gate["min_rows"]},
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "unmet_requirements": [c["name"] for c in checks if not c["passed"]],
        "checks": checks,
        "attempts_total": sum(len(v) for v in attempts_by_row.values()),
        "attempts_completed": len(completed),
        "attempts_not_run_or_error": sum(len(v) for v in attempts_by_row.values()) - len(completed),
        "successful_branches": len(wins),
        "success_rate": (len(wins) / len(completed)) if completed else None,
        "transient_full_crossings_without_settled_success": transient,
        "by_root": {k: {"successes": v[0], "attempts": v[1]} for k, v in sorted(by_root.items())},
        "by_horizon": {k: {"successes": v[0], "attempts": v[1]} for k, v in sorted(by_horizon.items())},
    }


def student_examples(trajectory: Path, chunk: int):
    """A student episode that settled successfully is itself a validated
    continuation from every state it visited."""
    with np.load(trajectory, allow_pickle=False) as raw:
        if not bool(np.asarray(raw["terminal_success"]).item()):
            return None
        idx = np.asarray(raw["step_index"], dtype=np.int64)
        stream = np.asarray(raw["executed_action_stream"], dtype=np.float32)
        keep = idx + chunk <= len(stream)
        idx = idx[keep]
        return (np.asarray(raw["images"], dtype=np.uint8)[:len(keep)][keep],
                np.asarray(raw["state"], dtype=np.float32)[:len(keep)][keep],
                np.stack([stream[t:t + chunk] for t in idx]) if len(idx) else np.zeros((0, chunk, 12), np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--gate-out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="read the smoke search row instead")
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = manifest["recovery_smoke"] if args.smoke else manifest["recovery_search"]
    base = root / ("smoke/recovery" if args.smoke else "recovery")
    attempts_by_row, sources = {}, []
    images, state, action, origin = [], [], [], []
    prov_cols = {k: [] for k in ("root", "candidate", "horizon", "candidate_seed", "step_index")}
    for row in rows:
        d = base / row["id"]
        record = json.loads((d / "recovery.json").read_text())
        attempts_by_row[row["id"]] = record["attempts"]
        with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
            if tuple(str(x) for x in ex["camera_keys"].tolist()) != CAMERAS:
                raise SystemExit(f"{d}: camera order changed")
            n = len(ex["action"])
            if n:
                if ex["images"].shape[1:] != (3, 480, 640, 3) or ex["action"].shape[1:] != (50, 12):
                    raise SystemExit(f"{d}: example shapes invalid")
                images.append(ex["images"]); state.append(ex["state"]); action.append(ex["action"])
                origin += [f"branch:{row['id']}"] * n
                for key in prov_cols:
                    prov_cols[key].append(np.asarray(ex[key], dtype=np.int64))
        student = student_examples(d / "trajectory.npz", 50)
        n_student = 0 if student is None else len(student[2])
        if n_student:
            images.append(student[0]); state.append(student[1]); action.append(student[2])
            origin += [f"student:{row['id']}"] * n_student
            for key in prov_cols:      # -1 marks a student example (no branch attempt)
                prov_cols[key].append(np.full(n_student, -1, dtype=np.int64))
        sources.append({"id": row["id"], "garment": row["garment"], "pose_key": row["pose_key"],
                        "branch_examples": int(n), "student_settled_success": student is not None,
                        "student_examples": n_student,
                        "recovery_json_sha256": digest(d / "recovery.json"),
                        "examples_sha256": digest(d / "recovery_examples.npz")})

    gate = coverage(attempts_by_row, manifest["recovery"]["coverage_gate"])
    args.gate_out.parent.mkdir(parents=True, exist_ok=True)
    args.gate_out.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({k: gate[k] for k in ("passed", "unmet_requirements", "successful_branches",
                                           "attempts_completed", "success_rate")}))
    if not gate["passed"]:
        return 4
    images = np.concatenate(images); state = np.concatenate(state); action = np.concatenate(action)
    if not (len(images) == len(state) == len(action) == len(origin)) or not np.isfinite(action).all():
        raise SystemExit("compiled recovery arrays lost alignment")
    n = len(action)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, images=images, state=state, action=action,
        awr_weight=np.ones(n, np.float32), advantage=np.zeros(n, np.float32), reward=np.ones(n, np.float32),
        origin=np.asarray(origin),
        **{k: np.concatenate(v) for k, v in prov_cols.items()},
        chunk=np.asarray(50, np.int32),
        task=np.asarray(manifest["task_prompt"]), camera_keys=np.asarray(CAMERAS))
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        "dataset": str(args.out), "dataset_sha256": digest(args.out), "samples": n,
        "origin_counts": dict(collections.Counter(o.split(":")[0] for o in origin)),
        "label_definition": ("observation rendered at a state paired with the next 50 actions actually "
                             "executed from it on a continuation that reached SETTLED success"),
        "weighting": "uniform; no episode-weighted AWR", "coverage_gate": gate, "sources": sources,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
