"""Compile v7's depth-qualified corpus from a fresh search with branch telemetry.

The v6 diagnosis (v6/analysis/diagnosis/synthesis.json) found the candidate
reaches the fold early but lands it shallow: across 128 matched episodes a
fold released >= 1 cm inside the closure threshold settled 37/41 vs 13/25
below. The labels it trained on were selected only by settled success, from
a baseline process that loses about 45% of its in-branch folds, so shallow
survivors were in the corpus and nothing marked them. This compiler changes
the label criterion, and nothing else:

  1. MECHANISM CHECK (preregistered, before any image is read): among search
     branches that reached all four conditions and have a landing window,
     landed-deep (>= LANDED_DEEP_CM) must settle more often than landed-
     shallow, one-sided Fisher p < 0.05. If not, landing depth is not a lever
     on these garments: exit 4, train nothing.
  2. LABEL RULE: a branch qualifies iff it settled, its terminal min(c1, c2)
     margin after the settle is >= TERMINAL_DEPTH_CM, and its first all-4
     step is <= FIRST_FOLD_MAX (so the filter cannot favour late folds that
     had no time to relax). Every label of a qualifying branch is kept; a
     student episode's labels only if the student itself qualifies.
  3. COVERAGE GATE per pose on qualifying branches, plus a total label floor.
     Unmet: exit 4 with the yield curve, train nothing.
  4. awr_weight balances the poses to 1/P of draws (the unchanged trainer
     samples proportional to it). No v4/v6 corpus is reused: their landing
     depth cannot be recovered and they contain shallow-landing survivors.

Landing (the diagnosis's definition, reproduced): the first window of >= 10
consecutive policy steps after the first all-4 step in which both gripper
link origins are > 12 cm from the cloth and the maximum particle lift is
< 5 cm; the landed margin is min(c1, c2) margin at the window's first step.
Pose is match_pose identity, never pose key.
"""
from __future__ import annotations

import argparse
import collections
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_recovery_dataset import CAMERAS, coverage, digest  # noqa: E402

PROV_COLS = ("root", "candidate", "horizon", "candidate_seed", "step_index")
LANDING = {"gripper_clear_m": 0.12, "lift_max_m": 0.05, "window": 10}


def pose_key(match_pose: str) -> tuple:
    return tuple(round(float(x), 4) for x in match_pose.split(":"))


def pose_of(match_pose: str, clusters: dict[str, str]) -> str:
    k = pose_key(match_pose)
    names = [name for name, mp in clusters.items() if pose_key(mp) == k]
    if len(names) != 1:
        raise SystemExit(f"match_pose {match_pose} is not exactly one development pose ({names})")
    return names[0]


def fisher_one_sided(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """P(A's success count >= observed | margins); A should exceed B."""
    total, successes = a_n + b_n, a_success + b_success
    denom = comb(total, a_n)
    return sum(comb(successes, k) * comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / denom


def landed_margin(attempt: dict, landing: dict = LANDING) -> float | None:
    """min(c1, c2) margin at the first landing window after the first all-4
    step, or None if the branch never reached all-4 or never landed."""
    first = attempt.get("first_full_step_offset")
    if first is None:
        return None
    grip, lift, marg = attempt["gripper_distance_trace"], attempt["lift_trace"], attempt["margin_trace"]
    clear = [min(g) > landing["gripper_clear_m"] and l < landing["lift_max_m"] for g, l in zip(grip, lift)]
    w = landing["window"]
    for k in range(first, len(clear) - w + 1):
        if all(clear[k:k + w]):
            return float(min(marg[k][0], marg[k][1]))
    return None


def branch_record(attempt: dict, rules: dict) -> dict:
    """Everything the rule and the mechanism check need about one attempt."""
    first = attempt.get("first_full_step_offset")
    first_abs = None if first is None else int(attempt["root"]) + int(first)
    terminal = attempt.get("terminal_margins")
    depth = None if terminal is None else float(min(terminal[0], terminal[1]))
    settled = bool(attempt.get("settled_success"))
    qualifies = (settled and depth is not None and depth >= rules["terminal_depth_cm"]
                 and first_abs is not None and first_abs <= rules["first_fold_max_step"])
    return {"root": int(attempt["root"]), "candidate": int(attempt["candidate"]), "settled": settled,
            "reached": first is not None, "first_fold_step": first_abs, "terminal_depth_cm": depth,
            "landed_margin_cm": landed_margin(attempt) if first is not None else None,
            "qualifies": bool(qualifies)}


def mechanism_check(records: list[dict], rules: dict) -> dict:
    """Among reached branches with a landing window: does landing deep predict settling?"""
    landed = [r for r in records if r["reached"] and r["landed_margin_cm"] is not None]
    deep = [r for r in landed if r["landed_margin_cm"] >= rules["landed_deep_cm"]]
    shallow = [r for r in landed if r["landed_margin_cm"] < rules["landed_deep_cm"]]
    ds, ss = sum(r["settled"] for r in deep), sum(r["settled"] for r in shallow)
    p = fisher_one_sided(ds, len(deep), ss, len(shallow)) if deep and shallow else 1.0
    return {"reached": sum(r["reached"] for r in records), "with_landing_window": len(landed),
            "deep": f"{ds}/{len(deep)}", "shallow": f"{ss}/{len(shallow)}",
            "deep_rate": ds / len(deep) if deep else None, "shallow_rate": ss / len(shallow) if shallow else None,
            "p_one_sided": p, "passed": bool(deep and shallow and p < rules["mechanism_p"]
                                             and ds / len(deep) > ss / len(shallow)),
            "threshold_cm": rules["landed_deep_cm"]}


def yield_curve(records_by_pose: dict[str, list[dict]], rules: dict) -> dict:
    out = {}
    for pose, recs in sorted(records_by_pose.items()):
        row = {}
        for cut in rules["yield_curve_cm"]:
            row[str(cut)] = sum(r["settled"] and r["terminal_depth_cm"] is not None
                                and r["terminal_depth_cm"] >= cut and r["first_fold_step"] is not None
                                and r["first_fold_step"] <= rules["first_fold_max_step"] for r in recs)
        row["settled_any_depth"] = sum(r["settled"] for r in recs)
        row["attempts"] = len(recs)
        out[pose] = row
    return out


def pose_gate(qualifying_attempts: dict[str, list[dict]], row_pose: dict[str, str], gates: dict) -> dict:
    """v4's coverage(), applied per pose to QUALIFYING branches only."""
    per_pose = {}
    for pose, gate in sorted(gates.items()):
        rows = {r: a for r, a in qualifying_attempts.items() if row_pose[r] == pose}
        per_pose[pose] = coverage(rows, gate)
    unmet = [f"{pose}:{req}" for pose, g in per_pose.items() for req in g["unmet_requirements"]]
    return {"passed": not unmet, "unmet_requirements": unmet, "per_pose": per_pose}


def balanced_weights(poses: np.ndarray) -> tuple[np.ndarray, dict]:
    counts = collections.Counter(poses.tolist())
    n, p = len(poses), len(counts)
    w = np.array([n / (p * counts[x]) for x in poses.tolist()], dtype=np.float32)
    share = {k: float(w[poses == k].sum() / w.sum()) for k in counts}
    return w, {"samples": dict(counts), "draw_share": share,
               "multiplier_vs_uniform": {k: n / (p * c) for k, c in counts.items()}}


def student_qualifies(row_dir: Path, rules: dict) -> tuple[bool, dict]:
    """A student episode qualifies by the same rule, from its own rollout record."""
    r = json.loads((row_dir / "rollout.json").read_text())
    geo = r["geometry_trajectory"]
    pol = [g for g in geo if g["phase"] == "policy"]
    settle = [g for g in geo if g["phase"] == "terminal_settle"]
    first = next((g["step"] for g in pol if g["success"]), None)
    depth = None
    if settle:
        d = settle[-1]["details"]
        depth = float(min(d["condition_1"]["margin_cm"], d["condition_2"]["margin_cm"]))
    ok = bool(r.get("terminal_success")) and depth is not None and depth >= rules["terminal_depth_cm"] \
        and first is not None and first <= rules["first_fold_max_step"]
    return ok, {"settled": bool(r.get("terminal_success")), "first_fold_step": first, "terminal_depth_cm": depth}


def student_examples(trajectory: Path, chunk: int):
    with np.load(trajectory, allow_pickle=False) as raw:
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
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    clusters, rows = manifest["pose_clusters"], manifest["recovery_search"]
    rules = manifest["depth_rules"]

    # ---- 1-3. mechanism check, label rule, coverage gate: attempt records only
    records, qualifying, row_pose, per_row = {}, {}, {}, {}
    for row in rows:
        pose = pose_of(row["match_pose"], clusters)
        if pose != row["pose"]:
            raise SystemExit(f"{row['id']}: declared pose {row['pose']} but match_pose is {pose}")
        row_pose[row["id"]] = pose
        rec = json.loads((root / "recovery" / row["id"] / "recovery.json").read_text())
        done = [a for a in rec["attempts"] if a.get("state") == "completed"]
        if any("margin_trace" not in a for a in done):
            raise SystemExit(f"{row['id']}: attempts lack v7 telemetry")
        records[row["id"]] = [branch_record(a, rules) for a in done]
        qualifying[row["id"]] = [dict(a, settled_success=True) for a, r in zip(done, records[row["id"]])
                                 if r["qualifies"]]
    everything = [r for recs in records.values() for r in recs]
    by_pose = collections.defaultdict(list)
    for rid, recs in records.items():
        by_pose[row_pose[rid]] += recs
    mech = mechanism_check(everything, rules)
    gate = pose_gate(qualifying, row_pose, manifest["recovery"]["coverage_gate"])
    curve = yield_curve(by_pose, rules)
    report = {"mechanism_check": mech, "coverage_gate": gate, "yield_curve": curve, "rules": rules,
              "per_pose_mechanism": {p: mechanism_check(r, rules) for p, r in sorted(by_pose.items())},
              "first_fold_step_median": {
                  "qualifying": float(np.median([r["first_fold_step"] for r in everything if r["qualifies"]]))
                  if any(r["qualifies"] for r in everything) else None,
                  "settled_not_qualifying": float(np.median([r["first_fold_step"] for r in everything
                                                             if r["settled"] and not r["qualifies"]
                                                             and r["first_fold_step"] is not None]))
                  if any(r["settled"] and not r["qualifies"] and r["first_fold_step"] is not None
                         for r in everything) else None},
              "branches": {rid: recs for rid, recs in records.items()}}
    passed = mech["passed"] and gate["passed"]
    report["passed"] = passed
    report["unmet_requirements"] = ([] if mech["passed"] else ["mechanism_check"]) + gate["unmet_requirements"]
    report["per_pose"] = {p: {"successful_branches": g["successful_branches"],
                              "attempts_completed": sum(len(records[r]) for r in records if row_pose[r] == p)}
                          for p, g in gate["per_pose"].items()}
    args.gate_out.parent.mkdir(parents=True, exist_ok=True)
    args.gate_out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": passed, "unmet": report["unmet_requirements"],
                      "mechanism": {k: mech[k] for k in ("deep", "shallow", "p_one_sided")},
                      "qualifying": {p: g["successful_branches"] for p, g in gate["per_pose"].items()}}),
          flush=True)
    if not passed:
        return 4

    # ---- sizes first (no images), then one allocation
    plan = []
    for row in rows:
        d = root / "recovery" / row["id"]
        keep = {(q["root"], q["candidate"]) for q in qualifying[row["id"]]}
        with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
            pairs = list(zip(np.asarray(ex["root"]).tolist(), np.asarray(ex["candidate"]).tolist()))
        sel = np.array([p in keep for p in pairs], dtype=bool)
        s_ok, s_info = student_qualifies(d, rules)
        n_student = 0
        if s_ok:
            with np.load(d / "trajectory.npz", allow_pickle=False) as raw:
                idx = np.asarray(raw["step_index"], dtype=np.int64)
                n_student = int((idx + 50 <= len(np.asarray(raw["executed_action_stream"]))).sum())
        plan.append((row, d, sel, s_ok, s_info, n_student))
    n = sum(int(sel.sum()) + ns for _, _, sel, _, _, ns in plan)
    if n < int(rules["min_total_labels"]):
        report["passed"] = False
        report["unmet_requirements"] = [f"total_labels {n} < {rules['min_total_labels']}"]
        args.gate_out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({"passed": False, "unmet": report["unmet_requirements"]}))
        return 4
    images = np.empty((n, 3, 480, 640, 3), dtype=np.uint8)
    state = np.empty((n, 12), dtype=np.float32)
    action = np.empty((n, 50, 12), dtype=np.float32)
    prov = {k: np.empty(n, dtype=np.int64) for k in PROV_COLS}
    depth = np.empty(n, dtype=np.float32)
    origin, pose, cursor, sources = [], [], 0, []
    for row, d, sel, s_ok, s_info, n_student in plan:
        branch_depth = {(r["root"], r["candidate"]): r["terminal_depth_cm"] for r in records[row["id"]]}
        m = int(sel.sum())
        if m:
            with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
                if tuple(str(x) for x in ex["camera_keys"].tolist()) != CAMERAS:
                    raise SystemExit(f"{d}: camera order changed")
                idx = np.flatnonzero(sel)
                images[cursor:cursor + m] = np.asarray(ex["images"])[idx]
                state[cursor:cursor + m] = np.asarray(ex["state"])[idx]
                action[cursor:cursor + m] = np.asarray(ex["action"])[idx]
                for k in PROV_COLS:
                    prov[k][cursor:cursor + m] = np.asarray(ex[k], dtype=np.int64)[idx]
                depth[cursor:cursor + m] = [branch_depth[(int(r), int(c))] for r, c in
                                            zip(np.asarray(ex["root"])[idx], np.asarray(ex["candidate"])[idx])]
            origin += [f"branch:{row['id']}"] * m
            cursor += m
        if n_student:
            st = student_examples(d / "trajectory.npz", 50)
            images[cursor:cursor + n_student] = st[0]
            state[cursor:cursor + n_student] = st[1]
            action[cursor:cursor + n_student] = st[2]
            for k in PROV_COLS:
                prov[k][cursor:cursor + n_student] = -1
            depth[cursor:cursor + n_student] = s_info["terminal_depth_cm"]
            origin += [f"student:{row['id']}"] * n_student
            cursor += n_student
        pose += [row_pose[row["id"]]] * (m + n_student)
        sources.append({"id": row["id"], "pose": row_pose[row["id"]], "garment": row["garment"],
                        "garment_class": row.get("garment_class"), "off_metadata_pose": row.get("off_metadata_pose", False),
                        "seed": row["seed"], "qualifying_branches": len(qualifying[row["id"]]),
                        "branch_labels": m, "student": s_info, "student_labels": n_student,
                        "recovery_json_sha256": digest(d / "recovery.json"),
                        "examples_sha256": digest(d / "recovery_examples.npz")})
    if cursor != n or len(origin) != n or not np.isfinite(action).all():
        raise SystemExit("compiled corpus lost alignment")
    pose = np.asarray(pose)
    weights, balance = balanced_weights(pose)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, images=images, state=state, action=action, awr_weight=weights,
        advantage=np.zeros(n, np.float32), reward=np.ones(n, np.float32),
        origin=np.asarray(origin), pose=pose, source=np.asarray(["v7"] * n), terminal_depth_cm=depth,
        **prov, chunk=np.asarray(50, np.int32), task=np.asarray(manifest["task_prompt"]),
        camera_keys=np.asarray(CAMERAS))
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        "dataset": str(args.out), "dataset_sha256": digest(args.out), "samples": n,
        "origin_counts": dict(collections.Counter(o.split(":")[0] for o in origin)),
        "pose_balance": balance,
        "label_definition": ("observation rendered at a state paired with the next 50 actions actually "
                             "executed from it, on a continuation that SETTLED with terminal min(c1, c2) "
                             f"margin >= {rules['terminal_depth_cm']} cm and first all-4 by step "
                             f"{rules['first_fold_max_step']}"),
        "weighting": "pose-balanced: awr_weight = N / (P * n_pose); uniform within a pose",
        "reused_corpora": "none",
        "gate_report": str(args.gate_out), "sources": sources,
    }, indent=2) + "\n")
    print(json.dumps({"samples": n, **balance["samples"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
