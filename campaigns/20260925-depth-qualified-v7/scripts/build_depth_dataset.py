"""Compile v7's depth-qualified corpus from a fresh search with branch telemetry.

The v6 diagnosis (v6/analysis/diagnosis/synthesis.json) found the candidate
reaches the fold early but lands it shallow: across 128 matched episodes a
fold released >= 1 cm inside the closure threshold settled 37/41 vs 13/25
below. The labels it trained on were selected only by settled success, from
a baseline process that loses about 45% of its in-branch folds, so shallow
survivors were in the corpus and nothing marked them. This compiler changes
the label criterion, and nothing else:

  1. MECHANISM CHECK (preregistered, before any image is read): among search
     branches that reached all four conditions and landed inside the branch
     (landings inherited from an already-folded root are excluded -- they
     are the root's, not the branch's), landed-deep (>= landed_deep_cm) must
     settle more often than landed-shallow: an exact conditional test
     stratified by pose (branches are pose-clustered; a pose with an empty
     group contributes nothing), p < mechanism_p, with the pooled deep rate
     above the pooled shallow rate. If not, landing depth is not a lever on
     these garments: exit 4, train nothing.
  2. LABEL RULE, per pose, with a preregistered cut ladder: a branch
     qualifies at cut c iff it settled, its terminal min(c1, c2) margin
     after the settle is >= c, and its first all-4 action is <= step
     first_fold_max_step (so the filter cannot favour late folds that had no
     time to relax). Each pose uses the STRICTEST cut in cut_ladder_cm
     (1.5, 1.0, 0.5, then 0.0 = settled-only) at which it meets the coverage
     gate; the chosen cut is recorded. The pre-launch review showed settled
     baseline P_A folds never exceed ~1.47 cm and P_B rarely do, so a single
     1.5 cm cut would stop the attempt for supply, not evidence. Every label
     of a qualifying branch is kept; a student episode's labels only if the
     student qualifies at its pose's cut.
  3. COVERAGE GATE per pose on qualifying branches (>= 24 from >= 8 (row,
     root) pairs and >= 3 rows), plus a total label floor. A pose that fails
     even settled-only is a supply shortfall: exit 4, train nothing.
  4. awr_weight balances the poses to 1/P of draws (the unchanged trainer
     samples proportional to it). No v4/v6 corpus is reused: their landing
     depth cannot be recovered and they contain shallow-landing survivors.
  Rows the search did not complete are excluded and recorded; more than one
  is an infrastructure failure (exit 5), never a scientific verdict.

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
ROW_FILES = ("status.json", "recovery.json", "recovery_examples.npz", "rollout.json", "trajectory.npz")


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


def stratified_p(strata: list[tuple[int, int, int, int]]) -> float:
    """Exact conditional one-sided test over strata (deep_succ, deep_n,
    shallow_succ, shallow_n): statistic = total deep successes, null = the
    convolution of each stratum's hypergeometric given its margins. Strata
    with an empty group contribute nothing; with none usable, p = 1."""
    dist, obs, used = {0: 1.0}, 0, 0
    for a, an, b, bn in strata:
        if an == 0 or bn == 0:
            continue
        used += 1
        s, tot = a + b, an + bn
        h = {k: comb(s, k) * comb(tot - s, an - k) / comb(tot, an)
             for k in range(max(0, s - bn), min(s, an) + 1)}
        obs += a
        new: dict[int, float] = {}
        for x, px in dist.items():
            for k, pk in h.items():
                new[x + k] = new.get(x + k, 0.0) + px * pk
        dist = new
    if not used:
        return 1.0
    return float(min(1.0, sum(p for x, p in dist.items() if x >= obs)))


def landing(attempt: dict, spec: dict) -> tuple[float | None, int | None]:
    """(min(c1, c2) margin, trace index) at the first landing window at or
    after the first all-4 step, or (None, None) if never reached or landed.
    `spec` is the manifest's depth_rules.landing -- read, never defaulted."""
    first = attempt.get("first_full_step_offset")
    if first is None:
        return None, None
    grip, lift, marg = attempt["gripper_distance_trace"], attempt["lift_trace"], attempt["margin_trace"]
    clear = [min(g) > spec["gripper_clear_m"] and l < spec["lift_max_m"] for g, l in zip(grip, lift)]
    w = spec["window"]
    for k in range(first, len(clear) - w + 1):
        if all(clear[k:k + w]):
            return float(min(marg[k][0], marg[k][1])), k
    return None, None


def landed_margin(attempt: dict, spec: dict) -> float | None:
    return landing(attempt, spec)[0]


def branch_record(attempt: dict, rules: dict) -> dict:
    """Everything the rule and the mechanism check need about one attempt.
    trace[k] is the state after action root+k+1, so first_fold_step counts
    completed actions (1-based) -- the same as geometry_trajectory 'step',
    the student check and the diagnosis's t0."""
    first = attempt.get("first_full_step_offset")
    first_abs = None if first is None else int(attempt["root"]) + int(first) + 1
    terminal = attempt.get("terminal_margins")
    depth = None if terminal is None else float(min(terminal[0], terminal[1]))
    settled = bool(attempt.get("settled_success"))
    landed, k = landing(attempt, rules["landing"])
    return {"root": int(attempt["root"]), "candidate": int(attempt["candidate"]), "settled": settled,
            "reached": first is not None, "first_fold_step": first_abs, "terminal_depth_cm": depth,
            "landed_margin_cm": landed,
            # the fold already existed at the root and the landing is the root's state
            "inherited_landing": bool(first == 0 and k == 0),
            "post_landing_root": bool(first == 0),
            "settled_early": bool(settled and first_abs is not None and first_abs <= rules["first_fold_max_step"])}


def qualifies(rec: dict, cut: float) -> bool:
    return bool(rec["settled_early"] and rec["terminal_depth_cm"] is not None and rec["terminal_depth_cm"] >= cut)


def mechanism_check(records_by_pose: dict[str, list[dict]], rules: dict) -> dict:
    """Among reached branches that landed inside the branch: does landing deep
    predict settling? Exact conditional test stratified by pose."""
    cut = rules["landed_deep_cm"]
    per_pose, strata, pooled = {}, [], [0, 0, 0, 0]
    for pose, recs in sorted(records_by_pose.items()):
        landed = [r for r in recs if r["reached"] and r["landed_margin_cm"] is not None and not r["inherited_landing"]]
        deep = [r for r in landed if r["landed_margin_cm"] >= cut]
        shallow = [r for r in landed if r["landed_margin_cm"] < cut]
        cell = (sum(r["settled"] for r in deep), len(deep), sum(r["settled"] for r in shallow), len(shallow))
        strata.append(cell)
        pooled = [x + y for x, y in zip(pooled, cell)]
        per_pose[pose] = {"deep": f"{cell[0]}/{cell[1]}", "shallow": f"{cell[2]}/{cell[3]}",
                          "inherited_landings_excluded": sum(r["inherited_landing"] for r in recs if r["reached"]),
                          "reached": sum(r["reached"] for r in recs), "with_landing_window": len(landed)}
    p = stratified_p(strata)
    ds, dn, ss, sn = pooled
    deep_rate = ds / dn if dn else None
    shallow_rate = ss / sn if sn else None
    return {"test": "exact conditional, stratified by pose, one-sided (deep settles more)",
            "threshold_cm": cut, "deep": f"{ds}/{dn}", "shallow": f"{ss}/{sn}",
            "deep_rate": deep_rate, "shallow_rate": shallow_rate, "p_one_sided": p,
            "per_pose": per_pose,
            "passed": bool(dn and sn and p < rules["mechanism_p"] and deep_rate > shallow_rate)}


def root_table(records_by_row: dict[str, list[dict]]) -> list[dict]:
    """One line per (row, root): so the clustering behind any verdict is auditable."""
    out = []
    for rid, recs in sorted(records_by_row.items()):
        roots = sorted({r["root"] for r in recs})
        for root in roots:
            rr = [r for r in recs if r["root"] == root]
            lm = [r["landed_margin_cm"] for r in rr if r["landed_margin_cm"] is not None]
            out.append({"row": rid, "root": root, "attempts": len(rr), "settled": sum(r["settled"] for r in rr),
                        "reached": sum(r["reached"] for r in rr), "inherited": sum(r["inherited_landing"] for r in rr),
                        "landed_median_cm": float(np.median(lm)) if lm else None})
    return out


def yield_curve(records_by_pose: dict[str, list[dict]], rules: dict) -> dict:
    out = {}
    for pose, recs in sorted(records_by_pose.items()):
        row = {str(c): sum(qualifies(r, c) for r in recs) for c in rules["yield_curve_cm"]}
        row["settled_any_depth_any_time"] = sum(r["settled"] for r in recs)
        row["attempts"] = len(recs)
        out[pose] = row
    return out


def ladder_gate(records_by_row: dict[str, list[dict]], attempts_by_row: dict[str, list[dict]],
                row_pose: dict[str, str], gates: dict, ladder: list[float]) -> dict:
    """Per pose: the strictest cut in the ladder at which the pose's qualifying
    branches meet v4's coverage() gate. Only the fields that truthfully
    describe the qualifying subset are reported."""
    per_pose = {}
    for pose, gate in sorted(gates.items()):
        rids = [r for r in records_by_row if row_pose[r] == pose]
        chosen = None
        tried = []
        for cut in ladder:
            q = {r: [a for a, rec in zip(attempts_by_row[r], records_by_row[r]) if qualifies(rec, cut)] for r in rids}
            cov = coverage({r: [dict(a, settled_success=True) for a in v] for r, v in q.items()}, gate)
            recs = [rec for r in rids for rec in records_by_row[r] if qualifies(rec, cut)]
            entry = {"cut_cm": cut, "passed": cov["passed"], "unmet_requirements": cov["unmet_requirements"],
                     "qualifying_branches": cov["successful_branches"],
                     "distinct_row_roots": len({(r, a["root"]) for r, v in q.items() for a in v}),
                     "rows": sum(1 for v in q.values() if v),
                     "post_landing_root_branches": sum(r["post_landing_root"] for r in recs),
                     "post_landing_root_pairs": len({(rid, rec["root"]) for rid in rids for rec in records_by_row[rid]
                                                     if qualifies(rec, cut) and rec["post_landing_root"]})}
            tried.append(entry)
            if cov["passed"]:
                chosen = entry
                break
        per_pose[pose] = {"chosen_cut_cm": None if chosen is None else chosen["cut_cm"],
                          "passed": chosen is not None,
                          "successful_branches": 0 if chosen is None else chosen["qualifying_branches"],
                          "unmet_requirements": [] if chosen is not None else tried[-1]["unmet_requirements"],
                          "ladder": tried}
    unmet = [f"{pose}:{req}" for pose, g in per_pose.items() for req in g["unmet_requirements"]]
    return {"counts": "depth-qualifying branches only, at each pose's chosen cut",
            "passed": not unmet, "unmet_requirements": unmet, "per_pose": per_pose}


def balanced_weights(poses: np.ndarray) -> tuple[np.ndarray, dict]:
    counts = collections.Counter(poses.tolist())
    n, p = len(poses), len(counts)
    w = np.array([n / (p * counts[x]) for x in poses.tolist()], dtype=np.float32)
    share = {k: float(w[poses == k].sum() / w.sum()) for k in counts}
    return w, {"samples": dict(counts), "draw_share": share,
               "multiplier_vs_uniform": {k: n / (p * c) for k, c in counts.items()}}


def student_qualifies(row_dir: Path, rules: dict, cut: float) -> tuple[bool, dict]:
    """A student episode qualifies by the same rule and its pose's cut, from its own rollout record."""
    r = json.loads((row_dir / "rollout.json").read_text())
    geo = r["geometry_trajectory"]
    pol = [g for g in geo if g["phase"] == "policy"]
    settle = [g for g in geo if g["phase"] == "terminal_settle"]
    first = next((g["step"] for g in pol if g["success"]), None)
    depth = None
    if settle:
        d = settle[-1]["details"]
        depth = float(min(d["condition_1"]["margin_cm"], d["condition_2"]["margin_cm"]))
    ok = bool(r.get("terminal_success")) and depth is not None and depth >= cut \
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

    # ---- 0. rows the search completed; the rest are recorded, never guessed
    excluded = []
    for row in rows:
        d = root / "recovery" / row["id"]
        sp = d / "status.json"
        ok = sp.is_file() and all((d / f).is_file() for f in ROW_FILES)
        if ok:
            try:
                ok = json.loads(sp.read_text()).get("state") == "completed"
                json.loads((d / "recovery.json").read_text())
            except ValueError:
                ok = False
        if not ok:
            excluded.append(row["id"])
    rows = [r for r in rows if r["id"] not in excluded]
    args.gate_out.parent.mkdir(parents=True, exist_ok=True)
    if len(excluded) > int(rules.get("max_excluded_rows", 1)):
        args.gate_out.write_text(json.dumps({"passed": False, "infrastructure": True, "excluded_rows": excluded,
                                             "unmet_requirements": [f"search rows missing: {excluded}"]}, indent=1) + "\n")
        print(json.dumps({"passed": False, "infrastructure": True, "excluded_rows": excluded}))
        return 5

    # ---- 1-3. mechanism check, label rule (ladder), coverage gate: attempt records only
    records, attempts_by_row, row_pose = {}, {}, {}
    for row in rows:
        pose = pose_of(row["match_pose"], clusters)
        if pose != row["pose"]:
            raise SystemExit(f"{row['id']}: declared pose {row['pose']} but match_pose is {pose}")
        row_pose[row["id"]] = pose
        rec = json.loads((root / "recovery" / row["id"] / "recovery.json").read_text())
        done = [a for a in rec["attempts"] if a.get("state") == "completed"]
        if any("margin_trace" not in a for a in done):
            raise SystemExit(f"{row['id']}: attempts lack v7 telemetry")
        attempts_by_row[row["id"]] = done
        records[row["id"]] = [branch_record(a, rules) for a in done]
    by_pose = collections.defaultdict(list)
    for rid, recs in records.items():
        by_pose[row_pose[rid]] += recs
    mech = mechanism_check(by_pose, rules)
    gate = ladder_gate(records, attempts_by_row, row_pose, manifest["recovery"]["coverage_gate"],
                       rules["cut_ladder_cm"])
    cut_by_pose = {p: g["chosen_cut_cm"] for p, g in gate["per_pose"].items()}
    qualifying = {rid: [a for a, r in zip(attempts_by_row[rid], records[rid])
                        if cut_by_pose.get(row_pose[rid]) is not None and qualifies(r, cut_by_pose[row_pose[rid]])]
                  for rid in records}
    curve = yield_curve(by_pose, rules)
    everything = [r for recs in records.values() for r in recs]
    chosen_q = [r for rid, recs in records.items() for r in recs
                if cut_by_pose.get(row_pose[rid]) is not None and qualifies(r, cut_by_pose[row_pose[rid]])]
    report = {"mechanism_check": mech, "qualifying_coverage_gate": gate, "cut_by_pose": cut_by_pose,
              "yield_curve": curve, "rules": rules, "excluded_rows": excluded,
              "first_fold_step_median": {
                  "qualifying_not_post_landing": float(np.median([r["first_fold_step"] for r in chosen_q
                                                                   if not r["post_landing_root"]]))
                  if any(not r["post_landing_root"] for r in chosen_q) else None,
                  "settled_early_not_qualifying": float(np.median([r["first_fold_step"] for r in everything
                                                                    if r["settled_early"] and r not in chosen_q]))
                  if any(r["settled_early"] and r not in chosen_q for r in everything) else None},
              "root_table": root_table(records),
              "branches": {rid: recs for rid, recs in records.items()}}
    passed = mech["passed"] and gate["passed"]
    report["passed"] = passed
    report["infrastructure"] = False
    report["unmet_requirements"] = ([] if mech["passed"] else ["mechanism_check"]) + gate["unmet_requirements"]
    report["per_pose"] = {p: {"successful_branches": g["successful_branches"], "chosen_cut_cm": g["chosen_cut_cm"],
                              "attempts_completed": sum(len(records[r]) for r in records if row_pose[r] == p)}
                          for p, g in gate["per_pose"].items()}
    args.gate_out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": passed, "unmet": report["unmet_requirements"],
                      "mechanism": {k: mech[k] for k in ("deep", "shallow", "p_one_sided")},
                      "cut_by_pose": cut_by_pose,
                      "qualifying": {p: g["successful_branches"] for p, g in gate["per_pose"].items()}}),
          flush=True)
    if not passed:
        return 4

    # ---- sizes first (no images), then one allocation
    plan = []
    for row in rows:
        d = root / "recovery" / row["id"]
        keep = {(int(q["root"]), int(q["candidate"])) for q in qualifying[row["id"]]}
        with np.load(d / "recovery_examples.npz", allow_pickle=False) as ex:
            pairs = list(zip(np.asarray(ex["root"]).tolist(), np.asarray(ex["candidate"]).tolist()))
        sel = np.array([p in keep for p in pairs], dtype=bool)
        s_ok, s_info = student_qualifies(d, rules, cut_by_pose[row_pose[row["id"]]])
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
        report["total_labels"] = n
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
                        "cut_cm": cut_by_pose[row_pose[row["id"]]],
                        "post_landing_root_qualifying": sum(r["post_landing_root"] for r in records[row["id"]]
                                                            if qualifies(r, cut_by_pose[row_pose[row["id"]]])),
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
                             "margin >= its pose's chosen cut and first all-4 by step "
                             f"{rules['first_fold_max_step']}"),
        "cut_by_pose": cut_by_pose, "excluded_rows": excluded,
        "weighting": "pose-balanced: awr_weight = N / (P * n_pose); uniform within a pose",
        "reused_corpora": "none",
        "gate_report": str(args.gate_out), "sources": sources,
    }, indent=2) + "\n")
    print(json.dumps({"samples": n, **balance["samples"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
