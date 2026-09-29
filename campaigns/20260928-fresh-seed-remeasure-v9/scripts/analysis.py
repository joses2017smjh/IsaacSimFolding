"""v9's preregistered analysis: matched counts, exact tests, behaviour detectors.

v9 is measurement only. The same two pinned policies (the untouched baseline
and v8's an1-step000300) run on 48 fresh-seed development rows in two blocks,
each pinned to one GPU model by a Slurm constraint. This module turns the
evaluation records of one block (or both) into the preregistered report. It
decides nothing about the deliverable, which stays the baseline.

Everything here is pure and CPU-only. It reads status.json, rollout.json,
rollout.json.behavior.jsonl and the rollout.log device table, and never
loads trajectory images.

Detectors (definitions frozen in the manifest with their calibration on the
256 v5-v8 development episodes):

  E1   right-only chunk-1 opening (endpoints.chunk1_opening, v8's definition)
  E2   maximum particle lift over actions 1-150 (endpoints.early_lift)
  drag the baseline's P_A failure step, as diagnosed in v8: a right-gripper
       close at table level (link z < 0.62 m) while the left link is parked
       (x < -0.20 m and z >= 0.62 m), followed within 45 actions by the
       right link crossing the midline (x < 0) and the cloth's maximum lift
       rising >= 3 cm above its value at the close
  EBL  early bimanual lift, the baseline's early-H50 plan: left and right
       grippers close within 4 actions of each other, both at table level,
       the later close within actions 1-150, followed within 45 actions by a
       lift gain >= 5 cm
"""
from __future__ import annotations

import collections
import json
from math import comb
from pathlib import Path
import re

import endpoints

TABLE_Z = 0.62
PARKED_X = -0.20
DRAG_WINDOW = 45
DRAG_LIFT_M = 0.03
EBL_PAIR = 4
EBL_EARLY = 150
EBL_WINDOW = 45
EBL_LIFT_M = 0.05

GPU_NAMES = {"a40": "NVIDIA A40", "rtx8000": "Quadro RTX 8000"}
_GPU_ROW = re.compile(r"^\|\s*\d+\s*\|\s*([^|]*?)\s*\|\s*Yes", re.M)
CONDITIONS = ("condition_1", "condition_2", "condition_3", "condition_4")


# ------------------------------------------------------------ hardware
def gpu_model(log_path: Path) -> str | None:
    """The GPU Isaac actually used, from the device table rollout.log prints
    at start-up; None if the table is absent or lists more than one name."""
    if not log_path.is_file():
        return None
    with log_path.open(errors="replace") as stream:
        head = stream.read(400_000)
    names = set(_GPU_ROW.findall(head))
    return names.pop() if len(names) == 1 else None


# ------------------------------------------------------------ detectors
def close_events(rows: list[dict]) -> list[tuple[int, str, int]]:
    """(action, side, row index) for every gripper-target crossing from open
    (> endpoints.OPEN_RAD) to closed (<= endpoints.OPEN_RAD)."""
    events = []
    for side in ("left", "right"):
        prev = None
        for i, r in enumerate(rows):
            g = r["gripper_target_rad"][side]
            if prev is not None and prev > endpoints.OPEN_RAD and g <= endpoints.OPEN_RAD:
                events.append((int(r["action"]), side, i))
            prev = g
    return sorted(events)


def _xyz(row: dict, side: str) -> list[float]:
    return row["nearest_particle"][side]["link_origin_xyz_m"]


def drag_events(rows: list[dict]) -> list[int]:
    hits = []
    for action, side, i in close_events(rows):
        if side != "right":
            continue
        r = rows[i]
        left = _xyz(r, "left")
        if _xyz(r, "right")[2] >= TABLE_Z or not (left[0] < PARKED_X and left[2] >= TABLE_Z):
            continue
        window = rows[i:i + DRAG_WINDOW + 1]
        lift0 = r["maximum_particle_lift_m"]
        if (min(_xyz(w, "right")[0] for w in window) < 0.0
                and max(w["maximum_particle_lift_m"] for w in window) - lift0 >= DRAG_LIFT_M):
            hits.append(action)
    return hits


def early_bimanual_lift(rows: list[dict]) -> list[int]:
    """Actions of the later close of every qualifying bimanual table-level close."""
    events = close_events(rows)
    hits = set()
    for a_left, s_left, i_left in events:
        if s_left != "left":
            continue
        for a_right, s_right, i_right in events:
            if s_right != "right" or abs(a_right - a_left) > EBL_PAIR:
                continue
            j = max(i_left, i_right)
            r = rows[j]
            if int(r["action"]) > EBL_EARLY:
                continue
            if _xyz(r, "left")[2] >= TABLE_Z or _xyz(r, "right")[2] >= TABLE_Z:
                continue
            window = rows[j:j + EBL_WINDOW + 1]
            if max(w["maximum_particle_lift_m"] for w in window) - r["maximum_particle_lift_m"] >= EBL_LIFT_M:
                hits.add(int(r["action"]))
    return sorted(hits)


def behaviour(rows: list[dict]) -> dict:
    return {"chunk1": endpoints.chunk1_opening(rows),
            "early_lift_m": round(endpoints.early_lift(rows), 4),
            "drag_actions": drag_events(rows),
            "ebl_actions": early_bimanual_lift(rows)}


# ------------------------------------------------------------ statistics
def fisher_one_sided(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """P(A's success count >= observed | margins), hypergeometric (driver's formula)."""
    total, successes = a_n + b_n, a_success + b_success
    denom = comb(total, a_n)
    return sum(comb(successes, k) * comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / denom


def _hypergeom_pmf(successes: int, a_n: int, b_n: int) -> dict[int, float]:
    total = a_n + b_n
    denom = comb(total, a_n)
    return {k: comb(successes, k) * comb(total - successes, a_n - k) / denom
            for k in range(max(0, successes - b_n), min(successes, a_n) + 1)}


def stratified_exact(strata: list[tuple[int, int, int, int]]) -> float:
    """One-sided exact test of the candidate's total successes, conditional on
    each stratum's margins: strata are (cand_success, cand_n, base_success, base_n).
    The null distribution is the convolution of the per-stratum hypergeometrics."""
    dist = {0: 1.0}
    observed = 0
    for cs, cn, bs, bn in strata:
        observed += cs
        pmf = _hypergeom_pmf(cs + bs, cn, bn)
        new: dict[int, float] = collections.defaultdict(float)
        for x, px in dist.items():
            for k, pk in pmf.items():
                new[x + k] += px * pk
        dist = new
    return sum(p for x, p in dist.items() if x >= observed - 1e-9)


def sign_flip(diffs: list[int]) -> float:
    """One-sided exact sign-flip test: the share of the 2^n sign assignments
    whose sum is at least the observed sum (rows as the unit)."""
    n = len(diffs)
    observed = sum(diffs)
    hits = 0
    for mask in range(1 << n):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))
        hits += s >= observed
    return hits / (1 << n)


def mcnemar_exact(cand_only: int, base_only: int) -> float:
    """One-sided exact McNemar on paired (row, seed) outcomes: P(X >= cand_only)
    for X ~ Binomial(cand_only + base_only, 1/2)."""
    n = cand_only + base_only
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(cand_only, n + 1)) / 2 ** n


# ------------------------------------------------------------ episodes
def episode(d: Path, expected_gpu: str | None = None) -> dict:
    """One episode's record. valid=False for anything that did not complete
    (an infrastructure event, never a policy failure)."""
    sp, rp = d / "status.json", d / "rollout.json"
    out = {"dir": str(d), "valid": False}
    if not sp.is_file() or not rp.is_file():
        return out
    try:
        status = json.loads(sp.read_text())
        if status.get("state") != "completed":
            out["state"] = status.get("state")
            return out
        r = json.loads(rp.read_text())
    except ValueError:
        return out
    gpu = status.get("gpu_model") or gpu_model(d / "rollout.log")
    details = r.get("terminal_checker", {}).get("details", {})
    out.update(valid=True, settled=bool(r.get("terminal_success")),
               reached=bool(r.get("geometric_ever_success")),
               first_reach_action=r.get("geometric_first_success_action"),
               conditions_passed=r.get("terminal_checker", {}).get("conditions_passed"),
               margins_cm=[round(details.get(c, {}).get("margin_cm", float("nan")), 3) for c in CONDITIONS],
               gpu_model=gpu, horizon=r.get("effective_n_action_steps"), seed=r.get("seed"))
    if expected_gpu is not None and gpu != GPU_NAMES[expected_gpu]:
        out.update(valid=False, state="wrong_gpu_model")
    bp = d / "rollout.json.behavior.jsonl"
    if out["valid"] and out["horizon"] == 50 and bp.is_file():
        out["behaviour"] = behaviour(endpoints.behavior(bp))
    return out


def pose_index(row: dict) -> int:
    return int(row["development_pose_slot"])


def cluster_of(row: dict, clusters: dict[str, str]) -> str:
    for name, pose in clusters.items():
        if row["match_pose"] == pose:
            return name
    raise ValueError(f"{row['id']}: match_pose is in no pose cluster")


# ------------------------------------------------------------ one block
def block_results(root: Path, manifest: dict, block: dict, labels: dict[str, str]) -> dict:
    """labels: {'baseline': ..., 'candidate': ...} (checkpoint labels, without the run suffix)."""
    rows = manifest["remeasure"]
    clusters = manifest["pose_clusters"]
    eps = {}
    for which, label in labels.items():
        base = root / "evaluation" / f"{label}-{block['key']}" / "remeasure"
        eps[which] = [episode(base / row["id"], block["key"]) for row in rows]
    n_valid = {w: sum(e["valid"] for e in eps[w]) for w in eps}
    out = {"block": block["key"], "gpu_name": block["gpu_name"], "rows": len(rows),
           "valid": n_valid, "complete": all(v == len(rows) for v in n_valid.values()),
           "invalid": {w: [rows[i]["id"] for i, e in enumerate(eps[w]) if not e["valid"]] for w in eps},
           "gpu_models_seen": dict(collections.Counter(e.get("gpu_model") for w in eps for e in eps[w]
                                                       if e.get("gpu_model")))}
    for h in (10, 50):
        idx = [i for i, r in enumerate(rows) if int(r["horizon"]) == h]
        paired = [(eps["candidate"][i], eps["baseline"][i], rows[i]) for i in idx
                  if eps["candidate"][i]["valid"] and eps["baseline"][i]["valid"]]
        cs = sum(eps["candidate"][i].get("settled", False) for i in idx if eps["candidate"][i]["valid"])
        bs = sum(eps["baseline"][i].get("settled", False) for i in idx if eps["baseline"][i]["valid"])
        cn = sum(eps["candidate"][i]["valid"] for i in idx)
        bn = sum(eps["baseline"][i]["valid"] for i in idx)
        creach = sum(eps["candidate"][i].get("reached", False) for i in idx if eps["candidate"][i]["valid"])
        breach = sum(eps["baseline"][i].get("reached", False) for i in idx if eps["baseline"][i]["valid"])
        by_pose = collections.defaultdict(lambda: [0, 0, 0, 0])
        for c, b, row in paired:
            k = pose_index(row)
            by_pose[k][0] += int(c["settled"]); by_pose[k][1] += 1
            by_pose[k][2] += int(b["settled"]); by_pose[k][3] += 1
        strata = [tuple(by_pose[k]) for k in sorted(by_pose)]
        diffs = [s[0] - s[2] for s in strata]
        cand_only = sum(c["settled"] and not b["settled"] for c, b, _ in paired)
        base_only = sum(b["settled"] and not c["settled"] for c, b, _ in paired)
        clusters_tab = collections.defaultdict(lambda: {"candidate": [0, 0, 0], "baseline": [0, 0, 0]})
        for i in idx:
            name = cluster_of(rows[i], clusters)
            for which in ("candidate", "baseline"):
                e = eps[which][i]
                if e["valid"]:
                    t = clusters_tab[name][which]
                    t[0] += int(e["settled"]); t[1] += int(e["reached"]); t[2] += 1
        p = fisher_one_sided(cs, cn, bs, bn) if cn and bn else 1.0
        hb = {"candidate": f"{cs}/{cn}", "baseline": f"{bs}/{bn}", "margin": cs - bs,
              "reached": {"candidate": f"{creach}/{cn}", "baseline": f"{breach}/{bn}"},
              "fisher_one_sided_p": p,
              "pose_stratified_exact_p": stratified_exact(strata) if strata else 1.0,
              "pose_sign_flip_p": sign_flip(diffs) if diffs else 1.0,
              "per_pose_diff": {f"dev{k:02d}": by_pose[k][0] - by_pose[k][2] for k in sorted(by_pose)},
              "paired_mcnemar": {"candidate_only": cand_only, "baseline_only": base_only,
                                 "one_sided_p": mcnemar_exact(cand_only, base_only)},
              "per_cluster_settled_reached_n": {k: {w: "/".join(map(str, v[w])) for w in v}
                                                for k, v in sorted(clusters_tab.items())},
              "per_row": {rows[i]["id"]: {w: {k: eps[w][i].get(k) for k in
                                              ("valid", "settled", "reached", "first_reach_action",
                                               "conditions_passed", "margins_cm", "gpu_model", "behaviour",
                                               "state")}
                                          for w in ("baseline", "candidate")} for i in idx}}
        if h == 10:
            hb["clause_margin_ge_4_and_p_lt_0.05"] = bool(cs - bs >= 4 and p < 0.05)
        else:
            hb["clause_candidate_ge_baseline"] = bool(cs >= bs)
            hb["behaviour"] = {}
            for which in ("baseline", "candidate"):
                beh = [(rows[i], eps[which][i].get("behaviour")) for i in idx if eps[which][i].get("behaviour")]
                hb["behaviour"][which] = {
                    "right_only_chunk1": sum(b["chunk1"] == "right" for _, b in beh),
                    "both_open_chunk1": sum(b["chunk1"] == "both" for _, b in beh),
                    "episodes": len(beh),
                    "drag_by_pose": dict(sorted(collections.Counter(f"dev{pose_index(r):02d}" for r, b in beh
                                                                    if b["drag_actions"]).items())),
                    "ebl_by_pose": dict(sorted(collections.Counter(f"dev{pose_index(r):02d}" for r, b in beh
                                                                   if b["ebl_actions"]).items())),
                    "dev03_early_lift_m": [b["early_lift_m"] for r, b in beh if pose_index(r) == 3],
                }
            hb["p_b_settled"] = {f"dev{k:02d}": {w: sum(eps[w][i].get("settled", False) for i in idx
                                                        if pose_index(rows[i]) == k and eps[w][i]["valid"])
                                                 for w in ("baseline", "candidate")} for k in (1, 3)}
        out[f"h{h}"] = hb
    out["rule"] = {"h10_clause": out["h10"]["clause_margin_ge_4_and_p_lt_0.05"],
                   "h50_clause": out["h50"]["clause_candidate_ge_baseline"],
                   "all_rows_valid": out["complete"],
                   "v5_rule_holds": bool(out["complete"] and out["h10"]["clause_margin_ge_4_and_p_lt_0.05"]
                                         and out["h50"]["clause_candidate_ge_baseline"])}
    return out


# ------------------------------------------------------------ both blocks
def classify(blocks: dict[str, dict | None]) -> dict:
    """The preregistered reading of the H10 gain across the two GPU-model blocks.
    A block counts only when complete (every row valid for both policies)."""
    measured = {k: v for k, v in blocks.items() if v and v.get("complete")}
    holds = sorted(k for k, v in measured.items() if v["rule"]["h10_clause"])
    if len(measured) < len(blocks):
        status = "partially measured"
    elif len(holds) == len(blocks):
        status = "replicates on both GPU models"
    elif holds:
        status = f"replicates on {holds[0]} only (hardware-conditional)"
    else:
        status = "does not replicate"
    return {"h10_gain": status, "h10_clause_holds_on": holds, "measured_blocks": sorted(measured),
            "v5_rule_holds_on": sorted(k for k, v in measured.items() if v["rule"]["v5_rule_holds"]),
            "deliverable": "baseline (v9 is measurement only)"}


def cross_block(root: Path, manifest: dict, labels: dict[str, str]) -> dict:
    """Same rows and seeds on both models: settled by model, per policy and pose."""
    rows = manifest["remeasure"]
    keys = [b["key"] for b in manifest["blocks"]]
    out = {}
    for which, label in labels.items():
        tab = collections.defaultdict(dict)
        for key in keys:
            base = root / "evaluation" / f"{label}-{key}" / "remeasure"
            for row in rows:
                e = episode(base / row["id"], key)
                cell = tab[f"dev{pose_index(row):02d}_h{row['horizon']}"].setdefault(key, [0, 0])
                if e["valid"]:
                    cell[0] += int(e["settled"]); cell[1] += 1
        out[which] = {k: {m: f"{v[0]}/{v[1]}" for m, v in d.items()} for k, d in sorted(tab.items())}
    return out


def compute(root: Path, manifest: dict, labels: dict[str, str], finished: list[str]) -> dict:
    blocks = {b["key"]: (block_results(root, manifest, b, labels) if b["key"] in finished else None)
              for b in manifest["blocks"]}
    combined = {}
    done = [v for v in blocks.values() if v and v["complete"]]
    if len(done) == len(blocks):
        for h in (10, 50):
            strata = []
            for v in done:
                per_pose = collections.defaultdict(lambda: [0, 0, 0, 0])
                for rid, rec in v[f"h{h}"]["per_row"].items():
                    k = int(rid[3:5])
                    per_pose[k][0] += int(rec["candidate"]["settled"]); per_pose[k][1] += 1
                    per_pose[k][2] += int(rec["baseline"]["settled"]); per_pose[k][3] += 1
                strata += [tuple(per_pose[k]) for k in sorted(per_pose)]
            combined[f"h{h}"] = {"candidate": sum(s[0] for s in strata), "baseline": sum(s[2] for s in strata),
                                 "n": sum(s[1] for s in strata),
                                 "block_x_pose_stratified_exact_p": stratified_exact(strata)}
    return {"blocks": blocks, "classification": classify(blocks), "combined": combined,
            "cross_block": cross_block(root, manifest, labels) if len(done) == len(blocks) else None}
