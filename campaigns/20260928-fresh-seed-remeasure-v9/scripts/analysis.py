"""v9's preregistered analysis: matched counts, exact tests, behaviour detectors.

v9 is measurement only. It runs the same two pinned policies:
- the untouched baseline;
- v8's an1-step000300.
They run on 48 fresh-seed development rows, in two blocks. Each block is
pinned to one hardware platform by a Slurm GPU-model constraint:
- a40 is the primary block, the user-approved design;
- rtx8000 is a secondary replication on a second platform.
This module turns the evaluation records into the preregistered report. It
decides nothing about the deliverable, which stays the baseline.

Everything here is pure and CPU-only. It reads status.json, request.json,
rollout.json, rollout.json.behavior.jsonl and the rollout.log device table,
and never loads trajectory images.

Detectors. The definitions are frozen in the manifest, together with their
calibration on the 128 v5-v8 H50 development episodes.

  E1   right-only chunk-1 opening (endpoints.chunk1_opening, v8's definition)
  E2   maximum particle lift over actions 1-150 (endpoints.early_lift)
  drag adapted from the v8 diagnosis's one-arm drag: a right-gripper close at
       table level (link z < 0.62 m) while the left link is parked
       (x < -0.20 m and z >= 0.62 m), followed within 45 actions by the
       right link crossing the midline (x < 0) and the cloth's maximum lift
       rising >= 3 cm above its value at the close. On the history it fires
       on the baseline at dev00/dev02/dev04 (P_A and P_C) in every run. Its
       first-event actions differ from v8's text because the criterion does.
  EBL  early bimanual table-level lift: left and right grippers close within
       4 actions of each other, both at table level, the later close within
       actions 1-150, followed within 45 actions by a lift gain >= 5 cm. On
       the history it fires on the baseline's early-H50 rows
       dev00/01/03/04/05, and for an1 also on dev06/dev07, where the
       baseline is 0/8.
"""
from __future__ import annotations

import collections
import datetime as dt
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
_DRIVER = re.compile(r"^\|\s*Driver Version:\s*([0-9.]+)", re.M)
_KERNEL = re.compile(r"Kernel:\s*(\S+)")
_CPU = re.compile(r"^\|\s*Processor:\s*(.+?)\s*$", re.M)
CONDITIONS = ("condition_1", "condition_2", "condition_3", "condition_4")


# ------------------------------------------------------------ hardware
def _log_head(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    with log_path.open(errors="replace") as stream:
        return stream.read(400_000)


def gpu_model(log_path: Path) -> str | None:
    """The GPU Isaac actually used, from the device table rollout.log prints
    at start-up; None if the table is absent or lists more than one name."""
    names = set(_GPU_ROW.findall(_log_head(log_path)))
    return names.pop() if len(names) == 1 else None


def platform(log_path: Path) -> dict:
    """GPU, driver, kernel and CPU from the same start-up table."""
    head = _log_head(log_path)
    first = lambda rx: (rx.findall(head) or [None])[0]  # noqa: E731
    return {"gpu": gpu_model(log_path), "driver": first(_DRIVER), "kernel": first(_KERNEL), "cpu": first(_CPU)}


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
    """P(A's success count >= observed | margins), hypergeometric (the v5-v8 driver's formula)."""
    total, successes = a_n + b_n, a_success + b_success
    denom = comb(total, a_n)
    return sum(comb(successes, k) * comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / denom


def h10_clause(cand: int, cand_n: int, base: int, base_n: int) -> bool:
    """matched_verdict's H10 clause, unchanged: margin >= 4 AND one-sided Fisher p < 0.05."""
    return cand - base >= 4 and fisher_one_sided(cand, cand_n, base, base_n) < 0.05


def clause_power(p0: float, p1: float, n: int = 24) -> float:
    """Exact probability that the H10 clause holds on n vs n when the baseline
    settles with probability p0 and the candidate with p1 (independent episodes)."""
    pb = [comb(n, k) * p0 ** k * (1 - p0) ** (n - k) for k in range(n + 1)]
    pc = [comb(n, k) * p1 ** k * (1 - p1) ** (n - k) for k in range(n + 1)]
    return sum(pb[b] * pc[c] for b in range(n + 1) for c in range(n + 1) if h10_clause(c, n, b, n))


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


def sign_flip_two_sided(diffs: list[int]) -> float:
    """Two-sided exact sign-flip test: share of assignments with |sum| >= |observed|."""
    n = len(diffs)
    observed = abs(sum(diffs))
    hits = sum(abs(sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))) >= observed
               for mask in range(1 << n))
    return hits / (1 << n)


def mcnemar_exact(cand_only: int, base_only: int) -> float:
    """One-sided exact McNemar on paired (row, seed) outcomes: P(X >= cand_only)
    for X ~ Binomial(cand_only + base_only, 1/2)."""
    n = cand_only + base_only
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(cand_only, n + 1)) / 2 ** n


# ------------------------------------------------------------ episodes
def parse_utc(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def episode(d: Path, expected_gpu: str | None = None, deadline_utc: str | None = None) -> dict:
    """One episode's record. valid=False for anything that did not complete in
    time on the pinned model (an infrastructure event, never a policy failure)."""
    sp, rp = d / "status.json", d / "rollout.json"
    out = {"dir": str(d), "valid": False}
    if not sp.is_file() or not rp.is_file():
        out["state"] = "missing"
        return out
    try:
        status = json.loads(sp.read_text())
        out["state"] = status.get("state")
        if status.get("state") != "completed":
            out["error"] = status.get("error")
            return out
        r = json.loads(rp.read_text())
        request = json.loads((d / "request.json").read_text()) if (d / "request.json").is_file() else {}
    except ValueError:
        out["state"] = "unreadable"
        return out
    plat = platform(d / "rollout.log")
    gpu = status.get("gpu_model") or plat["gpu"]
    details = r.get("terminal_checker", {}).get("details", {})
    margins = [details.get(c, {}).get("margin_cm") for c in CONDITIONS]
    out.update(valid=True, settled=bool(r.get("terminal_success")),
               reached=bool(r.get("geometric_ever_success")),
               first_reach_action=r.get("geometric_first_success_action"),
               conditions_passed=r.get("terminal_checker", {}).get("conditions_passed"),
               margins_cm=[round(m, 3) if isinstance(m, (int, float)) else None for m in margins],
               gpu_model=gpu, driver=plat["driver"], kernel=plat["kernel"], cpu=plat["cpu"],
               host=status.get("host"), completed_utc=status.get("completed_utc"),
               manifest_sha256=request.get("manifest_sha256"),
               horizon=r.get("effective_n_action_steps"), seed=r.get("seed"))
    if expected_gpu is not None and gpu != GPU_NAMES[expected_gpu]:
        out.update(valid=False, state="wrong_gpu_model")
    if deadline_utc is not None and (not out["completed_utc"]
                                     or parse_utc(out["completed_utc"]) > parse_utc(deadline_utc)):
        out.update(valid=False, state="after_deadline")
    bp = d / "rollout.json.behavior.jsonl"
    if out["valid"] and out["horizon"] == 50 and bp.is_file():
        out["behaviour"] = behaviour(endpoints.behavior(bp))
    return out


def row_index(row: dict) -> int:
    return int(row["development_pose_slot"])


def cluster_of(row: dict, clusters: dict[str, str]) -> str:
    for name, pose in clusters.items():
        if row["match_pose"] == pose:
            return name
    raise ValueError(f"{row['id']}: match_pose is in no pose cluster")


def _strata(paired, key) -> tuple[list[tuple[int, int, int, int]], list]:
    groups = collections.defaultdict(lambda: [0, 0, 0, 0])
    for c, b, row in paired:
        g = groups[key(row)]
        g[0] += int(c["settled"]); g[1] += 1
        g[2] += int(b["settled"]); g[3] += 1
    order = sorted(groups)
    return [tuple(groups[k]) for k in order], order


# ------------------------------------------------------------ one block
def block_results(root: Path, manifest: dict, block: dict, labels: dict[str, str]) -> dict:
    """labels: {'baseline': ..., 'candidate': ...} (checkpoint labels, without the run suffix)."""
    rows = manifest["remeasure"]
    clusters = manifest["pose_clusters"]
    deadline = manifest["protocol"]["block_deadline_utc"]
    eps = {}
    for which, label in labels.items():
        base = root / "evaluation" / f"{label}-{block['key']}" / "remeasure"
        eps[which] = [episode(base / row["id"], block["key"], deadline) for row in rows]
    digests = {e.get("manifest_sha256") for w in eps for e in eps[w] if e["valid"]}
    n_valid = {w: sum(e["valid"] for e in eps[w]) for w in eps}
    platforms = collections.Counter((e.get("gpu_model"), e.get("driver"), e.get("kernel"), e.get("host"))
                                    for w in eps for e in eps[w] if e["valid"])
    out = {"block": block["key"], "role": ("primary" if block["key"] == manifest["protocol"]["primary_block"]
                                           else "secondary"),
           "gpu_name": block["gpu_name"], "rows": len(rows), "valid": n_valid,
           "manifest_digests": sorted(d for d in digests if d) + (["<none>"] if None in digests else []),
           "complete": all(v == len(rows) for v in n_valid.values()) and len(digests) == 1 and None not in digests,
           "invalid": {w: {rows[i]["id"]: {"state": e.get("state"), "error": e.get("error")}
                           for i, e in enumerate(eps[w]) if not e["valid"]} for w in eps},
           "platforms": [{"gpu": g, "driver": dv, "kernel": k, "host": h, "episodes": n}
                         for (g, dv, k, h), n in sorted(platforms.items(), key=lambda x: -x[1])]}
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
        row_strata, row_keys = _strata(paired, row_index)
        cluster_strata, _ = _strata(paired, lambda r: cluster_of(r, clusters))
        diffs = [s[0] - s[2] for s in row_strata]
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
              "n_paired": len(paired),
              "row_stratified_exact_p": stratified_exact(row_strata) if row_strata else 1.0,
              "cluster_stratified_exact_p": stratified_exact(cluster_strata) if cluster_strata else 1.0,
              "row_sign_flip_p": sign_flip(diffs) if diffs else 1.0,
              "per_row_diff": {f"dev{k:02d}": d for k, d in zip(row_keys, diffs)},
              "paired_mcnemar": {"candidate_only": cand_only, "baseline_only": base_only,
                                 "one_sided_p": mcnemar_exact(cand_only, base_only)},
              "per_cluster_settled_reached_n": {k: {w: "/".join(map(str, v[w])) for w in v}
                                                for k, v in sorted(clusters_tab.items())},
              "per_row": {rows[i]["id"]: {w: {k: eps[w][i].get(k) for k in
                                              ("valid", "settled", "reached", "first_reach_action",
                                               "conditions_passed", "margins_cm", "gpu_model", "driver", "host",
                                               "behaviour", "state", "error")}
                                          for w in ("baseline", "candidate")} for i in idx}}
        if h == 10:
            hb["clause_margin_ge_4_and_p_lt_0.05"] = bool(cn and bn and h10_clause(cs, cn, bs, bn))
        else:
            hb["clause_candidate_ge_baseline"] = bool(cs >= bs)
            hb["behaviour"] = {}
            for which in ("baseline", "candidate"):
                beh = [(rows[i], eps[which][i].get("behaviour")) for i in idx if eps[which][i].get("behaviour")]
                hb["behaviour"][which] = {
                    "right_only_chunk1": sum(b["chunk1"] == "right" for _, b in beh),
                    "both_open_chunk1": sum(b["chunk1"] == "both" for _, b in beh),
                    "episodes": len(beh),
                    "drag_by_row": dict(sorted(collections.Counter(f"dev{row_index(r):02d}" for r, b in beh
                                                                   if b["drag_actions"]).items())),
                    "ebl_by_row": dict(sorted(collections.Counter(f"dev{row_index(r):02d}" for r, b in beh
                                                                  if b["ebl_actions"]).items())),
                    "dev03_early_lift_m": [b["early_lift_m"] for r, b in beh if row_index(r) == 3],
                }
            dev01 = {w: [eps[w][i].get("settled") for i in idx if row_index(rows[i]) == 1 and eps[w][i]["valid"]]
                     for w in ("baseline", "candidate")}
            hb["dev01"] = {"baseline": f"{sum(dev01['baseline'])}/{len(dev01['baseline'])}",
                           "candidate": f"{sum(dev01['candidate'])}/{len(dev01['candidate'])}",
                           "loss_recurs": sum(dev01["baseline"]) > sum(dev01["candidate"])}
            hb["p_b_settled"] = {f"dev{k:02d}": {w: sum(eps[w][i].get("settled", False) for i in idx
                                                        if row_index(rows[i]) == k and eps[w][i]["valid"])
                                                 for w in ("baseline", "candidate")} for k in (1, 3)}
        out[f"h{h}"] = hb
    out["rule"] = {"h10_clause": out["h10"]["clause_margin_ge_4_and_p_lt_0.05"],
                   "h50_clause": out["h50"]["clause_candidate_ge_baseline"],
                   "all_rows_valid": out["complete"],
                   "v5_rule_holds": bool(out["complete"] and out["h10"]["clause_margin_ge_4_and_p_lt_0.05"]
                                         and out["h50"]["clause_candidate_ge_baseline"])}
    return out


# ------------------------------------------------------------ both blocks
def _per_row_diff(block: dict) -> dict[str, int]:
    return block["h10"]["per_row_diff"]


def interaction(blocks: dict[str, dict]) -> dict:
    """Preregistered platform x policy interaction at H10: an exact two-sided
    sign-flip over the development rows of (cand - base on block 1) minus
    (cand - base on block 2), each summed over the row's 3 seeds."""
    a, b = (blocks[k] for k in sorted(blocks))
    da, db = _per_row_diff(a), _per_row_diff(b)
    rows = sorted(set(da) & set(db))
    diffs = [da[r] - db[r] for r in rows]
    return {"blocks": sorted(blocks), "rows": len(rows), "diff_of_diffs": dict(zip(rows, diffs)),
            "two_sided_p": sign_flip_two_sided(diffs) if diffs else 1.0}


def classify(blocks: dict[str, dict | None], protocol: dict) -> dict:
    """The preregistered reading. Each complete block is a reading in its own
    right (a40 primary, rtx8000 secondary); the cross-platform label needs both."""
    primary = protocol["primary_block"]
    complete = {k: v for k, v in blocks.items() if v and v.get("complete")}
    readings = {k: ("not measured (incomplete)" if k not in complete else
                    ("H10 clause holds" if complete[k]["rule"]["h10_clause"] else "H10 clause fails")
                    + ("; H50 clause holds" if complete[k]["rule"]["h50_clause"] else "; H50 clause fails"))
                for k in blocks}
    holds = sorted(k for k, v in complete.items() if v["rule"]["h10_clause"])
    inter = None
    power = protocol["power"]["headline"]
    if len(complete) < len(blocks):
        missing = sorted(set(blocks) - set(complete))
        cross = f"cross-platform classification not available ({', '.join(missing)} incomplete)"
    else:
        inter = interaction(complete)
        if len(holds) == len(blocks):
            cross = "replicates on both platforms"
        elif holds:
            other = sorted(set(blocks) - set(holds))[0]
            cross = (f"H10 clause holds on {holds[0]}, not on {other}; " +
                     (f"platform x policy interaction p={inter['two_sided_p']:.3f}: hardware-dependent"
                      if inter["two_sided_p"] < 0.05 else
                      f"platform x policy interaction not established (p={inter['two_sided_p']:.3f})"))
        else:
            cross = f"not replicated at this power ({power})"
    return {"primary_block": primary, "readings": readings, "h10_gain": cross,
            "headline": f"{primary} (primary): {readings[primary]} | " + " | ".join(
                f"{k} (secondary): {readings[k]}" for k in sorted(blocks) if k != primary) + f" | {cross}",
            "h10_clause_holds_on": holds, "measured_blocks": sorted(complete),
            "v5_rule_holds_on": sorted(k for k, v in complete.items() if v["rule"]["v5_rule_holds"]),
            "interaction": inter,
            "deliverable": "baseline (v9 is measurement only)"}


def cross_block(root: Path, manifest: dict, labels: dict[str, str]) -> dict:
    """Same rows and seeds on both platforms: settled by platform, per policy and row (unpaired)."""
    rows = manifest["remeasure"]
    deadline = manifest["protocol"]["block_deadline_utc"]
    keys = [b["key"] for b in manifest["blocks"]]
    out = {}
    for which, label in labels.items():
        tab = collections.defaultdict(dict)
        for key in keys:
            base = root / "evaluation" / f"{label}-{key}" / "remeasure"
            for row in rows:
                e = episode(base / row["id"], key, deadline)
                cell = tab[f"dev{row_index(row):02d}_h{row['horizon']}"].setdefault(key, [0, 0])
                if e["valid"]:
                    cell[0] += int(e["settled"]); cell[1] += 1
        out[which] = {k: {m: f"{v[0]}/{v[1]}" for m, v in d.items()} for k, d in sorted(tab.items())}
    return out


def compute(root: Path, manifest: dict, labels: dict[str, str], finished: list[str]) -> dict:
    blocks = {b["key"]: (block_results(root, manifest, b, labels) if b["key"] in finished else None)
              for b in manifest["blocks"]}
    done = [v for v in blocks.values() if v and v["complete"]]
    combined = {}
    if len(done) == len(blocks):
        for h in (10, 50):
            strata = []
            for v in done:
                per_row = collections.defaultdict(lambda: [0, 0, 0, 0])
                for rid, rec in v[f"h{h}"]["per_row"].items():
                    k = int(rid[3:5])
                    per_row[k][0] += int(rec["candidate"]["settled"]); per_row[k][1] += 1
                    per_row[k][2] += int(rec["baseline"]["settled"]); per_row[k][3] += 1
                strata += [tuple(per_row[k]) for k in sorted(per_row)]
            combined[f"h{h}"] = {"candidate": sum(s[0] for s in strata), "baseline": sum(s[2] for s in strata),
                                 "n": sum(s[1] for s in strata),
                                 "block_x_row_stratified_exact_p": stratified_exact(strata),
                                 "note": "descriptive; cannot override the per-block readings"}
    dev01 = [v["h50"]["dev01"] for v in done]
    if dev01:
        b = sum(int(x["baseline"].split("/")[0]) for x in dev01)
        c = sum(int(x["candidate"].split("/")[0]) for x in dev01)
        n = sum(int(x["baseline"].split("/")[1]) for x in dev01)
        m = sum(int(x["candidate"].split("/")[1]) for x in dev01)
        combined["dev01_h50"] = {"baseline": f"{b}/{n}", "candidate": f"{c}/{m}",
                                 "baseline_better_one_sided_p": fisher_one_sided(b, n, c, m) if n and m else 1.0,
                                 "loss_recurs_in": [v["block"] for v in done if v["h50"]["dev01"]["loss_recurs"]],
                                 "note": "descriptive, not gated; 3 vs 3 per block cannot reach significance"}
    return {"blocks": blocks, "classification": classify(blocks, manifest["protocol"]), "combined": combined,
            "cross_block": cross_block(root, manifest, labels) if len(done) == len(blocks) else None}
