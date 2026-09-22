"""Build conservative manipulation-event and matched-horizon analyses.

Inputs are the completed pilot's simulator-state telemetry. This script never
replays, mutates, or relabels the original rollouts. It emits raw observations
and separately named derived proxy events. The existing campaign did not
enable a validated particle-cloth contact sensor, so contact/grasp fields stay
unknown instead of being inferred from distance alone.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/failure-instrumentation"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def read_episode(row):
    directory = ROOT / "outputs" / row["id"]
    result = json.loads((directory / "rollout.json").read_text())
    behavior_path = Path(result["behavior_telemetry"]["path"])
    records = [json.loads(line) for line in behavior_path.read_text().splitlines()]
    return directory, result, records


def euclidean(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def first_or_none(values, predicate):
    return next((int(v["action"]) for v in values if predicate(v)), None)


def derived_motion(records, index):
    """Finite-difference simulator-state signals, kept distinct from raw data."""
    record = records[index]
    dt = float(record.get("simulation_seconds") or (1.0 / 90.0))
    result = {
        "gripper_origin_velocity_mps": {},
        "nearest_particle_velocity_mps": {},
        "velocity_alignment_cosine": {},
        "coupled_motion_proxy": False,
    }
    if index == 0:
        return result
    previous = records[index - 1]
    for side in ("left", "right"):
        current_nearest = record["nearest_particle"][side]
        previous_nearest = previous["nearest_particle"][side]
        g0 = previous_nearest["link_origin_xyz_m"]
        g1 = current_nearest["link_origin_xyz_m"]
        p0 = previous_nearest["particle_xyz_m"]
        p1 = current_nearest["particle_xyz_m"]
        gd = [float(b) - float(a) for a, b in zip(g0, g1)]
        pd = [float(b) - float(a) for a, b in zip(p0, p1)]
        gnorm = math.sqrt(sum(x * x for x in gd))
        pnorm = math.sqrt(sum(x * x for x in pd))
        result["gripper_origin_velocity_mps"][side] = gnorm / dt
        result["nearest_particle_velocity_mps"][side] = pnorm / dt
        if gnorm > 1e-6 and pnorm > 1e-6:
            result["velocity_alignment_cosine"][side] = sum(a * b for a, b in zip(gd, pd)) / (gnorm * pnorm)
        else:
            result["velocity_alignment_cosine"][side] = None
        aligned = result["velocity_alignment_cosine"][side]
        if (float(record["gripper_link_origin_distance_m"][side]) <= .03
                and gnorm / dt >= .01 and pnorm / dt >= .01
                and aligned is not None and aligned >= .5):
            result["coupled_motion_proxy"] = True
    return result


def derive(row, result, records):
    motion = [derived_motion(records, i) for i in range(len(records))]
    distances = {side: [float(r["gripper_link_origin_distance_m"][side]) for r in records]
                 for side in ("left", "right")}
    minimum = {side: min(values) for side, values in distances.items()}
    all_distances = [min(v["gripper_link_origin_distance_m"].values()) for v in records]
    first5 = first_or_none(records, lambda r: min(r["gripper_link_origin_distance_m"].values()) <= .05)
    first3 = first_or_none(records, lambda r: min(r["gripper_link_origin_distance_m"].values()) <= .03)
    first_boundary = first_or_none(records, lambda r: r["replan_boundary"] and r["action"] > 1)
    best = max(int(r["conditions_passed"]) for r in records)
    terminal = int(result["terminal_checker"]["conditions_passed"])
    if first5 is None:
        stage = "A_never_reached_cloth_proxy"
    else:
        stage = "B_reached_cloth_no_verified_contact"
    # The only available coupling signal is a named link origin and its
    # nearest particle. Keep this a proxy and never call it contact/acquisition.
    near = [r for r in records if min(r["gripper_link_origin_distance_m"].values()) <= .03]
    lift_after_near = max((r["maximum_particle_lift_m"] for r in near), default=0.0)
    # Require five consecutive aligned finite-difference samples. This is a
    # conservative motion-coupling proxy, not a contact or grasp label.
    coupled_run = coupled_proxy = False
    for signal in motion:
        coupled_run = coupled_run + 1 if signal["coupled_motion_proxy"] else 0
        if coupled_run >= 5:
            coupled_proxy = True
    if coupled_proxy:
        stage = "C_contact_or_coupling_unknown_proxy"
    if result["terminal_success"]:
        stage = "G_official_task_success"
    elif result["geometric_ever_success"]:
        stage = "F_local_success_then_terminal_failure"
    elif best > terminal:
        stage = "F_partial_condition_regression"
    return {
        "id": row["id"], "pose": row["development_pose_slot"],
        "garment": row["garment"], "local_demo": row["pose_source_local_episode_key"],
        "seed": row["seed"], "horizon": row["horizon"], "actions": len(records),
        "official_ever_success": bool(result["official_ever_success"]),
        "terminal_success": bool(result["terminal_success"]),
        "first_success_action": result["first_success_step"],
        "first_approach_proxy_action_le_5cm": first5,
        "first_near_proxy_action_le_3cm": first3,
        "first_replan_boundary_after_action_1": first_boundary,
        "minimum_gripper_origin_distance_m": minimum,
        "maximum_particle_lift_m": max((float(r["maximum_particle_lift_m"]) for r in records), default=0.0),
        "terminal_mean_particle_displacement_m": result["cloth_motion"]["terminal_mean_particle_displacement"],
        "best_policy_conditions": best, "terminal_conditions": terminal,
        "conditions_total": result["terminal_checker"]["conditions_total"],
        "replans": result["replanning_count"],
        "inference_seconds": sum(result["inference_seconds"]),
        "wall_seconds": result["wall_seconds"],
        "stage": stage,
        "native_contact_available": False,
        "native_contact_status": "unknown: original LeHome particle-cloth scene has no validated ContactSensor/contact-pair stream; rigid contact API was not retrofitted into the frozen pilot",
        "contact_start_action": None, "contact_end_action": None,
        "contacting_gripper": None, "acquisition_candidate_action": None,
        "retention_duration_actions": None, "release_drop_action": None,
        "grasp_label": "unknown/insufficient evidence",
        "proxy_coupling_action": next((int(records[i]["action"]) for i in range(len(records))
                                        if any(motion[j]["coupled_motion_proxy"] for j in range(max(0, i - 4), i + 1))
                                        and i >= 4), None),
        "proxy_coupling_definition": "five consecutive samples with named gripper/jaw origin <=3 cm, particle and gripper motion >=1 cm/s, and velocity cosine >=0.5; not contact or grasp ground truth",
        "maximum_particle_lift_after_near_m": lift_after_near,
        "records": records, "result": result,
    }


def action_difference(short, base):
    for a, b in zip(short["records"], base["records"]):
        diff = euclidean(a["executed_action_rad"], b["executed_action_rad"])
        if diff > .25:
            return int(a["action"]), diff
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    for row in MANIFEST["pilot_tasks"]:
        _, result, records = read_episode(row)
        episodes.append(derive(row, result, records))

    # events.jsonl contains every raw state/action row plus explicit derived
    # event fields. Raw and derived names remain separate for auditability.
    with (OUT / "events.jsonl").open("w") as stream:
        for episode in episodes:
            for i, record in enumerate(episode["records"]):
                motion = derived_motion(episode["records"], i)
                raw = {
                    "record_type": "raw_simulator_state_and_policy_action",
                    "episode": episode["id"], "pose": episode["pose"],
                    "horizon": episode["horizon"], "garment": episode["garment"],
                    "seed": episode["seed"], "raw": record,
                    "derived": {
                        "approach_le_5cm_proxy": min(record["gripper_link_origin_distance_m"].values()) <= .05,
                        "near_le_3cm_proxy": min(record["gripper_link_origin_distance_m"].values()) <= .03,
                        "particle_lift_ge_2cm_proxy": record["maximum_particle_lift_m"] >= .02,
                        "gripper_origin_velocity_mps": motion["gripper_origin_velocity_mps"],
                        "nearest_particle_velocity_mps": motion["nearest_particle_velocity_mps"],
                        "velocity_alignment_cosine": motion["velocity_alignment_cosine"],
                        "coupled_motion_proxy": motion["coupled_motion_proxy"],
                        "native_contact": None,
                        "acquisition": None,
                        "retention": None,
                        "release_drop": None,
                    },
                    "label_status": "raw fields measured; derived proximity/lift fields are proxies; contact/grasp labels unknown",
                }
                stream.write(json.dumps(raw, allow_nan=False) + "\n")

    by_pose = {}
    for episode in episodes:
        by_pose.setdefault(episode["pose"], {})[episode["horizon"]] = episode
    fields = ["pose", "garment", "local_demo", "seed", "horizon", "official_ever_success", "terminal_success",
              "first_success_action", "first_approach_proxy_action_le_5cm", "first_near_proxy_action_le_3cm",
              "minimum_gripper_origin_distance_m", "best_policy_conditions", "terminal_conditions", "conditions_total",
              "maximum_particle_lift_m", "terminal_mean_particle_displacement_m", "replans", "inference_seconds", "wall_seconds",
              "stage", "action_divergence_vs_h50_action", "action_l2_at_divergence_rad", "first_boundary_action",
              "boundary_jump_mean_rad", "first_camera_change_action", "native_contact_available", "grasp_label"]
    with (OUT / "matched-comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pose in sorted(by_pose):
            base = by_pose[pose][50]
            for horizon in (50, 10, 5):
                episode = by_pose[pose][horizon]
                divergence, diff = (None, None) if horizon == 50 else action_difference(episode, base)
                boundary = episode["result"]["behavior_telemetry"]["replan_boundary_jumps"]
                cameras = episode["result"]["behavior_telemetry"]["camera_changes_between_replans"]
                row = {key: episode.get(key) for key in fields}
                row.update({"action_divergence_vs_h50_action": divergence,
                            "action_l2_at_divergence_rad": diff,
                            "first_boundary_action": boundary[0]["action"] if boundary else None,
                            "boundary_jump_mean_rad": sum(x["l2_rad"] for x in boundary) / len(boundary) if boundary else None,
                            "first_camera_change_action": cameras[0]["action"] if cameras else None,
                            "minimum_gripper_origin_distance_m": json.dumps(episode["minimum_gripper_origin_distance_m"], sort_keys=True)})
                writer.writerow(row)

    # A compact pose-level machine-readable comparison, including earliest
    # divergence and both success outcomes.
    matched = []
    for pose in sorted(by_pose):
        base = by_pose[pose][50]
        item = {"pose": pose, "garment": base["garment"], "local_demo": base["local_demo"], "horizons": {}}
        for horizon in (50, 10, 5):
            e = by_pose[pose][horizon]
            divergence, diff = (None, None) if horizon == 50 else action_difference(e, base)
            item["horizons"][str(horizon)] = {key: e[key] for key in fields if key in e}
            item["horizons"][str(horizon)].update({"first_action_divergence_vs_h50": divergence, "action_l2_at_divergence_rad": diff})
        matched.append(item)
    (OUT / "matched-comparison.json").write_text(json.dumps(matched, indent=2) + "\n")

    # Restore only serializable summaries to avoid duplicating all raw records.
    summaries = []
    for e in episodes:
        summary = {key: value for key, value in e.items() if key not in ("records", "result")}
        summaries.append(summary)
    (OUT / "episode-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "events": sum(len(e["records"]) for e in episodes),
                      "output": str(OUT)}))


if __name__ == "__main__":
    main()
