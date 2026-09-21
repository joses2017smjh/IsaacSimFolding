"""Audit actual Release USDs, demonstrations, capture headers and raw results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(ROOT / "src"))
from lehome_fold.folding_geometry import authored_mesh, validate_indices, weld_correspondence
from audit_capture_manifest import read_capture


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, data):
    (ROOT / "audit" / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    data = WORKSPACE / "lehome-data"
    assets = data / "Assets"
    release = assets / "objects/Challenge_Garment/Release"
    demo_path = data / "Datasets/example/four_types_merged/meta/garment_info.json"
    demos = json.loads(demo_path.read_text())
    recent = WORKSPACE / "bhl-robustness-ladder/results/weekend-20260919"
    split_path = recent / "fold-adapt-s0/data_split.json"
    split = json.loads(split_path.read_text())
    split1 = json.loads((recent / "fold-adapt-s1/data_split.json").read_text())
    if split != split1:
        raise ValueError("Adaptation seeds do not share a split")
    train = {(r["garment"], r["episode"]) for r in split["train"]}
    validation = {(r["garment"], r["episode"]) for r in split["validation"]}
    if train & validation or {g for g, _ in train} & {g for g, _ in validation}:
        raise ValueError("Episode or garment leakage")
    if any("_Unseen_" in g for g, _ in train | validation):
        raise ValueError("Official Unseen identity in adaptation data")
    captures = [read_capture(p, 50) for p in sorted((data / "storm_capture100").glob("*.npz"))]
    records = []
    for f in sorted(REPO.glob("results/rollout_*.json")):
        d = json.loads(f.read_text())
        if "garment" in d:
            records.append({"path": str(f), "sha256": digest(f), "garment": d["garment"],
                            "mode": d.get("mode"), "success": d.get("success"),
                            "terminal_success": d.get("terminal_success"),
                            "steps": d.get("steps"), "policy": d.get("policy")})
    baselines = []
    policy_jobs = {"baseline": "21359573", "adapt0": "21359574", "adapt1": "21359575"}
    class_index = {"pant_short": 0, "top_short": 1, "top_long": 2, "pant_long": 3}
    for f in sorted(recent.glob("fold-*-s100.json")):
        d = json.loads(f.read_text())
        variant = f.name.split("-")[1]
        if variant not in policy_jobs:
            continue
        job = policy_jobs[variant] + "_" + str(class_index[d["garment_type"]])
        logs = list(recent.glob(f"wk-fold-*-{job}.out"))
        log = logs[0].read_text(errors="replace") if len(logs) == 1 else ""
        all_episodes = []
        for g in d["garments"]:
            for ordinal, ep in enumerate(g["episodes"], 1):
                all_episodes.append({"garment": g["garment"], "split": g["split"],
                    "episode_ordinal_within_garment_1based": ordinal,
                    "ever_success": ep["success"], "reported_pre_success_length": ep["length"],
                    "first_success_action_1based": ep["length"] + 1 if ep["success"] else None,
                    "executed_actions_derived": min(d["max_steps"], ep["length"] + 50) if ep["success"] else ep["length"],
                    "length_semantics_source": "scripts/utils/evaluation.py:175-200 excludes triggering action from length; 50-step tail includes triggering action",
                    "terminal_settled_success": None, "seed": d["seed"],
                    "physics_health": g.get("physics_health"),
                    "historical_evaluation_record_complete": True,
                    "new_strict_protocol_validated": False})
        error = d.get("error")
        cause = "completed" if d["completed"] else ("scorer_index_error" if error else "garment_switch_timeout")
        switch_lines = [s for s in log.splitlines() if "Switching garment to:" in s]
        baselines.append({"policy_variant": variant, "policy_path": d["policy"],
            "class": "_".join(s.capitalize() for s in d["garment_type"].split("_")),
            "source": str(f), "sha256": digest(f), "slurm_job_id": job,
            "log": str(logs[0]) if logs else None,
            "protocol_completed": d["completed"], "expected_episodes": d["expected_episodes"],
            "completed_recorded_episodes": len(all_episodes),
            "unrecorded_episodes": d["expected_episodes"] - len(all_episodes),
            "ever_successes": sum(e["ever_success"] for e in all_episodes),
            "terminal_settled_successes": None,
            "interruption": cause, "error": error,
            "last_switch": switch_lines[-1] if switch_lines else None,
            "last_log_lines": log.splitlines()[-12:],
            "episodes": all_episodes,
            "denominator_note": "Historical completed returned metrics only; missing episodes are not policy failures. Landmark semantics and terminal settling require a new corrected baseline."})
    garments = []
    for category in ("Pant_Short", "Pant_Long", "Top_Short", "Top_Long"):
        listing = release / category / (category + ".txt")
        names = listing.read_text().split()
        expected = [f"{category}_Seen_{n}" for n in range(10)] + [f"{category}_Unseen_{n}" for n in range(2)]
        if names != expected:
            raise ValueError(f"Unexpected Release identities/order for {category}: {names}")
        directories = {p.name for p in (release / category).iterdir() if p.is_dir()}
        if directories != set(names):
            raise ValueError(f"Release directory/list disagreement: {category}")
        for name in names:
            folder = release / category / name
            configs = list(folder.glob("*.json"))
            if len(configs) != 1:
                raise ValueError(f"Ambiguous garment config: {folder}")
            cfg = json.loads(configs[0].read_text())
            mesh = assets / cfg["asset_path"].removeprefix("/Assets/")
            materials = [assets / p.removeprefix("/Assets/") for p in cfg["visual_usd_paths"] if p]
            if not all(p.is_file() for p in [mesh, *materials]):
                raise ValueError(f"Missing asset for {name}")
            points, prim = authored_mesh(mesh)
            mapping, unique = weld_correspondence(points)
            indices = cfg["check_point"]
            validate_indices(indices, len(points))
            mapped = mapping[indices].tolist()
            validate_indices(mapped, len(unique))
            capture = [r for r in captures if r["garment"] == name]
            history = [r for r in records if r["garment"] == name]
            raw_eval = [{"source": b["source"], "policy": b["policy_variant"], **e}
                        for b in baselines for e in b["episodes"] if e["garment"] == name]
            media = sorted({str(p) for pattern in (f"results/*{name}*.gif", f"docs/gifs/*{name}*.gif")
                            for p in REPO.glob(pattern)})
            garments.append({"garment_id": name, "category": category,
                "split": "Unseen" if "_Unseen_" in name else "Seen",
                "listing": str(listing), "config": str(configs[0]), "config_sha256": digest(configs[0]),
                "mesh_path": str(mesh), "mesh_sha256": digest(mesh), "mesh_prim": prim,
                "vertex_count": len(points), "welded_particle_count_static": len(unique),
                "runtime_particle_count": None,
                "material_paths": [str(p) for p in materials], "color": None,
                "authored_landmarks": indices, "mapped_particle_landmarks": mapped,
                "legacy_index_out_of_bounds": any(i >= len(unique) for i in indices),
                "legacy_landmark_selection_changed": indices != mapped,
                "scorer_conditions_total": 5 if category.startswith("Top") else 4,
                "scorer_raw_thresholds": cfg["success_distance"], "default_scale": cfg["scale"],
                "demonstration_local_keys": sorted(map(int, demos.get(name, {}))),
                "demonstration_count": len(demos.get(name, {})),
                "captures": capture, "successful_capture_count": sum(r["success"] for r in capture),
                "failed_capture_count": sum(not r["success"] for r in capture),
                "adaptation_train_episode_ids": sorted(ep for g, ep in train if g == name),
                "adaptation_validation_episode_ids": sorted(ep for g, ep in validation if g == name),
                "historical_results": history, "strict_evaluation_episodes": raw_eval,
                "media_paths": media})
    if len(demos) != 40 or sum(len(v) for v in demos.values()) != 1000:
        raise ValueError("Unexpected demonstration counts")
    if {g["garment_id"] for g in garments if g["split"] == "Seen"} != set(demos):
        raise ValueError("Demo garment identities do not match official Seen identities")
    inventory = {"schema_version": 2, "actual_assets_verified": True, "garments": garments,
        "garment_count": len(garments), "demonstration_count": sum(len(v) for v in demos.values()),
        "demonstration_metadata": str(demo_path), "demonstration_metadata_sha256": digest(demo_path),
        "adaptation_split": str(split_path), "adaptation_split_sha256": digest(split_path),
        "capture_count": len(captures), "successful_captures": sum(r["success"] for r in captures),
        "class_counts": dict(Counter(g["category"] for g in garments)),
        "scorer_out_of_bounds_garments": [g["garment_id"] for g in garments if g["legacy_index_out_of_bounds"]],
        "scorer_changed_landmark_garments": [g["garment_id"] for g in garments if g["legacy_landmark_selection_changed"]],
        "static_preflight_passed": True, "runtime_correspondence_gate_passed": False,
        "missing_32": "Local pack contains Release only; no private Holdout assets present. Current upstream availability was not queried.",
        "capture_label_caveat": "Historical replay-success labels used legacy landmark selection; affected identities need corrected geometric revalidation."}
    write("garment_inventory.json", inventory)
    write("baseline_episodes.json", {"cells": baselines, "new_protocol_validated": False,
        "terminal_success_available": False, "legacy_counts_verified_from_raw_artifacts": True})
    write("phase_a.json", {"static_asset_scorer_preflight": True, "exact_counts": True,
        "whole_episode_and_garment_split": True, "runtime_landmark_correspondence": False,
        "phase_a_complete": False, "required_tests_passed": False})
    print(json.dumps({k: v for k, v in inventory.items() if k != "garments"}, indent=2))


if __name__ == "__main__":
    main()
