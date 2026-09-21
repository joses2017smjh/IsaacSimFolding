"""Freeze the smoke and matched horizon matrix before observing any new outcome."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1].parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if (ROOT / "manifest.json").exists():
        raise FileExistsError("Do not mutate an existing experiment manifest")
    data = WORKSPACE / "lehome-data"
    demos_path = data / "Datasets/example/four_types_merged/meta/garment_info.json"
    demos = json.loads(demos_path.read_text())
    ckpt = data / "outputs/train/bc_smolvla_raster_ft_full"
    def task(name, local_demo, seed, horizon, steps, tag):
        demo = demos[name][str(local_demo)]
        if len(set(demo["scale"])) != 1:
            raise ValueError("Anisotropic demo scale requires an explicit protocol")
        return {"id": tag, "garment": name, "pose_source": str(demos_path),
            "pose_source_local_episode_key": str(local_demo),
            "match_pose": ":".join(format(v, ".10g") for v in demo["object_initial_pose"]),
            "match_scale": demo["scale"][0], "seed": seed, "horizon": horizon, "steps": steps}
    smoke_names = ["Top_Short_Seen_3", "Pant_Short_Seen_0", "Pant_Long_Seen_0", "Top_Long_Seen_0", "Top_Short_Seen_0"]
    smokes = [task(name, 0, 100, 50, 120, "smoke_" + name) for name in smoke_names]
    switch = task("Top_Long_Seen_0", 0, 100, 50, 120, "diagnose_switch_Top_Long_0_to_1")
    switch["switch_to"] = "Top_Long_Seen_1"
    smokes.append(switch)
    pilot = []
    slot = 0
    for garment in ("Pant_Short_Seen_0", "Pant_Short_Seen_3", "Pant_Short_Seen_7", "Pant_Short_Seen_9"):
        for local_demo in (0, 1):
            for horizon in (50, 10, 5):
                row = task(garment, local_demo, 200 + slot, horizon, 600, f"dev{slot:02d}_h{horizon:02d}_{garment}")
                row["development_pose_slot"] = slot
                pilot.append(row)
            slot += 1
    source = {}
    for directory in ("src", "scripts", "tests", "external", "slurm"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".toml", ".sbatch"):
                source[str(path.relative_to(ROOT))] = digest(path)
    manifest = {"schema_version": 3, "repository": "https://github.com/joses2017smjh/IsaacSimFolding.git",
        "root_commit": "34a424c0a0a1e1d52cbd43beca4a4ad24738bc3b", "assets": str(data / "Assets"),
        "checkpoint": {"path": str(ckpt), "selection": "existing historical raster-adapted checkpoint; fixed before development",
            "sha256": {p.name: digest(p) for p in sorted(ckpt.iterdir()) if p.is_file()}},
        "task_prompt": "fold the garment on the table", "predicted_chunk_size": 50,
        "initial_settle_steps": 60, "terminal_settle_steps": 60,
        "scorer_protocol": "original pure predicates and thresholds; authored landmarks mapped to runtime-verified welded particles",
        "historical_results_directly_comparable": False,
        "physics_timestep": "unchanged frozen upstream configuration; actual value recorded in every result",
        "smoke_tasks": smokes, "pilot_tasks": pilot, "source_sha256": source,
        "asset_inventory_sha256": digest(ROOT / "audit/garment_inventory.json"),
        "pose_metadata_sha256": digest(demos_path),
        "submission_policy": "submit smoke first; no pilot until experiment-level smoke_gate.json passes",
        "maximum_simultaneous_gpu_tasks": 1,
        "storage": {"measured_home_du": "28G", "measured_home_df": "25G total,22G used,3.4G available",
            "measured_user_HPC_share_du": "1.4T", "existing_600_action_media_output": "47M",
            "projected_smoke_outputs_upper_bound_bytes": 1000000000,
            "projected_24_pilot_outputs_upper_bound_bytes": 3000000000,
            "raw_RGB_datasets_saved": False, "large_home_writes": False},
        "training_jobs_authorized_in_this_phase": False}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "audit/smoke_gate.json").write_text(json.dumps({"passed": False, "status": "not_run", "required_indices": list(range(5))}, indent=2) + "\n")
    print(json.dumps({"smoke_tasks": len(smokes), "pilot_tasks": len(pilot), "source_files": len(source),
                      "checkpoint_sha256": manifest["checkpoint"]["sha256"]["model.safetensors"]}, indent=2))


if __name__ == "__main__":
    main()
