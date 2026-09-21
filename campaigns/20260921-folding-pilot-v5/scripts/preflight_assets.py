"""Validate the frozen gate and asset bindings without starting Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "render"))
from asset_paths import robot_asset_path


def validate(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    robot = Path(robot_asset_path(manifest["assets"]))
    scene = Path(manifest["assets"]) / "scenes/marble/Scene_00_Apartment.usd"
    if not scene.is_file() or scene.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty apartment USD: {scene}")
    hashes = json.loads((root / "source_hashes.json").read_text())
    for rel, expected in hashes.items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source changed: {rel}")
    # Upstream garment JSON uses /Assets/... relative to the LeHome cwd.
    local = root / "external/lehome-challenge/Assets"
    assets = Path(manifest["assets"]).resolve()
    if (local / "objects").resolve() != (assets / "objects").resolve():
        raise ValueError("LeHome Assets/objects must resolve to the manifest asset root")
    for row in manifest["tasks"]:
        garment = Path(row["garment_dir"])
        configs = list(garment.glob("*.json"))
        if len(configs) != 1 or not list(garment.glob("*.usd")):
            raise ValueError(f"Missing or ambiguous garment files: {garment}")
        data = json.loads(configs[0].read_text())
        for rel in [data["asset_path"], *data.get("visual_usd_paths", [])]:
            path = root / "external/lehome-challenge" / rel.lstrip("/")
            if not path.is_file():
                raise FileNotFoundError(path)
    result = {"robot_asset": str(robot), "robot_sha256": hashlib.sha256(robot.read_bytes()).hexdigest(),
              "scene_asset": str(scene), "source_files_checked": len(hashes), "tasks": len(manifest["tasks"])}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    validate(parser.parse_args().campaign)
