"""Regression: checkout discovery must not choose the robot USD."""
import dataclasses
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "scripts/render/asset_paths.py"
spec = importlib.util.spec_from_file_location("asset_paths", path)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)

@dataclasses.dataclass
class Spawn:
    usd_path: str
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

class AssetPaths(unittest.TestCase):
    def test_explicit_asset_root_wins_over_checkout_cwd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = root / "shared/Assets/robots/lerobot/so101_follower_good.usd"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"usd fixture")
            checkout = root / "checkout"
            checkout.mkdir()
            before = Path.cwd()
            try:
                os.chdir(checkout)
                self.assertEqual(assets.robot_asset_path(root / "shared/Assets"), str(asset))
            finally:
                os.chdir(before)

    def test_both_arms_use_validated_asset_without_mutating_defaults(self):
        default = Spawn("/wrong/checkout/Assets/robot.usd")
        cfg = SimpleNamespace(left_robot=SimpleNamespace(spawn=default), right_robot=SimpleNamespace(spawn=default))
        assets.configure_robot_assets(cfg, "/shared/robot.usd")
        self.assertEqual(cfg.left_robot.spawn.usd_path, "/shared/robot.usd")
        self.assertEqual(cfg.right_robot.spawn.usd_path, "/shared/robot.usd")
        self.assertEqual(default.usd_path, "/wrong/checkout/Assets/robot.usd")
        self.assertIsNot(cfg.left_robot.spawn, cfg.right_robot.spawn)

    def test_missing_and_empty_robot_fail_before_simulator_start(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "--assets"):
                assets.robot_asset_path(temp)
            path = Path(temp) / "robots/lerobot/so101_follower_good.usd"
            path.parent.mkdir(parents=True)
            path.touch()
            with self.assertRaises(FileNotFoundError):
                assets.robot_asset_path(temp)

if __name__ == "__main__":
    unittest.main()
