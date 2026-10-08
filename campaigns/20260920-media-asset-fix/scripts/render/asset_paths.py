"""Resolve robot assets independently of Git discovery and the caller's cwd."""
from pathlib import Path


def robot_asset_path(assets):
    path = Path(assets).expanduser().resolve() / "robots/lerobot/so101_follower_good.usd"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty SO101 robot USD: {path}; check --assets")
    return str(path)


def configure_robot_assets(cfg, robot_usd):
    # Replace each spawn config independently; upstream module defaults stay intact.
    for name in ("left_robot", "right_robot"):
        arm = getattr(cfg, name)
        arm.spawn = arm.spawn.replace(usd_path=robot_usd)
