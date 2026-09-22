"""Static protocol checks for the pose-3 observation-component diagnostic."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (ROOT / "scripts/render/policy_rollout51.py").read_text()
LAUNCHER = (ROOT / "slurm/observation.sbatch").read_text()


def test_active_observation_components_and_probe_names():
    for key in (
        "observation.state",
        "observation.images.top_rgb",
        "observation.images.left_rgb",
        "observation.images.right_rgb",
        '"task": args.task',
    ):
        assert key in CONTROLLER
    for name in (
        "current-all",
        "old-all-images-current-state",
        "current-images-old-state",
        "old-top-only",
        "old-left-only",
        "old-right-only",
        "old-both-wrists",
    ):
        assert f'"{name}"' in CONTROLLER


def test_probe_is_gated_and_pose3_launcher_is_narrow():
    assert "meaningful_improvement_rad = 0.05" in CONTROLLER
    assert "[:2]" in CONTROLLER
    assert "--observation-diagnostic" in LAUNCHER
    assert "--observation-execute-best" in LAUNCHER
    assert "index=1" in LAUNCHER
    assert "--rtc-guidance" not in LAUNCHER
    assert "--queue-diagnostic" not in LAUNCHER


def test_prediction_audit_and_snapshot_restore_are_fail_closed():
    assert "inference_blocks_before_env_step" in CONTROLLER
    assert "simulator_stepped_during_prediction" in CONTROLLER
    assert "all_snapshot_restores_within_tolerance" in CONTROLLER
    assert "all_predictions_zero_sim_steps" in CONTROLLER
    assert "all_rng_restores_exact" in CONTROLLER
    assert "_snapshot_state(global_step - 1, images)" in CONTROLLER
