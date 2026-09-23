"""CPU tests for the iteration-2+ tooling: plan separation and horizon handling.

A later iteration's plan is where train/evaluation separation is easiest to
break, and a horizon change is where the compiler is easiest to fool. Both
are pinned here before any GPU time depends on them.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_plan = _load("make_plan")
builder = _load("build_rollout_dataset")


def _row(garment, key, seed):
    return {"id": f"x_{garment}_{key}_{seed}", "garment": garment, "pose_key": key, "seed": seed}


# ------------------------------------------------------------- separation
def test_plan_refuses_a_frozen_test_garment():
    garment = MANIFEST["frozen_test"][0]["garment"]
    problems = make_plan.separation_problems([_row(garment, 3, 70001)], MANIFEST, allow_dev_overlap=None)
    assert any("frozen-test garment" in p for p in problems)


def test_plan_refuses_a_development_garment_pose_pair_unless_justified():
    dev = MANIFEST["benchmark"][0]
    row = _row(dev["garment"], int(dev["pose_key"]), 70002)
    assert any("development" in p for p in
               make_plan.separation_problems([row], MANIFEST, allow_dev_overlap=None))
    assert make_plan.separation_problems([row], MANIFEST, allow_dev_overlap="stated reason") == []


def test_plan_refuses_preregistered_and_repeated_seeds():
    dev_seed = MANIFEST["benchmark"][0]["seed"]
    problems = make_plan.separation_problems(
        [_row("Pant_Short_Seen_4", 0, dev_seed), _row("Pant_Short_Seen_5", 0, 70003),
         _row("Pant_Short_Seen_6", 0, 70003)], MANIFEST, allow_dev_overlap=None)
    assert any("preregistered block" in p for p in problems)
    assert any("repeated" in p for p in problems)


def test_same_class_non_development_non_test_rows_are_accepted():
    rows = [_row("Pant_Short_Seen_4", 0, 70010), _row("Pant_Short_Seen_5", 1, 70011)]
    assert make_plan.separation_problems(rows, MANIFEST, allow_dev_overlap=None) == []


# --------------------------------------------------------------- horizons
def _episode(tmp_path: Path, horizon: int) -> Path:
    steps, every = 55, 5
    idx = np.arange(0, steps, every)
    stream = np.cumsum(np.ones((steps, 12), dtype=np.float32) * 0.01, axis=0)
    d = tmp_path / f"h{horizon}"
    d.mkdir()
    np.savez(d / "trajectory.npz",
             step_index=idx,
             images=np.zeros((len(idx), 3, 480, 640, 3), dtype=np.uint8),
             state=np.zeros((len(idx), 12), dtype=np.float32),
             executed_action=stream[idx],
             executed_action_stream=stream,
             camera_keys=np.asarray(["top_rgb", "left_rgb", "right_rgb"]),
             seed=np.asarray(7))
    (d / "rollout.json").write_text(json.dumps({
        "steps": steps, "garment": "Pant_Short_Seen_4", "effective_n_action_steps": horizon,
        "prediction_chunk_size": 50, "terminal_success": False,
        "terminal_checker": {"conditions_passed": 2, "conditions_total": 4, "success": False}}))
    (d / "request.json").write_text(json.dumps({
        "row": {"id": "r", "garment": "Pant_Short_Seen_4", "pose_key": 0, "seed": 7},
        "checkpoint": {"path": "/c"}}))
    return d / "trajectory.npz"


def test_undeclared_horizon_is_refused(tmp_path):
    path = _episode(tmp_path, 50)
    with pytest.raises(ValueError, match="declared horizons"):
        builder.load_episode(path, 50, (10,))


def test_declared_h50_collection_compiles_with_executed_targets(tmp_path):
    """An H50 rollout's label at t is still what it executed next, stream[t:t+50]."""
    path = _episode(tmp_path, 50)
    source, _, images, _, targets = builder.load_episode(path, 50, (50,))
    with np.load(path) as raw:
        stream = raw["executed_action_stream"]
    assert source["execution_horizon"] == 50
    assert targets.shape == (2, 50, 12)             # t=0 and t=5 have a full suffix
    assert np.array_equal(targets[1], stream[5:55])  # mid-chunk observation, true continuation
