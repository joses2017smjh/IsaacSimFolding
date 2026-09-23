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


# ------------------------------------------ iteration 3: advantage sharpness
import sys  # noqa: E402
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from lehome_fold import awr  # noqa: E402

ITER2_REWARDS = np.array([1.0] * 2 + [0.75] * 6 + [0.5] * 4 + [0.25] * 4)   # audited gate


def _mass(beta, w_max):
    adv = awr.success_residual(ITER2_REWARDS, np.full_like(ITER2_REWARDS, ITER2_REWARDS.mean()))
    w = awr.weights(adv, beta=beta, w_max=w_max, w_min=1e-6).astype(np.float64)
    return w, float(w[:2].sum() / w.sum())


def test_iteration2_weighting_left_successes_a_minority_of_the_mass():
    """The diagnosis A3 rests on: under beta=1, w_max=3 the two successes hit
    the cap at 3.0 while each 3/4 failure got ~1.88, so failures dominated."""
    w, success_mass = _mass(1.0, 3.0)
    assert success_mass == pytest.approx(0.286, abs=0.005)
    failure_3of4_mass = float(w[2:8].sum() / w.sum())
    assert failure_3of4_mass == pytest.approx(0.54, abs=0.01)


def test_lowering_beta_with_the_old_cap_makes_it_worse_not_better():
    """beta and w_max are coupled: with w_max=3 the 3/4 failures hit the cap too."""
    _, success_mass = _mass(0.5, 3.0)
    assert success_mass < 0.286


def test_iteration3_constants_concentrate_on_successes_and_still_pass_every_gate():
    w, success_mass = _mass(0.5, 20.0)
    assert success_mass == pytest.approx(0.63, abs=0.01)
    gates = MANIFEST["gates"]
    ess = awr.effective_sample_size(w)
    assert gates["ess_min"] <= ess <= len(w) - 1          # concentrated, not collapsed
    assert ess == pytest.approx(4.58, abs=0.05)
    assert len(np.unique(ITER2_REWARDS)) >= gates["distinct_rewards_min"]
    assert float(np.std(ITER2_REWARDS - ITER2_REWARDS.mean())) > gates["advantage_std_min"]
    # 111 usable samples per episode for every H50 row -> sample ESS fraction = ESS/16
    assert ess / len(w) <= gates["sample_ess_fraction_max"]


# ------------------------------------------------- reuse plumbing
def test_make_plan_refuses_a_partial_reused_source(tmp_path, monkeypatch):
    root = tmp_path / "campaigns" / "c"
    (root / "plans").mkdir(parents=True)
    (root / "rollouts" / "iteration2" / "a").mkdir(parents=True)
    (root / "rollouts" / "iteration2" / "a" / "trajectory.npz").write_bytes(b"x")
    (root / "plans" / "iteration2.resolved.json").write_text(json.dumps(
        {"collection": [{"id": "a"}, {"id": "b"}], "collection_horizons": [50]}))
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"iteration": 3, "collection_source": "iteration2",
                                "factor_changed": "x", "justification": "x",
                                "collection_checkpoint": "/b", "init_checkpoint": "/b"}))
    monkeypatch.setattr(sys, "argv", ["make_plan", "--campaign", str(root), "--spec", str(spec)])
    with pytest.raises(SystemExit, match="holds 1 trajectories"):
        make_plan.main()
