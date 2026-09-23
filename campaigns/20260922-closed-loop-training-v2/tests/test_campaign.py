"""CPU tests for the parts of this campaign that can fail without a simulator.

Every test here pins a specific failure that the 2026-09-22 review found in
the v1 draft. They run in about a second on a login node, which is the point:
the gate logic decides whether to spend forty GPU tasks, so it should not be
first exercised by spending them.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "src"))

from lehome_fold import awr  # noqa: E402


# --------------------------------------------------------------- AWR cap
def test_cap_binds_and_is_a_live_parameter():
    """v1 recorded w_max in provenance while it could never fire."""
    adv = [3.0, 1.0, 0.0, -1.0, -3.0]
    tight = awr.weights(adv, beta=1.0, w_max=2.0)
    loose = awr.weights(adv, beta=1.0, w_max=1000.0)
    assert tight.max() == pytest.approx(2.0, rel=1e-5)
    assert not np.allclose(tight, loose)
    assert awr.weight_summary(tight, w_max=2.0)["capped"] >= 1


def test_manifest_w_max_actually_binds_on_a_realistic_collection():
    """The cap must be live for THIS campaign's constants, not just in theory.

    An 8-episode all-failure collection is the expected case. If w_max sits
    outside that batch's z-range the cap is inert and recording it is a lie.
    """
    cfg = json.loads((ROOT / "manifest.json").read_text())["awr"]
    rewards = np.array([2 / 4, 2 / 4, 2 / 4, 1 / 4, 2 / 5, 3 / 5, 2 / 5, 2 / 5])
    adv = awr.success_residual(rewards, np.full_like(rewards, rewards.mean()))
    w = awr.weights(adv, beta=cfg["beta"], w_max=cfg["w_max"], w_min=cfg["w_min"])
    assert awr.weight_summary(w, w_max=cfg["w_max"])["capped"] >= 1, (
        f"w_max={cfg['w_max']} never binds at beta={cfg['beta']} on a realistic batch")


def test_weights_never_reach_zero_probability():
    w = awr.weights([0.0, -500.0], beta=0.001, w_max=3.0, w_min=1e-6)
    assert np.all(w > 0)
    assert (w.astype(np.float64) / w.astype(np.float64).sum()).sum() == pytest.approx(1.0)


# ------------------------------------------------------------ gate logic
def _gate_checks(rewards, cfg, gate_cfg):
    """Mirror of build_rollout_dataset's gate, exercised on synthetic rewards."""
    rewards = np.asarray(rewards, dtype=np.float32)
    adv = awr.success_residual(rewards, np.full_like(rewards, rewards.mean()))
    w = awr.weights(adv, beta=cfg["beta"], w_max=cfg["w_max"], w_min=cfg["w_min"])
    stats = awr.weight_summary(w, w_max=cfg["w_max"], w_min=cfg["w_min"])
    n = len(rewards)
    return {
        "reward_informative": int(np.unique(np.round(rewards, 6)).size) >= gate_cfg["distinct_rewards_min"],
        "advantage_informative": float(np.std(adv)) > gate_cfg["advantage_std_min"],
        "episode_weights_non_uniform": stats["ess"] <= n - 1,
        "episode_weights_not_concentrated": stats["ess"] >= gate_cfg["ess_min"],
    }


def _cfg():
    m = json.loads((ROOT / "manifest.json").read_text())
    return m["awr"], m["gates"]


def test_identical_rewards_are_rejected_as_uniform():
    """The failure mode the whole gate exists for.

    Eight identical outcomes give zero advantage, uniform weights and an ESS
    of exactly n -- the MAXIMUM. A one-sided "ESS is healthy" check passes
    this, and the run silently becomes uniform imitation of its own failures.
    """
    cfg, gate_cfg = _cfg()
    checks = _gate_checks([0.5] * 8, cfg, gate_cfg)
    assert not checks["reward_informative"]
    assert not checks["advantage_informative"]
    assert not checks["episode_weights_non_uniform"], "uniform weights must fail the gate"
    # ... and prove ESS alone would have waved it through.
    w = awr.weights(awr.success_residual(np.full(8, 0.5), np.full(8, 0.5)))
    assert awr.effective_sample_size(w) == pytest.approx(8.0)


def test_single_episode_domination_is_rejected():
    cfg, gate_cfg = _cfg()
    cfg = dict(cfg, beta=0.05)  # sharpen until one episode swamps the batch
    checks = _gate_checks([0.0] * 7 + [1.0], cfg, gate_cfg)
    assert not checks["episode_weights_not_concentrated"]


def test_realistic_all_failure_collection_passes():
    """An all-failure collection is still trainable if outcomes DIFFER."""
    cfg, gate_cfg = _cfg()
    checks = _gate_checks([2 / 4, 2 / 4, 2 / 4, 1 / 4, 2 / 5, 3 / 5, 2 / 5, 2 / 5], cfg, gate_cfg)
    assert all(checks.values()), checks


def test_two_distinct_rewards_fail_the_diversity_floor():
    cfg, gate_cfg = _cfg()
    checks = _gate_checks([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], cfg, gate_cfg)
    assert not checks["reward_informative"]


# --------------------------------------------------------------- manifest
def test_frozen_test_set_is_disjoint_from_training_and_development():
    m = json.loads((ROOT / "manifest.json").read_text())
    trained = {r["garment"] for r in m["collection"]} | {r["garment"] for r in m["collection_expansion"]}
    developed = {r["garment"] for r in m["benchmark"]}
    held = {r["garment"] for r in m["frozen_test"]}
    assert not (held & trained), held & trained
    assert not (held & developed), held & developed
    assert len({r["seed"] for r in m["frozen_test"]} & {r["seed"] for r in m["collection"]}) == 0


def test_every_protocol_decision_is_frozen_before_launch():
    m = json.loads((ROOT / "manifest.json").read_text())
    for key in ("budget", "gates", "training", "selection", "targets", "awr", "protocol"):
        assert key in m, f"{key} must be preregistered"
    assert m["selection"]["rule"].startswith("Evaluate step_000300")
    assert m["budget"]["max_gpu_tasks_iteration1"] > 0
    # RECAP's gradient update is unimplemented; the campaign must not claim it.
    assert "NOT used" in m["protocol"]["recap_conditioning"]


def test_collection_retains_failures_and_declares_no_seed_selection():
    m = json.loads((ROOT / "manifest.json").read_text())
    assert "no seed, pose or episode" in m["retention"]
    seeds = [r["seed"] for r in m["collection"]]
    assert len(seeds) == len(set(seeds)) == 8


def test_rollout_timeouts_are_distinct_and_bounded():
    """v1 had `900 if phase == "smoke" else 900` -- a lost distinction."""
    budget = json.loads((ROOT / "manifest.json").read_text())["budget"]
    t = budget["rollout_timeout_seconds"]
    assert t["smoke"] < t["collection"], "an 80-action smoke must not get a 600-action budget"
    for phase, seconds in t.items():
        assert 0 < seconds <= 3600, (phase, seconds)


# ------------------------------------------------------------ source pins
def test_manifest_pins_the_runner_it_executes():
    m = json.loads((ROOT / "manifest.json").read_text())
    assert m["runner_key"] in m["executed_sources"]
    assert (REPO / m["runner_key"]).is_file()


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_sources.py"),
         "--campaign", str(ROOT), "--skip-checkpoint"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
