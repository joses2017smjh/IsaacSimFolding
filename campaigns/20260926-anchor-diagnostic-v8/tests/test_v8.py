"""CPU tests for v8, the anchor diagnostic: the single changed variable (the
anchor), the byte-identical corpus reuse, the preregistered endpoints, the
confirmatory final-set rows and gate, the new runner phase, and the driver's
stage order. Carried v6/v7 pieces keep their v7 tests' checks where they run.
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
CAMPAIGNS = ROOT.parent
V7 = CAMPAIGNS / "20260925-depth-qualified-v7"
PY = sys.executable


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


driver = _load("driver")
ep = _load("endpoints")
rmt = _load("run_matched_task")


# ---------------------------------------------------------------- anchor
def test_the_anchor_is_the_preregistered_sixteen_pants_one_per_garment():
    a = MANIFEST["training"]["anchor"]
    recs = a["records"]
    assert len(recs) == 16 and len({r["garment"] for r in recs}) == 16
    assert all(r["garment"].startswith("Pant") for r in recs)
    assert sum(r["garment"].startswith("Pant_Short") for r in recs) == 8
    retention = set()
    for f in MANIFEST["training"]["retention"]["files"]:
        with np.load(f, allow_pickle=True) as z:
            retention.add(str(z["garment"]))
    final = {r["garment"] for r in MANIFEST["final_test"]}
    assert not {r["garment"] for r in recs} & (retention | final)
    assert (a["files"], a["frames"], MANIFEST["training"]["anchor_mode"]) == (16, 12, "whole_episode")


def test_the_trainers_glob_selects_exactly_the_pinned_anchor_files():
    a = MANIFEST["training"]["anchor"]
    selected = sorted(glob.glob(a["glob"]))[:a["files"]]
    assert [str(Path(p).resolve()) for p in selected] == sorted(r["target"] for r in a["records"])


def test_only_the_anchor_differs_from_v7s_training_block():
    v7 = json.loads((V7 / "manifest.json").read_text())["training"]
    mine = MANIFEST["training"]
    for k in ("lr", "batch_size", "grad_accum", "steps", "checkpoint_every", "rollout_fraction", "anchor_mode",
              "checkpoint_rule", "retention", "fit_precondition", "seed", "unfreeze", "anchor_fit_metric"):
        assert mine[k] == v7[k], k
    assert mine["anchor"]["glob"] != v7["anchor"]["glob"]


# ---------------------------------------------------------------- corpus
def test_the_corpus_is_v7s_byte_for_byte():
    prov = json.loads((V7 / "analysis/recovery-dataset-provenance.json").read_text())
    reuse = MANIFEST["corpus"]["reuse"]
    assert reuse["sha256"] == prov["dataset_sha256"]
    assert (ROOT / "datasets/recovery.npz").resolve() == (V7 / "datasets/recovery.npz").resolve()


# ------------------------------------------------------------- endpoints
def _beh(path, *, left_open_at=None, right_open_at=5, lift=0.0, lift_at=100):
    rows = []
    for a in range(1, 201):
        rows.append({"action": a,
                     "gripper_target_rad": {"left": 0.3 if left_open_at and a >= left_open_at else 0.0,
                                            "right": 0.3 if right_open_at and a >= right_open_at else 0.0},
                     "maximum_particle_lift_m": lift if a >= lift_at else 0.0})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_chunk1_opening_and_early_lift():
    rows = [{"action": a, "gripper_target_rad": {"left": 0.2 if a == 60 else 0.0, "right": 0.2 if a == 3 else 0.0},
             "maximum_particle_lift_m": 0.07 if a == 151 else 0.01} for a in range(1, 200)]
    assert ep.chunk1_opening(rows) == "right"           # left opens only at 60, outside chunk 1
    assert ep.early_lift(rows) == pytest.approx(0.01)   # the 0.07 lift is after action 150


def test_endpoint_prediction_needs_both_parts(tmp_path):
    for label, run in (("baseline", "r1"), ("cand", "r1")):
        for k in range(8):
            both = label == "baseline" or k < 5
            lift = 0.08 if k == 3 else 0.0
            _beh(tmp_path / f"evaluation/{label}-{run}/benchmark/dev0{k}_h50_G/rollout.json.behavior.jsonl",
                 left_open_at=5 if both else None, lift=lift)
    out = ep.compute(tmp_path, {"baseline": "baseline", "candidate": "cand"}, runs=("r1",))
    assert out["candidate-r1"]["both_open_rows"] == 5 and out["candidate-r1"]["dev03_early_lift_m"] == 0.08
    assert out["anchor_hypothesis_supported"]
    _beh(tmp_path / "evaluation/cand-r1/benchmark/dev03_h50_G/rollout.json.behavior.jsonl", left_open_at=5, lift=0.02)
    assert not ep.compute(tmp_path, {"baseline": "baseline", "candidate": "cand"}, runs=("r1",))["anchor_hypothesis_supported"]


# --------------------------------------------------- confirmatory gate
def test_confirm_rows_are_the_final_poses_at_both_horizons_with_fresh_seeds():
    rows = MANIFEST["final_confirm"]
    final = MANIFEST["final_test"]
    assert len(rows) == 48 and len({r["id"] for r in rows}) == 48 and len({r["seed"] for r in rows}) == 48
    assert sorted({r["horizon"] for r in rows}) == [10, 50]
    for r in rows:
        src = next(f for f in final if f["id"] == r["source_row"])
        assert (r["garment"], r["match_pose"], r["asset_config"]) == (src["garment"], src["match_pose"], src["asset_config"])
    earlier = {r["seed"] for r in MANIFEST["benchmark"] + final}
    assert not {r["seed"] for r in rows} & earlier


def _run(s, v=24, rows=24):
    return {"settled": s, "valid": v, "rows": rows}


def test_confirmatory_verdict_applies_the_rule_at_24_vs_24():
    good = driver.matched_verdict({"h10": [_run(14)], "h50": [_run(9)]}, {"h10": [_run(6)], "h50": [_run(9)]},
                                  True, True, expected_rows=24)
    assert good["improved"]
    weak = driver.matched_verdict({"h10": [_run(9)], "h50": [_run(9)]}, {"h10": [_run(6)], "h50": [_run(9)]},
                                  True, True, expected_rows=24)
    assert not weak["improved"]


@pytest.mark.parametrize("phase,index", [("final_confirm", 0), ("final_confirm", 95), ("benchmark", 1)])
def test_matched_runner_resolves_the_confirm_phase(tmp_path, phase, index):
    root = tmp_path / "camp"
    (root / "audit").mkdir(parents=True)
    (root / "scripts").symlink_to(ROOT / "scripts")
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    ck = V7 / "training/dq1/checkpoints/step_000200"
    rec = json.loads((V7 / "audit/candidate_checkpoint.json").read_text())
    (root / "audit/candidate_checkpoint.json").write_text(json.dumps(dict(rec, label="an1-step000200")))
    proc = subprocess.run([PY, str(root / "scripts/run_matched_task.py"), "--campaign", str(root), "--phase", phase,
                           "--matched-index", str(index), "--run", "c1", "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    req = json.loads(proc.stdout)
    assert req["row"] == MANIFEST[phase][index // 2]
    assert req["policy"] == ("baseline" if index % 2 == 0 else "candidate")


# ----------------------------------------------------------------- driver
def _temp(tmp_path):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "datasets").mkdir()
    (root / "datasets/recovery.npz").symlink_to(V7 / "datasets/recovery.npz")
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    return root


def test_first_tick_verifies_the_corpus_then_trains(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.tick()
    assert d.state["corpus"]["passed"] and set(d.state["stages"]) == {"an1.train"}


def test_a_mismatched_corpus_stops_before_training(tmp_path):
    root = _temp(tmp_path)
    m = json.loads((root / "manifest.json").read_text())
    m["corpus"]["reuse"]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(m))
    d = driver.Driver(root, dry=True)
    d.tick()
    assert d.state["done"] and "does not match" in d.state["stop_reason"] and not d.state["stages"]


def test_budget_funds_screen_retries_and_the_confirmatory_gate():
    b = MANIFEST["budget"]
    assert 6 + 64 + 64 + 96 + 96 <= b["gpu_tasks"] and b["reserve"]["final_set"] == 96


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run([PY, str(ROOT / "scripts/verify_sources.py"), "--campaign", str(ROOT), "--skip-checkpoint"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
