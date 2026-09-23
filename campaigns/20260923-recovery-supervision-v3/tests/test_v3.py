"""CPU tests for v3: the recovery coverage gate, the preregistered improvement
rule, the manifest's separation guarantees, and the driver's refusal to
invent a second attempt.

Each rule here decides whether GPU time is spent or a claim is made, so each
is pinned where being wrong costs nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brd = _load("build_recovery_dataset")
driver = _load("driver")
GATE = MANIFEST["recovery"]["coverage_gate"]


def _att(root, ok, full_entries=0, state="completed", horizon=10):
    return {"root": root, "settled_success": ok, "state": state, "horizon": horizon, "full_entries": full_entries}


# ------------------------------------------------------------ coverage gate
def test_two_lucky_branches_are_not_a_teacher():
    """A couple of successes from one root must not pass as coverage."""
    g = brd.coverage({"r1": [_att(50, True), _att(50, True)] + [_att(150, False)] * 14}, GATE)
    assert not g["passed"]
    assert set(g["unmet_requirements"]) >= {"successful_branches", "distinct_successful_roots"}


def test_spread_successes_pass_and_failures_are_counted():
    attempts = {f"r{i}": [_att(50, True), _att(150, True), _att(250, False), _att(350, False)] for i in range(3)}
    g = brd.coverage(attempts, GATE)
    assert g["passed"] and g["successful_branches"] == 6
    assert g["attempts_completed"] == 12 and g["success_rate"] == pytest.approx(0.5)


def test_transient_crossings_are_distinguished_from_settled_success():
    g = brd.coverage({"r": [_att(50, False, full_entries=2), _att(150, True, full_entries=1)]}, GATE)
    assert g["transient_full_crossings_without_settled_success"] == 1


def test_errors_and_unrun_attempts_are_reported_not_hidden():
    g = brd.coverage({"r": [_att(50, True), {"root": 150, "state": "error"}, {"root": 250, "state": "not_run"}]}, GATE)
    assert g["attempts_total"] == 3 and g["attempts_not_run_or_error"] == 2


# ------------------------------------------------------- improvement rule
def _run(s, v=8):
    return {"settled": s, "valid": v, "rows": 8}


def test_fisher_separates_a_real_gain_from_noise():
    assert driver.fisher_one_sided(10, 16, 2, 16) < 0.01
    assert driver.fisher_one_sided(4, 16, 2, 16) > 0.2


def test_one_favourable_run_cannot_carry_the_claim():
    """6/8 then 1/8 must not meet the target however it pools."""
    v = driver.improvement_verdict([_run(6), _run(1)], [_run(2), _run(0)], _run(4), 4, True, True)
    assert not v["target_6_of_8_both_runs"]


def test_a_real_repeated_gain_passes_every_check():
    v = driver.improvement_verdict([_run(6), _run(7)], [_run(2), _run(1)], _run(4), 4, True, True)
    assert v["improved"] and v["target_6_of_8_both_runs"] and not v["stretch_8_of_8_both_runs"]


def test_h50_regression_guard_and_reload_each_block_the_claim():
    good = ([_run(6), _run(7)], [_run(2), _run(1)])
    assert not driver.improvement_verdict(*good, _run(3), 4, True, True)["improved"]
    assert not driver.improvement_verdict(*good, _run(4), 4, False, True)["improved"]
    assert not driver.improvement_verdict(*good, _run(4), 4, True, False)["improved"]
    assert not driver.improvement_verdict([_run(6), _run(7, 7)], good[1], _run(4), 4, True, True)["improved"]


# ------------------------------------------------------------- separation
def test_search_final_development_and_retention_are_separated():
    search = {r["garment"] for r in MANIFEST["recovery_search"]}
    final = {r["garment"] for r in MANIFEST["final_test"]}
    dev = {r["garment"] for r in MANIFEST["benchmark"]}
    assert not search & final and not search & dev and not final & dev
    anchor_garments = {"Top_Short_Seen_0", "Top_Short_Seen_1", "Top_Long_Seen_0", "Top_Long_Seen_1",
                       "Pant_Short_Seen_0", "Pant_Long_Seen_0"}
    assert not final & anchor_garments and not search & anchor_garments


def test_final_set_is_untouched_by_any_earlier_evaluation():
    v2 = json.loads((ROOT.parent / "20260922-closed-loop-training-v2/manifest.json").read_text())
    earlier = {(r["garment"], int(r["pose_key"])) for k in ("benchmark", "frozen_test") for r in v2[k]}
    assert not {(r["garment"], r["pose_key"]) for r in MANIFEST["final_test"]} & earlier


def test_candidate_set_and_roots_are_fixed_before_search():
    rec = MANIFEST["recovery"]
    assert len(rec["candidates"]) == 4 and all(c["horizon"] in (10, 50) for c in rec["candidates"])
    assert all(0 < r < 600 and r % 10 == 0 for r in rec["roots"])


# ---------------------------------------------------------------- driver
def _temp(tmp_path):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "plans").mkdir()
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    return root


def test_no_second_attempt_without_a_committed_evidence_plan(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    assert d.attempt_plan(2) is None           # no default ladder in v3
    assert d.attempt_plan(1)["dataset"] == "recovery"


def test_the_final_set_reserve_is_protected(tmp_path):
    root = _temp(tmp_path)
    cap = int(MANIFEST["budget"]["gpu_tasks"])
    (root / "ledger/slurm-jobs.json").write_text(
        json.dumps({"jobs": [{"job_id": "1", "gpu_tasks": cap - 20}]}))
    d = driver.Driver(root, dry=True)
    assert not d.can_spend(8, reserve=16) and d.can_spend(4, reserve=16)


def test_attempt2_success_path_is_fundable_under_the_amended_cap():
    """The original 65-task cap could never fund the preregistered H50
    confirmation leg (69 minimum at zero overhead) -- the design flaw the
    dated amendment repairs. 29 used + 27 attempt-2 stages + 16 final <= cap."""
    cap = int(MANIFEST["budget"]["gpu_tasks"])
    assert 29 + (1 + 1 + 1 + 8 + 8 + 8) + 16 <= cap


def test_plan_deadline_is_manifest_declared():
    assert int(MANIFEST["automation"]["attempt2_plan_deadline_minutes"]) >= 120


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/verify_sources.py"),
                           "--campaign", str(ROOT), "--skip-checkpoint"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ------------------------------------------------------ retention guard
def test_retention_refuses_an_anchor_garment(tmp_path):
    trainer = _load("rollout_weighted_finetune")
    ep = tmp_path / "ep.npz"
    np.savez(ep, images=np.zeros((3, 3, 480, 640, 3), np.uint8), state=np.zeros((3, 12), np.float32),
             action=np.zeros((3, 50, 12), np.float32), garment=np.asarray("Pant_Long_Seen_0"))
    with pytest.raises(ValueError, match="shares garment"):
        trainer.load_whole_episodes([ep], 2, anchor=[{"garment": "Pant_Long_Seen_0", "sha256": "x"}])


def test_retention_samples_the_whole_episode_not_its_start(tmp_path):
    trainer = _load("rollout_weighted_finetune")
    ep = tmp_path / "ep.npz"
    np.savez(ep, images=np.zeros((5, 3, 480, 640, 3), np.uint8), state=np.zeros((5, 12), np.float32),
             action=np.zeros((5, 50, 12), np.float32), garment=np.asarray("Top_Long_Seen_4"))
    _, _, _, rec = trainer.load_whole_episodes([ep], 3, anchor=[])
    assert rec[0]["frames_used"] == [0, 2, 4]


@pytest.mark.parametrize("phase,index", [("recovery", 0), ("recovery", 7), ("recovery_smoke", 0),
                                         ("benchmark", 0), ("final_test", 7)])
def test_every_v3_phase_resolves_its_rows(phase, index):
    """The first dry run found `recovery` reading the search config as rows."""
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_rollout_task.py"), "--campaign", str(ROOT),
                           "--phase", phase, "--index", str(index), "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-500:]
    request = json.loads(proc.stdout)
    assert request["runner"]["path"].endswith("20260923-recovery-supervision-v3/scripts/render/policy_rollout51.py")
    if phase.startswith("recovery"):
        assert "--recovery_out" in request["command"] and "--trajectory_out" in request["command"]


# ------------------------------------------------- attempt-2 mechanics
def _ck(step, gate):
    return {"step": step, "path": f"/c/step_{step:06d}", "heldout_gate": gate, "heldout_loss": 0.08}


def test_latest_guard_passing_rule_takes_the_newest_passing_checkpoint():
    chosen, why = driver.latest_guard_passing({"checkpoints": [_ck(100, True), _ck(200, True), _ck(300, False)]})
    assert chosen["step"] == 200 and "latest guard-passing" in why
    chosen, _ = driver.latest_guard_passing({"checkpoints": [_ck(100, True), _ck(600, True)]})
    assert chosen["step"] == 600


def test_latest_guard_passing_selects_nothing_when_all_breach():
    chosen, why = driver.latest_guard_passing({"checkpoints": [_ck(100, False), _ck(200, False)]})
    assert chosen is None and "no checkpoint passes" in why


def test_grad_accum_flag_validates_and_defaults_to_attempt1_behaviour():
    trainer_src = (ROOT / "scripts/rollout_weighted_finetune.py").read_text()
    assert '"--grad-accum", type=int, default=1' in trainer_src
    assert "batch_size * args.grad_accum" in trainer_src   # effective batch recorded
