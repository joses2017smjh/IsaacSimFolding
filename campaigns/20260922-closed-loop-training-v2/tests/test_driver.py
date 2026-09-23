"""CPU tests for the campaign driver's decisions.

The driver submits GPU work unattended, so every rule it applies -- which
checkpoint to evaluate, what counts as a policy failure, what makes a
candidate ineligible, what the fallback next iteration is, and that a job is
never submitted twice -- is pinned here, where it costs nothing to be wrong.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("driver", ROOT / "scripts/driver.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


# ------------------------------------------------------------- Slurm states
def test_summarize_waits_on_any_live_task():
    assert driver.summarize(["COMPLETED", "RUNNING"]) == "active"
    assert driver.summarize(["PENDING"]) == "active"
    assert driver.summarize(["COMPLETED"] * 8) == "completed"
    assert driver.summarize(["COMPLETED", "FAILED"]) == "failed"
    assert driver.summarize(["CANCELLED by 12345"]) == "failed"
    assert driver.summarize([]) == "unknown"


# ------------------------------------------------------- checkpoint selection
def _ck(step, gate, loss=0.1):
    return {"step": step, "path": f"/c/step_{step:06d}", "heldout_gate": gate, "heldout_loss": loss}


def test_final_checkpoint_is_selected_when_it_passes_the_guard():
    chosen, _ = driver.select_checkpoint({"checkpoints": [_ck(100, True), _ck(200, True), _ck(300, True)]})
    assert chosen["step"] == 300


def test_fallback_to_step_200_only_on_a_final_breach():
    chosen, why = driver.select_checkpoint({"checkpoints": [_ck(100, True), _ck(200, True), _ck(300, False)]})
    assert chosen["step"] == 200 and "fallback" in why


def test_double_breach_selects_nothing_rather_than_step_100():
    """Amendment A1: a double breach is a diagnosis, not a hunt for step 100."""
    chosen, why = driver.select_checkpoint({"checkpoints": [_ck(100, True), _ck(200, False), _ck(300, False)]})
    assert chosen is None and "double breach" in why


# ------------------------------------------------------------------ scoring
ROWS = ([{"id": f"a{i}", "horizon": 10} for i in range(8)]
        + [{"id": f"b{i}", "horizon": 50} for i in range(8)])


def _res(settled, ever=None, cond=3):
    return {"terminal_success": settled, "ever_success": settled if ever is None else ever,
            "conditions_passed": cond}


def test_infrastructure_failure_is_not_counted_as_a_policy_failure():
    results = {r["id"]: _res(True, cond=4) for r in ROWS}
    results["a0"] = None                                  # infrastructure-invalid
    s = driver.score_rows(ROWS, results)
    assert s["h10"]["valid"] == 7 and s["h10"]["settled"] == 7
    assert s["h10"]["invalid"] == ["a0"]


def test_settled_and_latched_success_are_reported_separately():
    """The checker latches 'ever'; the target is stated on settled terminal."""
    results = {r["id"]: _res(False, ever=True, cond=3) for r in ROWS}
    s = driver.score_rows(ROWS, results)
    assert s["h10"]["settled"] == 0 and s["h10"]["ever"] == 8


# -------------------------------------------------------------- eligibility
BASE = {"h10": {"settled": 0, "valid": 8, "rows": 8}, "h50": {"settled": 4, "valid": 8, "rows": 8}}


def _cand(h10, h50, gate=True, reload_ok=True, valid=8, cond=3.0, loss=0.08):
    return {"heldout_gate": gate, "reload_ok": reload_ok, "heldout_loss": loss,
            "dev": {"h10": {"settled": h10, "valid": valid, "rows": 8, "mean_conditions": cond},
                    "h50": {"settled": h50, "valid": 8, "rows": 8, "mean_conditions": cond}}}


def test_h50_regression_makes_a_candidate_ineligible():
    reasons = driver.eligibility(_cand(6, 3), BASE)
    assert any("H50 regression" in r for r in reasons)


def test_guard_breach_and_bad_reload_are_disqualifying():
    assert any("guard" in r for r in driver.eligibility(_cand(6, 4, gate=False), BASE))
    assert any("reload" in r for r in driver.eligibility(_cand(6, 4, reload_ok=False), BASE))


def test_missing_rows_block_eligibility_instead_of_flattering_the_rate():
    assert any("infrastructure-invalid" in r for r in driver.eligibility(_cand(6, 4, valid=7), BASE))


def test_eligible_candidate_has_no_reasons():
    assert driver.eligibility(_cand(6, 4), BASE) == []


def test_ranking_is_h10_first_then_h50_then_conditions_then_loss():
    assert driver.rank_key(_cand(3, 4)) > driver.rank_key(_cand(2, 8))
    assert driver.rank_key(_cand(3, 5)) > driver.rank_key(_cand(3, 4))
    assert driver.rank_key(_cand(3, 4, cond=3.5)) > driver.rank_key(_cand(3, 4, cond=3.0))
    assert driver.rank_key(_cand(3, 4, loss=0.07)) > driver.rank_key(_cand(3, 4, loss=0.09))


# ------------------------------------------------------------- default ladder
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
RESERVED = ({r["seed"] for r in MANIFEST["benchmark"]} | {r["seed"] for r in MANIFEST["frozen_test"]}
            | {r["seed"] for r in MANIFEST["collection"]} | {r["seed"] for r in MANIFEST["collection_expansion"]})


@pytest.mark.parametrize("k", [2, 3])
def test_default_plan_seeds_never_touch_a_preregistered_block(k):
    plan = driver.default_plan(k, MANIFEST, None, [])
    seeds = [r["seed"] for r in plan["collection"] + plan["collection_expansion"]]
    assert not set(seeds) & RESERVED
    assert len(seeds) == len(set(seeds))
    ids = [r["id"] for r in plan["collection"] + plan["collection_expansion"]]
    assert len(ids) == len(set(ids))


def test_default_plan_never_collects_on_frozen_test_garments():
    test_garments = {r["garment"] for r in MANIFEST["frozen_test"]}
    for k in (2, 3):
        plan = driver.default_plan(k, MANIFEST, None, [])
        assert not {r["garment"] for r in plan["collection"]} & test_garments


def test_default_ladder_changes_exactly_one_named_factor_per_iteration():
    two = driver.default_plan(2, MANIFEST, None, [])
    three = driver.default_plan(3, MANIFEST, None, [])
    assert two["factor_changed"] == "collection_coverage" and len(two["collection"]) == 16
    assert three["factor_changed"] == "replay_balance"
    assert three["training"]["rollout_fraction"] != two["training"]["rollout_fraction"]
    # Coverage is held at 16 in iteration 3, so only the replay balance moved.
    assert len(three["collection"]) == len(two["collection"])


def test_default_plan_continues_on_policy_from_an_eligible_improvement():
    best = {"checkpoint": "/c/iter1", "label": "iter1-step000300"}
    plan = driver.default_plan(2, MANIFEST, best, [])
    assert plan["factor_changed"] == "collection_policy"
    assert plan["collection_checkpoint"] == plan["init_checkpoint"] == "/c/iter1"


# ------------------------------------------------ never submit a job twice
def _temp_campaign(tmp_path, jobs):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "amendments").mkdir()
    shutil.copy(ROOT / "manifest.json", root / "manifest.json")
    shutil.copy(ROOT / "amendments/A1-autonomous-mandate.json", root / "amendments/")
    (root / "ledger/slurm-jobs.json").write_text(json.dumps({"jobs": jobs}))
    return root


def test_pre_driver_jobs_are_adopted_not_resubmitted(tmp_path):
    root = _temp_campaign(tmp_path, [
        {"job_id": "111", "phase": "collect:iteration1"},
        {"job_id": "222", "phase": "compile:iteration1"},
        {"job_id": "333", "phase": "train:iteration1"}])
    d = driver.Driver(root, dry=True)
    assert d.state["stages"]["iter1.collect"]["job_ids"] == ["111"]
    assert d.state["stages"]["iter1.train"]["job_ids"] == ["333"]
    # submit() refuses any key that already holds a job.
    assert d.submit("iter1.collect", "collect.sbatch", [], gpu_tasks=8) is None


def test_budget_refuses_work_that_would_eat_the_final_reserve(tmp_path):
    root = _temp_campaign(tmp_path, [{"job_id": "1", "phase": "x", "gpu_tasks": 150}])
    d = driver.Driver(root, dry=True)
    cap = d.caps["max_gpu_tasks_total"]
    reserve = d.caps["reserved_for_final_evaluation_gpu_tasks"]
    assert not d.can_spend(cap - 150 - reserve + 1)
    assert d.can_spend(cap - 150, final=True)


def test_lock_prevents_two_concurrent_ticks(tmp_path):
    path = tmp_path / ".lock"
    with driver.Lock(path):
        with pytest.raises(SystemExit):
            with driver.Lock(path):
                pass
    assert not path.exists()
