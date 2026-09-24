"""CPU tests for v5: the interleaving that makes a run matched, the pinned
checkpoints, the verbatim rows, the preregistered verdict (including the
one-episode H50 case that decided v4), the settled-only gallery, and the
driver's first tick.

Each rule here decides whether GPU time is spent or a claim is made, so each
is pinned where being wrong costs nothing.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
V4 = ROOT.parent / "20260923-recovery-supervision-v4"


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rrt = _load("run_rollout_task")
driver = _load("driver")
gallery = _load("build_gallery")


# ----------------------------------------------------------- interleaving
def test_matched_index_interleaves_the_two_policies_on_the_same_row():
    for i in range(64):
        which, row = rrt.matched_index(i)
        assert which == ("baseline" if i % 2 == 0 else "candidate") and row == i // 2
    with pytest.raises(ValueError):
        rrt.matched_index(-1)


def test_sbatch_and_driver_share_the_single_mapping():
    src = (ROOT / "slurm/matched.sbatch").read_text()
    assert "--matched-index" in src and "SLURM_ARRAY_TASK_ID" in src
    assert "%8" in (ROOT / "scripts/driver.py").read_text(), "throttle keeps both policies in every batch"


def test_smoke_destination_separates_the_two_policies():
    row = {"id": "r"}
    a = rrt.destination(ROOT, "smoke", row, "baseline-s1")
    b = rrt.destination(ROOT, "smoke", row, "a2-step000250-s1")
    assert a != b and a.parent.parent == b.parent.parent == ROOT / "smoke"


# ------------------------------------------------------------- checkpoints
def test_both_checkpoints_are_pinned_and_the_candidate_matches_v4s_record():
    cand = MANIFEST["candidate_checkpoint"]
    record = json.loads((Path(cand["path"]) / "checkpoint.json").read_text())
    for name, sha in record["checkpoint_sha256"].items():
        assert cand["sha256"][name] == sha
    assert cand["immutable"] and MANIFEST["baseline_checkpoint"]["immutable"]
    v4 = json.loads((V4 / "manifest.json").read_text())
    assert MANIFEST["baseline_checkpoint"]["sha256"] == v4["baseline_checkpoint"]["sha256"]


def test_candidate_provenance_carries_the_two_recorded_clauses():
    prov = MANIFEST["candidate_checkpoint"]["provenance"]
    assert prov["retention_guard"]["passed"] is True and prov["reload"]["passed"] is True
    for key in ("retention_guard", "reload"):
        assert Path(prov[key]["record"]).is_file() and len(prov[key]["sha256"]) == 64


# -------------------------------------------------------------------- rows
def test_rows_are_v2_development_and_v3_final_verbatim():
    v2 = json.loads((ROOT.parent / "20260922-closed-loop-training-v2/manifest.json").read_text())
    v3 = json.loads((ROOT.parent / "20260923-recovery-supervision-v3/manifest.json").read_text())
    assert MANIFEST["benchmark"] == v2["benchmark"] and MANIFEST["smoke"] == v2["smoke"]
    assert MANIFEST["final_test"] == v3["final_test"]


def test_final_set_is_untouched_by_any_earlier_evaluation():
    v2 = json.loads((ROOT.parent / "20260922-closed-loop-training-v2/manifest.json").read_text())
    earlier = {(r["garment"], int(r["pose_key"])) for k in ("benchmark", "frozen_test") for r in v2[k]}
    assert not {(r["garment"], r["pose_key"]) for r in MANIFEST["final_test"]} & earlier


def test_n_is_fixed_at_two_runs_per_policy_per_horizon():
    p = MANIFEST["protocol"]
    assert p["runs_per_policy_per_horizon"] == 2 and p["run_labels"] == ["r1", "r2"]
    assert "no third run" in p["n_is_fixed"]
    assert len(driver.RUNS) == 2


# ----------------------------------------------------------------- verdict
def _run(settled, valid=8, rows=8):
    return {"settled": settled, "valid": valid, "rows": rows}


def _side(h10, h50):
    return {"h10": [_run(s) for s in h10], "h50": [_run(s) for s in h50]}


def test_v4s_one_episode_h50_shortfall_is_not_an_improvement():
    v = driver.matched_verdict(_side((4, 4), (3, 4)), _side((2, 0), (4, 4)), True, True)
    assert v["checks"]["pooled_h10_margin_ge_4"] and v["checks"]["fisher_one_sided_p_lt_0.05"]
    assert not v["checks"]["pooled_h50_not_below_baseline"] and not v["improved"]
    assert v["h50_margin"] == -1


def test_equal_pooled_h50_is_non_regression_and_passes():
    v = driver.matched_verdict(_side((4, 4), (4, 4)), _side((2, 0), (4, 4)), True, True)
    assert v["improved"] and v["p_value"] < 0.05


def test_h10_needs_both_margin_and_significance():
    assert not driver.matched_verdict(_side((3, 2), (4, 4)), _side((2, 0), (3, 3)), True, True)["improved"]  # margin 3
    assert not driver.matched_verdict(_side((3, 3), (4, 4)), _side((2, 0), (3, 3)), True, True)["improved"]  # p = 0.11
    assert not driver.matched_verdict(_side((4, 3), (4, 4)), _side((2, 0), (3, 3)), True, True)["improved"]   # 7 vs 2: p = 0.057
    assert driver.matched_verdict(_side((4, 4), (4, 4)), _side((2, 0), (3, 3)), True, True)["improved"]       # 8 vs 2: p = 0.027


def test_guard_reload_and_invalid_rows_each_block_the_claim():
    good = (_side((5, 5), (4, 4)), _side((1, 1), (3, 3)))
    assert driver.matched_verdict(*good, True, True)["improved"]
    assert not driver.matched_verdict(*good, False, True)["improved"]
    assert not driver.matched_verdict(*good, True, False)["improved"]
    short = _side((5, 5), (4, 4))
    short["h50"][1] = _run(4, valid=7)
    assert not driver.matched_verdict(short, good[1], True, True)["checks"]["all_rows_valid"]


def test_fisher_matches_known_values():
    assert driver.fisher_one_sided(8, 16, 2, 16) == pytest.approx(0.0269, abs=1e-4)
    assert driver.fisher_one_sided(7, 16, 2, 16) == pytest.approx(0.0567, abs=1e-4)


# ----------------------------------------------------------------- gallery
def _episode(root, label, row, *, settled, ever, conditions, policy="candidate"):
    d = root / "evaluation" / label / "benchmark" / row
    d.mkdir(parents=True)
    tag = "success" if ever else "failure"
    (d / "rollout.json").write_text(json.dumps({
        "terminal_success": settled, "success": ever, "garment": "G", "seed": 1,
        "effective_n_action_steps": 10,
        "terminal_checker": {"conditions_passed": conditions, "conditions_total": 4, "success": settled}}))
    (d / "status.json").write_text(json.dumps({"state": "completed", "slurm_job_id": "9", "slurm_array_task_id": "1"}))
    (d / "request.json").write_text(json.dumps({"policy": policy, "run": "r1", "checkpoint": {"sha256": {}}}))
    for name in (f"rollout_policy_{tag}_top.gif", f"rollout_policy_{tag}_left_wrist.gif",
                 f"rollout_policy_{tag}_final_triptych.png", f"rollout_policy_{tag}.mp4"):
        (d / name).write_bytes(b"GIF89a" + name.encode())


def test_gallery_shows_settled_successes_only_and_counts_the_rest(tmp_path):
    root = tmp_path / "c"
    _episode(root, "a2-step000250-r1", "dev00", settled=True, ever=True, conditions=4)
    _episode(root, "a2-step000250-r1", "dev01", settled=False, ever=True, conditions=3)   # latched, came undone
    _episode(root, "a2-step000250-r1", "dev02", settled=False, ever=False, conditions=2)
    _episode(root, "baseline-r1", "dev00", settled=False, ever=False, conditions=2, policy="baseline")
    summary = gallery.build(root)
    assert summary["valid"] == 4 and summary["settled"] == 1 and summary["ever"] == 2
    index = json.loads((root / "media/INDEX.json").read_text())
    copied = [f for e in index["entries"] for f in e["files"].values() if f["copied_to"]]
    assert len(copied) == 2 and all(Path(f["copied_to"]).is_file() for f in copied)
    assert all(gallery.sha256(Path(f["copied_to"])) == f["sha256"] for f in copied)
    page = (root / "gallery.html").read_text()
    assert page.count("<article>") == 1 and "dev01" not in page.split("<article>")[1]
    assert "1/3" in page and "2/3" in page      # settled / ever for the candidate group


# ------------------------------------------------------------ dry runs
@pytest.mark.parametrize("phase,index", [("smoke", 0), ("smoke", 1), ("benchmark", 0),
                                         ("benchmark", 31), ("final_test", 15)])
def test_dry_run_resolves_both_policies_from_the_matched_index(phase, index):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_rollout_task.py"), "--campaign", str(ROOT),
                           "--phase", phase, "--matched-index", str(index), "--run", "t", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    request = json.loads(proc.stdout)
    which = "baseline" if index % 2 == 0 else "candidate"
    block = MANIFEST[f"{which}_checkpoint"]
    assert request["policy"] == which and request["checkpoint"]["path"] == block["path"]
    assert request["row"] == MANIFEST[phase][index // 2]
    assert request["runner"]["path"].endswith("20260924-matched-comparison-v5/scripts/render/policy_rollout51.py")
    cmd = request["command"]
    assert cmd[cmd.index("--gif_every") + 1] == str(MANIFEST["media"]["gif_every"])
    assert cmd[cmd.index("--policy_path") + 1] == block["path"]


def test_dry_run_refuses_an_index_past_the_rows():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/run_rollout_task.py"), "--campaign", str(ROOT),
                           "--phase", "final_test", "--matched-index", "16", "--run", "t", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "out of range" in proc.stderr


# ----------------------------------------------------------------- driver
def _temp(tmp_path):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    return root


def test_first_tick_submits_the_smoke_and_nothing_else(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.tick()
    assert set(d.state["stages"]) == {"smoke"}
    assert d.state["stages"]["smoke"]["gpu_tasks"] == 2 and d.state["stages"]["smoke"]["array"] == "0-1"


def test_development_runs_are_one_array_of_32_each_and_final_needs_the_verdict(tmp_path):
    root = _temp(tmp_path)
    d = driver.Driver(root, dry=True)
    d.state["stages"]["smoke"] = {"job_ids": ["1"], "status": "completed"}
    # A completed smoke with results present lets the tick reach the runs.
    for i in range(2):
        dest = d.dest("smoke", i, "s1", MANIFEST["smoke"])
        dest.mkdir(parents=True)
        (dest / "status.json").write_text(json.dumps({"state": "completed"}))
        (dest / "rollout.json").write_text(json.dumps({"terminal_success": False, "success": False,
                                                        "terminal_checker": {}, "effective_n_action_steps": 10}))
    d.tick()
    assert {"run1", "run2"} <= set(d.state["stages"]) and "final" not in d.state["stages"]
    for key in ("run1", "run2"):
        assert d.state["stages"][key]["gpu_tasks"] == 32 and d.state["stages"][key]["array"] == "0-31"


def test_the_final_set_reserve_is_protected(tmp_path):
    root = _temp(tmp_path)
    cap = int(MANIFEST["budget"]["gpu_tasks"])
    (root / "ledger/slurm-jobs.json").write_text(json.dumps({"jobs": [{"job_id": "1", "gpu_tasks": cap - 20}]}))
    d = driver.Driver(root, dry=True)
    assert not d.can_spend(8, reserve=16) and d.can_spend(4, reserve=16)


def test_budget_funds_the_whole_path():
    b = MANIFEST["budget"]
    assert 2 + 64 + 16 + 8 <= b["gpu_tasks"] and b["reserve"]["final_set"] == 16


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/verify_sources.py"),
                           "--campaign", str(ROOT), "--skip-checkpoint"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
