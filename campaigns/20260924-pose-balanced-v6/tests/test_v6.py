"""CPU tests for v6: pose identity, the targeted search rows, the per-pose
gate and pose-balanced compiler (end to end on a synthetic campaign), the
offline-fit provenance path (checked against v4's reconstruction on v4's real
corpus), the candidate record the matched runner verifies, the unchanged
recipe and rule, and the driver's first tick.

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
CAMPAIGNS = ROOT.parent
V4 = CAMPAIGNS / "20260923-recovery-supervision-v4"
V5 = CAMPAIGNS / "20260924-matched-comparison-v5"
PY = sys.executable


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bbd = _load("build_balanced_dataset")
driver = _load("driver")
rmt = _load("run_matched_task")


def pid(mp):
    return tuple(round(float(x), 4) for x in mp.split(":"))


# ------------------------------------------------------------ pose identity
def test_development_poses_are_single_match_poses():
    c = MANIFEST["pose_clusters"]
    dev = {r["id"][:5]: r for r in MANIFEST["benchmark"] if "_h10_" in r["id"]}
    for name, members in (("P_A", ("dev00", "dev04", "dev06")), ("P_B", ("dev01", "dev03")),
                          ("P_C", ("dev02", "dev05", "dev07"))):
        assert {pid(dev[m]["match_pose"]) for m in members} == {pid(c[name])}
    # pose keys do NOT identify poses across garments (the v5 report error)
    assert dev["dev02"]["pose_key"] == 0 and dev["dev00"]["pose_key"] == 0
    assert pid(dev["dev02"]["match_pose"]) != pid(dev["dev00"]["match_pose"])


def test_search_rows_are_exact_pose_training_rows_with_fresh_seeds():
    rows = MANIFEST["recovery_search"]
    c = MANIFEST["pose_clusters"]
    assert sum(r["pose"] == "P_B" for r in rows) == 8 and sum(r["pose"] == "P_A" for r in rows) == 8
    for r in rows:
        assert pid(r["match_pose"]) == pid(c[r["pose"]]), r["id"]
    assert {(r["garment"], r["pose_key"]) for r in rows if r["pose"] == "P_B"} == \
        {("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)}
    evaluated = {r["garment"] for r in MANIFEST["benchmark"] + MANIFEST["final_test"]}
    assert not {r["garment"] for r in rows} & evaluated
    v4 = json.loads((V4 / "manifest.json").read_text())
    earlier = {r["seed"] for r in v4["recovery_search"] + v4["benchmark"] + v4["final_test"]}
    assert len({r["seed"] for r in rows}) == 16 and not {r["seed"] for r in rows} & earlier


def test_p_b_rows_search_early_roots_and_p_a_rows_keep_v4s():
    for r in MANIFEST["recovery_search"]:
        if r["pose"] == "P_B":
            assert r["recovery_roots"] == [30, 60, 90, 120, 150, 180]
        else:
            assert "recovery_roots" not in r
    assert MANIFEST["recovery"]["roots"] == [60, 120, 180, 240, 300, 360]


@pytest.mark.parametrize("index", [0, 15])
def test_search_dry_run_passes_the_row_roots(index):
    proc = subprocess.run([PY, str(ROOT / "scripts/run_rollout_task.py"), "--campaign", str(ROOT),
                           "--phase", "recovery", "--index", str(index), "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-600:]
    cmd = json.loads(proc.stdout)["command"]
    row = MANIFEST["recovery_search"][index]
    roots = row.get("recovery_roots") or MANIFEST["recovery"]["roots"]
    assert cmd[cmd.index("--recovery_roots") + 1] == ",".join(map(str, roots))
    assert cmd[cmd.index("--policy_path") + 1] == MANIFEST["baseline_checkpoint"]["path"]


# ------------------------------------------------------- balance and gate
def test_balanced_weights_give_every_pose_an_equal_share_and_mean_one():
    poses = np.array(["P_C"] * 50 + ["P_A"] * 30 + ["P_B"] * 20)
    w, info = bbd.balanced_weights(poses)
    assert w.mean() == pytest.approx(1.0)
    for p in ("P_A", "P_B", "P_C"):
        assert info["draw_share"][p] == pytest.approx(1 / 3)
    assert info["multiplier_vs_uniform"]["P_B"] == pytest.approx(100 / 60)


def test_pose_of_is_exact_identity_and_refuses_anything_else():
    c = MANIFEST["pose_clusters"]
    assert bbd.pose_of(c["P_B"], c) == "P_B"
    with pytest.raises(SystemExit):
        bbd.pose_of("0.027:0.010:0.67:0.11:4.0:90", c)    # near-P_B, 5.4 cm away: not P_B


def _att(root, ok):
    return {"root": root, "settled_success": ok, "state": "completed", "horizon": 10, "full_entries": int(ok)}


def test_pose_gate_fails_a_thin_pose_even_when_the_other_is_rich():
    gates = MANIFEST["recovery"]["coverage_gate"]
    rich = {f"a{i}": [_att(r, True) for r in (60, 120, 180, 240)] for i in range(8)}
    thin = {f"b{i}": [_att(30, i < 2)] + [_att(60, False)] * 5 for i in range(8)}
    row_pose = {**{k: "P_A" for k in rich}, **{k: "P_B" for k in thin}}
    g = bbd.pose_gate({**rich, **thin}, row_pose, gates)
    assert not g["passed"] and g["per_pose"]["P_A"]["passed"]
    assert any(u.startswith("P_B:") for u in g["unmet_requirements"])
    thick = {f"b{i}": [_att(r, True) for r in (30, 60, 90)] + [_att(120, False)] for i in range(8)}
    g = bbd.pose_gate({**rich, **thick}, row_pose, gates)
    assert g["passed"] and g["per_pose"]["P_B"]["successful_branches"] == 24


def _synthetic_campaign(tmp_path, *, p_b_wins):
    """A campaign with one P_B and one P_A search row plus a reused corpus."""
    c = MANIFEST["pose_clusters"]
    root = tmp_path / "camp"
    (root / "recovery").mkdir(parents=True)
    cams = np.asarray(("top_rgb", "left_rgb", "right_rgb"))
    rows = []
    for pose, n_ex in (("P_B", 2), ("P_A", 1)):
        rid = f"row_{pose}"
        d = root / "recovery" / rid
        d.mkdir()
        wins = p_b_wins if pose == "P_B" else 2
        (d / "recovery.json").write_text(json.dumps({"attempts": [_att(30 + 30 * k, k < wins) for k in range(3)]}))
        np.savez(d / "recovery_examples.npz", images=np.full((n_ex, 3, 480, 640, 3), 7, np.uint8),
                 state=np.zeros((n_ex, 12), np.float32), action=np.ones((n_ex, 50, 12), np.float32),
                 camera_keys=cams, **{k: np.arange(n_ex, dtype=np.int64) + 10 for k in bbd.PROV_COLS})
        np.savez(d / "trajectory.npz", terminal_success=np.asarray(False), step_index=np.arange(3),
                 executed_action_stream=np.zeros((600, 12), np.float32),
                 images=np.zeros((3, 3, 480, 640, 3), np.uint8), state=np.zeros((3, 12), np.float32))
        rows.append({"id": rid, "pose": pose, "match_pose": c[pose], "garment": "G", "pose_key": 0, "seed": 1})
    old_rows = [{"id": "v4row_c", "match_pose": c["P_C"]}]
    (tmp_path / "v4manifest.json").write_text(json.dumps({"recovery_search": old_rows}))
    old = tmp_path / "old.npz"
    np.savez(old, images=np.full((3, 3, 480, 640, 3), 3, np.uint8), state=np.zeros((3, 12), np.float32),
             action=np.zeros((3, 50, 12), np.float32), origin=np.asarray(["branch:v4row_c"] * 3),
             camera_keys=cams, **{k: np.full(3, -1, np.int64) for k in bbd.PROV_COLS})
    gate = {"min_successful_branches": 1, "min_distinct_roots": 1, "min_rows": 1}
    manifest = {"pose_clusters": c, "recovery_search": rows, "task_prompt": "fold",
                "recovery": {"coverage_gate": {"P_B": gate, "P_A": gate}},
                "corpus": {"reuse": {"path": str(old), "sha256": bbd.digest(old),
                                     "manifest": str(tmp_path / "v4manifest.json")}}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _compile(root):
    return subprocess.run([PY, str(ROOT / "scripts/build_balanced_dataset.py"), "--campaign", str(root),
                           "--out", str(root / "datasets/recovery.npz"),
                           "--provenance", str(root / "analysis/prov.json"),
                           "--gate-out", str(root / "audit/gate.json")], capture_output=True, text=True)


def test_compiler_merges_the_pinned_corpus_and_balances_poses(tmp_path):
    root = _synthetic_campaign(tmp_path, p_b_wins=1)
    proc = _compile(root)
    assert proc.returncode == 0, proc.stderr[-800:]
    with np.load(root / "datasets/recovery.npz", allow_pickle=False) as z:
        for member in bbd.TRAINER_MEMBERS + ("origin", "pose", "source") + bbd.PROV_COLS:
            assert member in z.files, member
        pose, w, src = z["pose"], z["awr_weight"], z["source"]
        assert len(pose) == 6 and list(src).count("v4") == 3
        assert int(z["images"][0, 0, 0, 0, 0]) == 3 and int(z["images"][-1, 0, 0, 0, 0]) == 7
        for p in ("P_A", "P_B", "P_C"):
            assert w[pose == p].sum() / w.sum() == pytest.approx(1 / 3)
    prov = json.loads((root / "analysis/prov.json").read_text())
    assert prov["samples_by_pose_and_source"] == {"P_A/v6": 1, "P_B/v6": 2, "P_C/v4": 3}


def test_compiler_fails_closed_before_reading_images_when_a_pose_is_thin(tmp_path):
    root = _synthetic_campaign(tmp_path, p_b_wins=0)
    proc = _compile(root)
    assert proc.returncode == 4
    assert not (root / "datasets/recovery.npz").exists()
    gate = json.loads((root / "audit/gate.json").read_text())
    assert not gate["passed"] and any(u.startswith("P_B:") for u in gate["unmet_requirements"])


def test_compiler_refuses_a_changed_reused_corpus(tmp_path):
    root = _synthetic_campaign(tmp_path, p_b_wins=1)
    m = json.loads((root / "manifest.json").read_text())
    m["corpus"]["reuse"]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(m))
    proc = _compile(root)
    assert proc.returncode != 0 and "changed" in proc.stderr


# ------------------------------------------------ offline fit, unchanged
class _Hide:
    """An npz view without the provenance columns, to force the old path."""
    def __init__(self, z):
        self.z = z
        self.files = [f for f in z.files if f not in ("horizon", "root", "step_index")]

    def __getitem__(self, k):
        return self.z[k]


def test_offline_fit_provenance_path_equals_v4s_reconstruction_on_v4s_corpus():
    of = _load("offline_fit")
    v4m = json.loads((V4 / "manifest.json").read_text())
    z = np.load(V4 / "datasets/recovery.npz", allow_pickle=False)
    n = len(z["origin"])
    h_new, r_new = of.example_classes(z, v4m, V4, n)
    h_old, r_old = of.example_classes(_Hide(z), v4m, V4, n)
    assert np.array_equal(h_new, h_old) and np.array_equal(r_new, r_old)


# ------------------------------------------------- the candidate record
def test_candidate_block_needs_the_driver_written_record(tmp_path):
    assert rmt.checkpoint_block(ROOT, MANIFEST, "baseline")["label"] == "baseline"
    (tmp_path / "audit").mkdir()
    with pytest.raises(SystemExit):
        rmt.checkpoint_block(tmp_path, MANIFEST, "candidate")


def _campaign_with_candidate(tmp_path, *, corrupt=False):
    root = tmp_path / "camp"
    (root / "audit").mkdir(parents=True)
    (root / "scripts").symlink_to(ROOT / "scripts")
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    ck = V4 / "training/a2/checkpoints/step_000250"          # any real checkpoint exercises the check
    record = {"path": str(ck), "label": "pb1-step000250",
              "sha256": json.loads((ck / "checkpoint.json").read_text())["checkpoint_sha256"]}
    if corrupt:
        record["sha256"]["model.safetensors"] = "0" * 64
    (root / "audit/candidate_checkpoint.json").write_text(json.dumps(record))
    return root, ck


@pytest.mark.parametrize("index,phase", [(1, "benchmark"), (31, "benchmark"), (15, "final_test"), (0, "benchmark")])
def test_matched_dry_run_resolves_both_policies(tmp_path, index, phase):
    root, ck = _campaign_with_candidate(tmp_path)
    proc = subprocess.run([PY, str(root / "scripts/run_matched_task.py"), "--campaign", str(root),
                           "--phase", phase, "--matched-index", str(index), "--run", "r1", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    req = json.loads(proc.stdout)
    which = "baseline" if index % 2 == 0 else "candidate"
    assert req["policy"] == which and req["row"] == MANIFEST[phase][index // 2]
    expect = MANIFEST["baseline_checkpoint"]["path"] if which == "baseline" else str(ck)
    assert req["checkpoint"]["path"] == expect


def test_matched_runner_refuses_a_candidate_that_drifted_from_its_record(tmp_path):
    root, _ = _campaign_with_candidate(tmp_path, corrupt=True)
    proc = subprocess.run([PY, str(root / "scripts/run_matched_task.py"), "--campaign", str(root),
                           "--phase", "benchmark", "--matched-index", "1", "--run", "r1", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "changed" in proc.stderr


# ------------------------------------------------- recipe and rule, unchanged
def test_training_is_a2s_recipe_verbatim_and_only_the_data_changes():
    t = MANIFEST["training"]
    a2 = json.loads((V4 / "training/a2/training.json").read_text())
    assert t["lr"] == a2["optimizer_hyperparameters"]["lr"] == 3.3e-6
    assert (t["batch_size"], t["grad_accum"], t["steps"], t["checkpoint_every"]) == \
        (a2["batch_size"], a2["grad_accum"], a2["steps"], 25)
    assert t["rollout_fraction"] == a2["rollout_sampling_fraction"] == 0.5
    assert t["anchor_mode"] == "whole_episode" and t["checkpoint_rule"] == "latest_guard_passing"
    assert t["retention"]["tolerance"] == 1.10 and t["fit_precondition"] is True
    assert [Path(r["path"]).name for r in a2["retention"]["records"]] == [Path(f).name for f in t["retention"]["files"]]


def test_rule_rows_and_final_set_are_v5s_verbatim():
    v5 = json.loads((V5 / "manifest.json").read_text())
    assert MANIFEST["protocol"]["improvement_rule"] == v5["protocol"]["improvement_rule"]
    assert MANIFEST["benchmark"] == v5["benchmark"] and MANIFEST["final_test"] == v5["final_test"]
    assert MANIFEST["protocol"]["runs_per_policy_per_horizon"] == 2 and driver.RUNS == ("r1", "r2")


def test_reused_corpus_is_v4s_exactly():
    v4p = json.loads((V4 / "analysis/recovery-dataset-provenance.json").read_text())
    assert MANIFEST["corpus"]["reuse"]["sha256"] == v4p["dataset_sha256"]


def _run(settled, valid=8, rows=8):
    return {"settled": settled, "valid": valid, "rows": rows}


def _side(h10, h50):
    return {"h10": [_run(s) for s in h10], "h50": [_run(s) for s in h50]}


def test_verdict_is_v5s():
    assert not driver.matched_verdict(_side((4, 4), (4, 3)), _side((3, 2), (4, 3)), True, True)["improved"]
    assert driver.matched_verdict(_side((6, 5), (4, 3)), _side((3, 2), (4, 3)), True, True)["improved"]
    assert not driver.matched_verdict(_side((6, 5), (3, 3)), _side((3, 2), (4, 3)), True, True)["improved"]


# ----------------------------------------------------------------- driver
def _temp(tmp_path):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    return root


def test_first_tick_submits_the_targeted_search_and_nothing_else(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.tick()
    assert set(d.state["stages"]) == {"search.collect"}
    st = d.state["stages"]["search.collect"]
    assert st["gpu_tasks"] == 16 and st["array"] == ",".join(map(str, range(16)))


def test_an_unmet_pose_gate_stops_without_training(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.state["search"] = {"passed": False, "unmet_requirements": ["P_B:successful_branches"]}
    d.tick()
    assert d.state["done"] and "still missing" in d.state["stop_reason"]
    assert not any(k.startswith("pb1") for k in d.state["stages"])
    assert d.state["final"]["selected"]["label"] == "baseline"


def test_budget_funds_the_whole_path_with_retries():
    b = MANIFEST["budget"]
    assert 16 + 16 + 6 + 64 + 64 + 16 <= b["gpu_tasks"] and b["reserve"]["final_set"] == 16


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run([PY, str(ROOT / "scripts/verify_sources.py"), "--campaign", str(ROOT),
                           "--skip-checkpoint"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
