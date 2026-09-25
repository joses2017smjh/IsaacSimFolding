"""CPU tests for v7: the search rows (exact poses, off-metadata large-garment
P_B), the depth rules (landing window, qualification, mechanism check, gate
on qualifying branches), the compiler end to end on a synthetic campaign,
the smoke telemetry checker, the runner's telemetry source, and the pieces
carried from v6 (offline-fit provenance path, candidate record, recipe,
rule, driver).

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
V6 = CAMPAIGNS / "20260924-pose-balanced-v6"
PY = sys.executable
RULES = MANIFEST["depth_rules"]


def _load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bdd = _load("build_depth_dataset")
ct = _load("check_telemetry")
driver = _load("driver")
rmt = _load("run_matched_task")


def pid(mp):
    return tuple(round(float(x), 4) for x in mp.split(":"))


# ------------------------------------------------------------- search rows
def test_rows_are_exact_poses_and_the_grid_is_as_preregistered():
    rows = MANIFEST["recovery_search"]
    c = MANIFEST["pose_clusters"]
    assert [sum(r["pose"] == p for r in rows) for p in ("P_A", "P_B", "P_C")] == [8, 12, 8]
    for r in rows:
        assert pid(r["match_pose"]) == pid(c[r["pose"]]), r["id"]
    small_pb = {(r["garment"], r["pose_key"]) for r in rows if r["pose"] == "P_B" and not r["off_metadata_pose"]}
    assert small_pb == {("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)}
    assert MANIFEST["recovery"]["roots"] == [30, 60, 90, 120, 150, 180]
    assert all("recovery_roots" not in r for r in rows)


def test_large_garment_p_b_rows_are_flagged_off_metadata_placements():
    demos = json.loads(Path(MANIFEST["pose_metadata"]["path"]).read_text())
    pb = pid(MANIFEST["pose_clusters"]["P_B"])
    off = [r for r in MANIFEST["recovery_search"] if r["off_metadata_pose"]]
    assert len(off) == 4 and {r["garment"] for r in off} == {"Pant_Short_Seen_4", "Pant_Short_Seen_5"}
    for r in off:
        assert r["pose"] == "P_B" and r["pose_key"] is None and r["garment_class"] == "large"
        assert not any(pid(":".join(format(v, ".10g") for v in d["object_initial_pose"])) == pb
                       for d in demos[r["garment"]].values())


def test_seeds_are_fresh_and_garments_are_training_only():
    rows = MANIFEST["recovery_search"]
    earlier = set()
    for m in (V6 / "manifest.json", V4 / "manifest.json", V5 / "manifest.json"):
        mm = json.loads(m.read_text())
        for k in ("recovery_search", "benchmark", "final_test"):
            earlier |= {r["seed"] for r in mm.get(k, [])}
    seeds = [r["seed"] for r in rows] + [MANIFEST["recovery_smoke"][0]["seed"]]
    assert len(set(seeds)) == len(seeds) and not set(seeds) & earlier
    evaluated = {r["garment"] for r in MANIFEST["benchmark"] + MANIFEST["final_test"]}
    assert not {r["garment"] for r in rows} & evaluated


@pytest.mark.parametrize("phase,index", [("recovery", 0), ("recovery", 27), ("recovery_smoke", 0)])
def test_search_dry_runs_resolve_rows_roots_and_pinned_pose(phase, index):
    proc = subprocess.run([PY, str(ROOT / "scripts/run_rollout_task.py"), "--campaign", str(ROOT),
                           "--phase", phase, "--index", str(index), "--dry-run"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-600:]
    req = json.loads(proc.stdout)
    cmd = req["command"]
    row = (MANIFEST["recovery_search"] if phase == "recovery" else MANIFEST["recovery_smoke"])[index]
    cfg = MANIFEST["recovery"] if phase == "recovery" else MANIFEST["recovery_smoke_config"]
    assert cmd[cmd.index("--recovery_roots") + 1] == ",".join(map(str, cfg["roots"]))
    assert "--match_pose=" + row["match_pose"] in cmd
    assert cmd[cmd.index("--policy_path") + 1] == MANIFEST["baseline_checkpoint"]["path"]


def test_runner_records_the_v7_branch_telemetry():
    src = (ROOT / "scripts/render/policy_rollout51.py").read_text()
    for key in ("margin_trace=margin_trace", "gripper_distance_trace=gripper_trace", "lift_trace=lift_trace",
                "terminal_margins=", "terminal_gripper_distance=", "terminal_lift=", '"schema": 2'):
        assert key in src, key
    # passive: the telemetry helper neither renders nor steps nor seeds
    helper = src[src.index("def _rec_measure"):src.index("def _rec_step")]
    for forbidden in ("render_images", "env.step", "manual_seed", "np.random"):
        assert forbidden not in helper


# --------------------------------------------------------- depth rules
def _attempt(root=30, cand=0, *, settled=True, first=10, n=60, landed=2.0, terminal=(2.0, 2.5, 5.0, 5.0),
             clear_from=None):
    """A synthetic completed attempt with telemetry: all-4 from step `first`,
    grippers clear (landing) from `clear_from`, closure margins `landed`."""
    if clear_from is None:
        clear_from = None if first is None else first + 5
    margins, grip, lift, trace = [], [], [], []
    for k in range(n):
        full = first is not None and k >= first
        m = [landed, landed + 1.0, 3.0, 3.0] if full else [-2.0, -1.0, 3.0, 3.0]
        margins.append(m)
        clear = clear_from is not None and k >= clear_from
        grip.append([0.2, 0.2] if clear else [0.02, 0.03])
        lift.append(0.01 if clear else 0.08)
        trace.append(sum(x >= 0 for x in m))
    return {"root": root, "candidate": cand, "state": "completed", "settled_success": settled,
            "first_full_step_offset": first, "margin_trace": margins, "gripper_distance_trace": grip,
            "lift_trace": lift, "policy_condition_trace": trace, "terminal_margins": list(terminal),
            "terminal_gripper_distance": [0.2, 0.2], "terminal_lift": 0.01, "actions_executed": n,
            "horizon": 10, "full_entries": 1}


def test_landed_margin_uses_the_manifest_window_after_the_fold():
    assert bdd.landed_margin(_attempt(landed=1.7), RULES["landing"]) == pytest.approx(1.7)
    assert bdd.landed_margin(_attempt(first=None), RULES["landing"]) is None                  # never reached
    assert bdd.landed_margin(_attempt(clear_from=55), RULES["landing"]) is None               # window of 10 does not fit
    a = _attempt(clear_from=0)                                                    # clear before the fold: ignored
    assert bdd.landed_margin(a, RULES["landing"]) == pytest.approx(a["margin_trace"][a["first_full_step_offset"]][0])
    wide = dict(RULES["landing"], window=100)                                                  # the manifest value is what counts
    assert bdd.branch_record(_attempt(), dict(RULES, landing=wide))["landed_margin_cm"] is None
    assert bdd.branch_record(_attempt(), RULES)["landed_margin_cm"] == pytest.approx(2.0)


def test_qualification_is_settled_early_and_at_least_the_cut():
    ok = bdd.branch_record(_attempt(root=30, first=10, terminal=(1.6, 2.0, 5, 5)), RULES)
    assert ok["first_fold_step"] == 41                                             # completed actions, 1-based
    assert bdd.qualifies(ok, 1.5) and not bdd.qualifies(ok, 2.0)
    assert not bdd.qualifies(bdd.branch_record(_attempt(settled=False), RULES), 0.0)
    assert bdd.qualifies(bdd.branch_record(_attempt(root=180, first=269, n=420), RULES), 1.5)      # step 450
    assert not bdd.qualifies(bdd.branch_record(_attempt(root=180, first=270, n=420), RULES), 0.0)  # step 451


def test_branch_and_student_count_the_450_bound_the_same_way(tmp_path):
    branch = bdd.branch_record(_attempt(root=180, first=269, n=420), RULES)
    d = tmp_path / "row"
    d.mkdir()
    geo = [{"phase": "policy", "step": s, "success": s >= 450,
             "details": {f"condition_{k}": {"margin_cm": 2.0} for k in (1, 2, 3, 4)}} for s in range(1, 601)]
    geo.append({"phase": "terminal_settle", "step": 660, "success": True,
                "details": {f"condition_{k}": {"margin_cm": 2.0} for k in (1, 2, 3, 4)}})
    (d / "rollout.json").write_text(json.dumps({"terminal_success": True, "geometry_trajectory": geo}))
    ok, info = bdd.student_qualifies(d, RULES, 1.5)
    assert info["first_fold_step"] == branch["first_fold_step"] == 450 and ok and bdd.qualifies(branch, 1.5)


def test_mechanism_check_is_stratified_and_excludes_inherited_landings():
    deep = [bdd.branch_record(_attempt(landed=2.0, settled=True), RULES) for _ in range(12)]
    shallow = [bdd.branch_record(_attempt(landed=0.3, settled=i < 3), RULES) for i in range(12)]
    m = bdd.mechanism_check({"P_C": deep + shallow}, RULES)
    assert m["passed"] and m["deep"] == "12/12" and m["shallow"] == "3/12" and m["p_one_sided"] < 0.05
    same = [bdd.branch_record(_attempt(landed=x, settled=(i // 2) % 2 == 0), RULES)   # settling independent of depth
            for i, x in enumerate([2.0, 0.3] * 12)]
    assert not bdd.mechanism_check({"P_C": same}, RULES)["passed"]
    assert not bdd.mechanism_check({"P_C": deep}, RULES)["passed"]                # no shallow group: cannot pass
    # a pose with only one group contributes nothing, but does not break the test
    assert bdd.mechanism_check({"P_C": deep + shallow, "P_A": deep}, RULES)["passed"]
    # landings inherited from an already-folded root are the root's, not the branch's
    inh = [bdd.branch_record(_attempt(first=0, clear_from=0, landed=2.0), RULES) for _ in range(12)]
    assert all(r["inherited_landing"] for r in inh)
    assert not bdd.mechanism_check({"P_C": inh + shallow}, RULES)["passed"]


def test_stratified_p_matches_a_single_fisher_table_and_ignores_empty_strata():
    assert bdd.stratified_p([(12, 12, 3, 12)]) == pytest.approx(bdd.fisher_one_sided(12, 12, 3, 12))
    assert bdd.stratified_p([(12, 12, 3, 12), (5, 5, 0, 0)]) == pytest.approx(bdd.fisher_one_sided(12, 12, 3, 12))
    assert bdd.stratified_p([(0, 0, 0, 0)]) == 1.0


def _recs(atts):
    return [bdd.branch_record(a, RULES) for a in atts]


def test_ladder_takes_each_poses_strictest_feasible_cut():
    gates = MANIFEST["recovery"]["coverage_gate"]
    atts, rp = {}, {}
    for i in range(8):                                    # P_C: deep supply -> 1.5
        atts[f"c{i}"] = [_attempt(root=r, terminal=(2.0, 2.0, 5, 5)) for r in (30, 60, 90)]; rp[f"c{i}"] = "P_C"
    for i in range(8):                                    # P_A: settled but shallow -> 0.0
        atts[f"a{i}"] = [_attempt(root=r, terminal=(0.3, 0.9, 5, 5)) for r in (30, 60, 90)]; rp[f"a{i}"] = "P_A"
    for i in range(2):                                    # P_B: too few rows at any cut
        atts[f"b{i}"] = [_attempt(root=r) for r in (30, 60)]; rp[f"b{i}"] = "P_B"
    recs = {r: _recs(a) for r, a in atts.items()}
    g = bdd.ladder_gate(recs, atts, rp, gates, RULES["cut_ladder_cm"])
    assert g["per_pose"]["P_C"]["chosen_cut_cm"] == 1.5 and g["per_pose"]["P_A"]["chosen_cut_cm"] == 0.0
    assert g["per_pose"]["P_B"]["chosen_cut_cm"] is None and not g["passed"]
    assert all(u.startswith("P_B:") for u in g["unmet_requirements"])
    assert [e["cut_cm"] for e in g["per_pose"]["P_A"]["ladder"]] == [1.5, 1.0, 0.5, 0.0]


# ---------------------------------------------------- compiler end to end
CAMS = np.asarray(("top_rgb", "left_rgb", "right_rgb"))


def _row_dir(root, rid, attempts, labels):
    """labels: list of (root, candidate) for examples of settled branches."""
    d = root / "recovery" / rid
    d.mkdir(parents=True)
    (d / "recovery.json").write_text(json.dumps({"schema": 2, "attempts": attempts}))
    n = len(labels)
    np.savez(d / "recovery_examples.npz", images=np.full((n, 3, 480, 640, 3), 5, np.uint8),
             state=np.zeros((n, 12), np.float32), action=np.ones((n, 50, 12), np.float32),
             camera_keys=CAMS, root=np.asarray([r for r, _ in labels], np.int64),
             candidate=np.asarray([c for _, c in labels], np.int64), horizon=np.full(n, 10, np.int64),
             candidate_seed=np.full(n, 9301, np.int64), step_index=np.arange(n, dtype=np.int64) + 40)
    np.savez(d / "trajectory.npz", terminal_success=np.asarray(False), step_index=np.arange(3),
             executed_action_stream=np.zeros((600, 12), np.float32),
             images=np.zeros((3, 3, 480, 640, 3), np.uint8), state=np.zeros((3, 12), np.float32))
    (d / "rollout.json").write_text(json.dumps({"terminal_success": False, "geometry_trajectory": []}))
    (d / "status.json").write_text(json.dumps({"state": "completed"}))


def _synthetic(tmp_path, *, shallow_settles=0):
    c = MANIFEST["pose_clusters"]
    root = tmp_path / "camp"
    rows = []
    for pose in ("P_A", "P_B", "P_C"):
        rid = f"row_{pose}"
        deep = [_attempt(root=30, cand=0, landed=2.0, terminal=(2.0, 2.0, 5, 5)),
                _attempt(root=60, cand=1, landed=2.2, terminal=(2.1, 2.4, 5, 5))]
        shallow = [_attempt(root=90, cand=k, landed=0.3, settled=k < shallow_settles,
                            terminal=(0.4, 1.0, 5, 5) if k < shallow_settles else (-1.0, 1.0, 5, 5))
                   for k in range(3)]
        labels = [(30, 0), (30, 0), (60, 1)] + [(90, k) for k in range(shallow_settles)]
        _row_dir(root, rid, deep + shallow, labels)
        rows.append({"id": rid, "pose": pose, "match_pose": c[pose], "garment": "G", "pose_key": 0, "seed": 1})
    gate = {"min_successful_branches": 2, "min_distinct_roots": 2, "min_rows": 1}
    rules = dict(RULES, min_total_labels=6, mechanism_p=0.5)
    (root / "manifest.json").write_text(json.dumps({
        "pose_clusters": c, "recovery_search": rows, "task_prompt": "fold", "depth_rules": rules,
        "recovery": {"coverage_gate": {p: gate for p in ("P_A", "P_B", "P_C")}}}))
    return root


def _compile(root):
    return subprocess.run([PY, str(ROOT / "scripts/build_depth_dataset.py"), "--campaign", str(root),
                           "--out", str(root / "datasets/recovery.npz"), "--provenance", str(root / "prov.json"),
                           "--gate-out", str(root / "gate.json")], capture_output=True, text=True)


def test_compiler_keeps_only_deep_qualifying_labels_and_balances_poses(tmp_path):
    root = _synthetic(tmp_path, shallow_settles=1)
    proc = _compile(root)
    assert proc.returncode == 0, proc.stderr[-800:] + proc.stdout[-400:]
    with np.load(root / "datasets/recovery.npz", allow_pickle=False) as z:
        assert len(z["pose"]) == 9                                   # 3 deep labels per pose; shallow dropped
        assert set(zip(z["root"].tolist(), z["candidate"].tolist())) == {(30, 0), (60, 1)}
        assert (z["terminal_depth_cm"] >= 1.5).all()                  # every pose chose the 1.5 cm cut
        w, pose = z["awr_weight"], z["pose"]
        for p in ("P_A", "P_B", "P_C"):
            assert w[pose == p].sum() / w.sum() == pytest.approx(1 / 3)
        for member in ("images", "state", "action", "advantage", "reward", "chunk", "task", "camera_keys", "origin"):
            assert member in z.files
    gate = json.loads((root / "gate.json").read_text())
    assert gate["passed"] and gate["mechanism_check"]["deep"] == "6/6"
    assert gate["cut_by_pose"] == {"P_A": 1.5, "P_B": 1.5, "P_C": 1.5} and gate["excluded_rows"] == []


def test_compiler_excludes_one_incomplete_row_and_stops_on_two(tmp_path):
    root = _synthetic(tmp_path, shallow_settles=1)
    m = json.loads((root / "manifest.json").read_text())
    extra = dict(m["recovery_search"][0], id="row_P_A_2")
    _row_dir(root, "row_P_A_2", [_attempt(root=30, cand=0), _attempt(root=60, cand=1)], [(30, 0), (60, 1)])
    m["recovery_search"].append(extra)
    (root / "recovery/row_P_A_2/status.json").write_text(json.dumps({"state": "infrastructure_timeout"}))
    (root / "manifest.json").write_text(json.dumps(m))
    proc = _compile(root)
    assert proc.returncode == 0, proc.stderr[-600:]
    assert json.loads((root / "gate.json").read_text())["excluded_rows"] == ["row_P_A_2"]
    (root / "datasets/recovery.npz").unlink()
    (root / "recovery/row_P_B/recovery_examples.npz").unlink()               # a second missing row
    proc = _compile(root)
    gate = json.loads((root / "gate.json").read_text())
    assert proc.returncode == 5 and gate["infrastructure"] and len(gate["excluded_rows"]) == 2


def test_compiler_fails_closed_when_depth_does_not_predict_settling(tmp_path):
    root = _synthetic(tmp_path, shallow_settles=3)                    # shallow settle as often as deep
    m = json.loads((root / "manifest.json").read_text())
    m["depth_rules"]["mechanism_p"] = 0.05
    (root / "manifest.json").write_text(json.dumps(m))
    proc = _compile(root)
    assert proc.returncode == 4 and not (root / "datasets/recovery.npz").exists()
    assert "mechanism_check" in json.loads((root / "gate.json").read_text())["unmet_requirements"]


def test_compiler_refuses_attempts_without_telemetry(tmp_path):
    root = _synthetic(tmp_path, shallow_settles=1)
    f = root / "recovery/row_P_A/recovery.json"
    rec = json.loads(f.read_text())
    del rec["attempts"][0]["margin_trace"]
    f.write_text(json.dumps(rec))
    proc = _compile(root)
    assert proc.returncode != 0 and "telemetry" in proc.stderr


def test_compiler_enforces_the_total_label_floor(tmp_path):
    root = _synthetic(tmp_path, shallow_settles=1)
    m = json.loads((root / "manifest.json").read_text())
    m["depth_rules"]["min_total_labels"] = 100
    (root / "manifest.json").write_text(json.dumps(m))
    proc = _compile(root)
    assert proc.returncode == 4 and not (root / "datasets/recovery.npz").exists()


# --------------------------------------------------------- smoke checker
def _smoke(tmp_path, *, attempt=None, start_margins=(-3.0, -5.0, 6.0, 6.0)):
    root = tmp_path / "camp"
    row = dict(MANIFEST["recovery_smoke"][0])
    d = root / "smoke/recovery" / row["id"]
    d.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    a = attempt or _attempt(root=60, first=10, n=90)
    (d / "recovery.json").write_text(json.dumps({"schema": 2, "attempts": [a]}))
    det = {f"condition_{k}": {"margin_cm": m} for k, m in zip((1, 2, 3, 4), start_margins)}
    (d / "rollout.json").write_text(json.dumps({"physics_finite": True, "robot_finite": True,
        "geometry_trajectory": [{"phase": "policy", "step": 1, "details": det, "success": False}]}))
    (d / "status.json").write_text(json.dumps({"state": "completed"}))
    return root


def test_smoke_checker_passes_good_telemetry_and_a_valid_spawn(tmp_path):
    assert ct.check(_smoke(tmp_path))["passed"]


def test_smoke_checker_reports_a_raised_runner_and_defers_a_transient_failure(tmp_path):
    root = _smoke(tmp_path)
    d = root / "smoke/recovery" / MANIFEST["recovery_smoke"][0]["id"]
    (d / "status.json").write_text(json.dumps({"state": "infrastructure_timeout"}))
    assert ct.check(root) is None                                   # no report: the driver retries once
    (d / "rollout.json.error.json").write_text(json.dumps({"error_type": "ValueError", "error": "nonfinite"}))
    rep = ct.check(root)
    assert not rep["passed"] and "ValueError" in rep["problems"][0]


@pytest.mark.parametrize("breakage", ["trace_length", "missing_terminal", "sign_mismatch", "folded_spawn"])
def test_smoke_checker_catches_each_defect(tmp_path, breakage):
    a = _attempt(root=60, first=10, n=90)
    start = (-3.0, -5.0, 6.0, 6.0)
    if breakage == "trace_length":
        a["lift_trace"] = a["lift_trace"][:-1]
    elif breakage == "missing_terminal":
        del a["terminal_margins"]
    elif breakage == "sign_mismatch":
        a["policy_condition_trace"] = [0] * len(a["policy_condition_trace"])
    else:
        start = (1.0, 2.0, 6.0, 6.0)
    assert not ct.check(_smoke(tmp_path, attempt=a, start_margins=start))["passed"]


# ------------------------------------------ carried from v6, unchanged
class _Hide:
    def __init__(self, z):
        self.z = z
        self.files = [f for f in z.files if f not in ("horizon", "root", "step_index")]

    def __getitem__(self, k):
        return self.z[k]


def test_fit_gate_pools_only_the_origin_kinds_present():
    of = _load("offline_fit")
    assert of.pooled_groups({"branch": [1, 2], "student": [3]}) == ["recovery_branch", "recovery_student"]
    assert of.pooled_groups({"branch": [1, 2]}) == ["recovery_branch"]
    assert of.pooled_groups({"branch": [1], "student": []}) == ["recovery_branch"]
    with pytest.raises(SystemExit):
        of.pooled_groups({})


def test_offline_fit_provenance_path_equals_v4s_reconstruction_on_v4s_corpus():
    of = _load("offline_fit")
    v4m = json.loads((V4 / "manifest.json").read_text())
    z = np.load(V4 / "datasets/recovery.npz", allow_pickle=False)
    n = len(z["origin"])
    h_new, r_new = of.example_classes(z, v4m, V4, n)
    h_old, r_old = of.example_classes(_Hide(z), v4m, V4, n)
    assert np.array_equal(h_new, h_old) and np.array_equal(r_new, r_old)


def _campaign_with_candidate(tmp_path, *, corrupt=False):
    root = tmp_path / "camp2"
    (root / "audit").mkdir(parents=True)
    (root / "scripts").symlink_to(ROOT / "scripts")
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    ck = V4 / "training/a2/checkpoints/step_000250"
    record = {"path": str(ck), "label": "dq1-step000250",
              "sha256": json.loads((ck / "checkpoint.json").read_text())["checkpoint_sha256"]}
    if corrupt:
        record["sha256"]["model.safetensors"] = "0" * 64
    (root / "audit/candidate_checkpoint.json").write_text(json.dumps(record))
    return root, ck


@pytest.mark.parametrize("index,phase", [(1, "benchmark"), (0, "benchmark"), (15, "final_test")])
def test_matched_dry_run_resolves_both_policies(tmp_path, index, phase):
    root, ck = _campaign_with_candidate(tmp_path)
    proc = subprocess.run([PY, str(root / "scripts/run_matched_task.py"), "--campaign", str(root),
                           "--phase", phase, "--matched-index", str(index), "--run", "r1", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-800:]
    req = json.loads(proc.stdout)
    which = "baseline" if index % 2 == 0 else "candidate"
    assert req["policy"] == which and req["row"] == MANIFEST[phase][index // 2]


def test_matched_runner_refuses_a_drifted_candidate(tmp_path):
    root, _ = _campaign_with_candidate(tmp_path, corrupt=True)
    proc = subprocess.run([PY, str(root / "scripts/run_matched_task.py"), "--campaign", str(root),
                           "--phase", "benchmark", "--matched-index", "1", "--run", "r1", "--dry-run"],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "changed" in proc.stderr


def test_recipe_rule_rows_and_final_set_are_v6s_verbatim():
    v6 = json.loads((V6 / "manifest.json").read_text())
    for k in ("lr", "batch_size", "grad_accum", "steps", "checkpoint_every", "rollout_fraction", "anchor_mode",
              "checkpoint_rule", "retention", "anchor", "fit_precondition", "seed", "unfreeze"):
        assert MANIFEST["training"][k] == v6["training"][k], k
    for k in ("h10", "h50", "all_rows_valid"):
        assert MANIFEST["protocol"]["improvement_rule"][k] == v6["protocol"]["improvement_rule"][k]
    assert MANIFEST["benchmark"] == v6["benchmark"] and MANIFEST["final_test"] == v6["final_test"]
    assert MANIFEST["corpus"]["reuse"] is None


# ----------------------------------------------------------------- driver
def _temp(tmp_path):
    root = tmp_path / "campaigns" / "c"
    (root / "ledger").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(MANIFEST))
    return root


def test_first_tick_submits_the_smoke_and_nothing_else(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.tick()
    assert set(d.state["stages"]) == {"smoke"} and d.state["stages"]["smoke"]["gpu_tasks"] == 1


def test_a_failed_smoke_stops_before_any_search(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.state["smoke"] = {"passed": False, "problems": ["x"]}
    d.tick()
    assert d.state["done"] and "smoke failed" in d.state["stop_reason"]
    assert "search.collect" not in d.state["stages"]


def test_after_a_passed_smoke_the_search_is_one_array_of_28(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.state["smoke"] = {"passed": True}
    d.tick()
    st = d.state["stages"]["search.collect"]
    assert st["gpu_tasks"] == 28 and st["array"] == ",".join(map(str, range(28)))


def test_stop_reasons_name_the_cause():
    mech = driver.gate_stop_reason({"unmet_requirements": ["mechanism_check"],
                                    "mechanism_check": {"passed": False, "deep": "5/9", "shallow": "5/8"}})
    assert "not a usable lever" in mech
    supply = driver.gate_stop_reason({"unmet_requirements": ["P_A:successful_branches"],
                                      "mechanism_check": {"passed": True, "deep": "30/34", "shallow": "10/22",
                                                          "p_one_sided": 0.001}})
    assert "30/34" in supply and "yield shortfall" in supply and "not a usable lever" not in supply
    infra = driver.gate_stop_reason({"infrastructure": True, "unmet_requirements": ["search incomplete"]})
    assert "infrastructure" in infra and "lever" not in infra


def test_a_raised_smoke_runner_is_not_retried(tmp_path):
    root = _temp(tmp_path)
    d = driver.Driver(root, dry=True)
    d.state["stages"]["smoke"] = {"job_ids": ["1"], "status": "failed"}
    err = root / "smoke/recovery" / MANIFEST["recovery_smoke"][0]["id"]
    err.mkdir(parents=True)
    (err / "rollout.json.error.json").write_text(json.dumps({"error_type": "ValueError", "error": "spawn"}))
    sm = d.smoke()
    assert sm["passed"] is False and "smoke.retry1" not in d.state["stages"]


def test_an_unmet_depth_gate_stops_without_training(tmp_path):
    d = driver.Driver(_temp(tmp_path), dry=True)
    d.state["smoke"] = {"passed": True}
    d.state["search"] = {"passed": False, "unmet_requirements": ["mechanism_check"], "mechanism_check": {}}
    d.tick()
    assert d.state["done"] and "depth gate unmet" in d.state["stop_reason"]
    assert not any(k.startswith("dq1") for k in d.state["stages"])
    assert d.state["final"]["selected"]["label"] == "baseline"


def test_budget_funds_the_whole_path_with_retries():
    b = MANIFEST["budget"]
    assert 2 + 28 + 28 + 6 + 64 + 64 + 16 <= b["gpu_tasks"] and b["reserve"]["final_set"] == 16


def test_verify_sources_passes_against_the_frozen_manifest():
    proc = subprocess.run([PY, str(ROOT / "scripts/verify_sources.py"), "--campaign", str(ROOT),
                           "--skip-checkpoint"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
