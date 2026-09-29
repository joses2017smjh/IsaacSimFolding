"""CPU tests for v9: rows, pins, statistics, detectors, GPU checks and the driver plan.

    /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest tests/test_v9.py -q
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT.parent
V8 = CAMPAIGNS / "20260926-anchor-diagnostic-v8"
sys.path.insert(0, str(ROOT / "scripts"))

import analysis as A  # noqa: E402
import build_manifest as B  # noqa: E402
import driver as D  # noqa: E402
import endpoints  # noqa: E402
from run_matched_task import POLICIES, destination, matched_index  # noqa: E402

V8_MANIFEST = json.loads((V8 / "manifest.json").read_text())
FROZEN = (ROOT / "manifest.json").is_file()


# ------------------------------------------------------------------ rows
def test_rows_are_fresh_shared_and_faithful_to_the_dev_poses():
    dev = V8_MANIFEST["benchmark"]
    rows = B.rows_from(dev)
    assert len(rows) == 48 and len({r["id"] for r in rows}) == 48
    assert collections.Counter(r["horizon"] for r in rows) == {10: 24, 50: 24}
    by_pose_block = collections.defaultdict(set)
    for r in rows:
        src = next(d for d in dev if d["id"] == r["source_row"])
        for key in ("garment", "asset_config", "match_pose", "match_scale", "steps", "horizon",
                    "development_pose_slot", "pose_key"):
            assert r[key] == src[key], (r["id"], key)
        by_pose_block[(r["development_pose_slot"], r["seed_block"])].add(r["seed"])
    # one seed per (pose, block), shared by H10 and H50
    assert all(len(s) == 1 for s in by_pose_block.values()) and len(by_pose_block) == 24
    used = B.used_seeds()
    assert 200 in used and 96000 in used, "the scan must see the dev seeds and v8's confirm seeds"
    assert not {r["seed"] for r in rows} & set(used)


def test_frozen_manifest_pins_both_policies_and_both_blocks():
    if not FROZEN:
        pytest.skip("manifest not frozen yet")
    m = json.loads((ROOT / "manifest.json").read_text())
    assert m["remeasure"] == B.rows_from(V8_MANIFEST["benchmark"])
    assert m["baseline_checkpoint"] == V8_MANIFEST["baseline_checkpoint"]
    record = json.loads((V8 / "audit/candidate_checkpoint.json").read_text())
    assert m["candidate_checkpoint"]["sha256"] == record["sha256"]
    assert m["candidate_checkpoint"]["label"] == "an1-step000300"
    assert [b["key"] for b in m["blocks"]] == ["a40", "rtx8000"]
    assert {b["constraint"] for b in m["blocks"]} == {"rtx8000", "a40"}
    assert m["protocol"]["primary_block"] == "a40"
    assert m["protocol"]["power"]["table"]["a40"]["baseline_history"] == "12/40"
    assert m["protocol"]["power"]["table"]["rtx8000"]["baseline_history"] == "3/24"
    assert m["seeds"]["scan"]["distinct_prior_seeds"] >= 182 and m["seeds"]["scan"]["max_prior_seed"] < 97000
    lehome = [k for k in m["executed_sources"] if "/external/lehome-challenge/" in k]
    assert any(k.endswith("success_checker_chanllege.py") for k in lehome) and len(lehome) >= 60
    env = m["environment_digests"]["lehome51-site/lerobot"]
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_sources import tree_digest
    assert tree_digest(Path(env["path"]), env["glob"]) == (env["sha256"], env["files"])
    runner = str((V8 / "scripts/render/policy_rollout51.py").relative_to(ROOT.parents[1]))
    assert m["executed_sources"][m["runner_key"]] == V8_MANIFEST["executed_sources"][runner]
    assert m["protocol"]["block_deadline_utc"] == B.BLOCK_DEADLINE_UTC


def test_interleaving_and_destinations_are_distinct_per_policy_block_and_row():
    rows = B.rows_from(V8_MANIFEST["benchmark"])
    seen = set()
    for block in ("rtx8000", "a40"):
        for i in range(2 * len(rows)):
            which, r = matched_index(i)
            assert which == POLICIES[i % 2] and r == i // 2
            seen.add(destination(ROOT, "remeasure", rows[r], f"{which}-{block}"))
    assert len(seen) == 2 * 2 * len(rows)


# ------------------------------------------------------------ statistics
def _v8_h10_strata():
    strata = []
    for k in range(8):
        cs = cn = bs = bn = 0
        for run in ("r1", "r2"):
            for label in ("an1-step000300", "baseline"):
                d = next((V8 / "evaluation" / f"{label}-{run}" / "benchmark").glob(f"dev{k:02d}_h10_*"))
                e = A.episode(d)
                assert e["valid"]
                if label == "baseline":
                    bs += e["settled"]; bn += 1
                else:
                    cs += e["settled"]; cn += 1
        strata.append((cs, cn, bs, bn))
    return strata


def test_statistics_reproduce_v8_published_values():
    strata = _v8_h10_strata()
    assert sum(s[0] for s in strata) == 9 and sum(s[2] for s in strata) == 2
    assert round(A.fisher_one_sided(9, 16, 2, 16), 4) == 0.0117
    assert round(A.stratified_exact(strata), 3) == 0.012
    assert round(A.sign_flip([s[0] - s[2] for s in strata]), 3) == 0.047
    assert abs(A.fisher_one_sided(9, 16, 2, 16) - D_fisher(9, 16, 2, 16)) < 1e-12


def D_fisher(a, an, b, bn):
    # the v5-v8 driver's formula, loaded from v8's own file, must be the same function
    import importlib.util
    spec = importlib.util.spec_from_file_location("v8_driver", V8 / "scripts/driver.py")
    v8d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v8d)
    return v8d.fisher_one_sided(a, an, b, bn)


def test_exact_tests_on_small_cases():
    assert A.stratified_exact([(3, 5, 1, 5)]) == pytest.approx(A.fisher_one_sided(3, 5, 1, 5))
    assert A.sign_flip([1]) == 0.5 and A.sign_flip([0, 0]) == 1.0
    assert A.sign_flip([1, 1, 1]) == 1 / 8
    assert A.mcnemar_exact(0, 0) == 1.0
    assert A.mcnemar_exact(5, 0) == 1 / 32
    assert A.mcnemar_exact(3, 3) == pytest.approx(sum(__import__("math").comb(6, k) for k in range(3, 7)) / 64)


# ------------------------------------------------------------- detectors
def test_detector_calibration_matches_the_v8_diagnosis():
    cal = B.calibration()
    assert cal["baseline"]["episodes"] == 64 and cal["an1"]["episodes"] == 16
    drags = {p: sorted(k for k in c if k.startswith("drag:")) for p, c in cal.items()}
    assert drags["baseline"] == ["drag:dev00", "drag:dev02", "drag:dev04"]
    assert all(cal["baseline"][k] == 8 for k in drags["baseline"])
    assert drags["an1"] == ["drag:dev00", "drag:dev02", "drag:dev04", "drag:dev06"]
    assert all(cal["an1"][k] == 2 for k in drags["an1"])
    assert drags["a2"] == drags["pb1"] == drags["dq1"] == []
    # E1: the diagnosed shift (fine-tunes 38/48 right-only; baseline 6/48 over v5-v7, 2 more in v8)
    assert cal["a2"]["right_only_chunk1"] + cal["pb1"]["right_only_chunk1"] + cal["dq1"]["right_only_chunk1"] == 38
    assert cal["baseline"]["right_only_chunk1"] == 8 and cal["an1"]["right_only_chunk1"] == 0
    # EBL, exactly: the baseline's early-H50 rows in every run; an1 those plus dev06/dev07; pb1 dev05 only
    ebl = {p: {k: v for k, v in c.items() if k.startswith("ebl:")} for p, c in cal.items()}
    assert ebl["baseline"] == {f"ebl:dev0{i}": 8 for i in (0, 1, 3, 4, 5)}
    assert ebl["an1"] == {f"ebl:dev0{i}": 2 for i in (0, 1, 3, 4, 5, 6, 7)}
    assert ebl["pb1"] == {"ebl:dev05": 2} and ebl["a2"] == {} and ebl["dq1"] == {}
    assert sum(c["episodes"] for c in cal.values()) == 128


def test_drag_detector_on_a_synthetic_stream():
    def row(a, gr, gl, rx, rz, lx, lz, lift):
        return {"action": a, "gripper_target_rad": {"left": gl, "right": gr}, "maximum_particle_lift_m": lift,
                "nearest_particle": {"left": {"link_origin_xyz_m": [lx, 0.0, lz]},
                                     "right": {"link_origin_xyz_m": [rx, 0.0, rz]}}}
    rows = [row(1, 0.5, 0.3, 0.15, 0.61, -0.26, 0.67, 0.01), row(2, -0.1, 0.3, 0.15, 0.61, -0.26, 0.67, 0.01)]
    rows += [row(3 + i, -0.1, 0.3, 0.15 - 0.01 * i, 0.65, -0.26, 0.67, 0.01 + 0.002 * i) for i in range(30)]
    assert A.drag_events(rows) == [2]
    parked_low = [dict(r, nearest_particle={**r["nearest_particle"],
                                            "left": {"link_origin_xyz_m": [-0.26, 0.0, 0.60]}}) for r in rows]
    assert A.drag_events(parked_low) == []          # left arm at the table: not the one-arm drag
    no_cross = [dict(r, nearest_particle={**r["nearest_particle"],
                                          "right": {"link_origin_xyz_m": [0.15, 0.0, 0.61]}}) for r in rows]
    assert A.drag_events(no_cross) == []            # never crosses the midline


# ------------------------------------------------------------------ GPU
def test_gpu_model_parses_from_every_v8_log():
    models = collections.Counter()
    for log in (V8 / "evaluation").glob("*/benchmark/*/rollout.log"):
        m = A.gpu_model(log)
        assert m in A.GPU_NAMES.values(), log
        models[m] += 1
    assert models == {"NVIDIA A40": 51, "Quadro RTX 8000": 13}
    dev06 = [A.gpu_model(p / "rollout.log") for p in (V8 / "evaluation").glob("*/benchmark/dev06_h10_*")]
    assert dev06 == ["NVIDIA A40"] * 4


# ------------------------------------------------------ synthetic block
SHA = "f" * 64
DEADLINE = "2026-10-06T00:00:00Z"
PROTOCOL = {"block_deadline_utc": DEADLINE, "primary_block": "a40",
            "power": {"headline": "power at p1=0.45: a40 20%, rtx8000 74%"}}


def _write_episode(d: Path, *, settled: bool, gpu: str, horizon: int, seed: int, state="completed",
                   completed="2026-09-30T12:00:00Z", manifest_sha=SHA):
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "gpu_model": gpu, "completed_utc": completed}))
    (d / "request.json").write_text(json.dumps({"manifest_sha256": manifest_sha}))
    details = {c: {"margin_cm": 1.0 if settled else -1.0} for c in A.CONDITIONS}
    (d / "rollout.json").write_text(json.dumps({
        "terminal_success": settled, "geometric_ever_success": settled, "geometric_first_success_action": 300,
        "terminal_checker": {"conditions_passed": 4 if settled else 3, "details": details},
        "effective_n_action_steps": horizon, "seed": seed}))
    if horizon == 50:
        lines = []
        for a in range(1, 161):
            lines.append(json.dumps({"action": a, "gripper_target_rad": {"left": 0.3, "right": 0.3},
                                     "maximum_particle_lift_m": 0.001 * a,
                                     "nearest_particle": {"left": {"link_origin_xyz_m": [-0.1, 0, 0.7]},
                                                          "right": {"link_origin_xyz_m": [0.1, 0, 0.7]}}}))
        (d / "rollout.json.behavior.jsonl").write_text("\n".join(lines) + "\n")


BLOCKS = [{"key": "a40", "constraint": "a40", "gpu_name": "NVIDIA A40", "role": "primary"},
          {"key": "rtx8000", "constraint": "rtx8000", "gpu_name": "Quadro RTX 8000", "role": "secondary"}]
LABELS = {"baseline": "baseline", "candidate": "an1-step000300"}


def _fake_campaign(tmp_path: Path, cand_settled, base_settled, gpu="Quadro RTX 8000", block="rtx8000", **kw):
    rows = B.rows_from(V8_MANIFEST["benchmark"])
    manifest = {"remeasure": rows, "pose_clusters": V8_MANIFEST["pose_clusters"], "blocks": BLOCKS,
                "protocol": PROTOCOL}
    for which, fn in (("candidate", cand_settled), ("baseline", base_settled)):
        for r in rows:
            _write_episode(tmp_path / "evaluation" / f"{LABELS[which]}-{block}" / "remeasure" / r["id"],
                           settled=fn(r), gpu=gpu, horizon=r["horizon"], seed=r["seed"], **kw)
    return manifest, dict(LABELS)


def test_block_results_counts_clauses_and_classification(tmp_path):
    cand = lambda r: r["horizon"] == 10 and r["development_pose_slot"] < 5 or (r["horizon"] == 50 and r["seed_block"] == 0)
    base = lambda r: r["horizon"] == 10 and r["development_pose_slot"] == 0 or (r["horizon"] == 50 and r["seed_block"] < 2)
    manifest, labels = _fake_campaign(tmp_path, cand, base)
    res = A.block_results(tmp_path, manifest, BLOCKS[1], labels)
    assert res["complete"] and res["valid"] == {"baseline": 48, "candidate": 48}
    assert res["h10"]["candidate"] == "15/24" and res["h10"]["baseline"] == "3/24"
    assert res["h10"]["clause_margin_ge_4_and_p_lt_0.05"]
    assert res["h50"]["candidate"] == "8/24" and res["h50"]["baseline"] == "16/24"
    assert not res["h50"]["clause_candidate_ge_baseline"] and not res["rule"]["v5_rule_holds"]
    assert res["h10"]["paired_mcnemar"] == {"candidate_only": 12, "baseline_only": 0,
                                            "one_sided_p": 1 / 2 ** 12}
    assert res["h50"]["behaviour"]["candidate"]["episodes"] == 24 and res["role"] == "secondary"
    assert res["h10"]["n_paired"] == 24 and res["manifest_digests"] == [SHA]
    assert res["h50"]["dev01"] == {"baseline": "2/3", "candidate": "1/3", "loss_recurs": True}
    c = A.classify({"a40": None, "rtx8000": res}, PROTOCOL)
    assert c["h10_gain"] == "cross-platform classification not available (a40 incomplete)"
    assert c["readings"]["rtx8000"].startswith("H10 clause holds") and c["h10_clause_holds_on"] == ["rtx8000"]
    assert c["headline"].startswith("a40 (primary): not measured")


def _block(h10_diffs: dict[int, tuple[int, int]], h10_holds: bool, h50_holds: bool = True) -> dict:
    """A minimal complete block record for classify(): per-row (cand, base) settled over 3 seeds."""
    return {"complete": True, "rule": {"h10_clause": h10_holds, "h50_clause": h50_holds,
                                       "v5_rule_holds": h10_holds and h50_holds},
            "h10": {"per_row_diff": {f"dev{k:02d}": c - b for k, (c, b) in h10_diffs.items()}}}


def test_classify_covers_all_four_cross_platform_outcomes_and_the_interaction():
    even = {k: (2, 1) for k in range(8)}
    both = A.classify({"a40": _block(even, True), "rtx8000": _block(even, True)}, PROTOCOL)
    assert both["h10_gain"] == "replicates on both platforms" and both["interaction"]["two_sided_p"] == 1.0
    neither = A.classify({"a40": _block(even, False), "rtx8000": _block(even, False)}, PROTOCOL)
    assert neither["h10_gain"] == "not replicated at this power (power at p1=0.45: a40 20%, rtx8000 74%)"
    one = A.classify({"a40": _block(even, False), "rtx8000": _block(even, True)}, PROTOCOL)
    assert one["h10_gain"].startswith("H10 clause holds on rtx8000, not on a40; platform x policy interaction "
                                      "not established")
    big = {k: (3, 0) for k in range(8)}
    none = {k: (1, 1) for k in range(8)}
    dep = A.classify({"a40": _block(none, False), "rtx8000": _block(big, True)}, PROTOCOL)
    assert dep["interaction"]["two_sided_p"] == 2 / 256 and dep["h10_gain"].endswith("hardware-dependent")
    assert dep["headline"].startswith("a40 (primary): H10 clause fails")


def test_wrong_gpu_model_or_unfinished_task_is_invalid_not_a_failure(tmp_path):
    manifest, labels = _fake_campaign(tmp_path, lambda r: True, lambda r: False, gpu="NVIDIA A40", block="rtx8000")
    res = A.block_results(tmp_path, manifest, BLOCKS[1], labels)
    assert res["valid"] == {"baseline": 0, "candidate": 0} and not res["complete"]
    assert set(s["state"] for s in res["invalid"]["baseline"].values()) == {"wrong_gpu_model"}


def test_episode_completed_after_the_deadline_is_invalid(tmp_path):
    manifest, labels = _fake_campaign(tmp_path, lambda r: True, lambda r: False, completed="2026-10-06T00:00:01Z")
    res = A.block_results(tmp_path, manifest, BLOCKS[1], labels)
    assert res["valid"] == {"baseline": 0, "candidate": 0} and not res["complete"]
    assert set(s["state"] for s in res["invalid"]["candidate"].values()) == {"after_deadline"}


def test_mixed_manifest_digests_make_a_block_incomplete(tmp_path):
    manifest, labels = _fake_campaign(tmp_path, lambda r: True, lambda r: False)
    first = tmp_path / "evaluation" / "baseline-rtx8000" / "remeasure" / manifest["remeasure"][0]["id"]
    (first / "request.json").write_text(json.dumps({"manifest_sha256": "0" * 64}))
    res = A.block_results(tmp_path, manifest, BLOCKS[1], labels)
    assert res["valid"] == {"baseline": 48, "candidate": 48} and not res["complete"]
    assert res["manifest_digests"] == ["0" * 64, SHA]


# ---------------------------------------------------------------- driver
def test_driver_plans_one_pinned_array_per_block(tmp_path, monkeypatch):
    if not FROZEN:
        pytest.skip("manifest not frozen yet")
    (tmp_path / "manifest.json").write_text((ROOT / "manifest.json").read_text())
    monkeypatch.setattr(D.Driver, "preflight", lambda self: True)
    drv = D.Driver(tmp_path, dry=True)
    assert drv.advance() == "wait"
    stages = drv.state["stages"]
    assert set(stages) == {"block:rtx8000", "block:a40"}
    assert all(s["gpu_tasks"] == 96 and s["array"] == "0-95" for s in stages.values())
    extras = [e["sbatch_extra"] for e in drv.state["log"] if e["message"].startswith("submitted")]
    assert extras == [["--constraint=a40"], ["--constraint=rtx8000"]]
    assert not (tmp_path / "ledger").exists(), "a dry run writes nothing"


def test_driver_dry_run_moves_nothing_and_enforces_the_deadline_on_completion_time(tmp_path, monkeypatch):
    if not FROZEN:
        pytest.skip("manifest not frozen yet")
    m = json.loads((ROOT / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    for which in ("baseline", "an1-step000300"):
        for r in m["remeasure"]:
            _write_episode(tmp_path / "evaluation" / f"{which}-a40" / "remeasure" / r["id"], settled=False,
                           gpu="NVIDIA A40", horizon=r["horizon"], seed=r["seed"])
    bad = tmp_path / "evaluation" / "baseline-a40" / "remeasure" / m["remeasure"][3]["id"]
    (bad / "status.json").write_text(json.dumps({"state": "infrastructure_error", "error": "x"}))
    late = tmp_path / "evaluation" / "baseline-a40" / "remeasure" / m["remeasure"][4]["id"]
    (late / "status.json").write_text(json.dumps({"state": "completed", "completed_utc": "2026-10-07T00:00:00Z"}))
    monkeypatch.setattr(D.Driver, "preflight", lambda self: True)
    monkeypatch.setattr(D, "job_states", lambda ids: {j: ["COMPLETED"] for j in ids})
    drv = D.Driver(tmp_path, dry=True)
    drv.state["stages"]["block:a40"] = {"job_ids": ["DRY-1"], "status": "completed"}
    assert drv.block_stage(m["blocks"][0], m["remeasure"]) == "wait"
    assert drv.state["stages"]["block:a40.retry1"]["array"] == "6,8"   # indices 2*3 and 2*4
    assert bad.exists() and late.exists() and not (tmp_path / "attempts").exists()


def test_deadline_parses_and_is_in_the_future_at_freeze():
    assert D.parse_utc(B.BLOCK_DEADLINE_UTC) > D.parse_utc("2026-09-29T00:00:00Z")


def test_task_dry_run_resolves_both_pinned_policies():
    if not FROZEN:
        pytest.skip("manifest not frozen yet")
    py = sys.executable
    sha = D.file_sha(ROOT / "manifest.json")
    for index in (0, 1):
        out = subprocess.run([py, str(ROOT / "scripts/run_matched_task.py"), "--campaign", str(ROOT),
                              "--phase", "remeasure", "--matched-index", str(index), "--run", "rtx8000",
                              "--gpu-model", "rtx8000", "--manifest-sha256", sha, "--dry-run"],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        req = json.loads(out.stdout)
        assert req["policy"] == POLICIES[index] and req["row"]["id"] == "dev00_h10_s97000"
        assert req["gpu"]["expected_name"] == "Quadro RTX 8000"
        assert "--seed" in req["command"] and req["command"][req["command"].index("--seed") + 1] == "97000"
        assert req["manifest_sha256"] == sha and len(req["checkpoint"]["sha256"]) == 8
    wrong = subprocess.run([py, str(ROOT / "scripts/run_matched_task.py"), "--campaign", str(ROOT),
                            "--phase", "remeasure", "--matched-index", "0", "--run", "rtx8000",
                            "--gpu-model", "rtx8000", "--manifest-sha256", "0" * 64, "--dry-run"],
                           capture_output=True, text=True)
    assert wrong.returncode != 0 and "refusing to run under a different protocol" in wrong.stderr
