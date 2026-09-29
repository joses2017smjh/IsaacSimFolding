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
    assert [b["key"] for b in m["blocks"]] == ["rtx8000", "a40"]
    assert {b["constraint"] for b in m["blocks"]} == {"rtx8000", "a40"}
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
    # EBL: the baseline's early plan on 5 rows, every campaign; an1 restores it; a2/dq1 never
    assert {k for k in cal["baseline"] if k.startswith("ebl:")} == {f"ebl:dev0{i}" for i in (0, 1, 3, 4, 5)}
    assert all(cal["baseline"][f"ebl:dev0{i}"] == 8 for i in (0, 1, 3, 4, 5))
    assert not any(k.startswith("ebl:") for k in list(cal["a2"]) + list(cal["dq1"]))
    assert {k for k in cal["an1"] if k.startswith("ebl:")} >= {f"ebl:dev0{i}" for i in (0, 1, 3, 4, 5)}


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
def _write_episode(d: Path, *, settled: bool, gpu: str, horizon: int, seed: int, state="completed"):
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "gpu_model": gpu}))
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


def _fake_campaign(tmp_path: Path, cand_settled, base_settled, gpu="Quadro RTX 8000", block="rtx8000"):
    rows = B.rows_from(V8_MANIFEST["benchmark"])
    manifest = {"remeasure": rows, "pose_clusters": V8_MANIFEST["pose_clusters"],
                "blocks": [{"key": "rtx8000", "constraint": "rtx8000", "gpu_name": "Quadro RTX 8000"},
                           {"key": "a40", "constraint": "a40", "gpu_name": "NVIDIA A40"}]}
    labels = {"baseline": "baseline", "candidate": "an1-step000300"}
    for which, fn in (("candidate", cand_settled), ("baseline", base_settled)):
        for r in rows:
            _write_episode(tmp_path / "evaluation" / f"{labels[which]}-{block}" / "remeasure" / r["id"],
                           settled=fn(r), gpu=gpu, horizon=r["horizon"], seed=r["seed"])
    return manifest, labels


def test_block_results_counts_clauses_and_classification(tmp_path):
    cand = lambda r: r["horizon"] == 10 and r["development_pose_slot"] < 5 or (r["horizon"] == 50 and r["seed_block"] == 0)
    base = lambda r: r["horizon"] == 10 and r["development_pose_slot"] == 0 or (r["horizon"] == 50 and r["seed_block"] < 2)
    manifest, labels = _fake_campaign(tmp_path, cand, base)
    res = A.block_results(tmp_path, manifest, manifest["blocks"][0], labels)
    assert res["complete"] and res["valid"] == {"baseline": 48, "candidate": 48}
    assert res["h10"]["candidate"] == "15/24" and res["h10"]["baseline"] == "3/24"
    assert res["h10"]["clause_margin_ge_4_and_p_lt_0.05"]
    assert res["h50"]["candidate"] == "8/24" and res["h50"]["baseline"] == "16/24"
    assert not res["h50"]["clause_candidate_ge_baseline"] and not res["rule"]["v5_rule_holds"]
    assert res["h10"]["paired_mcnemar"] == {"candidate_only": 12, "baseline_only": 0,
                                            "one_sided_p": 1 / 2 ** 12}
    assert res["h50"]["behaviour"]["candidate"]["episodes"] == 24
    c = A.classify({"rtx8000": res, "a40": None})
    assert c["h10_gain"] == "partially measured" and c["h10_clause_holds_on"] == ["rtx8000"]
    assert A.classify({"rtx8000": res, "a40": res})["h10_gain"] == "replicates on both GPU models"


def test_wrong_gpu_model_or_unfinished_task_is_invalid_not_a_failure(tmp_path):
    manifest, labels = _fake_campaign(tmp_path, lambda r: True, lambda r: False, gpu="NVIDIA A40", block="rtx8000")
    res = A.block_results(tmp_path, manifest, manifest["blocks"][0], labels)
    assert res["valid"] == {"baseline": 0, "candidate": 0} and not res["complete"]
    assert A.classify({"rtx8000": res, "a40": None})["h10_gain"] == "partially measured"


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
    assert extras == [["--constraint=rtx8000"], ["--constraint=a40"]]
    assert not (tmp_path / "ledger").exists(), "a dry run writes nothing"


def test_deadline_parses_and_is_in_the_future_at_freeze():
    assert D.parse_utc(B.BLOCK_DEADLINE_UTC) > D.parse_utc("2026-09-29T00:00:00Z")


def test_task_dry_run_resolves_both_pinned_policies():
    if not FROZEN:
        pytest.skip("manifest not frozen yet")
    py = sys.executable
    for index in (0, 1):
        out = subprocess.run([py, str(ROOT / "scripts/run_matched_task.py"), "--campaign", str(ROOT),
                              "--phase", "remeasure", "--matched-index", str(index), "--run", "rtx8000",
                              "--gpu-model", "rtx8000", "--dry-run"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        req = json.loads(out.stdout)
        assert req["policy"] == POLICIES[index] and req["row"]["id"] == "dev00_h10_s97000"
        assert req["gpu"]["expected_name"] == "Quadro RTX 8000"
        assert "--seed" in req["command"] and req["command"][req["command"].index("--seed") + 1] == "97000"
