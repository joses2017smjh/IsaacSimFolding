"""Freeze the v5 matched-comparison campaign before any GPU task runs.

Same discipline as v2-v4: every executed source hashed from the COMMITTED
blob, refusal on a dirty tree, refusal to overwrite a frozen manifest. The
protocol is evaluation only -- two pinned checkpoints, the same predeclared
rows, two runs per policy per horizon in interleaved Slurm arrays, and the
unchanged improvement rule applied to numbers measured in the same campaign.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
WORKSPACE = REPO.parent
PILOT = ROOT.parent / "20260921-horizon-pilot"
V2 = ROOT.parent / "20260922-closed-loop-training-v2"
V3 = ROOT.parent / "20260923-recovery-supervision-v3"
V4 = ROOT.parent / "20260923-recovery-supervision-v4"
DATA = WORKSPACE / "lehome-data"
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))
CANDIDATE = V4 / "training/a2/checkpoints/step_000250"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def committed_sha(commit: str, rel: str) -> str:
    return hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"],
                                         check=True, capture_output=True).stdout).hexdigest()


def executed_sources() -> list[str]:
    rels = []
    for directory in ("scripts", "slurm", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".sbatch", ".sh"):
                rels.append(str(path.relative_to(REPO)))
    for base in (PILOT / "src" / "lehome_fold", REPO / "src" / "lehome_fold"):
        rels += [str(p.relative_to(REPO)) for p in sorted(base.glob("*.py"))]
    return sorted(set(rels))


def main() -> int:
    out = ROOT / "manifest.json"
    if out.exists():
        raise SystemExit("refusing to mutate a frozen manifest; archive it to manifests/ first")
    commit = git("rev-parse", "HEAD")
    sources = executed_sources()
    dirty = {line[3:] for line in git("status", "--porcelain").splitlines()}
    if dirty & set(sources):
        raise SystemExit("executed sources are uncommitted:\n  " + "\n  ".join(sorted(dirty & set(sources))))
    executed = {}
    for rel in sources:
        sha = committed_sha(commit, rel)
        if sha != file_sha(REPO / rel):
            raise SystemExit(f"{rel}: working tree differs from commit")
        executed[rel] = sha

    v2 = json.loads((V2 / "manifest.json").read_text())
    v3 = json.loads((V3 / "manifest.json").read_text())
    v4 = json.loads((V4 / "manifest.json").read_text())
    demos_path = DATA / "Datasets/example/four_types_merged/meta/garment_info.json"
    baseline = DATA / "outputs/train/bc_smolvla_raster_ft_full"

    # Rows: v2's development rows and v3's untouched final set, both VERBATIM.
    dev = v2["benchmark"]
    final = v3["final_test"]
    smoke = v2["smoke"]
    if final != v4["final_test"]:
        raise SystemExit("the final set drifted between v3 and v4")
    if len(dev) != 16 or {int(r["horizon"]) for r in dev} != {10, 50} or len(final) != 8:
        raise SystemExit("row layout changed")
    trained = {r["garment"] for r in v4["recovery_search"]} | {
        "Pant_Short_Seen_0", "Pant_Long_Seen_0", "Top_Short_Seen_0", "Top_Long_Seen_0"}
    if {r["garment"] for r in final} & trained:
        raise SystemExit("final set touches a trained garment")
    used = {(r["garment"], int(r["pose_key"])) for k in ("benchmark", "frozen_test") for r in v2[k]}
    if {(r["garment"], r["pose_key"]) for r in final} & used:
        raise SystemExit("final set reuses an evaluated (garment, pose)")

    # The candidate: v4's a2-step000250, pinned file by file and cross-checked
    # against the hashes v4's trainer recorded when it saved the checkpoint.
    record = json.loads((CANDIDATE / "checkpoint.json").read_text())
    cand_sha = {p.name: file_sha(p) for p in sorted(CANDIDATE.iterdir()) if p.is_file()}
    for name, sha in record["checkpoint_sha256"].items():
        if cand_sha.get(name) != sha:
            raise SystemExit(f"candidate {name} changed since v4 recorded it")
    if record["step"] != 250 or not record["heldout_gate"]:
        raise SystemExit("candidate record is not the guard-passing step-250 checkpoint")
    training_json = V4 / "training/a2/training.json"
    ck250 = next(c for c in json.loads(training_json.read_text())["checkpoints"] if int(c["step"]) == 250)
    reload_json = V4 / "audit/reload_a2-step000250.json"
    reload = json.loads(reload_json.read_text())
    if not (ck250["heldout_gate"] and reload["finite"] and reload["config_matches_baseline"]):
        raise SystemExit("the candidate's v4 provenance does not say what the rule needs")
    v4_state = json.loads((V4 / "ledger/driver_state.json").read_text())
    v4_cand = v4_state["attempts"]["2"]["candidate"]

    manifest = {
        "schema_version": 1,
        "campaign": "20260924-matched-comparison-v5",
        "kind": "evaluation only: no training, no search, no new checkpoint",
        "predecessor": {"campaign": "20260923-recovery-supervision-v4",
                        "result": ("candidate a2-step000250: H10 8/16 vs 2/16 (p = 0.027) but H50 3/8 vs "
                                   "4/8 against baseline numbers from EARLIER campaigns; no improvement "
                                   "under the rule; the report names a same-wave comparison as the next step"),
                        "final_commit": git("log", "-1", "--format=%H", "--", str((V4 / "REPORT.md").relative_to(REPO)))},
        "objective": ("decide, with baseline and candidate measured side by side in the same Slurm "
                      "arrays on the same rows, whether a2-step000250 improves settled folding under "
                      "the unchanged preregistered rule; and record the resulting folds as media"),
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "runner_provenance": "byte-identical to v4's snapshot of the horizon-pilot runner",
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": {"path": str(demos_path), "sha256": file_sha(demos_path)},
        "baseline_checkpoint": {"path": str(baseline), "label": "baseline", "immutable": True,
                                "sha256": {p.name: file_sha(p) for p in sorted(baseline.iterdir()) if p.is_file()}},
        "candidate_checkpoint": {
            "path": str(CANDIDATE), "label": "a2-step000250", "immutable": True, "sha256": cand_sha,
            "provenance": {
                "campaign": "20260923-recovery-supervision-v4", "attempt": 2,
                "recipe": ("3x-scaled H10-only recovery supervision (81/384 settled), whole-episode anchor, "
                           "fp32 master weights, effective batch 32, lr 3.3e-6, latest guard-passing checkpoint"),
                "retention_guard": {"passed": bool(ck250["heldout_gate"]), "loss": ck250["heldout_loss"],
                                    "baseline_loss": ck250["heldout_loss_baseline"], "tolerance": 1.10,
                                    "record": str(training_json), "sha256": file_sha(training_json)},
                "reload": {"passed": bool(reload["finite"] and reload["config_matches_baseline"]),
                           "record": str(reload_json), "sha256": file_sha(reload_json)},
                "v4_measured": {k: v4_cand.get(k) for k in ("h10_r1", "h10_r2", "h50")},
            },
        },
        "protocol": {
            "rows": "v2's 16 development rows (8 poses x H10/H50, seeds 200-207), verbatim",
            "runs_per_policy_per_horizon": 2,
            "run_labels": ["r1", "r2"],
            "same_wave": ("each run is ONE Slurm array of 32 tasks over the 16 rows in which even indices "
                          "run the baseline and odd indices the candidate on the same row "
                          "(run_rollout_task.matched_index), throttled at 8 concurrent so every batch holds "
                          "four of each; r1 and r2 are submitted by the same driver tick"),
            "improvement_rule": {
                "h10": "pooled candidate H10 (16) vs pooled baseline H10 (16): margin >= 4 AND one-sided Fisher exact p < 0.05",
                "h50": "pooled candidate H50 (16) >= pooled baseline H50 (16), both measured in this campaign (non-regression)",
                "retention_guard": "as recorded in v4 for this checkpoint (candidate_checkpoint.provenance)",
                "reload": "as recorded in v4 for this checkpoint",
                "all_rows_valid": "every one of the 64 development episodes completed as a valid rollout",
                "function": "scripts/driver.py::matched_verdict, tested in tests/test_v5.py",
            },
            "n_is_fixed": ("two runs per policy per horizon, whatever they show; there is no third run, "
                           "no rerun of a failed clause, and no per-run selection"),
            "final_set": ("v3's untouched 8 rows, run once for BOTH policies in one matched array, "
                          "only if the improvement rule holds; descriptive, no further rule"),
            "baseline_numbers_from_earlier_campaigns": "not pooled in; v5's own baseline runs are the only baseline",
        },
        "media": {"gif_every": 6,
                  "note": ("the runner writes per-camera GIFs, an MP4 and snapshots for every episode; "
                           "gif_every only selects which rendered frames are kept and cannot change what "
                           "the policy observes; the gallery shows settled successes only and counts "
                           "every other episode alongside")},
        "budget": {"gpu_tasks": 96, "gpu_hours": 60.0, "reserve": {"final_set": 16},
                   "note": ("anti-runaway bounds only: the user lifted the GPU-hour constraint on "
                            "2026-09-24 ('at worst they just stay in queue'); the full path is "
                            "2 smoke + 64 development + 16 final + one retry wave"),
                   "rollout_timeout_seconds": {"benchmark": 1500, "smoke": 1500}},
        "smoke": smoke,
        "benchmark": dev,
        "final_test": final,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "development_rows": len(dev),
                      "final_rows": len(final), "candidate": str(CANDIDATE)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
