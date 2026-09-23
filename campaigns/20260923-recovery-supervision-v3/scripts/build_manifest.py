"""Freeze the v3 recovery-supervision campaign before any GPU task runs.

Same discipline as v2: every executed source hashed from the COMMITTED blob,
refusal on a dirty tree, refusal to overwrite a frozen manifest. What is new
is the protocol: a bounded recovery search, a coverage gate on it, a
retention set held out by whole episode and garment, and repeated matched
evaluation with predeclared decision rules, so no single favourable run can
carry a claim.
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
DATA = WORKSPACE / "lehome-data"
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))


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


def pose_row(demos, inventory, garment, key, seed, steps, tag, **extra):
    demo = demos[garment][str(key)]
    if len(set(demo["scale"])) != 1:
        raise ValueError(f"{garment} {key} anisotropic scale")
    row = {"id": tag, "garment": garment, "asset_config": inventory[garment]["config"],
           "pose_key": key, "match_pose": ":".join(format(v, ".10g") for v in demo["object_initial_pose"]),
           "match_scale": demo["scale"][0], "seed": seed, "steps": steps}
    row.update(extra)
    return row


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
    inventory = {g["garment_id"]: g for g in
                 json.loads((PILOT / "audit/garment_inventory.json").read_text())["garments"]}
    demos_path = DATA / "Datasets/example/four_types_merged/meta/garment_info.json"
    demos = json.loads(demos_path.read_text())
    ckpt = DATA / "outputs/train/bc_smolvla_raster_ft_full"

    # Recovery search on TRAINING-ONLY garments (never development, never any
    # test set), mirroring the development pose types: P_A x4, P_C x2, P_B x2.
    search = [pose_row(demos, inventory, g, k, 81000 + i, 600, f"search_{g.lower()}_key{k}",
                       horizon=10, trajectory_every=10)
              for i, (g, k) in enumerate([("Pant_Short_Seen_4", 0), ("Pant_Short_Seen_5", 0),
                                          ("Pant_Short_Seen_6", 0), ("Pant_Short_Seen_8", 0),
                                          ("Pant_Short_Seen_4", 1), ("Pant_Short_Seen_5", 1),
                                          ("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)])]
    # Development rows are v2's, unchanged, so v2's baseline run is a repeat.
    dev = v2["benchmark"]
    # A NEW untouched final set: garments never trained on by any campaign, at
    # pose keys never evaluated by any campaign. v2's frozen-test result on
    # key 0 of these garments stands as recorded and is not rerun.
    final = [pose_row(demos, inventory, g, k, 95000 + i, 600, f"final{i:02d}_h10_{g}_key{k}", horizon=10)
             for i, (g, k) in enumerate([(g, k) for g in ("Pant_Short_Seen_1", "Pant_Short_Seen_2")
                                          for k in (1, 2, 3, 4)])]
    trained = {r["garment"] for r in search} | {"Pant_Short_Seen_0", "Pant_Long_Seen_0",
                                               "Top_Short_Seen_0", "Top_Long_Seen_0"}
    if {r["garment"] for r in final} & trained or {r["garment"] for r in dev} & {r["garment"] for r in search}:
        raise SystemExit("separation violated")
    used = {(r["garment"], int(r["pose_key"])) for k in ("benchmark", "frozen_test") for r in v2[k]}
    if {(r["garment"], r["pose_key"]) for r in final} & used:
        raise SystemExit("final set reuses an evaluated (garment, pose)")
    retention = [str(DATA / "storm_capture100" / f) for f in ("ep833.npz", "ep864.npz", "ep176.npz", "ep354.npz")]

    manifest = {
        "schema_version": 1,
        "campaign": "20260923-recovery-supervision-v3",
        "predecessor": {"campaign": "20260922-closed-loop-training-v2", "result": "complete negative result",
                        "final_commit": "857d88e"},
        "objective": "improve settled H10 folding success with simulator-validated recovery supervision",
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": {"path": str(demos_path), "sha256": file_sha(demos_path)},
        "baseline_checkpoint": {"path": str(ckpt), "immutable": True,
                                "sha256": {p.name: file_sha(p) for p in sorted(ckpt.iterdir()) if p.is_file()}},
        "recovery": {
            "roots": [50, 150, 250, 350],
            "candidates": [{"horizon": 10, "seed": 9101}, {"horizon": 10, "seed": 9102},
                           {"horizon": 50, "seed": 9103}, {"horizon": 50, "seed": 9104}],
            "obs_every": 10, "max_seconds": 5400,
            "why": ("From each student-visited root on a training-only garment, every candidate is "
                    "EXECUTED to the end of the episode and scored by the unchanged 60-step settle. "
                    "Roots span early states (before any irreversible failure) to late ones. The "
                    "candidate set is fixed; every attempt is recorded; only settled successes label."),
            "label": ("observation rendered at a state + the next 50 actions actually executed from it, "
                      "on a continuation that reached settled success"),
            "coverage_gate": {"min_successful_branches": 6, "min_distinct_roots": 4, "min_rows": 3},
            "if_gate_fails": ("do not train; analyse the recorded per-step condition traces for a "
                              "validated progress signal (route 2) and proceed only by a committed plan"),
        },
        "training": {
            "steps": 300, "batch_size": 4, "rollout_fraction": 0.5, "lr": 1e-5, "optimizer": "AdamW",
            "unfreeze": "boundary", "seed": 4242, "checkpoint_every": 100,
            "anchor": v2["training"]["anchor"],
            "retention": {"files": retention, "frames_per_episode": 16, "tolerance": 1.10,
                          "definition": "whole demonstration episodes; garments disjoint from anchor, training, development and final sets"},
            "anchor_fit_metric": {"glob": v2["training"]["heldout"]["glob"], "files": 4, "frames": 8,
                                  "role": "the v2 guard, reported only"},
            "held_fixed_vs_v2": "optimisation, parameters, anchor and batch mix are identical; only the supervision changes",
            "weighting": "uniform over validated recovery examples; no episode-weighted AWR",
        },
        "evaluation": {
            "development_rows": "v2's 16 rows (8 poses x H10/H50, seeds 200-207)",
            "baseline_h10_runs": ["v2 run (2/8 settled, recorded)", "one fresh v3 run"],
            "baseline_h50_reference": "v2 run 4/8 (the horizon pilot also measured 4/8)",
            "candidate_runs": "screen: one H10 run; confirmation: a second H10 run and one H50 run",
            "screen_rule": "proceed to confirmation only if screen H10 settled >= 4/8",
            "improvement_rule": ("pooled candidate H10 (16 episodes) exceeds pooled baseline H10 (16 episodes) "
                                 "by >= 4 episodes with one-sided Fisher exact p < 0.05, AND candidate H50 >= 4/8, "
                                 "AND the retention guard passes, AND the checkpoint reloads"),
            "target": "both candidate H10 runs >= 6/8 settled (stretch: both 8/8), with the improvement rule",
            "final_set": "run once for the qualifying candidate and the baseline, only if the improvement rule holds",
            "final_rows": final,
            "v2_frozen_test": "unchanged, not rerun",
        },
        "budget": {"gpu_tasks": 65, "gpu_hours": 36.9,
                   "reserve": {"baseline_repeat": 8, "final_set": 16},
                   "source": "unused portion of v2's allocation: 170-105 tasks, 45-8.1 GPU-hours",
                   "rollout_timeout_seconds": {"benchmark": 1500, "recovery": 7000, "smoke": 1500,
                                               "collection": 1500, "boundary": 3000}},
        "second_attempt": "only from an explicit committed plan written from evidence; no default ladder",
        "recovery_search": search,
        "benchmark": dev,
        "final_test": final,
        "frozen_test": v2["frozen_test"],
        "recovery_smoke": [pose_row(demos, inventory, "Pant_Short_Seen_4", 0, 81900, 120,
                                    "smoke_search_pant_short_seen_4_key0", horizon=10, trajectory_every=10)],
        "recovery_smoke_config": {"roots": [50], "candidates": [{"horizon": 10, "seed": 9101},
                                                                {"horizon": 50, "seed": 9103}],
                                  "obs_every": 10, "max_seconds": 900},
        "collection": [], "collection_expansion": [], "smoke": v2["smoke"],
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "search": len(search),
                      "final": len(final)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
