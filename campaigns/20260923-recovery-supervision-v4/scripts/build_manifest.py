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
V3 = ROOT.parent / "20260923-recovery-supervision-v3"
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
    # Scaled 3x over v3: same 8 training-only (garment, pose) combinations,
    # FRESH episode seeds, 6 roots spanning the episode, 8 fixed candidate
    # seeds per root, ALL at H10 -- v3 measured H10 and H50 branches equally
    # successful (16/64 each), and H10 labels are the deployment protocol's
    # own closed-loop behaviour, avoiding the stale open-loop suffixes the
    # review flagged in H50 labels.
    search = [pose_row(demos, inventory, g, k, 82000 + i, 600, f"search_{g.lower()}_key{k}",
                       horizon=10, trajectory_every=10)
              for i, (g, k) in enumerate([("Pant_Short_Seen_4", 0), ("Pant_Short_Seen_5", 0),
                                          ("Pant_Short_Seen_6", 0), ("Pant_Short_Seen_8", 0),
                                          ("Pant_Short_Seen_4", 1), ("Pant_Short_Seen_5", 1),
                                          ("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)])]
    # Development rows are v2's, unchanged, so v2's baseline run is a repeat.
    dev = v2["benchmark"]
    # The final untouched set is v3's, carried over VERBATIM: v3 never spent
    # it (no candidate qualified), so it remains unseen by every training and
    # selection decision across all campaigns.
    final = json.loads((V3 / "manifest.json").read_text())["final_test"]
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
        "campaign": "20260923-recovery-supervision-v4",
        "predecessor": {"campaign": "20260923-recovery-supervision-v3",
                        "result": ("complete negative result; found and repaired the bf16 optimizer "
                                   "defect, validated the recovery-search supervision source (32/128 "
                                   "settled), and identified supervision breadth + the episode-start "
                                   "anchor as the binding constraints"),
                        "final_commit": "aa9dcca"},
        "objective": ("improve settled H10 folding success by pairing 3x-scaled H10-only recovery "
                      "supervision with a whole-episode anchor -- the two factors v3's final report "
                      "names as the remaining limitation"),
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
            "roots": [60, 120, 180, 240, 300, 360],
            "candidates": [{"horizon": 10, "seed": s} for s in
                           (9201, 9202, 9203, 9204, 9205, 9206, 9207, 9208)],
            "obs_every": 10, "max_seconds": 8100,
            "why": ("From each student-visited root on a training-only garment, every candidate is "
                    "EXECUTED to the end of the episode and scored by the unchanged 60-step settle. "
                    "Roots span early states (before any irreversible failure) to late ones. The "
                    "candidate set is fixed; every attempt is recorded; only settled successes label."),
            "label": ("observation rendered at a state + the next 50 actions actually executed from it, "
                      "on a continuation that reached settled success"),
            "coverage_gate": {"min_successful_branches": 18, "min_distinct_roots": 10, "min_rows": 4},
            "v3_labels_not_imported": ("deliberate: half of v3's branch labels are H50 continuations "
                                       "carrying stale open-loop suffixes, a flagged untested risk; "
                                       "v4 keeps an H10-only corpus matched to the deployment protocol"),
            "if_gate_fails": ("do not train; analyse the recorded per-step condition traces for a "
                              "validated progress signal (route 2) and proceed only by a committed plan"),
        },
        "training": {
            "steps": 300, "batch_size": 4, "rollout_fraction": 0.5, "lr": 1e-5, "optimizer": "AdamW",
            "unfreeze": "boundary", "seed": 4242,
            "grad_accum": 8,
            "checkpoint_every": 25,
            "checkpoint_rule": "latest_guard_passing",
            "checkpoint_rule_note": ("preregistered BEFORE any v4 result: v3 attempt 2 breached "
                                     "retention at every 100-step checkpoint while both training "
                                     "losses fell, so v4 checkpoints every 25 optimizer steps and "
                                     "selects the LATEST guard-passing checkpoint, else none"),
            "anchor_mode": "whole_episode",
            "anchor_mode_note": ("THE preregistered recipe factor: the episode-start anchor -- the "
                                 "first 8 frames of each file, the home-to-ready snap -- protected "
                                 "nothing the retention guard measures; whole-episode frames span "
                                 "approach, grasp, fold and release on the same 16 anchor files"),
            "anchor": dict(v2["training"]["anchor"], files=16, frames=12),
            "retention": {"files": retention, "frames_per_episode": 16, "tolerance": 1.10,
                          "definition": "whole demonstration episodes; garments disjoint from anchor, training, development and final sets"},
            "anchor_fit_metric": {"glob": v2["training"]["heldout"]["glob"], "files": 4, "frames": 8,
                                  "role": "the v2 guard, reported only"},
            "held_fixed_vs_v2": "optimisation, parameters, anchor and batch mix are identical; only the supervision changes",
            "weighting": "uniform over validated recovery examples; no episode-weighted AWR",
        },
        "evaluation": {
            "development_rows": "v2's 16 rows (8 poses x H10/H50, seeds 200-207)",
            "baseline_h10_runs": [
                "measured run 1 (v2 campaign): 2/8 settled, mean conditions 3.000",
                "measured run 2 (v3 campaign): 0/8 settled, mean conditions 2.500",
                "pooled preregistered baseline: 2/16; no new baseline run is spent"],
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
        "automation": {
            "attempt2_plan_deadline_minutes": 480,
            "note": ("if attempt 1 concludes without an improved candidate and no committed "
                     "plans/attempt2.json appears within this window, the driver finalizes on "
                     "the baseline; previously a 120-minute driver hardcode absent from the "
                     "manifest")},
        "amendments": ["amendments/2026-09-23-attempt2-budget.md"],
        "budget": {"gpu_tasks": 72, "gpu_hours": 28.0,
                   "amended": "2026-09-23-task-proxy: 60 -> 72, hours unchanged",
                   "reserve": {"final_set": 16},
                   "source": ("the remaining original allocation, measured from reconciled ledgers: "
                              "v2+v3 used 136 tasks and 12.17 of 45 GPU-hours. GPU-HOURS ARE THE "
                              "BINDING CAP (28.0 of the 32.8 remaining); the task count is an "
                              "anti-runaway proxy only, and cumulative tasks may exceed the informal "
                              "170 while hours stay within 45 -- declared upfront, not amended later"),
                   "rollout_timeout_seconds": {"benchmark": 1500, "recovery": 9600, "smoke": 1500,
                                               "collection": 1500, "boundary": 3000}},
        "second_attempt": "only from an explicit committed plan written from evidence; no default ladder",
        "recovery_search": search,
        "benchmark": dev,
        "final_test": final,
        "frozen_test": v2["frozen_test"],
        "recovery_smoke": [pose_row(demos, inventory, "Pant_Short_Seen_4", 0, 82900, 120,
                                    "smoke_search_pant_short_seen_4_key0", horizon=10, trajectory_every=10)],
        "recovery_smoke_config": {"roots": [50], "candidates": [{"horizon": 10, "seed": 9201},
                                                                {"horizon": 10, "seed": 9202}],
                                  "obs_every": 10, "max_seconds": 900},
        "collection": [], "collection_expansion": [], "smoke": v2["smoke"],
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "search": len(search),
                      "final": len(final)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
