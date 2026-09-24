"""Freeze the v6 pose-balanced campaign before any GPU task runs.

Same discipline as v2-v5: every executed source hashed from the COMMITTED
blob, refusal on a dirty tree, refusal to overwrite a frozen manifest. One
data factor changes against v4 attempt 2 -- supervision composition, from
the per-pose transfer analysis in analysis/pose-transfer.json -- and
everything downstream is v5's matched protocol, unchanged.
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
V3 = ROOT.parent / "20260923-recovery-supervision-v3"
V4 = ROOT.parent / "20260923-recovery-supervision-v4"
V5 = ROOT.parent / "20260924-matched-comparison-v5"
DATA = WORKSPACE / "lehome-data"
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))

# Targeted search rows: training-only garments at the EXACT development poses
# (match_pose identity; pose keys differ per garment -- see pose_clusters).
P_B_ROWS = [("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)]         # the only exact P_B rows
P_A_ROWS = [("Pant_Short_Seen_4", 0), ("Pant_Short_Seen_5", 0),
            ("Pant_Short_Seen_6", 0), ("Pant_Short_Seen_8", 0)]
P_B_SEEDS_PER_ROW, P_A_SEEDS_PER_ROW = 4, 2
P_B_ROOTS = [30, 60, 90, 120, 150, 180]        # v4: P_B roots >= 240 settled 2 of 48
DEFAULT_ROOTS = [60, 120, 180, 240, 300, 360]  # v4's, kept for P_A (late unfolding)


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


def pose_id(match_pose: str) -> tuple:
    return tuple(round(float(x), 4) for x in match_pose.split(":"))


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

    v3 = json.loads((V3 / "manifest.json").read_text())
    v4 = json.loads((V4 / "manifest.json").read_text())
    v5 = json.loads((V5 / "manifest.json").read_text())
    inventory = {g["garment_id"]: g for g in
                 json.loads((PILOT / "audit/garment_inventory.json").read_text())["garments"]}
    demos_path = DATA / "Datasets/example/four_types_merged/meta/garment_info.json"
    demos = json.loads(demos_path.read_text())
    baseline = DATA / "outputs/train/bc_smolvla_raster_ft_full"

    dev, final = v5["benchmark"], v5["final_test"]
    if final != v3["final_test"] or dev != v4["benchmark"]:
        raise SystemExit("development or final rows drifted from their sources")
    by_id = {r["id"][:5]: r for r in dev if "_h10_" in r["id"]}
    clusters = {"P_A": by_id["dev00"]["match_pose"], "P_B": by_id["dev01"]["match_pose"],
                "P_C": by_id["dev02"]["match_pose"]}
    for name, members in (("P_A", ("dev00", "dev04", "dev06")), ("P_B", ("dev01", "dev03")),
                          ("P_C", ("dev02", "dev05", "dev07"))):
        if {pose_id(by_id[m]["match_pose"]) for m in members} != {pose_id(clusters[name])}:
            raise SystemExit(f"development pose {name} is not one match_pose")

    search, i = [], 0
    for pose, grid, per_row, roots in (("P_B", P_B_ROWS, P_B_SEEDS_PER_ROW, P_B_ROOTS),
                                       ("P_A", P_A_ROWS, P_A_SEEDS_PER_ROW, None)):
        for g, k in grid:
            for _ in range(per_row):
                seed = 83000 + i
                row = pose_row(demos, inventory, g, k, seed, 600,
                               f"search_{pose.lower()}_{g.lower()}_key{k}_s{seed}",
                               horizon=10, trajectory_every=10, pose=pose)
                if roots:
                    row["recovery_roots"] = roots
                if pose_id(row["match_pose"]) != pose_id(clusters[pose]):
                    raise SystemExit(f"{row['id']} is not at the exact {pose} pose")
                search.append(row)
                i += 1
    search_g = {r["garment"] for r in search}
    if search_g & {r["garment"] for r in dev} or search_g & {r["garment"] for r in final}:
        raise SystemExit("separation violated: a search garment is evaluated")
    used_seeds = ({r["seed"] for r in v4["recovery_search"]} | {r["seed"] for r in v3["recovery_search"]}
                  | {r["seed"] for r in dev} | {r["seed"] for r in final})
    if {r["seed"] for r in search} & used_seeds:
        raise SystemExit("a search seed was used before")

    reused = V4 / "datasets/recovery.npz"
    v4_prov = json.loads((V4 / "analysis/recovery-dataset-provenance.json").read_text())
    reused_sha = file_sha(reused)
    if reused_sha != v4_prov["dataset_sha256"]:
        raise SystemExit("v4's corpus no longer matches its own provenance")
    v4_train = v4["training"]

    manifest = {
        "schema_version": 1,
        "campaign": "20260924-pose-balanced-v6",
        "kind": "one bounded training attempt: targeted supervision + pose-balanced sampling",
        "predecessor": {"campaign": "20260924-matched-comparison-v5",
                        "result": ("matched: a2-step000250 H10 8/16 vs baseline 5/16 (p = 0.24), H50 7/16 "
                                   "vs 7/16 -- not improved; gains on P_C (6/6 vs 1/6), P_B lost (0/8 vs 4/8)"),
                        "final_commit": git("log", "-1", "--format=%H", "--",
                                            str((V5 / "REPORT.md").relative_to(REPO)))},
        "analysis": "analysis/pose-transfer.json",
        "objective": ("keep a2's P_C gain and remove its P_B regression by supplying dense validated "
                      "supervision at the under-supplied poses (P_B, then P_A) and sampling every pose "
                      "equally; decide under v5's unchanged matched rule"),
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "runner_provenance": "byte-identical to v4's (and v5's) snapshot of the horizon-pilot runner",
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": {"path": str(demos_path), "sha256": file_sha(demos_path)},
        "baseline_checkpoint": {"path": str(baseline), "label": "baseline", "immutable": True,
                                "sha256": {p.name: file_sha(p) for p in sorted(baseline.iterdir()) if p.is_file()}},
        "pose_clusters": clusters,
        "pose_clusters_note": ("grouped by match_pose identity, never by pose key: key 0 is P_A on garments "
                               "0/7/9 but P_C on garment 3. P_A = dev00/04/06, P_B = dev01/03, P_C = dev02/05/07. "
                               "7 of the 8 final rows are poses absent from dev and from every search."),
        "recovery": {
            "roots": DEFAULT_ROOTS,
            "candidates": [{"horizon": 10, "seed": s} for s in range(9301, 9309)],
            "obs_every": 10, "max_seconds": 8100,
            "why": ("v4's search machinery unchanged (baseline as student, every candidate executed to a "
                    "settled verdict, every attempt recorded, labels from settled successes only), aimed at "
                    "the poses v5 showed under-supplied. P_B rows search early roots: in v4, P_B roots at "
                    "or after step 240 settled 2 of 48 attempts, roots 60-180 settled 12 of 48."),
            "coverage_gate": {
                "P_B": {"min_successful_branches": 24, "min_distinct_roots": 8, "min_rows": 3},
                "P_A": {"min_successful_branches": 24, "min_distinct_roots": 8, "min_rows": 3},
            },
            "coverage_gate_note": ("applied per pose to the NEW search only, before any image is read; "
                                   "thresholds are about half the yield v4's per-root rates predict "
                                   "(P_B ~58, P_A ~46), so a collapse fails and the attempt stops untrained"),
        },
        "recovery_search": search,
        "corpus": {
            "reuse": {"path": str(reused), "sha256": reused_sha, "samples": v4_prov["samples"],
                      "manifest": str(V4 / "manifest.json"),
                      "provenance": str(V4 / "analysis/recovery-dataset-provenance.json"),
                      "per_pose_samples": {"P_A": 838, "P_B": 580, "P_C": 1572}},
            "balance": ("awr_weight = N / (P * n_pose): each pose supplies 1/P of rollout draws; the trainer "
                        "samples proportional to awr_weight and is unchanged"),
        },
        "training": dict(
            {k: v for k, v in v4_train.items() if k not in ("held_fixed_vs_v2", "weighting")},
            lr=3.3e-6, fit_precondition=True,
            recipe_source="v4 attempt 2 (plans/attempt2.json over v4's manifest training block), verbatim",
            changed_factor=("supervision composition only: v4's corpus + the targeted P_B/P_A search, "
                            "pose-balanced; recipe, init (untouched baseline), steps, checkpoint rule and "
                            "retention guard identical to a2"),
            weighting="pose-balanced; see corpus.balance"),
        "protocol": {
            "rows": "v2's 16 development rows (8 poses x H10/H50, seeds 200-207), verbatim",
            "runs_per_policy_per_horizon": 2, "run_labels": ["r1", "r2"],
            "same_wave": ("each run is ONE Slurm array of 32 tasks in which even indices run the baseline and "
                          "odd the candidate on the same row, throttled at 8 concurrent; r1 and r2 are "
                          "submitted by the same driver tick (v5's protocol)"),
            "improvement_rule": v5["protocol"]["improvement_rule"],
            "n_is_fixed": "two runs per policy per horizon, whatever they show; no third run, no second attempt",
            "final_set": ("v3's untouched 8 rows (v5's final_test, verbatim), run once for BOTH policies in one "
                          "matched array, only if the improvement rule holds; descriptive"),
            "baseline_numbers_from_earlier_campaigns": "not pooled in; v6's own baseline runs are the only baseline",
        },
        "media": {"gif_every": 12},
        "budget": {"gpu_tasks": 150, "gpu_hours": 40.0, "reserve": {"final_set": 16},
                   "expected": "search 16 (~9 GPU-h) + train/reload/fit 3 (~1) + evaluation 64 (~5) + final 16 (~1.2)",
                   "note": ("anti-runaway bounds; the user lifted the GPU-hour constraint on 2026-09-24. For "
                            "reference, v2-v5 used 24.8 of the original 45 GPU-hours."),
                   "rollout_timeout_seconds": {"benchmark": 1500, "recovery": 9600, "smoke": 1500}},
        "smoke": v5["smoke"],
        "benchmark": dev,
        "final_test": final,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "search_rows": len(search),
                      "P_B_rows": sum(r["pose"] == "P_B" for r in search),
                      "P_A_rows": sum(r["pose"] == "P_A" for r in search)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
