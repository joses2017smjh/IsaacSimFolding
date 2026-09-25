"""Freeze the v7 depth-qualified campaign before any GPU task runs.

Same discipline as v2-v6: every executed source hashed from the COMMITTED
blob, refusal on a dirty tree, refusal to overwrite a frozen manifest. One
data factor changes against v6 -- the label criterion (settled AND landed
deep, from a fresh search that records landing telemetry), chosen by the
verified v6 diagnosis (v6/analysis/diagnosis/synthesis.json) -- and
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
V6 = ROOT.parent / "20260924-pose-balanced-v6"
DATA = WORKSPACE / "lehome-data"
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))

# Fresh search rows on the training-only garments, at the EXACT development
# poses (match_pose identity; pose keys differ per garment). All rows use the
# early roots where v6 found reach being learned. Large-garment P_B rows are
# off-metadata placements: Seen_4/5 list no P_B key, so the P_B match_pose is
# pinned on them -- every earlier P_B label came from the small Seen_6/8, and
# the diagnosis traced dev03's (large Seen_3) P_B failure to that.
GRID = [("P_A", [("Pant_Short_Seen_4", 0), ("Pant_Short_Seen_5", 0),
                 ("Pant_Short_Seen_6", 0), ("Pant_Short_Seen_8", 0)], 2),
        ("P_C", [("Pant_Short_Seen_4", 1), ("Pant_Short_Seen_5", 1),
                 ("Pant_Short_Seen_6", 1), ("Pant_Short_Seen_8", 1)], 2),
        ("P_B", [("Pant_Short_Seen_6", 2), ("Pant_Short_Seen_8", 2)], 4)]
OFF_METADATA_P_B = [("Pant_Short_Seen_4", 2), ("Pant_Short_Seen_5", 2)]   # (garment, seeds)
GARMENT_CLASS = {"Pant_Short_Seen_4": "large", "Pant_Short_Seen_5": "large",
                 "Pant_Short_Seen_6": "small", "Pant_Short_Seen_8": "small"}
ROOTS = [30, 60, 90, 120, 150, 180]
DEPTH_RULES = {"landed_deep_cm": 1.0, "mechanism_p": 0.05, "cut_ladder_cm": [1.5, 1.0, 0.5, 0.0],
               "first_fold_max_step": 450, "min_total_labels": 3000, "max_excluded_rows": 1,
               "yield_curve_cm": [0.0, 0.5, 1.0, 1.5, 2.0],
               "landing": {"gripper_clear_m": 0.12, "lift_max_m": 0.05, "window": 10}}


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
    for pose, grid, per_row in GRID:
        for g, k in grid:
            for _ in range(per_row):
                seed = 84000 + i
                row = pose_row(demos, inventory, g, k, seed, 600,
                               f"search_{pose.lower()}_{g.lower()}_key{k}_s{seed}",
                               horizon=10, trajectory_every=10, pose=pose,
                               garment_class=GARMENT_CLASS[g], off_metadata_pose=False)
                if pose_id(row["match_pose"]) != pose_id(clusters[pose]):
                    raise SystemExit(f"{row['id']} is not at the exact {pose} pose")
                search.append(row)
                i += 1
    for g, seeds in OFF_METADATA_P_B:
        if any(pose_id(":".join(format(v, ".10g") for v in d["object_initial_pose"])) == pose_id(clusters["P_B"])
               for d in demos[g].values()):
            raise SystemExit(f"{g} lists the P_B pose; it is not an off-metadata placement")
        for _ in range(seeds):
            seed = 84000 + i
            row = pose_row(demos, inventory, g, 0, seed, 600, f"search_p_b_{g.lower()}_pinned_s{seed}",
                           horizon=10, trajectory_every=10, pose="P_B", garment_class=GARMENT_CLASS[g],
                           off_metadata_pose=True)
            row["match_pose"], row["pose_key"] = clusters["P_B"], None
            search.append(row)
            i += 1
    search_g = {r["garment"] for r in search}
    if search_g & {r["garment"] for r in dev} or search_g & {r["garment"] for r in final}:
        raise SystemExit("separation violated: a search garment is evaluated")
    v6m = json.loads((V6 / "manifest.json").read_text())
    used_seeds = ({r["seed"] for r in v6m["recovery_search"]} | {r["seed"] for r in v4["recovery_search"]}
                  | {r["seed"] for r in v3["recovery_search"]} | {r["seed"] for r in dev} | {r["seed"] for r in final})
    if {r["seed"] for r in search} & used_seeds or len({r["seed"] for r in search}) != len(search):
        raise SystemExit("a search seed was used before or repeats")
    smoke_row = pose_row(demos, inventory, "Pant_Short_Seen_4", 0, 84900, 150,
                         "smoke_search_p_b_pant_short_seen_4_pinned", horizon=10, trajectory_every=10,
                         pose="P_B", garment_class="large", off_metadata_pose=True)
    smoke_row["match_pose"], smoke_row["pose_key"] = clusters["P_B"], None
    v4_train = v4["training"]

    manifest = {
        "schema_version": 1,
        "campaign": "20260925-depth-qualified-v7",
        "kind": "one bounded training attempt: depth-qualified supervision from a fresh telemetry search",
        "predecessor": {"campaign": "20260924-pose-balanced-v6",
                        "result": ("matched: pb1-step000200 H10 4/16 vs 5/16, H50 8/16 vs 7/16 -- not improved; "
                                   "reached the fold 13/16 at H10 but landed it shallow (median 1.0 cm vs "
                                   "baseline 2.45); P_B H50 lost on the large garment"),
                        "final_commit": git("log", "-1", "--format=%H", "--",
                                            str((V6 / "REPORT.md").relative_to(REPO)))},
        "analysis": str(V6 / "analysis/diagnosis/synthesis.json"),
        "objective": ("keep v6's early reach and land the fold deep: labels only from branches that "
                      "settled with terminal closure margin >= 1.5 cm and folded by step 450, gated on a "
                      "preregistered check that landing depth predicts settling in this search; plus "
                      "large-garment P_B supervision; decide under v5's unchanged matched rule"),
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "runner_provenance": ("v6's runner snapshot (itself byte-identical to v4/v5) plus passive recovery-branch "
                              "telemetry (recovery.json schema 2: margin/gripper/lift traces, terminal_*); no "
                              "physics, render or RNG change; pinned in executed_sources"),
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": {"path": str(demos_path), "sha256": file_sha(demos_path)},
        "baseline_checkpoint": {"path": str(baseline), "label": "baseline", "immutable": True,
                                "sha256": {p.name: file_sha(p) for p in sorted(baseline.iterdir()) if p.is_file()}},
        "pose_clusters": clusters,
        "depth_rules": DEPTH_RULES,
        "depth_rules_note": ("preregistered from the v6 diagnosis (landed >= 1 cm settled 37/41 vs 13/25 below "
                             "across 128 matched episodes) and the pre-launch review. Mechanism check: among "
                             "reached branches that landed inside the branch (landings inherited from a folded "
                             "root excluded), landed-deep must settle more often -- exact conditional test "
                             "stratified by pose, p < 0.05, pooled deep rate above shallow -- or nothing is "
                             "trained. Label rule per pose: settled, first all-4 action <= 450 (1-based), terminal "
                             "min(c1,c2) >= the strictest cut in cut_ladder_cm at which the pose meets the gate "
                             "(0.0 = settled-only; settled baseline P_A folds never exceeded ~1.47 cm, so a single "
                             "1.5 cm cut would stop P_A for supply). Landing: first >= 10-step window at/after the "
                             "first all-4 step with both gripper link origins > 12 cm from the cloth and max "
                             "particle lift < 5 cm."),
        "pose_clusters_note": ("grouped by match_pose identity, never by pose key: key 0 is P_A on garments "
                               "0/7/9 but P_C on garment 3. P_A = dev00/04/06, P_B = dev01/03, P_C = dev02/05/07. "
                               "7 of the 8 final rows are poses absent from dev and from every search."),
        "recovery": {
            "roots": ROOTS,
            "candidates": [{"horizon": 10, "seed": s} for s in range(9301, 9309)],
            "obs_every": 10, "max_seconds": 8100,
            "why": ("v4/v6 search machinery unchanged (baseline as student and brancher, every candidate "
                    "executed to a settled verdict, every attempt recorded), plus passive per-step branch "
                    "telemetry (margins, gripper distances, lift) so landing depth can select labels."),
            "coverage_gate": {p: {"min_successful_branches": 24, "min_distinct_roots": 8, "min_rows": 3}
                              for p in ("P_A", "P_B", "P_C")},
            "coverage_gate_note": "counts DEPTH-QUALIFYING branches only, per pose; plus depth_rules.min_total_labels",
        },
        "recovery_search": search,
        "recovery_smoke": [smoke_row],
        "recovery_smoke_config": {"roots": [60], "candidates": [{"horizon": 10, "seed": 9301},
                                                                {"horizon": 10, "seed": 9302}],
                                  "obs_every": 10, "max_seconds": 1200},
        "corpus": {"reuse": None,
                   "note": "no earlier corpus is reused: its landing depth cannot be recovered",
                   "balance": "awr_weight = N / (P * n_pose); the trainer samples proportional to it (unchanged)"},
        "training": dict(
            {k: v for k, v in v4_train.items() if k not in ("held_fixed_vs_v2", "weighting")},
            lr=3.3e-6, fit_precondition=True,
            recipe_source="v4 attempt 2 (plans/attempt2.json over v4's manifest training block), verbatim",
            changed_factor=("label criterion only: depth-qualified branches from a fresh search (with "
                            "large-garment P_B rows); recipe, init (untouched baseline), steps, checkpoint rule, "
                            "retention guard and pose balance identical to v6"),
            weighting="pose-balanced; see corpus.balance"),
        "protocol": {
            "rows": "v2's 16 development rows (8 poses x H10/H50, seeds 200-207), verbatim",
            "runs_per_policy_per_horizon": 2, "run_labels": ["r1", "r2"],
            "same_wave": ("each run is ONE Slurm array of 32 tasks in which even indices run the baseline and "
                          "odd the candidate on the same row, throttled at 8 concurrent; r1 and r2 are "
                          "submitted by the same driver tick (v5's protocol)"),
            "improvement_rule": {**v5["protocol"]["improvement_rule"],
                                 "retention_guard": "heldout_gate of the selected checkpoint from this campaign's training",
                                 "reload": "this campaign's audit/reload_<label>.json: finite and config_matches_baseline",
                                 "function": "scripts/driver.py::matched_verdict (unchanged from v5/v6)"},
            "n_is_fixed": "two runs per policy per horizon, whatever they show; no third run, no second attempt",
            "final_set": ("v3's untouched 8 rows (v5's final_test, verbatim), run once for BOTH policies in one "
                          "matched array, only if the improvement rule holds; descriptive"),
            "baseline_numbers_from_earlier_campaigns": "not pooled in; v7's own baseline runs are the only baseline",
        },
        "media": {"gif_every": 12},
        "budget": {"gpu_tasks": 260, "gpu_hours": 70.0, "reserve": {"final_set": 16},
                   "expected": ("smoke 1 + search 28 (~20 GPU-h) + train/reload/fit 3 (~1) + evaluation 64 (~5) "
                                "+ final 16 (~1.2)"),
                   "note": "anti-runaway bounds; the user lifted the GPU-hour constraint on 2026-09-24",
                   "rollout_timeout_seconds": {"benchmark": 1500, "recovery": 9600, "smoke": 1500}},
        "smoke": v5["smoke"],
        "benchmark": dev,
        "final_test": final,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "search_rows": len(search),
                      **{p: sum(r["pose"] == p for r in search) for p in ("P_A", "P_B", "P_C")},
                      "off_metadata": sum(r["off_metadata_pose"] for r in search)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
