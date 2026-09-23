"""Freeze the closed-loop iteration before a single GPU task is submitted.

Everything that could later be chosen to flatter a result is decided here:
the compute budget, the collection distribution, the AWR constants, the
degeneracy thresholds, the checkpoint-selection rule and the evaluation
protocol. The manifest is written once and refuses to overwrite itself.

Source hashes are taken from the COMMITTED blob (``git show <commit>:<path>``),
not from the working tree, and the builder refuses to run while any hashed path
is dirty. That is the difference between "these are the files I happened to
have" and "this campaign can be reconstructed from this commit" -- and it is
the specific failure the horizon pilot hit, where its frozen manifest recorded
hashes that no longer matched either HEAD or the working tree.
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
DATA = WORKSPACE / "lehome-data"

# Every file this campaign actually executes, repo-relative. The runner and its
# import graph live in the frozen pilot; this campaign does not re-vendor 3300
# lines of controller, it pins the revision it calls.
RUNNER_KEY = "campaigns/20260921-horizon-pilot/scripts/render/policy_rollout51.py"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def committed_blob_sha(commit: str, rel: str) -> str:
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"],
                          check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def executed_sources() -> list[str]:
    """The campaign's own scripts, plus the pilot code they invoke."""
    rels = []
    for directory in ("scripts", "slurm", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            # .sh matters: slurm/_common.sh carries the actual apptainer
            # invocation, bind mounts and PYTHONPATH. Leaving it unhashed
            # would let the execution environment change without the manifest
            # noticing -- which is most of what a source pin is for.
            if path.is_file() and path.suffix in (".py", ".sbatch", ".sh"):
                rels.append(str(path.relative_to(REPO)))
    for path in sorted((PILOT / "scripts" / "render").glob("*.py")):
        rels.append(str(path.relative_to(REPO)))
    for path in sorted((PILOT / "src" / "lehome_fold").glob("*.py")):
        rels.append(str(path.relative_to(REPO)))
    for path in sorted((REPO / "src" / "lehome_fold").glob("*.py")):
        rels.append(str(path.relative_to(REPO)))
    return sorted(set(rels))


def pose_row(demos: dict, garment: str, local_key: int, seed: int, steps: int,
             tag: str, inventory: dict, **extra) -> dict:
    demo = demos[garment][str(local_key)]
    if len(set(demo["scale"])) != 1:
        raise ValueError(f"{garment} demo {local_key} has anisotropic scale")
    row = {
        "id": tag,
        "garment": garment,
        "asset_config": inventory[garment]["config"],
        "pose_key": local_key,
        "match_pose": ":".join(format(v, ".10g") for v in demo["object_initial_pose"]),
        "match_scale": demo["scale"][0],
        "seed": seed,
        "steps": steps,
    }
    row.update(extra)
    return row


def main() -> int:
    out = ROOT / "manifest.json"
    if out.exists():
        raise SystemExit("refusing to mutate an existing frozen manifest")

    commit = git("rev-parse", "HEAD")
    sources = executed_sources()

    # Refuse to freeze a manifest against a dirty tree: a hash taken from a
    # commit is only meaningful if the file on disk is that commit's content.
    dirty = [line[3:] for line in git("status", "--porcelain").splitlines()]
    conflicted = sorted(set(dirty) & set(sources))
    if conflicted:
        raise SystemExit("executed sources are uncommitted:\n  " + "\n  ".join(conflicted))

    executed = {}
    for rel in sources:
        committed = committed_blob_sha(commit, rel)
        live = file_sha(REPO / rel)
        if committed != live:
            raise SystemExit(f"{rel}: working tree {live} != committed {committed}")
        executed[rel] = committed

    inventory = {g["garment_id"]: g
                 for g in json.loads((PILOT / "audit/garment_inventory.json").read_text())["garments"]}
    demos_path = DATA / "Datasets/example/four_types_merged/meta/garment_info.json"
    demos = json.loads(demos_path.read_text())
    ckpt = DATA / "outputs/train/bc_smolvla_raster_ft_full"

    # ---------------------------------------------------------- collection
    # Four garment classes x two real demonstration poses. Fixed before any
    # outcome is seen, and every trajectory is retained whatever it scores --
    # AWR needs the failures, and seed selection after the fact would be
    # exactly the cherry-picking this protocol exists to prevent.
    collection_garments = ("Pant_Short_Seen_0", "Pant_Long_Seen_0",
                           "Top_Short_Seen_0", "Top_Long_Seen_0")
    collection, slot = [], 0
    for garment in collection_garments:
        for pose in (0, 1):
            collection.append(pose_row(demos, garment, pose, 4200 + slot, 600,
                                       f"collect_{garment.lower()}_pose{pose}", inventory,
                                       trajectory_every=5))
            slot += 1

    # Preregistered expansion. Declared NOW so that, if the first collection's
    # learning signal is degenerate, the response is a fixed second draw rather
    # than a set of seeds chosen once the first outcomes are known.
    expansion, slot = [], 0
    for garment in collection_garments:
        for pose in (2, 3):
            expansion.append(pose_row(demos, garment, pose, 4300 + slot, 600,
                                      f"expand_{garment.lower()}_pose{pose}", inventory,
                                      trajectory_every=5))
            slot += 1

    # ---------------------------------------------------------- evaluation
    # Development set: the same eight poses and seeds the horizon pilot used,
    # so its baseline measurement applies. These are DEVELOPMENT data -- the
    # failure they exhibit is what motivated this campaign, so a gain on them
    # is not an unbiased estimate of anything.
    dev, slot = [], 0
    for garment in ("Pant_Short_Seen_0", "Pant_Short_Seen_3",
                    "Pant_Short_Seen_7", "Pant_Short_Seen_9"):
        for pose in (0, 1):
            for horizon in (10, 50):
                dev.append(pose_row(demos, garment, pose, 200 + slot, 600,
                                    f"dev{slot:02d}_h{horizon:02d}_{garment}", inventory,
                                    horizon=horizon, development_pose_slot=slot))
            slot += 1

    # Frozen test set: garments that appear in NO collection row and NO
    # development row, fixed here and excluded from every training and
    # selection decision. Two per class so a single class cannot carry it.
    test_garments = ("Pant_Short_Seen_1", "Pant_Short_Seen_2",
                     "Pant_Long_Seen_1", "Pant_Long_Seen_2",
                     "Top_Short_Seen_1", "Top_Short_Seen_2",
                     "Top_Long_Seen_1", "Top_Long_Seen_2")
    reserved = {r["garment"] for r in collection} | {r["garment"] for r in dev} \
        | {r["garment"] for r in expansion}
    leaked = sorted(set(test_garments) & reserved)
    if leaked:
        raise SystemExit(f"frozen test garments appear in training/development: {leaked}")
    test = [pose_row(demos, garment, 0, 9000 + i, 600, f"test{i:02d}_h10_{garment}",
                     inventory, horizon=10)
            for i, garment in enumerate(test_garments)]

    smoke = [pose_row(demos, "Pant_Short_Seen_0", 0, 3100, 80,
                      "smoke_pant_short_seen0_pose0", inventory, trajectory_every=2)]

    manifest = {
        "schema_version": 1,
        "campaign": "20260922-closed-loop-training-v2",
        "supersedes": "20260922-closed-loop-training-v1 (pre-review draft, never launched)",
        "objective": (
            "One bounded rollout-driven AWR iteration from the untouched BC baseline, "
            "measured on closed-loop folding rather than offline loss."),
        "git": {
            "commit": commit,
            "repository": "https://github.com/joses2017smjh/IsaacSimFolding.git",
            "source_hashes_taken_from": "committed blob at this commit, verified equal to the working tree",
        },
        "runner_key": RUNNER_KEY,
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": {"path": str(demos_path), "sha256": file_sha(demos_path)},
        "garment_inventory_sha256": file_sha(PILOT / "audit/garment_inventory.json"),
        "baseline_checkpoint": {
            "path": str(ckpt),
            "immutable": True,
            "sha256": {p.name: file_sha(p) for p in sorted(ckpt.iterdir()) if p.is_file()},
        },
        "protocol": {
            "execution_horizon": 10,
            "prediction_chunk_size": 50,
            "initial_settle_steps": 60,
            "terminal_settle_steps": 60,
            "action_budget": 600,
            "reward": "official terminal geometric conditions_passed / conditions_total",
            "advantage": "terminal reward minus the fixed-collection mean (lehome_fold.awr.success_residual)",
            "objective": ("AWR importance RESAMPLING of rollout frames plus an unchanged "
                          "supervised loss, because SmolVLA's flow-matching forward exposes "
                          "one scalar batch loss and no per-example likelihood"),
            "recap_conditioning": ("NOT used. lehome_fold.recap has no implemented gradient "
                                   "update in this repository and no campaign has ever run one."),
            "h50_labels": "prohibited; the pilot's fresh-H50 oracle recovered 0/6 student-visited roots",
        },
        "awr": {"beta": 1.0, "w_max": 3.0, "w_min": 1e-6,
                "w_max_rationale": (
                    "advantages are standardised, so the exponent is a z-score and the cap "
                    "binds where z > ln(w_max) = 1.0986. An 8-episode batch has a z-range of "
                    "roughly +/-2, so w_max=3 is inside it and the cap is a live parameter. "
                    "The previous default of 20 (ln 3.0) would never have bound.")},
        "gates": {
            "distinct_rewards_min": 3,
            "advantage_std_min": 0.02,
            "ess_min": 3.0,
            "sample_ess_fraction_max": 0.95,
            "rationale": ("ESS is MAXIMISED by uniform weights, so a high ESS is equally "
                          "consistent with a strong signal and with no signal at all. The "
                          "gate is therefore two-sided: ESS must be below n (non-uniform) "
                          "and above a floor (not collapsed onto one episode)."),
            "on_failure": ("run the preregistered collection_expansion once, recompile over "
                           "both draws, and if the gate still fails STOP and report the "
                           "unmet requirement rather than training on uniform weights"),
        },
        "training": {
            "steps": 300, "batch_size": 4, "rollout_fraction": 0.5, "lr": 1e-5,
            "optimizer": "AdamW", "grad_clip": 1.0, "unfreeze": "boundary",
            "seed": 4242, "checkpoint_every": 100,
            "anchor": {"glob": str(DATA / "storm_capture/ep*.npz"), "files": 16, "frames": 8},
            "heldout": {"glob": str(DATA / "storm_capture100/ep*.npz"), "files": 4, "frames": 8,
                        "tolerance": 1.10,
                        "role": "raster retention regression guard, never an optimisation target"},
        },
        "selection": {
            "rule": ("Evaluate step_000300 on the full development set. Fall back to "
                     "step_000200 ONLY if step_000300 breaches the raster retention guard. "
                     "One candidate is evaluated; this is fixed before training."),
            "criterion": "terminal settled geometric success at H10 on the development set",
            "tie_break": "higher mean terminal conditions_passed, then lower raster held-out loss",
        },
        "targets": {
            "primary": "development H10 terminal settled success >= 4/8",
            "stretch": "development H10 >= 6/8 with no H50 regression below the baseline 4/8",
            "confirmation": ("the frozen test set, 8 garments excluded from collection, "
                             "development and selection, run for baseline and candidate"),
            "baseline_development": {"h10": "0/8", "h50": "4/8",
                                     "source": "campaigns/20260921-horizon-pilot pilot-results",
                                     "reuse_justification": (
                                         "identical garments, poses, seeds 200-207, checkpoint SHA, "
                                         "600/60/60 protocol and runner. The only runner change since "
                                         "is trajectory capture, which is inert unless --trajectory_out "
                                         "is passed and is not passed for benchmark rows.")},
        },
        "budget": {
            "max_gpu_tasks_iteration1": 44,
            "breakdown": {"smoke": 1, "collect": 8, "train": 1,
                          "development_candidate": 16, "frozen_test": 16, "boundary": 1,
                          "compile_cpu": 1},
            "expansion_gpu_tasks": 8,
            "iteration2_gpu_tasks": 30,
            "iteration2_condition": ("only if iteration 1 improves development closed-loop "
                                     "success AND passes the raster retention guard"),
            "rollout_timeout_seconds": {"smoke": 900, "collection": 1500,
                                        "benchmark": 1500, "boundary": 3000},
            "timeout_rationale": ("the pilot measured 315 s mean wall time for a 600-action H10 "
                                  "episode and 326 s at H50; 1500 s is roughly 4x headroom and "
                                  "still bounded well inside the 30-minute allocation"),
        },
        "smoke": smoke,
        "collection": collection,
        "collection_expansion": expansion,
        "benchmark": dev,
        "frozen_test": test,
        "retention": ("every rollout is kept, successful or not; no seed, pose or episode "
                      "is dropped after its outcome is known"),
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(out), "commit": commit, "executed_sources": len(executed),
        "smoke": len(smoke), "collection": len(collection),
        "expansion": len(expansion), "development": len(dev), "frozen_test": len(test),
        "baseline_model_sha256": manifest["baseline_checkpoint"]["sha256"]["model.safetensors"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
