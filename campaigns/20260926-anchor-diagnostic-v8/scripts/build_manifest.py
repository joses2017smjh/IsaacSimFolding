"""Freeze the v8 anchor diagnostic before any GPU task runs.

Same discipline as v2-v7: every executed source hashed from the COMMITTED
blob, refusal on a dirty tree, refusal to overwrite a frozen manifest. One
variable changes against v7 -- the BC anchor (16 pants demonstrations, same
size and frame schedule, instead of 12 tops + 4 pants) -- chosen by the
verified v7 diagnosis (v7/analysis/diagnosis/diagnosis.json). The v7 corpus
is reused byte for byte; recipe and evaluation are v7's verbatim, plus
preregistered mechanistic endpoints and a confirmatory final-set gate.
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
V7 = ROOT.parent / "20260925-depth-qualified-v7"
DATA = WORKSPACE / "lehome-data"
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))
# Preregistered anchor: the lowest-numbered storm_capture100 episode of each of
# 8 Pant_Short and 8 Pant_Long garments, excluding the retention garments
# (Pant_Long_Seen_3/4), the final-set garments (Pant_Short_Seen_1/2) and the
# anchor-fit metric's files (the sorted glob's first 4).
ANCHOR_EPISODES = [500, 583, 604, 635, 656, 676, 708, 728, 750, 781, 802, 885, 906, 926, 958, 978]
CONFIRM_SEED_BASE = 96000


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
    import numpy as np
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
    v6 = json.loads((V6 / "manifest.json").read_text())
    v7 = json.loads((V7 / "manifest.json").read_text())
    dev, final, clusters = v7["benchmark"], v7["final_test"], v7["pose_clusters"]
    if final != v3["final_test"] or dev != v6["benchmark"]:
        raise SystemExit("development or final rows drifted")
    baseline = DATA / "outputs/train/bc_smolvla_raster_ft_full"

    # ---- the reused corpus, pinned byte for byte
    corpus = ROOT / "datasets/recovery.npz"
    v7_prov = json.loads((V7 / "analysis/recovery-dataset-provenance.json").read_text())
    corpus_sha = file_sha(corpus)
    if corpus_sha != v7_prov["dataset_sha256"] or corpus.resolve() != (V7 / "datasets/recovery.npz").resolve():
        raise SystemExit("datasets/recovery.npz is not v7's corpus")
    if file_sha(ROOT / "analysis/recovery-dataset-provenance.json") != file_sha(V7 / "analysis/recovery-dataset-provenance.json"):
        raise SystemExit("the provenance copy differs from v7's")

    # ---- the one changed variable: the anchor
    retention_garments, anchor_records = set(), []
    for f in v7["training"]["retention"]["files"]:
        with np.load(f, allow_pickle=True) as z:
            retention_garments.add(str(z["garment"]))
    final_garments = {r["garment"] for r in final}
    links = sorted((ROOT / "anchor_pants").glob("ep*.npz"))
    if [int(p.stem[2:]) for p in links] != sorted(ANCHOR_EPISODES):
        raise SystemExit("anchor_pants/ does not hold exactly the preregistered episodes")
    fit_first4 = [str(p) for p in sorted((DATA / "storm_capture100").glob("ep*.npz"))[:4]]
    for link in links:
        target = link.resolve()
        if target.parent != (DATA / "storm_capture100").resolve() or str(target) in fit_first4:
            raise SystemExit(f"{link}: not an eligible storm_capture100 episode")
        with np.load(target, allow_pickle=True) as z:
            garment = str(z["garment"])
            frames = len(z["action"])
        if not garment.startswith("Pant") or garment in retention_garments or garment in final_garments:
            raise SystemExit(f"{link}: {garment} is not an eligible pants garment")
        anchor_records.append({"link": str(link.relative_to(REPO)), "target": str(target), "garment": garment,
                               "frames": frames, "sha256": file_sha(target)})
    if len({r["garment"] for r in anchor_records}) != 16:
        raise SystemExit("the anchor must hold one episode per garment")

    # ---- confirmatory final set: 8 unseen poses x H10/H50 x 3 distinct seeds
    confirm = []
    for k in range(3):
        for h in (10, 50):
            for i, r in enumerate(final):
                seed = CONFIRM_SEED_BASE + 100 * k + i
                confirm.append(dict(r, id=f"confirm_{r['id'][:7]}_h{h}_s{seed}", horizon=h, seed=seed,
                                    source_row=r["id"]))
    used = set()
    for m in (v3, v4, v6, v7):
        for key in ("recovery_search", "benchmark", "final_test", "frozen_test", "recovery_smoke"):
            used |= {r["seed"] for r in m.get(key) or []}
    if {r["seed"] for r in confirm} & used or len({r["id"] for r in confirm}) != 48:
        raise SystemExit("confirmatory seeds collide or ids repeat")

    training = dict(v7["training"])
    training["anchor"] = dict(v7["training"]["anchor"], glob=str(ROOT / "anchor_pants/ep*.npz"),
                              files=16, frames=12, records=anchor_records,
                              composition="16 pants (8 Pant_Short + 8 Pant_Long), one episode per garment")
    training["changed_factor"] = ("the BC anchor: 16 pants demonstrations instead of v7's 12 tops + 4 pants, same size "
                                  "(16 files), frame schedule (whole_episode, 12 frames) and 50% batch fraction; corpus, "
                                  "recipe, init, steps, checkpoint rule and guard are v7's")
    training["second_factor_declared"] = ("the anchor SOURCE also moves, unavoidably: from storm_capture/ (a separate "
                                          "16-episode capture holding only 4 pants episodes) to storm_capture100/ (the "
                                          "baseline raster fine-tune's own 91-episode capture). Shared episodes have "
                                          "identical state and action; later frames differ slightly (mean abs ~0.7-1.3/255, "
                                          "3-4% of pixels > 8). Composition and source cannot be separated in this run.")
    training["v7_anchor"] = "storm_capture/ep*.npz first 16 (12 tops, 4 pants; path-list sha1 57a584de2e)"

    manifest = {
        "schema_version": 1,
        "campaign": "20260926-anchor-diagnostic-v8",
        "kind": "single-variable diagnostic: the BC anchor; a positive claim needs the confirmatory final set",
        "predecessor": {"campaign": "20260925-depth-qualified-v7",
                        "result": ("matched: dq1-step000200 H10 7/16 vs 3/16 (p = 0.126), H50 6/16 vs 7/16 -- not "
                                   "improved; H50 deficit is P_B (0/12 across all fine-tunes vs 9/12), from an "
                                   "early-H50 shift every fine-tune shares"),
                        "final_commit": git("log", "-1", "--format=%H", "--", str((V7 / "REPORT.md").relative_to(REPO)))},
        "analysis": str(V7 / "analysis/diagnosis/diagnosis.json"),
        "question": ("does the shared BC anchor (12/16 tops; every right-only frame-0 target is from a top) drive "
                     "the fine-tunes' first-H50-chunk change (right-only gripper opening, first lift 150-200 "
                     "actions later) and the resulting P_B H50 loss?"),
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "runner_provenance": "v7's runner snapshot, unchanged (no search runs in v8)",
        "executed_sources": executed,
        "task_prompt": "fold the garment on the table",
        "assets": str(DATA / "Assets"),
        "lehome": str(PILOT / "external/lehome-challenge"),
        "pose_metadata": v7["pose_metadata"],
        "baseline_checkpoint": v7["baseline_checkpoint"],
        "pose_clusters": clusters,
        "corpus": {"reuse": {"path": str(corpus), "resolves_to": str(corpus.resolve()), "sha256": corpus_sha,
                             "samples": v7_prov["samples"], "provenance": str(V7 / "analysis/recovery-dataset-provenance.json")},
                   "note": "v7's depth-qualified corpus, byte for byte; no search and no compile in v8"},
        "training": training,
        "endpoints": {
            "script": "scripts/endpoints.py",
            "E1": ("right-only chunk-1 opening per H50 development row: the right gripper target > 0.10 rad at some "
                   "action in 1-50 and the left never (the diagnosed shift)"),
            "E2": "maximum particle lift over actions 1-150 at dev03 H50",
            "prediction": ("SUPPORTED iff the candidate (run r1) is right-only on <= 3 of 8 H50 rows AND lifts dev03 "
                           "above 0.05 m by action 150; REFUTED iff not, with all rows valid and the positive control "
                           "met; INDETERMINATE if any r1 H50 row of either policy is missing/unfinished or the "
                           "baseline in the same arrays does not show the same pattern"),
            "reference": "v7 diagnosis: baseline opens both in 42/48 H50 episodes; fine-tunes right-only 38/48",
            "calibration": ("before launch the rule classifies the baseline as supported and a2, pb1, dq1 as refuted in "
                            "every one of v5, v6, v7 (tests/test_v8.py)"),
            "order": "recorded before the settled-count verdict; they decide the diagnostic question, not the deliverable",
        },
        "protocol": dict(v7["protocol"], screen="the v5 rule on the 16 development rows, 2 runs per policy: a SCREEN only",
                         final_set=("used only as the confirmatory gate (confirmatory_gate / final_confirm); the 8-row "
                                    "descriptive final_test run is not executed in v8"),
                         baseline_numbers_from_earlier_campaigns=("not pooled in; v8's own baseline runs (r1, r2, c1) "
                                                                  "are the only baseline"),
                         confirmatory_gate=("only if the screen passes: the 8 untouched final-set poses (Seen_1/2, mesh "
                                            "PS_049, 7 of 8 poses absent from dev and every search) x H10/H50 x 3 "
                                            "distinct seeds per policy in ONE interleaved array (96 episodes); a "
                                            "positive claim needs the same rule on 24 vs 24 (H10 margin >= 4 and "
                                            "one-sided Fisher p < 0.05, H50 non-regression, all rows valid); spent once"),
                         confirmatory_power=("stated in advance: realistically low (the v7 diagnosis estimates at most "
                                             "~27% given candidates' 0/16 on PS_049 dev rows at H10)"),
                         reported_not_gated=("post hoc, written after three candidates lost P_B H50: whether P_B H50 "
                                             "reaches >= 2/4 and whether any H50 row the baseline settles in both runs "
                                             "drops to 0/2; reach counted from the per-step geometric trace")),
        "media": {"gif_every": 12},
        "budget": {"gpu_tasks": 360, "gpu_hours": 50.0, "reserve": {"final_set": 96},
                   "expected": "train/reload/fit 3 (~1 GPU-h) + evaluation 64 (~5) + confirmatory 96 only if screened in (~7)",
                   "note": "anti-runaway bounds; the user lifted the GPU-hour constraint on 2026-09-24",
                   "rollout_timeout_seconds": {"benchmark": 1500, "smoke": 1500}},
        "smoke": v7["smoke"],
        "benchmark": dev,
        "final_test": final,
        "final_confirm": confirm,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "anchor_files": len(anchor_records),
                      "confirm_rows": len(confirm), "corpus_sha256": corpus_sha[:12]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
