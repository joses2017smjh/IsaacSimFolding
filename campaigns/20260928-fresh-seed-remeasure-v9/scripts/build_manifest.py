"""Freeze the v9 re-measurement before any GPU task runs.

Same discipline as v2-v8:
- every executed source is hashed from the COMMITTED blob;
- a dirty tree is refused;
- overwriting a frozen manifest is refused.

v9 trains nothing and chooses nothing. It measures two byte-pinned policies:
- the untouched baseline;
- v8's an1-step000300, checked here file by file against v8's candidate record.

It runs them on 48 fresh-seed development rows, in two blocks each pinned to
one GPU model. The seeds are proven unused by every earlier campaign's
manifests and plans. The behaviour detectors are calibrated on the v5-v8
history at freeze time, and that calibration is recorded here.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CAMPAIGNS = ROOT.parent
PILOT = CAMPAIGNS / "20260921-horizon-pilot"
V8 = CAMPAIGNS / "20260926-anchor-diagnostic-v8"
HISTORY = {"v5": "20260924-matched-comparison-v5", "v6": "20260924-pose-balanced-v6",
           "v7": "20260925-depth-qualified-v7", "v8": "20260926-anchor-diagnostic-v8"}
RUNNER_KEY = str((ROOT / "scripts/render/policy_rollout51.py").relative_to(REPO))
SEED_BASE = 97000
SEED_BLOCKS = 3
BLOCK_DEADLINE_UTC = "2026-10-06T00:00:00Z"

sys.path.insert(0, str(ROOT / "scripts"))
import analysis  # noqa: E402
import endpoints  # noqa: E402


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
            if path.is_file() and path.suffix in (".py", ".sbatch", ".sh") and "__pycache__" not in path.parts:
                rels.append(str(path.relative_to(REPO)))
    for base in (PILOT / "src" / "lehome_fold", REPO / "src" / "lehome_fold"):
        rels += [str(p.relative_to(REPO)) for p in sorted(base.glob("*.py"))]
    return sorted(set(rels))


def used_seeds() -> dict[int, list[str]]:
    """Every integer under a seed-like key in every earlier campaign's manifests and plans."""
    seeds: dict[int, set[str]] = collections.defaultdict(set)

    def walk(x, src):
        if isinstance(x, dict):
            for k, v in x.items():
                if "seed" in k.lower():
                    for s in (v if isinstance(v, list) else [v]):
                        if isinstance(s, int) and not isinstance(s, bool):
                            seeds[s].add(src)
                walk(v, src)
        elif isinstance(x, list):
            for v in x:
                walk(v, src)

    for path in sorted(CAMPAIGNS.glob("*/manifest.json")) + sorted(CAMPAIGNS.glob("*/manifests/*.json")) \
            + sorted(CAMPAIGNS.glob("*/plans/*.json")):
        if path.parent == ROOT or path.parents[1] == ROOT:
            continue
        try:
            walk(json.loads(path.read_text()), str(path.relative_to(CAMPAIGNS)))
        except ValueError:
            continue
    return {s: sorted(v) for s, v in seeds.items()}


def rows_from(dev: list[dict]) -> list[dict]:
    """8 development poses x H10/H50 x 3 fresh seeds; a seed is shared by both
    horizons and both policies of one pose, as seeds 200-207 were on the dev rows."""
    by_key = {r["id"][:9]: r for r in dev}
    rows = []
    for k in range(SEED_BLOCKS):
        for h in (10, 50):
            for i in range(8):
                base = by_key[f"dev{i:02d}_h{h}"]
                seed = SEED_BASE + 100 * k + i
                rows.append(dict(base, id=f"dev{i:02d}_h{h}_s{seed}", seed=seed, seed_block=k,
                                 source_row=base["id"]))
    return rows


def calibration() -> dict:
    """The detectors on every v5-v8 H50 development episode, per policy and row."""
    tab: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for tag, camp in HISTORY.items():
        for d in sorted((CAMPAIGNS / camp / "evaluation").glob("*/benchmark/dev0?_h50_*")):
            label = d.parts[-3]
            policy = "baseline" if label.startswith("baseline") else label.split("-")[0]
            b = analysis.behaviour(endpoints.behavior(d / "rollout.json.behavior.jsonl"))
            row = d.name[:5]
            tab[policy]["episodes"] += 1
            tab[policy]["right_only_chunk1"] += b["chunk1"] == "right"
            if b["drag_actions"]:
                tab[policy][f"drag:{row}"] += 1
            if b["ebl_actions"]:
                tab[policy][f"ebl:{row}"] += 1
    return {p: dict(sorted(c.items())) for p, c in sorted(tab.items())}


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
    if executed[RUNNER_KEY] != json.loads((V8 / "manifest.json").read_text())["executed_sources"][
            str((V8 / "scripts/render/policy_rollout51.py").relative_to(REPO))]:
        raise SystemExit("the runner differs from v8's")

    v8 = json.loads((V8 / "manifest.json").read_text())
    dev = v8["benchmark"]
    if len(dev) != 16:
        raise SystemExit("v8 development rows changed")

    # ---- both policies pinned byte for byte
    record_path = V8 / "audit/candidate_checkpoint.json"
    record = json.loads(record_path.read_text())
    ckpt = Path(record["path"])
    for name, sha in record["sha256"].items():
        if file_sha(ckpt / name) != sha:
            raise SystemExit(f"v8 candidate {name} no longer matches v8's record")
    if file_sha(ROOT / "audit/v8_candidate_checkpoint.json") != file_sha(record_path):
        raise SystemExit("audit/v8_candidate_checkpoint.json is not v8's record")
    v8_state = json.loads((V8 / "ledger/driver_state.json").read_text())
    candidate = {"path": record["path"], "label": record["label"], "step": record["step"],
                 "sha256": record["sha256"], "immutable": True,
                 "source": {"campaign": HISTORY["v8"], "record": str(record_path.relative_to(REPO)),
                            "record_sha256": file_sha(record_path)},
                 "v8_gates": {k: v8_state["attempt"].get(k) for k in
                              ("selection", "guard", "retention_loss", "retention_baseline", "reload_ok", "fit_gate")}}
    baseline = v8["baseline_checkpoint"]
    base_dir = Path(baseline["path"])
    for name, sha in baseline["sha256"].items():
        if file_sha(base_dir / name) != sha:
            raise SystemExit(f"baseline {name} no longer matches its pin")

    # ---- fresh rows
    rows = rows_from(dev)
    seen = used_seeds()
    collide = sorted({r["seed"] for r in rows} & set(seen))
    if collide:
        raise SystemExit(f"seeds already used: {collide} by {[seen[s] for s in collide]}")
    if len({r["id"] for r in rows}) != 48 or len({r["seed"] for r in rows}) != 24:
        raise SystemExit("rows must be 48 distinct ids over 24 distinct seeds")

    blocks = [{"key": "rtx8000", "constraint": "rtx8000", "gpu_name": analysis.GPU_NAMES["rtx8000"]},
              {"key": "a40", "constraint": "a40", "gpu_name": analysis.GPU_NAMES["a40"]}]

    manifest = {
        "schema_version": 1,
        "campaign": ROOT.name,
        "kind": "measurement only: fresh seeds, GPU model pinned per block; no training, no final set",
        "predecessor": {"campaign": HISTORY["v8"],
                        "result": ("dev screen: an1-step000300 H10 9/16 vs 2/16 (p = 0.0117), H50 6/16 vs 8/16 -- "
                                   "not improved; H10 fragile (dev04/dev06 carry it; dev06's gain is A40-only), "
                                   "GPU model an uncontrolled quasi-seed; every campaign reused seeds 200-207"),
                        "report": str((V8 / "REPORT.md").relative_to(REPO)),
                        "diagnosis": str((V8 / "analysis/diagnosis/diagnosis.json").relative_to(REPO))},
        "question": ("do an1's H10 gain and its dev01 H50 loss survive fresh seeds once the GPU model is "
                     "controlled, and does the answer depend on the GPU model?"),
        "authorization": "user, 2026-09-28 ('yes proceed' to the v8 report's recommended next step)",
        "git": {"commit": commit, "source_hashes_taken_from": "committed blob, verified equal to the working tree"},
        "runner_key": RUNNER_KEY,
        "runner_provenance": "v8's runner snapshot (itself v7's), unchanged; hash verified equal to v8's pin",
        "executed_sources": executed,
        "task_prompt": v8["task_prompt"],
        "assets": v8["assets"],
        "lehome": v8["lehome"],
        "pose_metadata": v8["pose_metadata"],
        "pose_clusters": v8["pose_clusters"],
        "baseline_checkpoint": baseline,
        "candidate_checkpoint": candidate,
        "gpu_models": analysis.GPU_NAMES,
        "blocks": blocks,
        "remeasure": rows,
        "seeds": {"base": SEED_BASE, "rule": "seed = 97000 + 100*k + pose index (k = 0..2), shared across "
                  "horizons and policies within a block and across the two blocks",
                  "fresh_against": "every seed-like integer in every earlier campaign's manifests and plans "
                                   f"({len(seen)} distinct)"},
        "protocol": {
            "design": ("two blocks, each ONE interleaved Slurm array (even index = baseline, odd = candidate on "
                       "row index // 2, throttled %8) pinned to one GPU model by --constraint; the same 48 rows "
                       "and seeds in both blocks"),
            "why_two_blocks": ("the approved design pinned A40; at launch every public A40 node was drained for "
                               "maintenance and both RTX 8000 nodes were full, so the RTX 8000 block was added: it "
                               "runs whenever capacity appears, and it measures directly whether the H10 gain "
                               "depends on the GPU model (v8: dev06's gain came from an A40-only baseline failure)"),
            "per_block_primary": ("matched_verdict's clauses on 24 vs 24 per horizon: H10 margin >= 4 AND one-sided "
                                  "Fisher p < 0.05; H50 candidate >= baseline; all 96 episodes valid on the pinned "
                                  "model. Retention guard and reload are v8's (same bytes) and not re-run"),
            "classification": ("H10 gain 'replicates on both GPU models' iff the H10 clause holds in both complete "
                               "blocks; 'replicates on <model> only (hardware-conditional)' iff in exactly one; "
                               "'does not replicate' iff in neither; 'partially measured' if a block is incomplete"),
            "secondary_reported_not_gated": [
                "pose-stratified exact test and pose sign-flip test per horizon per block",
                "paired (row, seed) exact McNemar per horizon per block",
                "block x pose stratified exact test over both blocks",
                "per pose-cluster settled/reached/n; reach from the per-step geometric trace",
                "P_B H50 per row (dev01, dev03)",
                "E1 right-only chunk-1 opening and E2 dev03 lift by action 150 (v8's endpoints.py)",
                "drag and early-bimanual-lift detectors per H50 row (scripts/analysis.py)",
                "terminal margins per episode; settled by GPU model per policy and row (same seeds)",
            ],
            "deliverable": "the baseline, whatever the result: v9 is measurement only",
            "final_set": "untouched",
            "stop_rule": ("one array per block; one outcome-blind resubmission only for tasks that left no "
                          "completed rollout or ran on the wrong model; no extra seeds, no re-running valid rows, "
                          "no other checkpoint, no rule change after launch"),
            "block_deadline_utc": BLOCK_DEADLINE_UTC,
            "deadline_rule": ("a block not finished by the deadline has its outstanding jobs cancelled and is "
                              "reported as incomplete (descriptive only, excluded from the classification)"),
            "cannot_show": ["generalization to held-out poses, garments or meshes (dev poses are corpus poses; nine "
                            "checkpoints were screened on them)",
                            "anchor causation (composition, source and step are confounded)",
                            "an unbiased effect size (an1 was selected on these poses: winner's curse)",
                            "performance of the policy as shipped unless H50 holds (checkpoints ship at H50)"],
            "later_deliverable_change": ("would need a separately preregistered holdout that includes P_B H50 and "
                                         "a declared shipping horizon"),
        },
        "detectors": {
            "script": "scripts/analysis.py",
            "E1_E2": "scripts/endpoints.py (v8's, byte for byte)",
            "drag": analysis.__doc__.split("drag ", 1)[1].split("  EBL")[0].strip(),
            "ebl": analysis.__doc__.split("EBL  ", 1)[1].split('"""')[0].strip(),
            "constants": {"table_z_m": analysis.TABLE_Z, "parked_x_m": analysis.PARKED_X,
                          "drag_window": analysis.DRAG_WINDOW, "drag_lift_m": analysis.DRAG_LIFT_M,
                          "ebl_pair_actions": analysis.EBL_PAIR, "ebl_early_actions": analysis.EBL_EARLY,
                          "ebl_window": analysis.EBL_WINDOW, "ebl_lift_m": analysis.EBL_LIFT_M},
            "calibration_v5_v8_h50": calibration(),
        },
        "media": v8["media"],
        "budget": {"gpu_tasks": 400, "gpu_hours": 40.0,
                   "expected": "2 blocks x 96 episodes (~6 GPU-h each) + at most one retry of failed tasks",
                   "note": "anti-runaway bounds; the user lifted the GPU-hour constraint on 2026-09-24",
                   "rollout_timeout_seconds": {"remeasure": 1500}},
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "rows": len(rows),
                      "seeds_checked_against": len(seen)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
