"""Freeze the v9 re-measurement before any GPU task runs.

Same discipline as v2-v8:
- every executed source is hashed from the COMMITTED blob;
- a dirty tree is refused;
- overwriting a frozen manifest is refused.

v9 trains nothing and chooses nothing. It measures two byte-pinned policies:
- the untouched baseline;
- v8's an1-step000300, checked here file by file against v8's candidate record.

It runs them on 48 fresh-seed development rows, in two platform blocks each
pinned to one GPU model:
- a40 is primary (the approved design);
- rtx8000 is a secondary replication.
The seeds are checked unused against every earlier campaign's manifests,
plans, per-episode request/status records, analysis JSON and '--seed'
command arguments. The behaviour detectors are calibrated on the 128 v5-v8
H50 development episodes at freeze time, and that calibration is recorded
here, along with the per-model power of the H10 clause.
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

LEHOME = PILOT / "external/lehome-challenge"
LEROBOT = REPO.parent / "lehome51-site/lerobot"
PRIMARY_BLOCK = "a40"

sys.path.insert(0, str(ROOT / "scripts"))
import analysis  # noqa: E402
import endpoints  # noqa: E402
from verify_sources import tree_digest  # noqa: E402


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
    # The LeHome checkout on PYTHONPATH (scorer, env, garment loader): every tracked .py.
    rels += [line for line in git("ls-files", "--", str(LEHOME.relative_to(REPO))).splitlines()
             if line.endswith(".py")]
    return sorted(set(rels))


SEED_SCAN_MAX_BYTES = 8 << 20


def seed_scan_files() -> list[Path]:
    """Manifests, plans, per-episode request/status records and analysis JSON of
    every earlier campaign (files over 8 MB -- rollout geometry dumps -- are skipped
    and counted)."""
    pats = ("*/manifest.json", "*/manifests/*.json", "*/plans/*.json", "*/**/request.json",
            "*/**/status.json", "*/analysis/**/*.json", "*/audit/**/*.json", "*/ledger/*.json")
    files = set()
    for pat in pats:
        files |= {p for p in CAMPAIGNS.glob(pat) if ROOT not in p.parents and p.is_file()}
    return sorted(files)


def used_seeds(stats: dict | None = None) -> dict[int, list[str]]:
    """Every integer under a seed-like key, and every '--seed N' in a command list."""
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
            for i, v in enumerate(x):
                if v == "--seed" and i + 1 < len(x) and str(x[i + 1]).lstrip("-").isdigit():
                    seeds[int(x[i + 1])].add(src)
                walk(v, src)

    scanned = skipped = 0
    for path in seed_scan_files():
        if path.stat().st_size > SEED_SCAN_MAX_BYTES:
            skipped += 1
            continue
        try:
            walk(json.loads(path.read_text()), str(path.relative_to(CAMPAIGNS).parts[0]))
            scanned += 1
        except (ValueError, UnicodeDecodeError):
            skipped += 1
    if stats is not None:
        stats.update(files_scanned=scanned, files_skipped=skipped)
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


def baseline_h10_history() -> dict[str, list[int]]:
    """The matched baseline's H10 settled/n per GPU model over v5-v8 (rollout.log device tables)."""
    tab = collections.defaultdict(lambda: [0, 0])
    for camp in HISTORY.values():
        for d in (CAMPAIGNS / camp / "evaluation").glob("baseline-r?/benchmark/dev0?_h10_*"):
            model = analysis.gpu_model(d / "rollout.log")
            r = json.loads((d / "rollout.json").read_text())
            tab[model][0] += int(bool(r.get("terminal_success")))
            tab[model][1] += 1
    return {k: v for k, v in sorted(tab.items())}


def power_table() -> dict:
    """Per-model power of the H10 clause at 24 vs 24, p0 from the matched baseline's
    v5-v8 H10 history on that model, for three plausible candidate rates."""
    hist = baseline_h10_history()
    pooled = [sum(v[0] for v in hist.values()), sum(v[1] for v in hist.values())]
    p0 = {"a40": hist["NVIDIA A40"], "rtx8000": hist["Quadro RTX 8000"], "pooled": pooled}
    table = {k: {"baseline_history": f"{v[0]}/{v[1]}", "p0": round(v[0] / v[1], 3),
                 "power": {str(p1): round(analysis.clause_power(v[0] / v[1], p1), 3) for p1 in (0.35, 0.45, 0.54)}}
             for k, v in p0.items()}
    minimum = {str(b): next(c for c in range(25) if analysis.h10_clause(c, 24, b, 24)) for b in range(8)}
    headline = (f"power at p1=0.45: a40 {table['a40']['power']['0.45']:.0%}, "
                f"rtx8000 {table['rtx8000']['power']['0.45']:.0%}")
    return {"n": 24, "table": table, "minimum_candidate_to_pass_by_baseline": minimum, "headline": headline,
            "note": ("exact, independent episodes; at n=24 one-sided Fisher p < 0.05 already needs a margin of at "
                     "least 5, so 'margin >= 4' does not bind")}


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
    scan = {}
    seen = used_seeds(scan)
    collide = sorted({r["seed"] for r in rows} & set(seen))
    if collide:
        raise SystemExit(f"seeds already used: {collide} by {[seen[s] for s in collide]}")
    if len({r["id"] for r in rows}) != 48 or len({r["seed"] for r in rows}) != 24:
        raise SystemExit("rows must be 48 distinct ids over 24 distinct seeds")

    blocks = [{"key": "a40", "constraint": "a40", "gpu_name": analysis.GPU_NAMES["a40"], "role": "primary"},
              {"key": "rtx8000", "constraint": "rtx8000", "gpu_name": analysis.GPU_NAMES["rtx8000"],
               "role": "secondary"}]
    power = power_table()
    lerobot_sha, lerobot_files = tree_digest(LEROBOT, "*.py")

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
                  "fresh_against": ("every seed-like key and '--seed N' argument in every earlier campaign's "
                                    "manifests, plans, request/status records, analysis, audit and ledger JSON"),
                  "scan": dict(scan, distinct_prior_seeds=len(seen), max_prior_seed=max(seen))},
        "environment_digests": {"lehome51-site/lerobot": {"path": str(LEROBOT), "glob": "*.py",
                                                          "files": lerobot_files, "sha256": lerobot_sha}},
        "protocol": {
            "primary_block": PRIMARY_BLOCK,
            "design": ("two platform blocks, each ONE interleaved Slurm array (even index = baseline, odd = candidate "
                       "on row index // 2) pinned to one GPU model by --constraint; the same 48 rows and seeds in "
                       "both. a40 is PRIMARY (the user-approved design); rtx8000 is a SECONDARY replication"),
            "why_two_blocks": ("the approved design pinned A40; at launch every public A40 node was drained for "
                               "maintenance and both RTX 8000 nodes were full, so an RTX 8000 block was added under "
                               "the user's standing lift of the GPU-hour constraint: it can start before the A40 "
                               "maintenance ends. The two blocks compare hardware PLATFORMS (GPU model plus its node "
                               "pool: CPU, ISA, driver), unpaired and balanced on rows and seeds; driver version and "
                               "host are reported per block"),
            "per_block_reading": ("each complete block is a reading in its own right: matched_verdict's clauses on 24 "
                                  "vs 24 per horizon -- H10 margin >= 4 AND one-sided Fisher p < 0.05; H50 candidate "
                                  ">= baseline; all 96 episodes valid on the pinned model under one manifest digest. "
                                  "Retention guard and reload are v8's (same bytes) and not re-run. The a40 reading "
                                  "is the headline"),
            "cross_platform_label": ("'replicates on both platforms' iff the H10 clause holds in both complete blocks; "
                                     "'H10 clause holds on X, not on Y' iff in exactly one, with hardware dependence "
                                     "claimed ONLY if the preregistered interaction test gives p < 0.05, else "
                                     "'interaction not established' (which is not evidence of no hardware effect; a "
                                     "platform-dependent gain can also come from the baseline's own hardware "
                                     "sensitivity); 'not replicated at this power (...)' iff in neither; 'not "
                                     "available' if a block is incomplete"),
            "interaction_test": ("exact two-sided sign-flip over the 8 development rows of (cand - base settled, "
                                 "summed over the row's 3 seeds) on a40 minus the same on rtx8000; hardware "
                                 "dependence iff p < 0.05"),
            "familywise": "P(the H10 clause passes in at least one block | no effect) is about 5% (per block ~2-3%)",
            "power": power,
            "dev01_h50_rule": ("descriptive, not gated: the dev01 H50 loss 'recurs' in a complete block iff baseline "
                               "dev01 H50 settled > candidate (3 vs 3); pooled over complete blocks with a one-sided "
                               "Fisher p (baseline better); 3 vs 3 per block cannot reach significance"),
            "secondary_reported_not_gated": [
                "row-stratified (8 development slots) and cluster-stratified (P_A/P_B/P_C) exact tests, row "
                "sign-flip test and paired (row, seed) exact McNemar per horizon per block, each with n_paired",
                "block x row stratified exact test over both blocks (descriptive; cannot override the per-block "
                "readings)",
                "per pose-cluster settled/reached/n; reach from the per-step geometric trace",
                "P_B H50 per row (dev01, dev03)",
                "E1 right-only chunk-1 opening and E2 dev03 lift by action 150 (v8's endpoints.py)",
                "drag and early-bimanual-lift detectors per H50 row (scripts/analysis.py)",
                "terminal margins per episode; settled by platform per policy and row (same seeds, unpaired); "
                "driver/kernel/host per block",
            ],
            "multiplicity": "secondary p-values are unadjusted and descriptive",
            "naming": "row_* tests are stratified by development slot, not comparable with v8's 'by pose 0.011'",
            "deliverable": "the baseline, whatever the result: v9 is measurement only",
            "final_set": "untouched",
            "retry_states": ("a task is retried once, outcome-blind, iff its status is not 'completed': no status or "
                             "rollout (killed, node failure), infrastructure_error (wrong GPU model, runner exit, "
                             "validation failure such as a non-finite simulator state), infrastructure_timeout. "
                             "Every retried directory and its state is recorded. If the retry also fails, that "
                             "episode is invalid and its block incomplete (historical rate: 0 of 256 v5-v8 "
                             "episodes failed validation)"),
            "stop_rule": ("one array per block and the single retry above; no extra seeds, no re-running valid rows, "
                          "no other checkpoint, no rule change after launch; no manual cancellation of a block -- "
                          "any cancellation other than at the deadline is logged as a protocol deviation"),
            "block_deadline_utc": BLOCK_DEADLINE_UTC,
            "deadline_rule": ("enforced on each episode's completed_utc: an episode completed after the deadline is "
                              "invalid; a block still running at the deadline has its jobs cancelled and is reported "
                              "as incomplete (excluded from the cross-platform label)"),
            "provenance_rule": ("every task verifies the manifest digest the driver submitted under, every executed "
                                "source and all 8 pinned files of its checkpoint before Isaac starts; a block whose "
                                "episodes carry more than one manifest digest is incomplete"),
            "concurrency_and_cost": ("QoS caps: gpu partition 5 concurrent tasks (cpu=40 at 8 CPUs each), ampere 2 "
                                     "GPUs; expected ~5.7 GPU-h (a40) + ~7.8 GPU-h (rtx8000) from v8 timings. The "
                                     "user's own queued A40 jobs from other projects share the ampere cap and are "
                                     "left untouched (standing instruction)"),
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
            "drag": " ".join(analysis.__doc__.split("  drag ", 1)[1].split("  EBL")[0].split()),
            "ebl": " ".join(analysis.__doc__.split("  EBL  ", 1)[1].split()),
            "calibration_scope": "the 128 v5-v8 H50 development episodes (baseline 64; a2, pb1, dq1, an1 16 each)",
            "constants": {"table_z_m": analysis.TABLE_Z, "parked_x_m": analysis.PARKED_X,
                          "drag_window": analysis.DRAG_WINDOW, "drag_lift_m": analysis.DRAG_LIFT_M,
                          "ebl_pair_actions": analysis.EBL_PAIR, "ebl_early_actions": analysis.EBL_EARLY,
                          "ebl_window": analysis.EBL_WINDOW, "ebl_lift_m": analysis.EBL_LIFT_M},
            "calibration_v5_v8_h50": calibration(),
        },
        "media": v8["media"],
        "budget": {"gpu_tasks": 400, "gpu_hours": 40.0,
                   "expected": "2 blocks x 96 episodes (~5.7 + ~7.8 GPU-h) + at most one retry of failed tasks",
                   "note": "anti-runaway bounds; the user lifted the GPU-hour constraint on 2026-09-24",
                   "rollout_timeout_seconds": {"remeasure": 1500}},
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"commit": commit, "executed_sources": len(executed), "rows": len(rows),
                      "seed_scan": manifest["seeds"]["scan"], "power": power["headline"],
                      "lerobot_files": lerobot_files}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
