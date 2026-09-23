"""Submit one bounded iteration as a dependency chain, and record every job id.

Enforces the frozen budget before submitting: if the requested tasks would
exceed manifest["budget"], nothing is submitted. Every id lands in
ledger/slurm-jobs.json as it is created, so a chain that dies halfway is still
traceable.

Dependency choice, deliberately not uniform:
  compile  afterany  -- a partial collection SHOULD reach the compiler, which
                        exits 2 with an explicit count. afterok would leave the
                        job silently unscheduled and much harder to diagnose.
  train    afterok   -- training must never start on a failed compile, and the
                        gate file it reads would not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def sbatch(args: list[str]) -> str:
    out = subprocess.run(["sbatch", "--parsable", *args], check=True,
                         capture_output=True, text=True).stdout.strip()
    return out.split(";")[0]


def record(ledger: Path, **row) -> None:
    data = json.loads(ledger.read_text())
    row["submitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row["state"] = "submitted"
    data["jobs"].append(row)
    ledger.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--scope", default="iteration1", choices=("iteration1", "expanded"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    ledger = root / "ledger/slurm-jobs.json"

    phase = "collection" if args.scope == "iteration1" else "collection_expansion"
    script = "collect.sbatch" if args.scope == "iteration1" else "expand.sbatch"
    n_rows = len(manifest[phase])

    planned = n_rows + 2  # rollouts + compile (cpu) + train
    spent = sum(1 for j in json.loads(ledger.read_text())["jobs"]
                if j.get("counts_against_budget", True))
    cap = manifest["budget"]["max_gpu_tasks_iteration1"] + manifest["budget"]["expansion_gpu_tasks"]
    if spent + planned > cap:
        raise SystemExit(f"frozen budget exceeded: {spent} spent + {planned} planned > {cap}")

    chain = [
        ("collect", ["--array", f"0-{n_rows - 1}", str(root / "slurm" / script), str(root)]),
        ("compile", ["--dependency", "afterany:{prev}", str(root / "slurm/compile.sbatch"),
                     str(root), args.scope]),
        ("train", ["--dependency", "afterok:{prev}", str(root / "slurm/train.sbatch"),
                   str(root), args.scope]),
    ]
    if args.dry_run:
        print(json.dumps({"scope": args.scope, "rows": n_rows, "planned_tasks": planned,
                          "budget_spent": spent, "budget_cap": cap,
                          "chain": [c[0] for c in chain]}, indent=2))
        return 0

    prev, submitted = None, {}
    for name, argv in chain:
        argv = [a.replace("{prev}", prev) if prev else a for a in argv]
        job = sbatch(argv)
        submitted[name] = job
        record(ledger, job_id=job, phase=f"{name}:{args.scope}",
               script=argv[-3] if name != "collect" else argv[2],
               manifest_commit=manifest["git"]["commit"],
               depends_on=prev, array=f"0-{n_rows - 1}" if name == "collect" else None)
        prev = job
    print(json.dumps(submitted, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
