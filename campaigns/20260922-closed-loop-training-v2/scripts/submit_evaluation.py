"""Submit the closed-loop evaluation for one checkpoint, under the frozen budget.

Phases, all preregistered in the manifest:
  benchmark   16 development rows, 8 poses x {H10, H50}. Development data.
  frozen_test  8 rows on garments excluded from collection, expansion,
               development and selection. Run for BOTH the baseline and the
               candidate, because no prior measurement of this set exists.
  boundary     the two validated pose-3 replanning roots.

The development baseline is NOT resubmitted: the horizon pilot measured the
same garments, poses, seeds 200-207, checkpoint SHA and 600/60/60 protocol,
and the manifest records why that is reusable. The frozen test baseline has no
such prior and is always run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PHASES = {
    "benchmark": ("benchmark.sbatch", "benchmark"),
    "frozen_test": ("test.sbatch", "frozen_test"),
    "boundary": ("boundary.sbatch", None),
}


def sbatch(args: list[str]) -> str:
    out = subprocess.run(["sbatch", "--parsable", *args], check=True,
                         capture_output=True, text=True).stdout.strip()
    return out.split(";")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--phases", nargs="+", default=["benchmark", "frozen_test"],
                    choices=sorted(PHASES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.campaign.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    ledger = root / "ledger/slurm-jobs.json"

    plan = []
    for phase in args.phases:
        script, rows_key = PHASES[phase]
        argv = [str(root / "slurm" / script), str(root), args.checkpoint, args.label]
        if rows_key:
            n = len(manifest[rows_key])
            argv = ["--array", f"0-{n - 1}"] + argv
        plan.append((phase, argv))

    if args.dry_run:
        print(json.dumps({"label": args.label, "checkpoint": args.checkpoint,
                          "plan": [{"phase": p, "argv": a} for p, a in plan]}, indent=2))
        return 0

    data = json.loads(ledger.read_text())
    submitted = {}
    for phase, argv in plan:
        job = sbatch(argv)
        submitted[phase] = job
        data["jobs"].append({
            "job_id": job, "phase": f"{phase}:{args.label}",
            "script": argv[-3], "checkpoint": args.checkpoint, "label": args.label,
            "manifest_commit": manifest["git"]["commit"],
            "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "submitted"})
        ledger.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(submitted, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
