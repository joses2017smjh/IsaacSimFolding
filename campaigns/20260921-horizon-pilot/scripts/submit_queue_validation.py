#!/usr/bin/env python3
"""Submit only the conditional pose-1/3/5/7 validation after the gate passes."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "analysis" / "queue-continuity" / "validation-gate.json"
RECEIPT = ROOT / "analysis" / "queue-continuity" / "validation-submission.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    gate = json.loads(GATE.read_text())
    delay = gate.get("smallest_passing_delay")
    base = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "selected_delay": delay, "poses": [1, 3, 5, 7],
            "submitted": False, "existing_jobs_modified": False}
    if not gate.get("queue_gate_pass") or delay is None:
        base["reason"] = "pose-3 queue gate did not pass; recommend observation-component ablation"
        RECEIPT.write_text(json.dumps(base, indent=2) + "\n")
        print(json.dumps(base, indent=2))
        return 0
    command = ["sbatch", "--parsable", "--job-name=lh-queue-val", "--array=0-3%1",
               f"--output={ROOT}/logs/queue-validation-%A_%a.out",
               f"--error={ROOT}/logs/queue-validation-%A_%a.err",
               str(ROOT / "slurm/queue_validation.sbatch"), str(ROOT), str(delay)]
    base["command"] = command
    if not args.submit:
        base["reason"] = "dry run; pass --submit to launch the four conditional validation poses"
    else:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"unexpected sbatch receipt: {result.stdout!r}")
        base.update(submitted=True, job_id=job_id, stdout=result.stdout.strip())
    RECEIPT.write_text(json.dumps(base, indent=2) + "\n")
    print(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
